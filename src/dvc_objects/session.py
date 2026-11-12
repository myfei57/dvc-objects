"""Resumable, object-level transfer sessions between two ObjectDBs.

A :class:`TransferSession` binds a fixed pair of source/destination
:class:`~dvc_objects.db.ObjectDB` instances to a finite set of opaque OIDs and
drives every object through the ``pending -> in_flight -> verified`` state
machine (with ``failed`` as a terminal error state). Every transition is
durably checkpointed in a SQLite ledger so that a rebuilt process only has to
redo work which was never confirmed.

Guarantees:

* OIDs are treated as opaque identifiers. Content is only trusted after an
  explicit :class:`Verifier` check or an independent SHA256 of the transported
  bytes; an OID that happens to look like a hex MD5 is never interpreted.
* Versioned sources bind the exact version/generation captured before the
  copy. Unversioned sources are fingerprinted before and after the transfer
  and a drift is reported explicitly via :class:`SourceChangedError`; no
  snapshot semantics are promised for arbitrary mutable sources.
* Bytes always land in a temporary object first and are only published to the
  final path through the filesystem's declared atomic move capability.
  Filesystems without that capability either reject strict mode or produce a
  explicitly degraded result, and verification of the final path still gates
  the ``verified`` state.
* Checkpoints are written inside SQLite transactions and a single session may
  own a given ledger at a time. Recovery always verifies the real destination
  before retransmitting, so an object published just before a checkpoint
  failure is neither retransmitted nor deleted.
"""

from __future__ import annotations

import ctypes
import enum
import errno
import hashlib
import json
import logging
import os
import queue
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    runtime_checkable,
)

from .errors import (
    AtomicPublishUnsupportedError,
    LedgerMismatchError,
    ObjectDBError,
    ObjectDBPermissionError,
    ObjectFormatError,
    SessionLockedError,
    SourceChangedError,
    TransferCancelledError,
    TransferSessionError,
)

if TYPE_CHECKING:
    from .db import ObjectDB
    from .fs.base import AnyFSPath, FileSystem

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
CHUNK_SIZE = 1024 * 1024
DEFAULT_MAX_INFLIGHT_BYTES = 64 * 1024 * 1024

# Errors which must not be retried: retrying cannot fix missing permissions,
# malformed content, a drifted source or an explicitly absent source object.
_PERMANENT_ERRORS = (
    PermissionError,
    ObjectDBPermissionError,
    ObjectDBError,
    ObjectFormatError,
    SourceChangedError,
)


class Status(str, enum.Enum):
    """Per-object transfer state persisted in the ledger."""

    PENDING = "pending"
    IN_FLIGHT = "in_flight"
    VERIFIED = "verified"
    FAILED = "failed"


class ErrorKind(str, enum.Enum):
    """Classification of a recorded object failure."""

    TRANSIENT = "transient"
    PERMANENT = "permanent"
    SOURCE_CHANGED = "source_changed"
    FATAL = "fatal"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def _hash_file(fs: FileSystem, path: AnyFSPath) -> str:
    hasher = hashlib.sha256()
    with fs.open(path, "rb") as fobj:
        while True:
            chunk = fobj.read(CHUNK_SIZE)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _scalar_info(info: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in sorted(info.items())
        if isinstance(value, (str, int, float, bool)) or value is None
    }


@dataclass(frozen=True)
class SourceIdentity:
    """Obtainable identity of a source object at transfer time."""

    versioned: bool
    version: str | None
    fingerprint: str | None

    def to_json(self) -> str:
        return json.dumps(
            {
                "versioned": self.versioned,
                "version": self.version,
                "fingerprint": self.fingerprint,
            }
        )

    @classmethod
    def from_json(cls, text: str | None) -> SourceIdentity | None:
        if text is None:
            return None
        data = json.loads(text)
        return cls(
            versioned=data["versioned"],
            version=data.get("version"),
            fingerprint=data.get("fingerprint"),
        )


def capture_identity(fs: FileSystem, path: AnyFSPath) -> SourceIdentity:
    """Capture the currently obtainable identity of a source object.

    Version-aware filesystems bind the exact ``version_id``/``generation``.
    Other filesystems are described by their ``ukey()`` (or a deterministic
    digest of the scalar ``info()`` entries when ``ukey()`` is empty).
    """
    info = fs.info(path)
    version = info.get("version_id") or info.get("generation")
    if fs.version_aware and version is not None:
        return SourceIdentity(versioned=True, version=str(version), fingerprint=None)

    fingerprint: str | None = None
    with suppress(Exception):
        fingerprint = fs.ukey(path)
    if not fingerprint:
        fingerprint = "info:" + _sha256_hex(
            _canonical_json(_scalar_info(info)).encode("utf-8")
        )
    return SourceIdentity(versioned=False, version=None, fingerprint=fingerprint)


@dataclass
class ObjectState:
    """A single ledger row describing one object."""

    oid: str
    status: Status
    attempts: int = 0
    src_version: str | None = None
    src_identity: SourceIdentity | None = None
    tmp_path: str | None = None
    size: int | None = None
    expected_digest: str | None = None
    observed_digest: str | None = None
    error_kind: ErrorKind | None = None
    error: str | None = None


@dataclass
class TransferResult:
    """Outcome of one :meth:`TransferSession.run` invocation."""

    verified: set[str] = field(default_factory=set)
    failed: dict[str, str] = field(default_factory=dict)
    pending: set[str] = field(default_factory=set)
    # Objects accepted from an already-valid destination (zero retransmit).
    reused: set[str] = field(default_factory=set)
    # Objects actually copied from the source during this run.
    retransferred: set[str] = field(default_factory=set)
    degraded: bool = False
    cancelled: bool = False

    @property
    def ok(self) -> bool:
        return not (self.failed or self.pending or self.cancelled)


@dataclass(frozen=True)
class SessionPlan:
    """Immutable namespace binding of one ledger to one transfer plan."""

    src_id: str
    src_config_digest: str
    dest_id: str
    dest_config_digest: str
    oids_digest: str
    verifier_id: str
    oids: set[str]


@runtime_checkable
class Verifier(Protocol):
    """Content verification protocol.

    Implementations must never infer expected content from the OID itself.
    """

    def verifier_id(self) -> str:
        """Stable identifier of this verifier configuration for the ledger."""

    def digest(self, oid: str) -> str | None:
        """Return the expected digest for ``oid`` or ``None`` when the
        transported-bytes digest is the only available reference."""

    def verify(self, oid: str, fs: FileSystem, path: AnyFSPath) -> str:
        """Verify ``path`` and return its observed digest.

        Raises :class:`~dvc_objects.errors.ObjectFormatError` when the
        content does not satisfy the verifier.
        """


class Sha256Verifier:
    """Verifier that independently re-reads and hashes destination content.

    The reference digest is the SHA256 computed over the bytes as they were
    read from the source, so an OID looking like an MD5 never influences the
    outcome.
    """

    __slots__ = ()

    def verifier_id(self) -> str:
        return "sha256:transport"

    def digest(self, oid: str) -> str | None:
        return None

    def verify(self, oid: str, fs: FileSystem, path: AnyFSPath) -> str:
        return _hash_file(fs, path)


class MapVerifier:
    """Explicit verifier mapping opaque OIDs to expected SHA256 digests."""

    __slots__ = ("_digests", "_id")

    def __init__(self, digests: dict[str, str]):
        self._digests = dict(digests)
        self._id = "map:sha256:" + _sha256_hex(
            _canonical_json(self._digests).encode("utf-8")
        )

    def verifier_id(self) -> str:
        return self._id

    def digest(self, oid: str) -> str | None:
        return self._digests[oid]

    def verify(self, oid: str, fs: FileSystem, path: AnyFSPath) -> str:
        observed = _hash_file(fs, path)
        expected = self._digests[oid]
        if observed != expected:
            raise ObjectFormatError(
                f"object {oid!r}: expected digest {expected}, got {observed}"
            )
        return observed


class _ByteSemaphore:
    """Bounds the total number of bytes copied concurrently."""

    __slots__ = ("_cond", "_limit", "_used", "max_seen")

    def __init__(self, limit: int | None):
        self._limit = limit
        self._used = 0
        self.max_seen = 0
        self._cond = threading.Condition()

    def acquire(self, size: int) -> None:
        if self._limit is None:
            with self._cond:
                self._used += size
                self.max_seen = max(self.max_seen, self._used)
            return
        with self._cond:
            # An object larger than the whole budget is allowed to take the
            # full capacity once every other transfer has drained; blocking
            # forever would deadlock the plan on a single big object.
            while self._used and self._used + size > self._limit:
                self._cond.wait()
            self._used += size
            self.max_seen = max(self.max_seen, self._used)

    def release(self, size: int) -> None:
        with self._cond:
            self._used -= size
            self._cond.notify_all()

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()


def _windows_pid_alive(pid: int) -> bool:
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    # PROCESS_SYNCHRONIZE lets us wait on / inspect the process handle.
    handle = kernel32.OpenProcess(0x00100000, False, pid)
    if not handle:
        return False
    kernel32.CloseHandle(handle)
    return True


def _posix_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    platform: str = sys.platform
    if platform.startswith("win"):
        return _windows_pid_alive(pid)
    return _posix_pid_alive(pid)


class _SessionFileLock:
    """Cross-process single-owner lock guarded by an O_EXCL marker file."""

    __slots__ = ("_held", "_token", "path")

    def __init__(self, path: str, force: bool):
        self.path = path
        self._token = uuid.uuid4().hex
        self._held = False
        self.acquire(force=force)

    def acquire(self, force: bool = False) -> None:
        if self._held:
            return
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if force or not self._owner_alive():
                    with suppress(FileNotFoundError):
                        os.remove(self.path)
                    force = False
                    continue
                raise SessionLockedError(
                    f"transfer ledger {self.path!r} is locked by another session"
                ) from None
            else:
                with os.fdopen(fd, "w") as fobj:
                    json.dump(
                        {
                            "token": self._token,
                            "pid": os.getpid(),
                            "host": os.name,
                            "created": _now(),
                        },
                        fobj,
                    )
                self._held = True
                return
        raise SessionLockedError(
            f"transfer ledger {self.path!r} is locked by another session"
        )

    def _owner_alive(self) -> bool:
        try:
            with open(self.path, encoding="utf-8") as fobj:
                payload = json.load(fobj)
        except (OSError, ValueError):
            return False
        return _pid_alive(int(payload.get("pid", -1)))

    def release(self) -> None:
        if not self._held:
            return
        with suppress(FileNotFoundError):
            try:
                with open(self.path, encoding="utf-8") as fobj:
                    payload = json.load(fobj)
            except (OSError, ValueError):
                payload = {}
            if payload.get("token") in (None, self._token):
                os.remove(self.path)
        self._held = False

    def __enter__(self) -> _SessionFileLock:  # noqa: PYI034 - no typing_extensions dep
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def _now() -> float:
    return time.time()


class Ledger:
    """SQLite-backed checkpoint store for transfer sessions.

    All state changes run inside SQLite transactions. Use :meth:`open` for a
    durable file-backed ledger (with a cross-process session lock) or
    :meth:`memory` for an in-process ledger.
    """

    SCHEMA_VERSION = SCHEMA_VERSION
    _COLUMNS = (
        "status",
        "attempts",
        "src_version",
        "src_identity",
        "tmp_path",
        "size",
        "expected_digest",
        "observed_digest",
        "error_kind",
        "error",
    )

    def __init__(
        self,
        conn: sqlite3.Connection,
        path: str | None,
        file_lock: _SessionFileLock | None,
    ):
        self.conn = conn
        self.path = path
        self._file_lock = file_lock
        self._lock = threading.RLock()
        self._owner: str | None = None

    @classmethod
    def open(cls, path: str | os.PathLike[str], force: bool = False) -> Ledger:
        path = os.fspath(path)
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        file_lock = _SessionFileLock(path + ".lock", force=force)
        conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            cls._create_schema(conn)
        except Exception:
            conn.close()
            file_lock.release()
            raise
        return cls(conn, path, file_lock)

    @classmethod
    def memory(cls) -> Ledger:
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        cls._create_schema(conn)
        return cls(conn, None, None)

    @staticmethod
    def _create_schema(conn: sqlite3.Connection) -> None:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session (
                session_id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                src_id TEXT NOT NULL,
                src_config_digest TEXT NOT NULL,
                dest_id TEXT NOT NULL,
                dest_config_digest TEXT NOT NULL,
                oids_digest TEXT NOT NULL,
                verifier_id TEXT NOT NULL,
                token TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS object (
                session_id TEXT NOT NULL,
                oid TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                src_version TEXT,
                src_identity TEXT,
                tmp_path TEXT,
                size INTEGER,
                expected_digest TEXT,
                observed_digest TEXT,
                error_kind TEXT,
                error TEXT,
                updated_at REAL NOT NULL,
                PRIMARY KEY (session_id, oid)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_object_status ON object(session_id, status)"
        )
        conn.commit()

    def close(self) -> None:
        with self._lock:
            self._owner = None
            self.conn.close()
            if self._file_lock is not None:
                self._file_lock.release()

    def acquire_owner(self, token: str) -> None:
        with self._lock:
            if self._owner is not None and self._owner != token:
                raise SessionLockedError("ledger is owned by another active session")
            self._owner = token

    def release_owner(self, token: str) -> None:
        with self._lock:
            if self._owner == token:
                self._owner = None

    def begin_session(
        self, session_id: str, plan: SessionPlan
    ) -> tuple[str, bool, str]:
        """Create or validate the single session row and its object rows.

        A ledger describes exactly one transfer plan. Returns
        ``(effective_session_id, is_new, session_token)``. Reusing a ledger
        created for a different source/destination namespace, OID set or
        verifier configuration raises :class:`LedgerMismatchError`.
        """
        with self._lock, self.conn:
            row = self.conn.execute(
                "SELECT * FROM session ORDER BY created_at LIMIT 1"
            ).fetchone()
            now = _now()
            if row is None:
                token = uuid.uuid4().hex
                self.conn.execute(
                    """
                    INSERT INTO session (
                        session_id, schema_version, src_id, src_config_digest,
                        dest_id, dest_config_digest, oids_digest, verifier_id,
                        token, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        SCHEMA_VERSION,
                        plan.src_id,
                        plan.src_config_digest,
                        plan.dest_id,
                        plan.dest_config_digest,
                        plan.oids_digest,
                        plan.verifier_id,
                        token,
                        now,
                        now,
                    ),
                )
                self.conn.executemany(
                    "INSERT INTO object (session_id, oid, status, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        (session_id, oid, Status.PENDING.value, now)
                        for oid in sorted(plan.oids)
                    ],
                )
                return session_id, True, token

            record = dict(
                zip(
                    (
                        "session_id",
                        "schema_version",
                        "src_id",
                        "src_config_digest",
                        "dest_id",
                        "dest_config_digest",
                        "oids_digest",
                        "verifier_id",
                        "token",
                        "created_at",
                        "updated_at",
                    ),
                    row,
                )
            )
            expected = {
                "schema_version": SCHEMA_VERSION,
                "src_id": plan.src_id,
                "src_config_digest": plan.src_config_digest,
                "dest_id": plan.dest_id,
                "dest_config_digest": plan.dest_config_digest,
                "oids_digest": plan.oids_digest,
                "verifier_id": plan.verifier_id,
            }
            for key, value in expected.items():
                if record[key] != value:
                    raise LedgerMismatchError(
                        f"ledger {self.path or ':memory:'} was created with "
                        f"{key}={record[key]!r}, refusing to reuse it for "
                        f"{key}={value!r}"
                    )

            effective_id = str(record["session_id"])
            known = {
                existing[0]
                for existing in self.conn.execute(
                    "SELECT oid FROM object WHERE session_id = ?",
                    (effective_id,),
                )
            }
            if known != set(plan.oids):
                raise LedgerMismatchError(
                    "ledger object set does not match the requested OID set"
                )
            return effective_id, False, str(record["token"])

    def states(self, session_id: str) -> dict[str, ObjectState]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT oid, status, attempts, src_version, src_identity, "
                "tmp_path, size, expected_digest, observed_digest, "
                "error_kind, error FROM object WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        return {row[0]: self._row_to_state(row) for row in rows}

    def get(self, session_id: str, oid: str) -> ObjectState:
        with self._lock:
            row = self.conn.execute(
                "SELECT oid, status, attempts, src_version, src_identity, "
                "tmp_path, size, expected_digest, observed_digest, "
                "error_kind, error FROM object WHERE session_id = ? AND oid = ?",
                (session_id, oid),
            ).fetchone()
        if row is None:
            raise KeyError(oid)
        return self._row_to_state(row)

    @staticmethod
    def _row_to_state(row: tuple[Any, ...]) -> ObjectState:
        (
            oid,
            status,
            attempts,
            src_version,
            src_identity,
            tmp_path,
            size,
            expected_digest,
            observed_digest,
            error_kind,
            error,
        ) = row
        return ObjectState(
            oid=oid,
            status=Status(status),
            attempts=attempts,
            src_version=src_version,
            src_identity=SourceIdentity.from_json(src_identity),
            tmp_path=tmp_path,
            size=size,
            expected_digest=expected_digest,
            observed_digest=observed_digest,
            error_kind=ErrorKind(error_kind) if error_kind else None,
            error=error,
        )

    def reset_inflight(self, session_id: str) -> None:
        """In-flight rows never reached a terminal state; redo them."""
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE object SET status = ?, updated_at = ? "
                "WHERE session_id = ? AND status = ?",
                (Status.PENDING.value, _now(), session_id, Status.IN_FLIGHT.value),
            )

    def reset_retryable_failures(self, session_id: str) -> None:
        """Reopen transient/drift failures on a new run.

        Permission, format and fatal failures stay terminal: re-running a
        session must not silently retry a condition that cannot self-heal.
        """
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE object SET status = ?, attempts = 0, updated_at = ? "
                "WHERE session_id = ? AND status = ? "
                "AND error_kind IN (?, ?)",
                (
                    Status.PENDING.value,
                    _now(),
                    session_id,
                    Status.FAILED.value,
                    ErrorKind.TRANSIENT.value,
                    ErrorKind.SOURCE_CHANGED.value,
                ),
            )

    def update(self, session_id: str, oid: str, **fields: Any) -> None:
        if not fields:
            return
        unknown = set(fields) - set(self._COLUMNS)
        if unknown:
            raise ValueError(f"unknown ledger columns: {sorted(unknown)}")
        assignments: list[str] = []
        values: list[Any] = []
        for column in self._COLUMNS:
            if column not in fields:
                continue
            value = fields[column]
            if column == "status":
                value = value.value if isinstance(value, Status) else value
            elif column == "src_identity":
                value = value.to_json() if isinstance(value, SourceIdentity) else value
            elif column == "error_kind":
                value = value.value if isinstance(value, ErrorKind) else value
            assignments.append(f"{column} = ?")
            values.append(value)
        assignments.append("updated_at = ?")
        values.append(_now())
        values.extend((session_id, oid))
        with self._lock, self.conn:
            # Column names come from the internal _COLUMNS allowlist only.
            self.conn.execute(
                f"UPDATE object SET {', '.join(assignments)} "  # noqa: S608
                "WHERE session_id = ? AND oid = ?",
                values,
            )
            self.conn.execute(
                "UPDATE session SET updated_at = ? WHERE session_id = ?",
                (_now(), session_id),
            )


class TransferSession:
    """Resumable object-level transfer of a fixed OID set between two ODBs."""

    def __init__(  # noqa: PLR0913 - all knobs are part of one coherent plan
        self,
        src: ObjectDB,
        dest: ObjectDB,
        oids: set[str],
        *,
        ledger: str | os.PathLike[str] | Ledger,
        verifier: Verifier | None = None,
        jobs: int | None = None,
        max_inflight_bytes: int | None = DEFAULT_MAX_INFLIGHT_BYTES,
        max_retries: int = 3,
        strict: bool = True,
        session_id: str | None = None,
        force: bool = False,
        callback: Any = None,
    ):
        self.src = src
        self.dest = dest
        self.oids = set(oids)
        self._ledger_arg = ledger
        self._force = force
        self.verifier: Verifier = verifier or Sha256Verifier()
        self.jobs = max(1, jobs or dest.fs.jobs)
        self.max_inflight_bytes = max_inflight_bytes
        self.max_retries = max_retries
        self.strict = strict
        self.callback = callback
        self.session_id = session_id or f"xfer-{uuid.uuid4().hex}"

        self._ledger: Ledger | None = None
        self._token: str | None = None
        self._cancel = threading.Event()
        self._inflight = _ByteSemaphore(max_inflight_bytes)
        self._notified: set[tuple[str, str]] = set()
        self._fatal: BaseException | None = None
        self._degraded = False
        self._reused: set[str] = set()
        self._retransferred: set[str] = set()

    @property
    def token(self) -> str | None:
        return self._token

    def __enter__(self) -> TransferSession:  # noqa: PYI034 - no typing_extensions dep
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._ledger is None:
            return
        ledger = self._ledger
        owner = self.session_id
        self._ledger = None
        self._token = None
        ledger.release_owner(owner)
        if isinstance(self._ledger_arg, Ledger):
            return
        ledger.close()

    def cancel(self) -> None:
        """Ask :meth:`run` to stop scheduling new objects."""
        self._cancel.set()
        self._inflight.wake()

    def states(self) -> dict[str, Status]:
        if self._ledger is None or self._token is None:
            raise TransferSessionError("session has not been started yet")
        return {
            oid: state.status
            for oid, state in self._ledger.states(self.session_id).items()
        }

    def _open_ledger(self) -> Ledger:
        if self._ledger is not None:
            return self._ledger
        if isinstance(self._ledger_arg, Ledger):
            ledger = self._ledger_arg
            owns_connection = False
        else:
            ledger = Ledger.open(self._ledger_arg, force=self._force)
            owns_connection = True
        try:
            ledger.acquire_owner(self.session_id)
        except Exception:
            # Do not leak the connection/lock when another owner wins.
            if owns_connection:
                ledger.close()
            raise
        self._ledger = ledger
        return ledger

    def _check_capabilities(self) -> bool:
        if self.dest.read_only:
            raise ObjectDBPermissionError("Cannot transfer into a read-only ODB")
        if not getattr(self.dest.fs, "CAN_ATOMIC_PUBLISH", False):
            if self.strict:
                raise AtomicPublishUnsupportedError(
                    f"{type(self.dest.fs).__name__} does not declare atomic "
                    "publish support; rerun with strict=False to record a "
                    "degraded transfer result"
                )
            logger.warning(
                "destination filesystem %s does not declare atomic publish; "
                "transfer will be marked degraded and the final path is "
                "re-verified before any object is confirmed",
                type(self.dest.fs).__name__,
            )
            return True
        return False

    def _build_plan(self) -> SessionPlan:
        src_id = (
            f"{self.src.fs.protocol}::{self.src.path}::read_only={self.src.read_only}"
        )
        dest_id = (
            f"{self.dest.fs.protocol}::{self.dest.path}::"
            f"read_only={self.dest.read_only}"
        )
        return SessionPlan(
            src_id=src_id,
            src_config_digest=_sha256_hex(
                _canonical_json(self.src.fs.config).encode("utf-8")
            ),
            dest_id=dest_id,
            dest_config_digest=_sha256_hex(
                _canonical_json(self.dest.fs.config).encode("utf-8")
            ),
            oids_digest=_sha256_hex("\x00".join(sorted(self.oids)).encode("utf-8")),
            verifier_id=self.verifier.verifier_id(),
            oids=set(self.oids),
        )

    def run(self) -> TransferResult:
        """Execute (or resume) the transfer plan until it settles."""
        ledger = self._open_ledger()
        try:
            self._degraded = self._check_capabilities()
            effective_id, _, token = ledger.begin_session(
                self.session_id, self._build_plan()
            )
            # The ledger owns the plan identity: rebuilt sessions adopt the
            # persisted session id instead of silently forking a new plan.
            self.session_id = effective_id
            self._token = token
            self._reused.clear()
            self._retransferred.clear()
            for oid, state in ledger.states(self.session_id).items():
                if state.status is Status.VERIFIED:
                    # Confirmed in a previous process/run: no bytes are copied
                    # for it during this run.
                    self._reused.add(oid)
            ledger.reset_retryable_failures(self.session_id)
            self._recover()
            self._dispatch()
            return self._build_result(cancelled=False)
        except TransferCancelledError as exc:
            exc.result = self._build_result(cancelled=True)
            raise

    def _recover(self) -> None:
        ledger = self._ledger
        assert ledger is not None
        assert self._token is not None
        ledger.reset_inflight(self.session_id)
        self._sweep_tmp_objects()
        for oid, state in ledger.states(self.session_id).items():
            if state.status is not Status.PENDING:
                continue
            final_path = self.dest.oid_to_path(oid)
            if not self.dest.fs.isfile(final_path):
                continue
            # A finished destination object (e.g. one published just before
            # the checkpoint write failed) is verified before anything is
            # retransmitted or deleted.
            try:
                observed = self.verifier.verify(oid, self.dest.fs, final_path)
                expected = self.verifier.digest(oid)
                if expected is not None and observed != expected:
                    raise ObjectFormatError(
                        f"existing object {oid!r} failed verification"
                    )
                if state.src_identity is not None:
                    self._assert_identity_unchanged(oid, state.src_identity)
            except SourceChangedError as exc:
                self._record_failure(oid, ErrorKind.SOURCE_CHANGED, exc)
                continue
            except ObjectFormatError:
                logger.warning("removing corrupt pre-existing object %r", final_path)
                with suppress(FileNotFoundError):
                    self.dest.fs.rm_file(final_path)
                continue

            ledger.update(
                self.session_id,
                oid,
                status=Status.VERIFIED,
                observed_digest=observed,
                expected_digest=expected,
                tmp_path=None,
                error=None,
                error_kind=None,
            )
            self._reused.add(oid)
            self._notify(oid, Status.VERIFIED)

    def _sweep_tmp_objects(self) -> None:
        assert self._token is not None
        prefix = f".xfer.{self._token}."
        try:
            paths = list(self.dest.fs.find(self.dest.path))
        except Exception:  # noqa: BLE001 - missing odb root is a no-op sweep
            return
        for path in paths:
            name = self.dest.fs.name(path)
            if name.startswith(prefix) and name.endswith(".tmp"):
                logger.debug("removing orphaned temporary object %r", path)
                with suppress(FileNotFoundError):
                    self.dest.fs.rm_file(path)

    def _dispatch(self) -> None:
        assert self._ledger is not None
        pending = sorted(
            oid
            for oid, state in self._ledger.states(self.session_id).items()
            if state.status is Status.PENDING
        )
        if not pending:
            return
        work: queue.Queue[str] = queue.Queue()
        for oid in pending:
            work.put(oid)
        workers = [
            threading.Thread(
                target=self._worker, args=(work,), name=f"transfer-{index}"
            )
            for index in range(min(self.jobs, len(pending)))
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        if self._fatal is not None:
            raise self._fatal
        if self._cancel.is_set():
            raise TransferCancelledError

    def _worker(self, work: queue.Queue[str]) -> None:
        while self._fatal is None and not self._cancel.is_set():
            try:
                oid = work.get_nowait()
            except queue.Empty:
                return
            try:
                self._process_oid(oid)
            except TransferCancelledError:
                state = self._safe_get_state(oid)
                if state is not None and state.tmp_path:
                    with suppress(FileNotFoundError):
                        self.dest.fs.rm_file(state.tmp_path)
                return
            except BaseException as exc:  # noqa: BLE001 - fatal escalation
                self._fatal = exc
                self._cancel.set()
                self._inflight.wake()
                return

    def _safe_get_state(self, oid: str) -> ObjectState | None:
        assert self._ledger is not None
        with suppress(Exception):
            return self._ledger.get(self.session_id, oid)
        return None

    def _process_oid(self, oid: str) -> None:
        assert self._ledger is not None
        state = self._ledger.get(self.session_id, oid)
        for attempt in range(state.attempts, self.max_retries + 1):
            if self._cancel.is_set():
                raise TransferCancelledError
            try:
                self._attempt(oid, attempt)
                return
            except TransferCancelledError:
                raise
            except _PERMANENT_ERRORS as exc:
                kind = (
                    ErrorKind.SOURCE_CHANGED
                    if isinstance(exc, SourceChangedError)
                    else ErrorKind.PERMANENT
                )
                self._record_failure(oid, kind, exc)
                return
            except OSError as exc:
                if exc.errno == errno.EMFILE:
                    # Mirrors fs.generic.log_exceptions: EMFILE is fatal,
                    # retrying would only hit the same descriptor exhaustion.
                    self._record_failure(oid, ErrorKind.FATAL, exc)
                    raise
                if attempt >= self.max_retries:
                    self._record_failure(oid, ErrorKind.TRANSIENT, exc)
                    return
                logger.warning(
                    "transient failure transferring %r (attempt %d/%d): %s",
                    oid,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
            except Exception as exc:  # noqa: BLE001 - bounded retry envelope
                if attempt >= self.max_retries:
                    self._record_failure(oid, ErrorKind.TRANSIENT, exc)
                    return
                logger.warning(
                    "unexpected failure transferring %r (attempt %d/%d): %s",
                    oid,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )

    def _attempt(self, oid: str, attempt: int) -> None:
        assert self._ledger is not None
        assert self._token is not None
        src_path = self.src.oid_to_path(oid)
        final_path = self.dest.oid_to_path(oid)

        # Record the attempt (and clear any previous error) before touching
        # the source so permanent failures still carry a meaningful count.
        self._ledger.update(
            self.session_id,
            oid,
            status=Status.IN_FLIGHT,
            attempts=attempt + 1,
            error=None,
            error_kind=None,
        )
        if not self.src.fs.isfile(src_path):
            raise ObjectDBError(f"source object {oid!r} is missing")

        identity = capture_identity(self.src.fs, src_path)
        state = self._ledger.get(self.session_id, oid)
        if state.src_identity is not None and identity != state.src_identity:
            raise SourceChangedError(
                f"source object {oid!r} changed since the transfer started"
            )

        size = self.src.fs.size(src_path)
        tmp_path = self._tmp_path(final_path, oid)
        self._ledger.update(
            self.session_id,
            oid,
            src_identity=identity,
            src_version=identity.version,
            tmp_path=tmp_path,
            size=size,
        )

        hasher = hashlib.sha256()
        reservation = size if size and size > 0 else CHUNK_SIZE
        self._inflight.acquire(reservation)
        try:
            self._stream_copy(oid, src_path, tmp_path, identity.version, hasher)
        finally:
            self._inflight.release(reservation)

        transport_digest = hasher.hexdigest()
        observed = self.verifier.verify(oid, self.dest.fs, tmp_path)
        expected = self.verifier.digest(oid) or transport_digest
        if observed != expected:
            raise ObjectFormatError(
                f"temporary object {oid!r}: expected {expected}, got {observed}"
            )

        # The source must still describe the same bytes before publishing.
        self._assert_identity_unchanged(oid, identity)
        self._publish(tmp_path, final_path)

        # Even when the move is not strictly atomic, the final path is only
        # trusted after an independent verification of its content.
        published = self.verifier.verify(oid, self.dest.fs, final_path)
        if published != expected:
            with suppress(FileNotFoundError):
                self.dest.fs.rm_file(final_path)
            raise ObjectFormatError(
                f"published object {oid!r}: expected {expected}, got {published}"
            )
        self._assert_identity_unchanged(oid, identity)

        self._ledger.update(
            self.session_id,
            oid,
            status=Status.VERIFIED,
            tmp_path=None,
            expected_digest=expected,
            observed_digest=published,
            error=None,
            error_kind=None,
        )
        self._retransferred.add(oid)
        self._notify(oid, Status.VERIFIED)

    def _stream_copy(
        self,
        oid: str,
        src_path: AnyFSPath,
        tmp_path: AnyFSPath,
        version: str | None,
        hasher: hashlib._Hash,
    ) -> None:
        self.dest.fs.makedirs(self.dest.fs.parent(tmp_path), exist_ok=True)
        with (
            self._open_source(src_path, version) as fsrc,
            self.dest.fs.open(tmp_path, "wb") as fdst,
        ):
            while True:
                if self._cancel.is_set():
                    raise TransferCancelledError
                chunk = fsrc.read(CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
                fdst.write(chunk)

    def _open_source(self, src_path: AnyFSPath, version: str | None):
        if version:
            try:
                return self.src.fs.open(src_path, "rb", version_id=version)
            except TypeError:
                logger.debug(
                    "source filesystem ignores bound version_id for %r",
                    src_path,
                )
        return self.src.fs.open(src_path, "rb")

    def _assert_identity_unchanged(self, oid: str, expected: SourceIdentity) -> None:
        if expected.versioned:
            # The bound version is opened explicitly during the copy, so a
            # rotating head cannot change the bytes being transferred.
            return
        src_path = self.src.oid_to_path(oid)
        try:
            current = capture_identity(self.src.fs, src_path)
        except Exception as exc:
            raise SourceChangedError(
                f"source object {oid!r} became unavailable during transfer"
            ) from exc
        if current != expected:
            raise SourceChangedError(
                f"source object {oid!r} changed during transfer; refusing to "
                "confirm a snapshot of a mutable source"
            )

    def _publish(self, tmp_path: AnyFSPath, final_path: AnyFSPath) -> None:
        try:
            self.dest.fs.mv(tmp_path, final_path)
        except Exception:
            with suppress(FileNotFoundError):
                self.dest.fs.rm_file(tmp_path)
            raise

    def _tmp_path(self, final_path: AnyFSPath, oid: str) -> AnyFSPath:
        assert self._token is not None
        tag = hashlib.sha256(oid.encode("utf-8")).hexdigest()[:16]
        name = f".xfer.{self._token}.{tag}.tmp"
        return self.dest.fs.join(self.dest.fs.parent(final_path), name)

    def _record_failure(self, oid: str, kind: ErrorKind, exc: BaseException) -> None:
        assert self._ledger is not None
        state = self._safe_get_state(oid)
        if state is not None and state.tmp_path:
            with suppress(FileNotFoundError):
                self.dest.fs.rm_file(state.tmp_path)
        logger.warning("transfer of %r failed (%s): %s", oid, kind.value, exc)
        self._ledger.update(
            self.session_id,
            oid,
            status=Status.FAILED,
            error_kind=kind,
            error=str(exc),
            tmp_path=None,
        )
        self._notify(oid, Status.FAILED)

    def _notify(self, oid: str, status: Status) -> None:
        key = (oid, status.value)
        if key in self._notified:
            return
        self._notified.add(key)
        if self.callback is None:
            return
        on_state = getattr(self.callback, "on_state", None)
        if callable(on_state):
            on_state(oid, status)
        relative_update = getattr(self.callback, "relative_update", None)
        if callable(relative_update):
            relative_update()

    def _build_result(self, cancelled: bool) -> TransferResult:
        assert self._ledger is not None
        verified: set[str] = set()
        failed: dict[str, str] = {}
        pending: set[str] = set()
        for oid, state in self._ledger.states(self.session_id).items():
            if state.status is Status.VERIFIED:
                verified.add(oid)
            elif state.status is Status.FAILED:
                failed[oid] = state.error or "unknown error"
            else:
                pending.add(oid)
        return TransferResult(
            verified=verified,
            failed=failed,
            pending=pending,
            reused=set(self._reused),
            retransferred=set(self._retransferred),
            degraded=self._degraded,
            cancelled=cancelled,
        )
