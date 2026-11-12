"""Atomic conditional-write primitives.

This module contains the low-level building blocks used by
:class:`dvc_objects.cond.ConditionalObjectDB`:

* :class:`Capability` -- an explicit, per-filesystem declaration of what
  atomicity/durability guarantees a writer actually provides.
* :class:`ConditionalWriter` -- the adapter contract every conditional
  backend implements.
* :class:`LocalConditionalWriter` -- a local-disk implementation based on
  same-directory temporary files, a truly exclusive publish and a CAS
  lock protocol.
* :class:`RemoteConditionalAdapter` -- a skeleton for remote adapters
  that gate writes with ETag/generation preconditions.

Revision tokens returned here are *opaque*: callers may only compare them
for equality. They are never content digests and must never be parsed.
"""

import base64
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from io import BytesIO
from secrets import token_urlsafe
from typing import Optional, Protocol, runtime_checkable

from dvc_objects.errors import (
    AmbiguousCommitError,
    ConditionalWriteUnsupportedError,
    PreconditionFailedError,
    ShortWriteError,
)

from .utils import umask

logger = logging.getLogger(__name__)

#: Default bounded chunk size used while spooling a stream into a
#: temporary object (8 MiB).
DEFAULT_CONDITIONAL_CHUNK_SIZE = 8 * 2**20

#: Per-target CAS lock tuning.
DEFAULT_LOCK_TIMEOUT = 30.0
DEFAULT_LOCK_POLL_INTERVAL = 0.05
DEFAULT_LOCK_STALE_AFTER = 30.0

_LOCAL_REVISION_SCHEME = "local"
_REVISION_ENVELOPE_PREFIX = "rev1"


@runtime_checkable
class StreamReader(Protocol):
    """The only stream capability a conditional writer requires."""

    def read(self, size: int = -1) -> bytes: ...


@runtime_checkable
class StreamWriter(Protocol):
    """Sink interface used while spooling into a temporary object."""

    def write(self, data: bytes) -> int: ...


@dataclass(frozen=True)
class Capability:
    """Explicit guarantee matrix for a conditional writer.

    The three durability/visibility flags are declared independently on
    purpose: a filesystem may publish atomically without being able to
    fsync a directory, or acknowledge a write without offering local
    crash-durability semantics. Writers must not claim guarantees they
    cannot make.

    Attributes:
        conditional_writes: ``put_if_absent`` / ``compare_and_put`` are
            enforced atomically by the backend (no exists-then-put race).
        atomic_visibility: a committed object appears all-at-once;
            readers can never observe half-written bytes.
        fsync_before_publish: data is flushed to stable storage before
            the object becomes visible.
        crash_durable: a successfully published object survives an
            OS/power crash (implies metadata persistence as well).
    """

    conditional_writes: bool
    atomic_visibility: bool
    fsync_before_publish: bool
    crash_durable: bool


def opaque_token(kind: str, value: str) -> str:
    """Wrap a backend token (ETag, generation, ...) as an opaque revision.

    The token is encoded rather than returned raw so that callers cannot
    mistake an ETag for a content digest. Only equality comparisons are
    meaningful; the encoding is an implementation detail.
    """
    payload = json.dumps(
        {"v": 1, "kind": kind, "token": value}, separators=(",", ":")
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return f"{_REVISION_ENVELOPE_PREFIX}:{encoded}"


def copy_bounded(
    src: StreamReader,
    dst: StreamWriter,
    size: Optional[int],
    chunk_size: int = DEFAULT_CONDITIONAL_CHUNK_SIZE,
) -> int:
    """Copy ``src`` into ``dst`` in bounded chunks.

    When ``size`` is given it is a hard contract: fewer bytes raise
    :class:`ShortWriteError` (the caller must discard the partial temp
    object), and any byte beyond ``size`` raises as well so that an
    oversized stream can never be silently truncated. Returns the number
    of bytes copied.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    written = 0
    while size is None or written < size:
        read_size = chunk_size if size is None else min(chunk_size, size - written)
        if read_size == 0:
            break
        chunk = src.read(read_size)
        if not chunk:
            break
        dst.write(chunk)
        written += len(chunk)

    if size is not None and written < size:
        raise ShortWriteError(
            f"stream ended after {written} bytes; {size} bytes were declared"
        )
    if size is not None and src.read(1):
        raise ShortWriteError(
            f"stream produced more bytes than the declared size of {size}"
        )
    return written


class ConditionalWriter(ABC):
    """Backend contract for atomic conditional writes.

    Implementations operate on concrete filesystem paths. All revision
    tokens are opaque strings; only equality is permitted.

    ``put_if_absent`` creates *only* when nothing exists at ``path`` and
    raises :class:`FileExistsError` otherwise -- it must never overwrite.
    ``compare_and_put`` replaces the object only while its current
    revision matches ``expected_revision``, raising
    :class:`PreconditionFailedError` (which carries the observed current
    revision) otherwise.

    Implementations are responsible for removing any temporary object
    they created, and must never remove objects (including temp objects
    and locks) they do not own.
    """

    @property
    @abstractmethod
    def capability(self) -> Capability:
        """Return the guarantee matrix declared by this writer."""

    @abstractmethod
    def get_revision(self, path: str) -> Optional[str]:
        """Return the current opaque revision, or ``None`` if absent."""

    @abstractmethod
    def put_if_absent(
        self,
        path: str,
        fobj: StreamReader,
        size: Optional[int] = None,
    ) -> str:
        """Create ``path`` exclusively; return the new opaque revision.

        Raises :class:`FileExistsError` if ``path`` already exists,
        without modifying it.
        """

    @abstractmethod
    def compare_and_put(
        self,
        path: str,
        expected_revision: str,
        fobj: StreamReader,
        size: Optional[int] = None,
    ) -> str:
        """Replace ``path`` if its revision matches; return a new revision.

        Raises :class:`PreconditionFailedError` on any mismatch,
        including a missing object.
        """


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


class LocalConditionalWriter(ConditionalWriter):
    """Conditional writer for a local POSIX/NTFS filesystem.

    Publish protocol:

    * Bytes are spooled into an *owned* same-directory temporary file
      (``.<leaf>.<pid>.<random>.tmp``) in bounded chunks and fsynced.
    * ``put_if_absent`` publishes with ``os.link`` (hard link,
      ``O_CREAT|O_EXCL`` semantics): a truly exclusive create that fails
      with :class:`FileExistsError` instead of overwriting. A rename is
      never used to emulate no-replace semantics.
    * ``compare_and_put`` takes a per-target lock, re-reads the revision
      under the lock, then publishes with ``os.replace`` -- an actual
      atomic replacement, only performed after the revision check.
    * Every cleanup path only ever removes the exact temp path/lock that
      this instance created; locks are released only when their owner
      token still matches.

    Capability declaration: atomic visibility and pre-publish fsync are
    always available. Crash durability additionally requires a directory
    fsync, which has no portable equivalent on Windows, so it is declared
    separately and disabled there.

    Requires a hard-link-capable local filesystem (POSIX local disks and
    NTFS). Filesystems without hard links cannot provide the exclusive
    publish primitive and must not be wired to this writer.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CONDITIONAL_CHUNK_SIZE,
        lock_timeout: float = DEFAULT_LOCK_TIMEOUT,
        lock_poll_interval: float = DEFAULT_LOCK_POLL_INTERVAL,
        lock_stale_after: float = DEFAULT_LOCK_STALE_AFTER,
    ) -> None:
        self.chunk_size = chunk_size
        self._lock_timeout = lock_timeout
        self._lock_poll_interval = lock_poll_interval
        self._lock_stale_after = lock_stale_after
        self._capability = Capability(
            conditional_writes=True,
            atomic_visibility=True,
            fsync_before_publish=True,
            # Directory fsync (metadata persistence) is POSIX-only.
            crash_durable=os.name != "nt",
        )

    @property
    def capability(self) -> Capability:
        return self._capability

    @staticmethod
    def _revision_from_stat(stat_result: "os.stat_result") -> str:
        return (
            f"{_LOCAL_REVISION_SCHEME}:{stat_result.st_dev}:"
            f"{stat_result.st_ino}:{stat_result.st_mtime_ns}:"
            f"{stat_result.st_size}"
        )

    def get_revision(self, path: str) -> Optional[str]:
        try:
            stat_result = os.stat(path)
        except FileNotFoundError:
            return None
        return self._revision_from_stat(stat_result)

    def _tmp_path(self, parent: str, leaf: str) -> str:
        # The pid + random token make ownership explicit: only the
        # process that generated this name is allowed to remove it.
        return os.path.join(
            parent, f".{leaf}.{os.getpid()}.{token_urlsafe(8)}.tmp"
        )

    @staticmethod
    def _lock_path(parent: str, leaf: str) -> str:
        return os.path.join(parent, f".{leaf}.lock")

    @staticmethod
    def _remove_owned(tmp_path: str) -> None:
        # Remove an exact, self-created temp path. Never glob or sweep
        # the directory, which could delete another writer's temp file.
        with suppress(FileNotFoundError):
            os.unlink(tmp_path)

    @staticmethod
    def _fsync_dir(parent: str) -> None:
        if os.name == "nt":
            # No directory-fsync primitive on Windows; crash durability
            # is therefore not declared for this platform.
            return
        try:
            dir_fd = os.open(parent, os.O_RDONLY)
        except OSError:
            logger.debug("could not open directory %r for fsync", parent)
            return
        try:
            os.fsync(dir_fd)
        except OSError:
            logger.debug("directory fsync failed for %r", parent)
        finally:
            os.close(dir_fd)

    def _publish_exclusive(self, tmp_path: str, path: str) -> None:
        # Hard link = atomic, create-only publish. FileExistsError means
        # another writer won the race; we must not fall back to rename.
        os.link(tmp_path, path)

    def _publish_replace(self, tmp_path: str, path: str) -> None:
        # Used only while holding the CAS lock after re-checking the
        # revision: a genuine atomic replacement, not a no-replace fake.
        os.replace(tmp_path, path)

    def put_if_absent(
        self,
        path: str,
        fobj: StreamReader,
        size: Optional[int] = None,
    ) -> str:
        parent = os.path.dirname(path)
        leaf = os.path.basename(path)
        os.makedirs(parent, exist_ok=True)

        tmp_path = self._tmp_path(parent, leaf)
        fd = os.open(tmp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666 & ~umask)
        try:
            with os.fdopen(fd, "wb") as dst:
                copy_bounded(fobj, dst, size, self.chunk_size)
                dst.flush()
                os.fsync(dst.fileno())

            # Hard link = atomic, create-only publish. FileExistsError
            # means another writer won the race; never fall back to rename.
            self._publish_exclusive(tmp_path, path)
            self._fsync_dir(parent)
            revision = self.get_revision(path)
            assert revision is not None
            return revision
        finally:
            # Covers success (the hard link gave the object its own name,
            # the temp name is now redundant), conflicts, short writes and
            # cancellation: only our exact, owned temp path is touched.
            self._remove_owned(tmp_path)

    def _acquire_lock(self, lock_path: str, owner_token: str) -> None:
        deadline = time.monotonic() + self._lock_timeout
        while True:
            try:
                fd = os.open(
                    lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666 & ~umask
                )
            except FileExistsError:
                self._break_stale_lock(lock_path)
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for CAS lock {lock_path!r}"
                    ) from None
                time.sleep(self._lock_poll_interval)
                continue

            try:
                _write_all(fd, owner_token.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            return

    def _break_stale_lock(self, lock_path: str) -> None:
        try:
            age = time.time() - os.stat(lock_path).st_mtime
        except FileNotFoundError:
            return
        if age < self._lock_stale_after:
            return
        # Only age-qualified locks are removed; a live lock is never
        # touched and the owner check in _release_lock still applies.
        with suppress(FileNotFoundError):
            os.unlink(lock_path)

    def _release_lock(self, lock_path: str, owner_token: str) -> None:
        try:
            with open(lock_path, "rb") as lock_file:
                owner = lock_file.read()
        except FileNotFoundError:
            return
        if owner == owner_token.encode("utf-8"):
            with suppress(FileNotFoundError):
                os.unlink(lock_path)

    def compare_and_put(
        self,
        path: str,
        expected_revision: str,
        fobj: StreamReader,
        size: Optional[int] = None,
    ) -> str:
        parent = os.path.dirname(path)
        leaf = os.path.basename(path)
        os.makedirs(parent, exist_ok=True)

        lock_path = self._lock_path(parent, leaf)
        owner_token = f"{os.getpid()}:{token_urlsafe(16)}"
        self._acquire_lock(lock_path, owner_token)
        try:
            current_revision = self.get_revision(path)
            if current_revision is None or current_revision != expected_revision:
                raise PreconditionFailedError(
                    path, expected_revision, current_revision
                )

            tmp_path = self._tmp_path(parent, leaf)
            fd = os.open(
                tmp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666 & ~umask
            )
            try:
                with os.fdopen(fd, "wb") as dst:
                    copy_bounded(fobj, dst, size, self.chunk_size)
                    dst.flush()
                    os.fsync(dst.fileno())

                self._publish_replace(tmp_path, path)
                self._fsync_dir(parent)
                revision = self.get_revision(path)
                assert revision is not None
                return revision
            finally:
                # On success os.replace consumed the temp name (ENOENT is
                # suppressed); on failure or an ambiguous publish only
                # our exact, owned temp path is removed.
                self._remove_owned(tmp_path)
        finally:
            self._release_lock(lock_path, owner_token)


class RemoteConditionalAdapter(ConditionalWriter):
    """Skeleton for ETag/generation-gated remote adapters.

    Concrete transports (HTTP, S3, GCS, ...) implement the three hooks
    below. Preconditions are expressed in terms of the backend's own
    ETag/generation, wrapped via :func:`opaque_token`; the wrapped token
    is compared opaquely and is never treated as a checksum of the
    contents.

    Each adapter declares its own :class:`Capability` explicitly -- a
    remote cannot be assumed to share local-disk durability semantics.
    """

    def __init__(self, capability: Capability) -> None:
        if not capability.conditional_writes:
            raise ConditionalWriteUnsupportedError(
                f"{type(self).__name__} was constructed without the "
                "conditional_writes capability"
            )
        self._capability = capability

    @property
    def capability(self) -> Capability:
        return self._capability

    @abstractmethod
    def _head_revision(self, path: str) -> Optional[str]:
        """Fetch the current wrapped revision token, or ``None``."""

    @abstractmethod
    def _put_if_none_match(
        self,
        path: str,
        data: bytes,
    ) -> str:
        """Create ``path`` only if nothing exists (If-None-Match: *).

        Returns the new wrapped revision. A backend precondition failure
        must be raised as :class:`FileExistsError`.
        """

    @abstractmethod
    def _put_if_match(
        self,
        path: str,
        expected_revision: str,
        data: bytes,
    ) -> str:
        """Replace ``path`` only if the revision still matches.

        Returns the new wrapped revision. A backend precondition failure
        must be raised as :class:`PreconditionFailedError` carrying the
        observed current revision.
        """

    def get_revision(self, path: str) -> Optional[str]:
        return self._head_revision(path)

    @staticmethod
    def _drain(fobj: StreamReader, size: Optional[int]) -> bytes:
        sink = BytesIO()
        copy_bounded(fobj, sink, size)
        return sink.getvalue()

    def put_if_absent(
        self,
        path: str,
        fobj: StreamReader,
        size: Optional[int] = None,
    ) -> str:
        data = self._drain(fobj, size)
        return self._put_if_none_match(path, data)

    def compare_and_put(
        self,
        path: str,
        expected_revision: str,
        fobj: StreamReader,
        size: Optional[int] = None,
    ) -> str:
        data = self._drain(fobj, size)
        return self._put_if_match(path, expected_revision, data)


__all__ = [
    "DEFAULT_CONDITIONAL_CHUNK_SIZE",
    "DEFAULT_LOCK_POLL_INTERVAL",
    "DEFAULT_LOCK_STALE_AFTER",
    "DEFAULT_LOCK_TIMEOUT",
    "AmbiguousCommitError",
    "Capability",
    "ConditionalWriter",
    "LocalConditionalWriter",
    "RemoteConditionalAdapter",
    "StreamReader",
    "StreamWriter",
    "copy_bounded",
    "opaque_token",
]
