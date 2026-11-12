"""Race-free conditional object database.

:class:`ConditionalObjectDB` adds atomic create / compare-and-swap
semantics on top of the regular :class:`dvc_objects.db.ObjectDB`
layout. Unlike an ``exists()`` followed by ``put_file()`` -- which is a
check-then-act race -- every write here is a single conditional
operation enforced by the underlying
:class:`~dvc_objects.fs.conditional.ConditionalWriter`.

OIDs remain caller-provided opaque identifiers. They are *not* assumed
to be MD5 hashes, but every OID is validated before path construction so
that it cannot escape the store root through separators or
``.``/``..`` segments, and cannot collide with the reserved
temporary/lock namespace.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import TYPE_CHECKING, Any, BinaryIO, Optional, Union, cast

from fsspec.callbacks import DEFAULT_CALLBACK

from .db import ObjectDB
from .errors import (
    ConditionalWriteUnsupportedError,
    InvalidOIDError,
    ObjectDBPermissionError,
    PreconditionFailedError,
    ShortWriteError,
)
from .fs.conditional import (
    Capability,
    ConditionalWriter,
    StreamReader,
)

if TYPE_CHECKING:
    from fsspec import Callback

    from .fs.base import FileSystem

logger = logging.getLogger(__name__)

#: Suffixes reserved for same-directory temp files and CAS locks.
_RESERVED_SUFFIXES = (".tmp", ".lock")


class ConditionalStatus(str, Enum):
    """Outcome of a conditional write."""

    #: The object did not exist and was published by this call.
    CREATED = "created"
    #: An object already existed; it was left untouched.
    ALREADY_EXISTS = "already_exists"
    #: The supplied opaque revision no longer matched the live object.
    PRECONDITION_FAILED = "precondition_failed"


@dataclass(frozen=True)
class ConditionalResult:
    """Result of a conditional write.

    ``revision`` always carries the revision that is current *after* the
    call: the freshly minted one on :attr:`ConditionalStatus.CREATED`,
    otherwise the revision observed on the untouched object.
    """

    oid: str
    status: ConditionalStatus
    revision: Optional[str]

    @property
    def published(self) -> bool:
        """True only when this call actually published new bytes."""
        return self.status is ConditionalStatus.CREATED


def validate_oid(oid: Any, fs: "FileSystem") -> str:
    """Validate an opaque OID before mapping it onto a store path.

    The OID is treated as an arbitrary identifier rather than an MD5
    hash. It is rejected when it:

    * is not a non-empty string,
    * is too short for the ``xx/yyyy`` two-level layout,
    * contains a path separator (``/``, ``\\``, the filesystem's own
      separator) or a NUL byte,
    * contains an out-of-bounds segment (``.`` or ``..``),
    * enters the reserved temp-file / lock namespace (a leaf starting
      with ``.`` or ending in ``.tmp``/``.lock``).
    """
    if not isinstance(oid, str) or not oid:
        raise InvalidOIDError(f"oid must be a non-empty string, got {oid!r}")

    separators = {sep for sep in ("/", "\\", "\x00", fs.sep, fs.flavour.sep) if sep}
    bad_separator = next((sep for sep in separators if sep in oid), None)
    if bad_separator is not None:
        raise InvalidOIDError(
            f"oid {oid!r} contains a path separator or NUL byte; "
            "OIDs are opaque single-segment identifiers"
        )

    prefix, leaf = oid[:2], oid[2:]
    if not prefix or not leaf:
        raise InvalidOIDError(
            f"oid {oid!r} is too short for the two-level 'xx/yyyy' store layout"
        )

    for part in (prefix, leaf):
        if part in (".", ".."):
            raise InvalidOIDError(
                f"oid {oid!r} contains an out-of-bounds path segment ('.'/'..')"
            )

    if (
        oid.startswith(".")
        or leaf.startswith(".")
        or leaf.endswith(_RESERVED_SUFFIXES)
    ):
        raise InvalidOIDError(
            f"oid {oid!r} collides with the reserved temporary-file/lock "
            "namespace (leading dot or '.tmp'/'.lock' suffix)"
        )

    return oid


class _CountingStream(StreamReader):
    """Thin read wrapper tracking how many bytes were consumed."""

    def __init__(self, fobj: StreamReader) -> None:
        self._fobj = fobj
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._fobj.read(size)
        self.bytes_read += len(chunk)
        return chunk

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._fobj, attr)


class ConditionalObjectDB(ObjectDB):
    """ObjectDB with atomic ``put_if_absent`` / ``compare_and_put``.

    Args:
        fs: Backing filesystem.
        path: Store root.
        read_only: When true, every write path raises
            :class:`ObjectDBPermissionError` before touching the store.
        writer: Explicit :class:`ConditionalWriter`. When omitted, the
            filesystem must declare one itself via an
            ``conditional_writer`` attribute; otherwise construction
            fails with :class:`ConditionalWriteUnsupportedError` instead
            of silently falling back to a racy exists-then-put sequence.
    """

    def __init__(
        self,
        fs: "FileSystem",
        path: str,
        read_only: bool = False,
        *,
        writer: Optional[ConditionalWriter] = None,
        **config: Any,
    ) -> None:
        super().__init__(fs, path, read_only, **config)

        resolved = writer if writer is not None else getattr(
            fs, "conditional_writer", None
        )
        if not isinstance(resolved, ConditionalWriter):
            raise ConditionalWriteUnsupportedError(
                f"{fs!r} does not declare an atomic conditional-write "
                "capability; refusing to emulate one with exists()+put()"
            )
        if not resolved.capability.conditional_writes:
            raise ConditionalWriteUnsupportedError(
                f"{type(resolved).__name__} declares conditional_writes=False"
            )

        self._writer = resolved

    @property
    def writer(self) -> ConditionalWriter:
        return self._writer

    @property
    def capability(self) -> Capability:
        """Guarantee matrix declared by the backing writer."""
        return self._writer.capability

    def _require_writable(self) -> None:
        if self.read_only:
            raise ObjectDBPermissionError(
                "Cannot perform a conditional write on a read-only ODB"
            )

    def _checked_path(self, oid: str) -> str:
        validate_oid(oid, self.fs)
        return self.oid_to_path(oid)

    @staticmethod
    def _as_stream(
        data: Union[bytes, BinaryIO],
    ) -> tuple[StreamReader, Optional[int]]:
        if isinstance(data, bytes):
            return BytesIO(data), len(data)
        return data, cast("Optional[int]", getattr(data, "size", None))

    def _confirm(
        self,
        stream: _CountingStream,
        size: Optional[int],
        callback: "Callback",
    ) -> None:
        # Defensive check in addition to the writer's bounded-copy
        # contract: a confirmed publish must carry exactly the declared
        # number of bytes.
        if size is not None and stream.bytes_read != size:
            raise ShortWriteError(
                f"published {stream.bytes_read} bytes but {size} were declared"
            )
        # Only bytes of an actually confirmed publish are reported.
        callback.relative_update(stream.bytes_read)

    def revision(self, oid: str) -> Optional[str]:
        """Return the current opaque revision of ``oid`` or ``None``."""
        return self._writer.get_revision(self._checked_path(oid))

    def put_if_absent(
        self,
        oid: str,
        data: Union[bytes, BinaryIO],
        callback: "Callback" = DEFAULT_CALLBACK,
    ) -> ConditionalResult:
        """Atomically create ``oid`` if it does not already exist.

        An existing object -- even one with different contents -- is
        never overwritten: the result is
        :attr:`ConditionalStatus.ALREADY_EXISTS` with its live revision.
        """
        self._require_writable()
        path = self._checked_path(oid)
        fobj, size = self._as_stream(data)
        stream = _CountingStream(fobj)

        self._init(self._oid_parts(oid)[0])

        try:
            revision = self._writer.put_if_absent(path, stream, size)
        except FileExistsError:
            # The backend rejected the create; read the winner's
            # revision without attempting any write of our own.
            return ConditionalResult(
                oid,
                ConditionalStatus.ALREADY_EXISTS,
                self._writer.get_revision(path),
            )

        self._confirm(stream, size, callback)
        return ConditionalResult(oid, ConditionalStatus.CREATED, revision)

    def compare_and_put(
        self,
        oid: str,
        expected_revision: str,
        data: Union[bytes, BinaryIO],
        callback: "Callback" = DEFAULT_CALLBACK,
    ) -> ConditionalResult:
        """Atomically replace ``oid`` while its revision matches.

        ``expected_revision`` is an opaque token previously returned by
        :meth:`revision`, :meth:`put_if_absent` or this method. A stale
        token -- or a missing object -- yields
        :attr:`ConditionalStatus.PRECONDITION_FAILED` and leaves the
        live object untouched.
        """
        self._require_writable()
        if not isinstance(expected_revision, str) or not expected_revision:
            raise ValueError(
                "expected_revision must be a non-empty opaque revision token"
            )

        path = self._checked_path(oid)
        fobj, size = self._as_stream(data)
        stream = _CountingStream(fobj)

        self._init(self._oid_parts(oid)[0])

        try:
            revision = self._writer.compare_and_put(
                path, expected_revision, stream, size
            )
        except PreconditionFailedError as exc:
            return ConditionalResult(
                oid,
                ConditionalStatus.PRECONDITION_FAILED,
                exc.current_revision,
            )

        self._confirm(stream, size, callback)
        return ConditionalResult(oid, ConditionalStatus.CREATED, revision)


__all__ = [
    "ConditionalObjectDB",
    "ConditionalResult",
    "ConditionalStatus",
    "validate_oid",
]
