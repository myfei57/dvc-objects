from typing import Optional


class ObjectFormatError(Exception):
    pass


class ObjectDBError(Exception):
    pass


class ObjectDBPermissionError(ObjectDBError):
    pass


class InvalidOIDError(ObjectDBError):
    """The supplied OID cannot be mapped to a safe in-store path."""


class ConditionalWriteUnsupportedError(ObjectDBError):
    """The underlying filesystem does not declare atomic conditional writes.

    A plain ``exists()`` followed by ``put_file()`` (or a rename that
    overwrites an existing object) must not be used to emulate a
    no-replace publish, so callers fail explicitly instead.
    """


class ShortWriteError(ObjectDBError):
    """The input stream ended before (or after) its declared size."""


class PreconditionFailedError(ObjectDBError):
    """A compare-and-swap revision precondition did not match.

    ``current_revision`` is the revision observed at commit time (``None``
    when the object did not exist). It is an opaque token and must not be
    interpreted as a content digest.
    """

    def __init__(
        self,
        path: str,
        expected_revision: str,
        current_revision: Optional[str],
    ) -> None:
        self.path = path
        self.expected_revision = expected_revision
        self.current_revision = current_revision
        super().__init__(
            f"precondition failed for {path!r}: expected revision "
            f"{expected_revision!r}, current revision is {current_revision!r}"
        )


class AmbiguousCommitError(ObjectDBError):
    """The publish may or may not have happened.

    Raised when the transport returned an error after issuing the
    conditional commit, so the local outcome cannot be inferred. The
    caller must reconcile by reading the current opaque revision instead
    of retrying blindly.
    """

    def __init__(self, path: str, reason: Optional[BaseException] = None) -> None:
        self.path = path
        self.reason = reason
        super().__init__(
            f"commit outcome for {path!r} is unknown; "
            f"reconcile via the current revision before retrying"
        )
