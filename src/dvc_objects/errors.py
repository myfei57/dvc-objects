class ObjectFormatError(Exception):
    pass


class ObjectDBError(Exception):
    pass


class ObjectDBPermissionError(ObjectDBError):
    pass


class TransferSessionError(Exception):
    """Base class for resumable transfer-session errors."""


class SessionLockedError(TransferSessionError):
    """Raised when another active session already owns the ledger."""


class LedgerMismatchError(TransferSessionError):
    """Raised when a ledger was created for a different transfer plan."""


class SourceChangedError(TransferSessionError):
    """Raised when an unversioned source object changes during a transfer."""


class AtomicPublishUnsupportedError(TransferSessionError):
    """Raised in strict mode when a destination cannot publish atomically."""


class TransferCancelledError(TransferSessionError):
    """Raised when a session run is stopped via :meth:`TransferSession.cancel`."""

    def __init__(self, message=None, result=None):
        super().__init__(message or "transfer session was cancelled")
        self.result = result
