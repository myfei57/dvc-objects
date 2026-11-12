from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .base import FileSystem


class AuthError(Exception):
    pass


class ConfigError(Exception):
    pass


class RemoteMissingDepsError(Exception):
    def __init__(
        self,
        fs: "FileSystem",
        protocol: str,
        url: str,
        missing: Optional[list[str]] = None,
    ) -> None:
        self.protocol = protocol
        self.fs = fs
        self.url = url
        self.missing_deps = missing or []
        super().__init__(
            f"filesystem for '{protocol}': '{type(fs)}' missing dependencies"
        )


class PlanOverlappingTransferError(Exception):
    """Base class for errors raised by :class:`PlanOverlappingTransfer`."""


class PlanPrecheckError(PlanOverlappingTransferError):
    """Raised when a planned transfer fails the read-only precheck.

    Precheck failures always happen before anything is written.
    """


class PlanUnsupportedFileError(PlanPrecheckError):
    """Raised when a planned path is not a supported regular file.

    Symlinks and non-regular files (directories, devices, fifos, ...) are
    rejected explicitly.
    """


class PlanCrossDeviceError(PlanPrecheckError):
    """Raised when an atomic replace strategy would cross device boundaries."""


class PlanStagingSpaceError(PlanOverlappingTransferError):
    """Raised when the owned staging area cannot hold the required bytes.

    This is always raised before any destination is overwritten.
    """


class PlanConflictError(PlanOverlappingTransferError):
    """Raised when a source is modified externally while the plan is running."""


class PlanJournalError(PlanOverlappingTransferError):
    """Raised when a recovery journal is missing, foreign or inconsistent."""
