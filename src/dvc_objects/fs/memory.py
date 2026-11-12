from .base import FileSystem


class MemoryFileSystem(FileSystem):
    protocol = "memory"
    PARAM_CHECKSUM = "md5"
    # The in-memory store is mutated through single, GIL-protected dictionary
    # operations, so publishing a tmp entry never exposes partial content.
    CAN_ATOMIC_PUBLISH = True

    def __init__(self, global_store=True, fs=None, **kwargs):
        super().__init__(fs=fs, **kwargs)
        if fs is None:
            import fsspec

            self.fs = fsspec.filesystem("memory", **self.fs_args)
            if not global_store:
                self.fs.store = {}
                self.fs.pseudo_dirs = [""]

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.fs.store is other.fs.store

    __hash__ = FileSystem.__hash__
