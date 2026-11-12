from io import BytesIO
from uuid import uuid4

import pytest

from dvc_objects.cond import (
    ConditionalObjectDB,
    ConditionalResult,
    ConditionalStatus,
    validate_oid,
)
from dvc_objects.db import ObjectDB
from dvc_objects.errors import (
    AmbiguousCommitError,
    ConditionalWriteUnsupportedError,
    InvalidOIDError,
    ObjectDBPermissionError,
    PreconditionFailedError,
    ShortWriteError,
)
from dvc_objects.fs.conditional import (
    Capability,
    RemoteConditionalAdapter,
    copy_bounded,
    opaque_token,
)
from dvc_objects.fs.local import LocalFileSystem
from dvc_objects.fs.memory import MemoryFileSystem


class RecordingCallback:
    """Minimal fsspec-like callback recording confirmed byte ticks."""

    def __init__(self):
        self.updates = []
        self.sizes = []

    def set_size(self, size):
        self.sizes.append(size)

    def relative_update(self, inc=1):
        self.updates.append(inc)


class ControllableRemoteAdapter(RemoteConditionalAdapter):
    """In-memory remote adapter modelling ETag/generation-gated writes.

    Every revision is an opaque token (see ``opaque_token``); the adapter
    never interprets it as a content digest. Faults can be injected to
    simulate token rotation, ambiguous commits and cleanup failures.
    """

    def __init__(self, capability=None, fail_cleanup=False):
        super().__init__(capability or Capability(True, True, False, False))
        self.store = {}
        self.deletions = []
        self.ambiguous_next = False
        self.before_commit = None
        self._fail_cleanup = fail_cleanup

    @staticmethod
    def _new_revision():
        return opaque_token("etag", uuid4().hex)

    def _head_revision(self, path):
        entry = self.store.get(path)
        return entry[0] if entry is not None else None

    def _put_if_none_match(self, path, data):
        if path in self.store:
            raise FileExistsError(path)
        revision = self._new_revision()
        self.store[path] = (revision, data)
        if self.ambiguous_next:
            self.ambiguous_next = False
            raise AmbiguousCommitError(path)
        return revision

    def _put_if_match(self, path, expected_revision, data):
        if self.before_commit is not None:
            self.before_commit(path)
        entry = self.store.get(path)
        if entry is None or entry[0] != expected_revision:
            current = entry[0] if entry is not None else None
            raise PreconditionFailedError(path, expected_revision, current)
        revision = self._new_revision()
        self.store[path] = (revision, data)
        if self.ambiguous_next:
            self.ambiguous_next = False
            raise AmbiguousCommitError(path)
        return revision

    def _remove_tmp(self, staging):
        self.deletions.append(staging)
        self.store.pop(staging, None)
        if self._fail_cleanup:
            self._fail_cleanup = False
            raise OSError(f"simulated cleanup failure for {staging}")

    def put_if_absent(self, path, fobj, size=None):
        data = self._drain(fobj, size)
        staging = f"{path}#staging#{uuid4().hex}"
        self.store[staging] = (opaque_token("tmp", uuid4().hex), data)
        try:
            revision = self._put_if_none_match(path, data)
        except BaseException:
            self._safe_remove(staging)
            raise
        self._safe_remove(staging)
        return revision

    def compare_and_put(self, path, expected_revision, fobj, size=None):
        data = self._drain(fobj, size)
        staging = f"{path}#staging#{uuid4().hex}"
        self.store[staging] = (opaque_token("tmp", uuid4().hex), data)
        try:
            revision = self._put_if_match(path, expected_revision, data)
        except BaseException:
            self._safe_remove(staging)
            raise
        self._safe_remove(staging)
        return revision

    def _safe_remove(self, staging):
        try:
            self._remove_tmp(staging)
        except OSError:
            # Real adapters log and surface this; tests assert that no
            # committed object was ever the target of a deletion.
            pass

    def committed_keys(self):
        return {key for key in self.store if "#staging#" not in key}


class NoCapabilityWriter(ControllableRemoteAdapter):
    @property
    def capability(self):
        return Capability(False, False, False, False)


@pytest.fixture
def adapter():
    return ControllableRemoteAdapter()


@pytest.fixture
def odb(adapter):
    return ConditionalObjectDB(
        MemoryFileSystem(global_store=False), "/odb", writer=adapter
    )


def test_construction_requires_declared_capability(memfs):
    with pytest.raises(ConditionalWriteUnsupportedError):
        ConditionalObjectDB(memfs, "/odb")


def test_construction_rejects_writer_without_capability(memfs):
    writer = NoCapabilityWriter()
    with pytest.raises(ConditionalWriteUnsupportedError):
        ConditionalObjectDB(memfs, "/odb", writer=writer)


def test_capability_is_exposed(odb, adapter):
    assert odb.capability is adapter.capability
    assert odb.capability.conditional_writes is True


@pytest.mark.parametrize(
    "bad_oid",
    [
        "",
        "x",
        "ab",
        "a/bcd",
        "a\\bcd",
        "a\x00bcd",
        "..x",
        "ab..",
        "ab.",
        "ab.cd.tmp",
        "ab.cd.lock",
        ".hidden",
    ],
)
def test_invalid_oids_rejected(memfs, bad_oid):
    with pytest.raises(InvalidOIDError):
        validate_oid(bad_oid, memfs)


def test_opaque_non_md5_oids_are_accepted(odb):
    # OIDs are opaque identifiers, not assumed to be MD5: long tokens,
    # base32-ish alphabets and dashes must all work.
    for oid in ["z9-z9-z9-z9-z9", "SHA256_x" * 4, "0" * 64]:
        result = odb.put_if_absent(oid, b"payload")
        assert result.status is ConditionalStatus.CREATED
        assert result.published is True
        assert isinstance(result.revision, str)
        assert result.revision


def test_put_if_absent_create_then_conflict_never_overwrites(odb, adapter):
    first = odb.put_if_absent("abcdef1234", b"original")
    assert first.status is ConditionalStatus.CREATED

    conflict = odb.put_if_absent("abcdef1234", b"totally different bytes")
    assert conflict.status is ConditionalStatus.ALREADY_EXISTS
    assert conflict.published is False
    # The existing object's revision is reported, untouched.
    assert conflict.revision == first.revision
    assert adapter.store[odb.oid_to_path("abcdef1234")][1] == b"original"


def test_compare_and_put_lifecycle(odb):
    created = odb.put_if_absent("cafe000011", b"v1")
    rev1 = created.revision

    # CAS on a missing object fails the precondition.
    missing = odb.compare_and_put("cafe000022", "anything", b"v")
    assert missing.status is ConditionalStatus.PRECONDITION_FAILED
    assert missing.revision is None

    # Stale token cannot replace the current generation.
    stale = odb.compare_and_put("cafe000011", "stale-token", b"v2")
    assert stale.status is ConditionalStatus.PRECONDITION_FAILED
    assert stale.revision == rev1

    swapped = odb.compare_and_put("cafe000011", rev1, b"v2-bytes")
    assert swapped.status is ConditionalStatus.CREATED
    assert swapped.revision != rev1

    # The old revision is now rejected even though it was valid before.
    old = odb.compare_and_put("cafe000011", rev1, b"v3")
    assert old.status is ConditionalStatus.PRECONDITION_FAILED
    assert old.revision == swapped.revision

    assert odb.revision("cafe000011") == swapped.revision


def test_token_rotation_between_check_and_commit(odb, adapter):
    created = odb.put_if_absent("abcd000099", b"v1")

    def rotate(path):
        # Another writer bumps the generation before our commit lands.
        adapter.store[path] = (opaque_token("gen", "99"), b"concurrent")

    adapter.before_commit = rotate
    result = odb.compare_and_put("abcd000099", created.revision, b"mine")
    assert result.status is ConditionalStatus.PRECONDITION_FAILED
    assert result.revision != created.revision
    assert adapter.store[odb.oid_to_path("abcd000099")][1] == b"concurrent"
    adapter.before_commit = None


def test_callback_only_counts_confirmed_publishes(odb):
    callback = RecordingCallback()
    odb.put_if_absent("beef000001", b"a" * 123, callback=callback)
    odb.put_if_absent("beef000001", b"b" * 999, callback=callback)
    rev = odb.revision("beef000001")
    odb.compare_and_put("beef000001", "wrong", b"c", callback=callback)

    assert callback.updates == [123]

    odb.compare_and_put("beef000001", rev, b"d" * 7, callback=callback)
    assert callback.updates == [123, 7]


def test_short_write_and_oversize_leave_nothing_behind(odb, adapter):
    class ShortStream(BytesIO):
        size = 64

    with pytest.raises(ShortWriteError):
        odb.put_if_absent("dead000001", ShortStream(b"only 16 bytes...."))

    class OversizeStream(BytesIO):
        size = 4

    with pytest.raises(ShortWriteError):
        odb.put_if_absent("dead000002", OversizeStream(b"six!!!"))

    assert odb.revision("dead000001") is None
    assert odb.revision("dead000002") is None
    assert not adapter.committed_keys()
    # All staging temp objects were cleaned up.
    assert not [key for key in adapter.store if "#staging#" in key]


def test_copy_bounded_direct_contract():
    sink = BytesIO()
    written = copy_bounded(BytesIO(b"x" * 10), sink, 10, chunk_size=3)
    assert written == 10
    assert sink.getvalue() == b"x" * 10

    with pytest.raises(ShortWriteError):
        copy_bounded(BytesIO(b"x" * 2), BytesIO(), 5)
    with pytest.raises(ShortWriteError):
        copy_bounded(BytesIO(b"x" * 6), BytesIO(), 5)

    # Without a declared size the whole stream is consumed.
    sink = BytesIO()
    assert copy_bounded(BytesIO(b"abc"), sink, None) == 3


def test_ambiguous_commit_requires_reconciliation(odb, adapter):
    adapter.ambiguous_next = True
    callback = RecordingCallback()
    with pytest.raises(AmbiguousCommitError):
        odb.put_if_absent("face000001", b"maybe", callback=callback)

    # The commit actually landed: reconcile via the current revision
    # instead of retrying blindly.
    revision = odb.revision("face000001")
    assert revision is not None
    assert callback.updates == []

    again = odb.put_if_absent("face000001", b"maybe")
    assert again.status is ConditionalStatus.ALREADY_EXISTS
    assert again.revision == revision
    assert not [key for key in adapter.store if "#staging#" in key]


def test_cleanup_failure_never_deletes_committed_objects():
    adapter = ControllableRemoteAdapter(fail_cleanup=True)
    odb = ConditionalObjectDB(
        MemoryFileSystem(global_store=False), "/odb", writer=adapter
    )

    odb.put_if_absent("feed000001", b"keep-me")
    committed = odb.oid_to_path("feed000001")

    # A failed create (conflict) and a failed CAS both run cleanup paths.
    adapter._fail_cleanup = True
    odb.put_if_absent("feed000001", b"other")
    adapter._fail_cleanup = True
    odb.compare_and_put("feed000001", "nope", b"other")

    assert committed not in adapter.deletions
    assert all("#staging#" in key for key in adapter.deletions)
    assert adapter.store[committed][1] == b"keep-me"


def test_readonly_blocks_all_writes(adapter):
    odb = ConditionalObjectDB(
        MemoryFileSystem(global_store=False), "/odb", writer=adapter, read_only=True
    )
    with pytest.raises(ObjectDBPermissionError):
        odb.put_if_absent("abcd000001", b"x")
    with pytest.raises(ObjectDBPermissionError):
        odb.compare_and_put("abcd000001", "rev", b"x")
    # Reads remain available.
    assert odb.revision("abcd000001") is None
    assert not adapter.store


def test_opaque_revision_tokens_are_never_parsed(odb):
    # ETag-style values with quotes, spaces and hashes round-trip as
    # opaque tokens; the DB only ever compares them for equality.
    weird = opaque_token("etag", '"a b # 1: v/2"')
    created = odb.put_if_absent("1234aaaaaa", b"x")
    result = odb.compare_and_put("1234aaaaaa", weird, b"y")
    assert result.status is ConditionalStatus.PRECONDITION_FAILED
    assert result.revision == created.revision

    ok = odb.compare_and_put("1234aaaaaa", created.revision, b"y")
    assert ok.status is ConditionalStatus.CREATED


def test_local_end_to_end(tmp_path):
    fs = LocalFileSystem()
    odb = ConditionalObjectDB(fs, str(tmp_path))

    created = odb.put_if_absent("9f8e7d6c5b", b"local-bytes")
    assert created.revision is not None
    target = tmp_path / "9f" / "8e7d6c5b"
    assert target.read_bytes() == b"local-bytes"

    conflict = odb.put_if_absent("9f8e7d6c5b", b"other")
    assert conflict.status is ConditionalStatus.ALREADY_EXISTS
    assert target.read_bytes() == b"local-bytes"

    swapped = odb.compare_and_put("9f8e7d6c5b", created.revision, b"fresh")
    assert swapped.status is ConditionalStatus.CREATED
    assert target.read_bytes() == b"fresh"

    leftovers = [
        node.name
        for node in (tmp_path / "9f").iterdir()
        if node.name.startswith(".")
    ]
    assert leftovers == []


def test_legacy_objectdb_add_remains_compatible(tmp_path):
    fs = LocalFileSystem()
    legacy = ObjectDB(fs, str(tmp_path))
    legacy.add_bytes("1122334455", b"legacy")

    conditional = ConditionalObjectDB(fs, str(tmp_path))
    # The conditional DB observes the legacy object and must not overwrite.
    seen = conditional.put_if_absent("1122334455", b"new")
    assert seen.status is ConditionalStatus.ALREADY_EXISTS
    assert (tmp_path / "11" / "22334455").read_bytes() == b"legacy"

    # And a legacy add of a fresh oid still works alongside the new API.
    legacy.add_bytes("5544332211", b"legacy2")
    assert conditional.revision("5544332211") is not None
    assert isinstance(seen, ConditionalResult)
