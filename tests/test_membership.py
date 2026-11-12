import json
import logging
import random
import threading
import time
from typing import Any

import pytest

from dvc_objects.db import ObjectDB
from dvc_objects.fs.memory import MemoryFileSystem
from dvc_objects.membership import (
    _HEADER_LEN,
    MAX_OID_BYTES,
    SHARD_VERSION,
    MembershipIndex,
    MembershipIndexDisabledError,
    MembershipIndexError,
    MembershipIndexLimitError,
    _pack_frame,
    _parse_journal,
    _Sketch,
)


def oid(value: int) -> str:
    return f"{value:032x}"


class CountingMemoryFileSystem(MemoryFileSystem):
    """Memory FS that counts remote-ish HEAD/LIST style operations.

    ``head`` counts individual object-existence checks at the raw fsspec
    level (i.e. what a remote would serve as one HEAD request), while
    ``find``/``ls`` count listing operations.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = {"head": 0, "find": 0, "ls": 0, "isfile": 0}
        raw_exists = self.fs.exists

        def count_exists(path, *args, **kwargs):
            self.calls["head"] += 1
            return raw_exists(path, *args, **kwargs)

        self.fs.exists = count_exists

    def reset(self):
        for key in self.calls:
            self.calls[key] = 0

    def find(self, path, prefix=False, batch_size=None, **kwargs):
        self.calls["find"] += 1
        yield from super().find(path, prefix=prefix, batch_size=batch_size, **kwargs)

    def ls(self, path: str, detail: bool = False, **kwargs: Any):
        self.calls["ls"] += 1
        # ``detail`` is a runtime bool; the base class models it as a
        # pair of literal overloads.
        return super().ls(  # type: ignore[call-overload]
            path, detail=detail, **kwargs
        )

    def isfile(self, path: str) -> bool:
        self.calls["isfile"] += 1
        return super().isfile(path)


class NoMoveFileSystem(MemoryFileSystem):
    @property
    def mv(self):
        raise AttributeError("mv is not supported on this backend")


class NoAppendFileSystem(MemoryFileSystem):
    def open(self, path, mode="r", **kwargs):
        if "a" in mode:
            raise OSError("append is not supported on this backend")
        return super().open(path, mode=mode, **kwargs)


class SlowListingODB(ObjectDB):
    """ODB whose full listing waits on an event mid-rebuild."""

    def __init__(self, *args, **kwargs):
        self.listing_event = kwargs.pop("listing_event")
        super().__init__(*args, **kwargs)

    def all(self, jobs=None):
        self.listing_event.wait(5)
        yield from super().all(jobs=jobs)


def make_odb(fs, path="/odb", index=None):
    return ObjectDB(fs, path, membership_index=index)


def make_index(odb, **kwargs):
    kwargs.setdefault("writer_id", "writer-1")
    kwargs.setdefault("seed", 1234)
    kwargs.setdefault("num_shards", 16)
    return MembershipIndex(odb, **kwargs)


def assert_not_ready(index: MembershipIndex) -> None:
    # Goes through a function call so mypy does not narrow the mutable
    # ``ready`` property across a later ``rebuild()``.
    assert index.ready is False


# ---------------------------------------------------------------------------
# Construction / fallback behaviour
# ---------------------------------------------------------------------------


def test_index_files_live_outside_odb_tree(memfs):
    plain = make_odb(memfs)
    index = make_index(plain)
    odb = make_odb(memfs, index=index)
    for i in range(10):
        odb.add_bytes(oid(i), b"content")
    index.rebuild()

    assert set(odb.all()) == {oid(i) for i in range(10)}
    assert memfs.exists("/odb.membership/manifest.json")


def test_index_not_ready_before_rebuild(memfs):
    plain = make_odb(memfs)
    index = make_index(plain)
    odb = make_odb(memfs, index=index)
    odb.add_bytes(oid(1), b"content")

    assert_not_ready(index)
    negative, uncertain = index.check({oid(2), oid(3)})
    assert negative == set()
    assert uncertain == {oid(2), oid(3)}


def test_unsupported_backend_disables_index(memfs):
    fs = NoMoveFileSystem(global_store=False)
    plain = ObjectDB(fs, "/odb")
    index = make_index(plain)

    assert_not_ready(index)
    assert index.disabled_reason is not None
    assert index.check({oid(1)}) == (set(), {oid(1)})
    plain.add_bytes(oid(1), b"content")
    assert plain.oids_exist([oid(1), oid(2)]) == [oid(1)]


def test_backend_without_journal_append_falls_back(memfs):
    fs = NoAppendFileSystem(global_store=False)
    plain = ObjectDB(fs, "/odb")
    index = make_index(plain)
    odb = ObjectDB(fs, "/odb", membership_index=index)

    # The object itself is published; only the journal write fails, so the
    # index invalidates itself instead of breaking the add.
    odb.add_bytes(oid(1), b"content")
    assert_not_ready(index)
    assert odb.oids_exist([oid(1), oid(2)]) == [oid(1)]
    # A rebuild (full listing, no journal dependency) restores service.
    index.rebuild(adopt=True)
    assert index.ready is True


def test_unknown_writer_falls_back_and_adopts(memfs):
    plain = make_odb(memfs)
    index = make_index(plain, writer_id="writer-a")
    odb = make_odb(memfs, index=index)
    odb.add_bytes(oid(1), b"content")
    index.rebuild()
    index.close()

    other = make_index(plain, writer_id="writer-b")
    assert_not_ready(other)
    assert isinstance(other.disabled_reason, str)
    assert other.check({oid(2)}) == (set(), {oid(2)})

    other.rebuild(adopt=True)
    assert other.ready is True
    indexed = make_odb(memfs, index=other)
    assert indexed.oids_exist([oid(1), oid(2)]) == [oid(1)]


def test_namespace_mismatch_disables_index(memfs):
    shared_path = "/shared-membership"
    first = make_odb(memfs, "/odb-a")
    index = make_index(first, path=shared_path)
    first.add_bytes(oid(1), b"content")
    index.rebuild()
    index.close()

    second = ObjectDB(memfs, "/odb-b")
    reopened = make_index(second, path=shared_path)
    assert_not_ready(reopened)
    assert "namespace" in (reopened.disabled_reason or "")
    assert reopened.check({oid(2)}) == (set(), {oid(2)})


# ---------------------------------------------------------------------------
# Correctness: no false negatives, positives always confirmed
# ---------------------------------------------------------------------------


def test_reliable_negatives_skip_all_remote_calls(memfs):
    fs = CountingMemoryFileSystem(global_store=False)
    plain = ObjectDB(fs, "/odb")
    index = make_index(plain)
    odb = ObjectDB(fs, "/odb", membership_index=index)
    present = {oid(i) for i in range(40)}
    missing = {oid(10_000 + i) for i in range(40)}
    for value in present:
        odb.add_bytes(value, b"content")
    index.rebuild()

    fs.reset()
    assert odb.oids_exist(missing) == []
    assert fs.calls == {"head": 0, "find": 0, "ls": 0, "isfile": 0}

    fs.reset()
    result = set(odb.oids_exist(present | missing))
    assert result == present
    # Probabilistic hits still go to the real ODB.
    assert fs.calls["head"] + fs.calls["find"] > 0


def test_existing_oids_are_never_reported_from_bloom_alone(memfs):
    plain = make_odb(memfs)
    index = make_index(plain)
    odb = make_odb(memfs, index=index)
    odb.add_bytes(oid(7), b"content")
    index.rebuild()

    negative, uncertain = index.check({oid(7), oid(8)})
    assert negative == {oid(8)}
    assert uncertain == {oid(7)}
    assert set(odb.oids_exist([oid(7), oid(8)])) == {oid(7)}


def test_random_workload_matches_plain_odb(memfs):
    fs = CountingMemoryFileSystem(global_store=False)
    plain = ObjectDB(fs, "/odb")
    index = make_index(plain, num_shards=32)
    odb = ObjectDB(fs, "/odb", membership_index=index)

    # Deterministic, non-cryptographic workload; security is irrelevant here.
    rng = random.Random(20261112)  # noqa: S311
    universe = [oid(i) for i in range(400)]
    stored = set()
    for step in range(12):
        batch = rng.sample(universe, 25)
        for value in batch:
            if value not in stored:
                odb.add_bytes(value, b"content")
                stored.add(value)
        if step % 3 == 0:
            index.rebuild()
        query = set(rng.sample(universe, 60)) | {oid(900_000 + i) for i in range(20)}
        assert set(odb.oids_exist(query)) == set(plain.oids_exist(query))

    # Every currently stored oid must be reported: zero false negatives.
    assert set(odb.oids_exist(stored)) == stored
    # Fully absent queries are served without touching the backend.
    fs.reset()
    absent = {oid(800_000 + i) for i in range(50)}
    assert odb.oids_exist(absent) == []
    assert fs.calls["head"] == 0
    assert fs.calls["find"] == 0


def test_incremental_journal_adds_keep_coverage(memfs):
    plain = make_odb(memfs)
    index = make_index(plain)
    odb = make_odb(memfs, index=index)
    odb.add_bytes(oid(1), b"content")
    index.rebuild()

    later = oid(2)
    later_missing = oid(3)
    odb.add_bytes(later, b"content")
    negative, uncertain = index.check({later, later_missing})
    assert uncertain == {later}
    assert negative == {later_missing}

    # Journal replay across a clean reopen keeps the same guarantees.
    index.close()
    reopened = make_index(plain)
    assert reopened.ready is True
    assert reopened.check({later, later_missing}) == ({later_missing}, {later})


def test_delete_only_adds_false_positives(memfs):
    fs = CountingMemoryFileSystem(global_store=False)
    plain = ObjectDB(fs, "/odb")
    index = make_index(plain)
    odb = ObjectDB(fs, "/odb", membership_index=index)
    odb.add_bytes(oid(1), b"content")
    index.rebuild()

    odb.delete(oid(1))
    # The shared bits remain: the deleted oid stays "uncertain", never
    # negative, and the real query filters the false positive out.
    negative, uncertain = index.check({oid(1)})
    assert negative == set()
    assert uncertain == {oid(1)}
    fs.reset()
    assert odb.oids_exist([oid(1)]) == []
    assert fs.calls["head"] + fs.calls["find"] > 0

    # A rebuild from the current listing drops the stale bits.
    index.rebuild()
    assert index.check({oid(1)}) == ({oid(1)}, set())


def test_counting_fs_proves_reduced_head_and_list(memfs):
    fs = CountingMemoryFileSystem(global_store=False)
    # Force the per-object HEAD path so call counts map 1:1 to oids.
    fs.CAN_TRAVERSE = False
    plain = ObjectDB(fs, "/odb")
    index = make_index(plain, num_shards=16)
    odb = ObjectDB(fs, "/odb", membership_index=index)
    present = {oid(i) for i in range(30)}
    missing = {oid(20_000 + i) for i in range(60)}
    for value in present:
        odb.add_bytes(value, b"content")
    index.rebuild()
    query = present | missing

    fs.reset()
    plain_result = set(plain.oids_exist(query))
    plain_calls = fs.calls.copy()

    fs.reset()
    indexed_result = set(odb.oids_exist(query))
    indexed_calls = fs.calls.copy()

    assert indexed_result == plain_result == present
    # Every present oid is confirmed via a real HEAD; proven-absent oids
    # never reach the backend (apart from a handful of bloom false
    # positives, which are still strictly fewer than the full query).
    assert len(present) <= indexed_calls["head"] < len(query)
    assert plain_calls["head"] == len(query)
    assert indexed_calls["head"] < plain_calls["head"]

    fs.reset()
    assert odb.oids_exist(missing) == []
    assert fs.calls["head"] == 0
    assert fs.calls["find"] == 0


def test_counting_fs_proves_reduced_listing_calls(memfs):
    fs = CountingMemoryFileSystem(global_store=False)
    plain = ObjectDB(fs, "/odb")
    index = make_index(plain, num_shards=16)
    odb = ObjectDB(fs, "/odb", membership_index=index)
    for i in range(30):
        odb.add_bytes(oid(i), b"content")
    index.rebuild()
    missing = {oid(20_000 + i) for i in range(60)}

    fs.reset()
    assert plain.oids_exist(missing) == []
    plain_calls = fs.calls.copy()
    assert plain_calls["find"] > 0

    fs.reset()
    assert odb.oids_exist(missing) == []
    indexed_calls = fs.calls.copy()
    assert indexed_calls["find"] == 0
    assert indexed_calls["head"] == 0


def test_empty_query_short_circuits(memfs):
    fs = CountingMemoryFileSystem(global_store=False)
    plain = ObjectDB(fs, "/odb")
    index = make_index(plain)
    odb = ObjectDB(fs, "/odb", membership_index=index)
    index.rebuild()

    fs.reset()
    assert odb.oids_exist([]) == []
    assert fs.calls == {"head": 0, "find": 0, "ls": 0, "isfile": 0}


# ---------------------------------------------------------------------------
# Crash / corruption recovery
# ---------------------------------------------------------------------------


def test_unclean_shutdown_falls_back_until_rebuild(memfs):
    plain = make_odb(memfs)
    index = make_index(plain)
    odb = make_odb(memfs, index=index)
    odb.add_bytes(oid(1), b"content")
    index.rebuild()
    # Simulate a crash: leave the session marker in place.
    memfs.pipe_file("/odb.membership/dirty.lock", b"crashed")

    reopened = make_index(plain)
    assert_not_ready(reopened)
    assert reopened.check({oid(2)}) == (set(), {oid(2)})
    assert make_odb(memfs, index=reopened).oids_exist([oid(1), oid(2)]) == [oid(1)]

    reopened.rebuild()
    assert reopened.ready is True
    assert reopened.check({oid(2)}) == ({oid(2)}, set())


def test_truncated_journal_falls_back(memfs):
    plain = make_odb(memfs)
    index = make_index(plain)
    odb = make_odb(memfs, index=index)
    odb.add_bytes(oid(1), b"content")
    index.rebuild()
    odb.add_bytes(oid(2), b"content")
    index.close()

    journal_path = "/odb.membership/journal.log"
    data = memfs.cat_file(journal_path)
    memfs.pipe_file(journal_path, data[:-3])

    reopened = make_index(plain)
    assert_not_ready(reopened)
    assert reopened.check({oid(3)}) == (set(), {oid(3)})
    reopened.rebuild()
    assert reopened.ready is True
    indexed = make_odb(memfs, index=reopened)
    assert set(indexed.oids_exist([oid(1), oid(2), oid(3)])) == {
        oid(1),
        oid(2),
    }


def test_corrupt_shard_falls_back_and_rebuilds(memfs):
    plain = make_odb(memfs)
    index = make_index(plain, num_shards=8)
    odb = make_odb(memfs, index=index)
    for i in range(12):
        odb.add_bytes(oid(i), b"content")
    index.rebuild()
    index.close()

    shard_path = "/odb.membership/shards/0000.bf"
    blob = bytearray(memfs.cat_file(shard_path))
    blob[-5] ^= 0xFF  # flip a payload bit, CRC must catch it
    memfs.pipe_file(shard_path, bytes(blob))

    reopened = make_index(plain, num_shards=8)
    # Eager verification detects the bad shard before any query is served.
    assert_not_ready(reopened)
    negative, _ = reopened.check({oid(50_000)})
    assert negative == set()

    reopened.rebuild()
    assert reopened.ready is True
    assert set(
        make_odb(memfs, index=reopened).oids_exist(
            [oid(i) for i in range(12)] + [oid(50_000)]
        )
    ) == {oid(i) for i in range(12)}


def test_corrupt_manifest_falls_back(memfs):
    plain = make_odb(memfs)
    index = make_index(plain)
    odb = make_odb(memfs, index=index)
    odb.add_bytes(oid(1), b"content")
    index.rebuild()
    index.close()

    memfs.pipe_file("/odb.membership/manifest.json", b"{not json")
    reopened = make_index(plain)
    assert_not_ready(reopened)
    assert reopened.check({oid(2)}) == (set(), {oid(2)})
    reopened.rebuild(adopt=True)
    assert reopened.ready is True


def test_rebuild_temp_limit_aborts_without_losing_state(memfs):
    plain = make_odb(memfs)
    index = make_index(plain, max_temp_bytes=64)
    odb = make_odb(memfs, index=index)
    odb.add_bytes(oid(1), b"content")

    with pytest.raises(MembershipIndexLimitError):
        index.rebuild()
    assert_not_ready(index)
    # The ODB itself is untouched and plain queries still agree.
    assert plain.oids_exist([oid(1), oid(2)]) == [oid(1)]


# ---------------------------------------------------------------------------
# Bounded resources
# ---------------------------------------------------------------------------


def test_shard_count_is_bounded(memfs):
    plain = make_odb(memfs)
    index = make_index(plain, num_shards=16)
    odb = make_odb(memfs, index=index)
    for i in range(300):
        odb.add_bytes(oid(i), b"content")
    index.rebuild()

    shard_files = [
        path for path in memfs.find("/odb.membership/shards") if path.endswith(".bf")
    ]
    assert len(shard_files) == 16


def test_shard_cache_is_bounded(memfs):
    plain = make_odb(memfs)
    index = make_index(plain, num_shards=16, max_cache_shards=2)
    odb = make_odb(memfs, index=index)
    for i in range(64):
        odb.add_bytes(oid(i), b"content")
    index.rebuild()

    query = {oid(10_000 + i) for i in range(128)}
    negative, uncertain = index.check(query)
    assert negative == query
    assert uncertain == set()
    assert len(index._cache) <= 2


def test_oversized_and_non_string_oids_are_rejected(memfs):
    plain = make_odb(memfs)
    index = make_index(plain, num_shards=8)
    with pytest.raises(ValueError):  # noqa: PT011
        index.add_oids(["x" * (MAX_OID_BYTES + 1)])
    with pytest.raises(ValueError):  # noqa: PT011
        index.add_oids([""])
    with pytest.raises(ValueError):  # noqa: PT011
        index.add_oids([1234])  # type: ignore[list-item]

    # Valid oids still work after a rejected one.
    index.add_oids([oid(1)])
    index.rebuild()
    assert index.check({oid(2)}) == ({oid(2)}, set())


def test_malicious_oids_cannot_create_shards(memfs):
    plain = make_odb(memfs)
    index = make_index(plain, num_shards=4)
    for i in range(100):
        index.add_oids([f"{i}-{i * 7:0100x}"])
    index.rebuild()
    assert len(index._manifest["shards"]) == 4


# ---------------------------------------------------------------------------
# Serialization validation
# ---------------------------------------------------------------------------


def _make_blob(**overrides):
    sketch = _Sketch(
        shard_id=3,
        num_shards=8,
        num_bits=64,
        num_hashes=3,
        seed=99,
        generation=5,
        count=2,
    )
    sketch.add(b"oid-a")
    sketch.add(b"oid-b")
    blob = bytearray(sketch.serialize())
    if overrides:
        offset, value = overrides["offset"], overrides["value"]
        blob[offset : offset + len(value)] = value
    return bytes(blob)


def test_shard_roundtrip_and_lookups():
    sketch = _Sketch(3, 8, 64, 3, 99)
    sketch.add(b"oid-a")
    blob = sketch.serialize()
    restored = _Sketch.deserialize(
        blob,
        expected_shard=3,
        expected_num_shards=8,
        expected_num_bits=64,
        max_shard_bytes=1024,
        max_generation=10,
    )
    assert restored.contains(b"oid-a")
    assert not restored.contains(b"oid-missing")


def test_shard_deserialization_rejects_bad_inputs():
    good = _make_blob()
    kwargs = {
        "expected_shard": 3,
        "expected_num_shards": 8,
        "expected_num_bits": 64,
        "max_shard_bytes": 1024,
        "max_generation": 10,
    }
    with pytest.raises(MembershipIndexError):
        _Sketch.deserialize(good[:-5], **kwargs)  # truncated
    bad_magic = _make_blob(offset=0, value=b"XXXXXX")
    with pytest.raises(MembershipIndexError):
        _Sketch.deserialize(bad_magic, **kwargs)
    bad_version = _make_blob(offset=6, value=(SHARD_VERSION + 1).to_bytes(2, "little"))
    with pytest.raises(MembershipIndexError):
        _Sketch.deserialize(bad_version, **kwargs)
    with pytest.raises(MembershipIndexError):
        _Sketch.deserialize(
            good, **{**kwargs, "expected_shard": 4}
        )  # shard id mismatch
    with pytest.raises(MembershipIndexError):
        _Sketch.deserialize(
            good, **{**kwargs, "expected_num_bits": 128}
        )  # bit length mismatch
    corrupt = _make_blob(offset=_HEADER_LEN, value=b"\xff")
    with pytest.raises(MembershipIndexError):
        _Sketch.deserialize(corrupt, **kwargs)  # payload crc mismatch
    with pytest.raises(MembershipIndexError):
        _Sketch.deserialize(good, **{**kwargs, "max_generation": 1})


def test_shard_count_overflow_is_rejected():
    sketch = _Sketch(0, 8, 64, 3, 1)
    sketch.count = 2**32
    with pytest.raises(MembershipIndexError):
        sketch.serialize()


def test_journal_frame_validation():
    first = _pack_frame(1, b"oid-1")
    second = _pack_frame(2, b"oid-2")
    frames = _parse_journal(first + second, 1)
    assert frames == [(1, b"oid-1"), (2, b"oid-2")]

    with pytest.raises(MembershipIndexError):
        _parse_journal(first + second[:-2], 1)  # truncated tail
    with pytest.raises(MembershipIndexError):
        _parse_journal(second, 1)  # generation gap
    corrupted = bytearray(first)
    corrupted[-1] ^= 0xFF
    with pytest.raises(MembershipIndexError):
        _parse_journal(bytes(corrupted), 1)


def test_manifest_rejects_tampered_fields(memfs):
    plain = make_odb(memfs)
    index = make_index(plain, num_shards=8)
    odb = make_odb(memfs, index=index)
    odb.add_bytes(oid(1), b"content")
    index.rebuild()
    index.close()

    path = "/odb.membership/manifest.json"
    manifest = json.loads(memfs.cat_file(path))
    manifest["shard_bits"] = manifest["shard_bits"] * 2
    memfs.pipe_file(path, json.dumps(manifest).encode())
    reopened = make_index(plain, num_shards=8)
    assert_not_ready(reopened)


# ---------------------------------------------------------------------------
# Logging safety
# ---------------------------------------------------------------------------


def test_raw_oids_never_reach_default_logs(memfs, caplog):
    plain = make_odb(memfs)
    index = make_index(plain, num_shards=8)
    odb = make_odb(memfs, index=index)
    secret_oids = [f"deadbeef{i:024x}" for i in range(5)]
    with caplog.at_level(logging.DEBUG, logger="dvc_objects.membership"):
        for value in secret_oids:
            odb.add_bytes(value, b"content")
        index.rebuild()
        odb.delete(secret_oids[0])
        index.check({secret_oids[1], "cafebabe" * 4})
        index.invalidate("synthetic invalidation")
        index.rebuild()

    messages = "\n".join(record.getMessage() for record in caplog.records)
    for value in secret_oids:
        assert value not in messages
    # Only structural identifiers are logged, never raw oid material.
    assert "generation" in messages


# ---------------------------------------------------------------------------
# Concurrency: adds interleaved with rebuilds
# ---------------------------------------------------------------------------


def test_concurrent_adds_and_rebuilds_keep_parity(memfs):
    plain = ObjectDB(memfs, "/odb")
    index = make_index(plain, num_shards=16, seed=7)
    odb = ObjectDB(memfs, "/odb", membership_index=index)
    n_threads, per_thread = 6, 40
    barrier = threading.Barrier(n_threads + 1)
    errors: list[Exception] = []
    rebuild_errors: list[Exception] = []

    def adder(thread_id):
        barrier.wait()
        for i in range(thread_id * per_thread, (thread_id + 1) * per_thread):
            try:
                odb.add_bytes(oid(i), b"content")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [
        threading.Thread(target=adder, args=(thread_id,))
        for thread_id in range(n_threads)
    ]
    for thread in threads:
        thread.start()
    barrier.wait()  # release all adders together
    for _ in range(5):
        time.sleep(0.02)
        try:
            index.rebuild()
        except Exception as exc:  # noqa: BLE001 - transient listing race
            rebuild_errors.append(exc)
    for thread in threads:
        thread.join()
    assert not errors
    # A racing listing may fail mid-rebuild, but it must never corrupt the
    # committed index; the final rebuild below is required to succeed.
    assert not any(isinstance(exc, MembershipIndexError) for exc in rebuild_errors)

    index.rebuild()
    total = n_threads * per_thread
    present = {oid(i) for i in range(total)}
    assert set(odb.oids_exist(present)) == present
    missing = {oid(90_000 + i) for i in range(100)}
    assert odb.oids_exist(missing) == []
    assert set(plain.oids_exist(present | missing)) == present


def test_concurrent_add_during_rebuild_is_merged_after_cut(memfs):
    event = threading.Event()
    slow_odb = SlowListingODB(memfs, "/odb", listing_event=event)
    index = make_index(slow_odb, num_shards=8)
    odb = ObjectDB(memfs, "/odb", membership_index=index)
    odb.add_bytes(oid(1), b"content")
    index.rebuild()

    concurrent = oid(2)
    concurrent_missing = oid(3)

    def rebuilder():
        index.rebuild()

    thread = threading.Thread(target=rebuilder)
    thread.start()
    # The rebuild took its cut and is blocked inside the ODB listing.
    time.sleep(0.05)
    odb.add_bytes(concurrent, b"content")  # journaled after the cut
    event.set()
    thread.join()

    assert index.ready is True
    negative, uncertain = index.check({concurrent, concurrent_missing})
    assert uncertain == {concurrent}
    assert negative == {concurrent_missing}
    assert set(odb.oids_exist([concurrent, concurrent_missing])) == {concurrent}


def test_rebuild_failure_does_not_destroy_previous_index(memfs):
    plain = make_odb(memfs)
    index = make_index(plain, num_shards=8)
    odb = make_odb(memfs, index=index)
    odb.add_bytes(oid(1), b"content")
    index.rebuild()
    assert index.ready is True

    def boom(jobs=None):
        raise RuntimeError("listing exploded")

    index._odb.all = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="listing exploded"):
        index.rebuild()
    # The previously committed index still serves correct negatives.
    assert index.ready is True
    assert index.check({oid(2)}) == ({oid(2)}, set())
    assert set(odb.oids_exist([oid(1), oid(2)])) == {oid(1)}


def test_adopt_rebuild_requires_explicit_flag(memfs):
    plain = make_odb(memfs)
    index = make_index(plain, writer_id="writer-a")
    odb = make_odb(memfs, index=index)
    odb.add_bytes(oid(1), b"content")
    index.rebuild()
    index.close()

    foreign = make_index(plain, writer_id="writer-b")
    with pytest.raises(MembershipIndexDisabledError):
        foreign.rebuild()
    assert_not_ready(foreign)
