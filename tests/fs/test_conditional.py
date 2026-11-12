import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Optional

import pytest

from dvc_objects.cond import ConditionalObjectDB, ConditionalStatus
from dvc_objects.errors import (
    AmbiguousCommitError,
    ObjectDBPermissionError,
    PreconditionFailedError,
    ShortWriteError,
)
from dvc_objects.fs.conditional import LocalConditionalWriter
from dvc_objects.fs.local import LocalFileSystem


class BlockingStream:
    """Yields one chunk, then blocks until an event is set."""

    def __init__(self, data: bytes, event: threading.Event):
        self._bio = BytesIO(data)
        self._event = event
        self._opened = False

    def read(self, size: int = -1) -> bytes:
        if not self._opened:
            self._opened = True
            return self._bio.read(size)
        self._event.wait(5)
        return self._bio.read(size)


class AmbiguousLocalWriter(LocalConditionalWriter):
    """Publish succeeds at the OS level but is reported as uncertain."""

    def _publish_replace(self, tmp_path, path):
        super()._publish_replace(tmp_path, path)
        raise AmbiguousCommitError(path)


def _mp_put_worker(args):
    base, oid, data = args
    odb = ConditionalObjectDB(LocalFileSystem(), base)
    result = odb.put_if_absent(oid, data)
    return result.status.value, result.revision


def test_capability_flags_are_declared_separately():
    cap = LocalConditionalWriter().capability
    assert cap.conditional_writes is True
    assert cap.atomic_visibility is True
    assert cap.fsync_before_publish is True
    # Crash durability is an independent, platform-specific claim.
    assert cap.crash_durable is (os.name != "nt")


def test_put_if_absent_is_exclusive_and_never_overwrites(tmp_path):
    writer = LocalConditionalWriter()
    path = str(tmp_path / "ab" / "object")

    revision = writer.put_if_absent(path, BytesIO(b"first"), 5)
    assert isinstance(revision, str)
    assert revision
    assert writer.get_revision(path) == revision

    with pytest.raises(FileExistsError):
        writer.put_if_absent(path, BytesIO(b"second"), 6)
    assert Path(path).read_bytes() == b"first"
    assert writer.get_revision(path) == revision
    assert writer.get_revision(str(tmp_path / "nope")) is None


def test_compare_and_put_enforces_revision_generation(tmp_path):
    writer = LocalConditionalWriter()
    path = str(tmp_path / "ab" / "object")
    rev1 = writer.put_if_absent(path, BytesIO(b"v1"), 2)

    with pytest.raises(PreconditionFailedError) as exc:
        writer.compare_and_put(path, "stale-token", BytesIO(b"v2"), 2)
    assert exc.value.current_revision == rev1
    assert Path(path).read_bytes() == b"v1"

    rev2 = writer.compare_and_put(path, rev1, BytesIO(b"v2"), 2)
    assert rev2 != rev1
    assert Path(path).read_bytes() == b"v2"
    assert writer.get_revision(path) == rev2

    # The previous generation can no longer replace the new one.
    with pytest.raises(PreconditionFailedError):
        writer.compare_and_put(path, rev1, BytesIO(b"v3"), 2)
    assert Path(path).read_bytes() == b"v2"


def test_compare_and_put_on_missing_object_fails_precondition(tmp_path):
    writer = LocalConditionalWriter()
    path = str(tmp_path / "ab" / "object")
    with pytest.raises(PreconditionFailedError) as exc:
        writer.compare_and_put(path, "anything", BytesIO(b"v"), 1)
    assert exc.value.current_revision is None
    assert not os.path.exists(path)


def test_short_write_cleans_temp_and_publishes_nothing(tmp_path):
    writer = LocalConditionalWriter()
    parent = tmp_path / "ab"
    path = str(parent / "object")

    with pytest.raises(ShortWriteError):
        writer.put_if_absent(path, BytesIO(b"x"), 10)

    assert not os.path.exists(path)
    assert [node.name for node in parent.iterdir()] == []


def test_no_half_published_object_while_stream_is_open(tmp_path):
    writer = LocalConditionalWriter()
    path = tmp_path / "ab" / "object"
    event = threading.Event()
    stream = BlockingStream(b"x" * 1024 + b"y" * 1024, event)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(writer.put_if_absent, str(path), stream, None)

        deadline = time.monotonic() + 5
        while (
            not list((tmp_path / "ab").glob(".*.tmp"))
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        # Spooling in progress: only the owned temp exists, the formal
        # object must never expose partial bytes.
        assert not path.exists()
        assert len(list((tmp_path / "ab").glob(".*.tmp"))) == 1

        event.set()
        revision = future.result(5)

    assert isinstance(revision, str)
    assert path.read_bytes() == b"x" * 1024 + b"y" * 1024
    assert not list((tmp_path / "ab").glob(".*.tmp"))


def test_foreign_temp_and_lock_are_never_removed(tmp_path):
    writer = LocalConditionalWriter()
    parent = tmp_path / "ab"
    parent.mkdir()
    foreign_tmp = parent / ".object.999999.foreign.tmp"
    foreign_tmp.write_bytes(b"not ours")
    foreign_lock = parent / ".object.lock"
    foreign_lock.write_bytes(b"other-owner-token")

    with pytest.raises(ShortWriteError):
        writer.put_if_absent(str(parent / "object"), BytesIO(b"x"), 10)

    # Releasing a lock we do not own must leave it in place.
    writer._release_lock(str(foreign_lock), "our-different-token")
    assert foreign_lock.exists()
    assert foreign_tmp.exists()

    revision = writer.put_if_absent(str(parent / "object"), BytesIO(b"ok"), 2)
    assert revision
    assert foreign_tmp.exists()
    assert foreign_lock.exists()
    assert (parent / "object").read_bytes() == b"ok"


def test_stale_lock_takeover_and_fresh_lock_timeout(tmp_path):
    lock_path = str(tmp_path / ".object.lock")

    # A fresh lock owned by someone else blocks acquisition. A large
    # stale threshold keeps this independent of filesystem mtime
    # granularity (notably coarse NTFS timestamps).
    blocking = LocalConditionalWriter(
        lock_timeout=0.3, lock_poll_interval=0.02, lock_stale_after=3600
    )
    with open(lock_path, "wb") as lock_file:
        lock_file.write(b"other")
    with pytest.raises(TimeoutError):
        blocking._acquire_lock(lock_path, "us")
    assert os.path.exists(lock_path)

    # Once the lock ages past the stale threshold it may be taken over.
    old = time.time() - 3600
    os.utime(lock_path, (old, old))
    takeover = LocalConditionalWriter(
        lock_timeout=1.0, lock_poll_interval=0.02, lock_stale_after=0.05
    )
    takeover._acquire_lock(lock_path, "us")
    takeover._release_lock(lock_path, "us")
    assert not os.path.exists(lock_path)


def test_thread_race_put_if_absent_single_winner(tmp_path):
    writer = LocalConditionalWriter()
    path = str(tmp_path / "ab" / "object")
    barrier = threading.Barrier(8)
    outcomes: list[tuple[str, Optional[str]]] = []

    def worker(index):
        barrier.wait()
        try:
            revision = writer.put_if_absent(
                path, BytesIO(f"payload-{index}".encode() * 100), None
            )
            outcomes.append(("created", revision))
        except FileExistsError:
            outcomes.append(("exists", None))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    created = [outcome for outcome in outcomes if outcome[0] == "created"]
    assert len(created) == 1
    assert len(outcomes) == 8
    assert Path(path).read_bytes().startswith(b"payload-")
    assert not list((tmp_path / "ab").glob(".*.tmp"))
    assert not list((tmp_path / "ab").glob(".*.lock"))


def test_thread_race_cas_only_one_generation_advances(tmp_path):
    writer = LocalConditionalWriter()
    path = str(tmp_path / "ab" / "object")
    rev1 = writer.put_if_absent(path, BytesIO(b"v1"), 2)

    barrier = threading.Barrier(6)
    outcomes: list[tuple[str, Optional[str]]] = []

    def worker(index):
        barrier.wait()
        try:
            revision = writer.compare_and_put(
                path, rev1, BytesIO(f"v{index}".encode() * 10), None
            )
            outcomes.append(("created", revision))
        except PreconditionFailedError:
            outcomes.append(("failed", None))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    created = [outcome for outcome in outcomes if outcome[0] == "created"]
    assert len(created) == 1
    assert len([outcome for outcome in outcomes if outcome[0] == "failed"]) == 5
    assert writer.get_revision(path) == created[0][1]
    assert not list((tmp_path / "ab").glob(".*.lock"))


def test_ambiguous_publish_is_reconcilable_and_cleans_up(tmp_path):
    writer = AmbiguousLocalWriter()
    path = str(tmp_path / "ab" / "object")
    rev1 = writer.put_if_absent(path, BytesIO(b"v1"), 2)

    with pytest.raises(AmbiguousCommitError):
        writer.compare_and_put(path, rev1, BytesIO(b"v2"), 2)

    # The replacement actually happened: reconcile via the live revision.
    rev2 = writer.get_revision(path)
    assert rev2 != rev1
    assert Path(path).read_bytes() == b"v2"
    assert not list((tmp_path / "ab").glob(".*.tmp"))
    assert not list((tmp_path / "ab").glob(".*.lock"))


def test_fsync_runs_before_publish(tmp_path, mocker):
    fsync = mocker.patch("dvc_objects.fs.conditional.os.fsync")
    writer = LocalConditionalWriter()
    writer.put_if_absent(str(tmp_path / "ab" / "object"), BytesIO(b"d"), 1)
    assert fsync.called


def test_multiprocess_put_if_absent_race(tmp_path):
    context = multiprocessing.get_context("spawn")
    oid = "abcd1234ef"
    payloads = [
        (str(tmp_path), oid, f"writer-{index}".encode() * 1000)
        for index in range(6)
    ]

    with context.Pool(6) as pool:
        results = pool.map(_mp_put_worker, payloads)

    created = [result for result in results if result[0] == "created"]
    assert len(created) == 1
    assert {result[0] for result in results} == {"created", "already_exists"}

    final_path = tmp_path / "ab" / "cd1234ef"
    final_bytes = final_path.read_bytes()
    assert final_bytes in {payload[2] for payload in payloads}

    # Every observer agrees on the single winning revision.
    assert all(result[1] == created[0][1] for result in results)

    leftovers = [
        node.name
        for node in (tmp_path / "ab").iterdir()
        if node.name.startswith(".")
    ]
    assert leftovers == []


def test_conditional_db_readonly_and_status_on_local(tmp_path):
    odb = ConditionalObjectDB(LocalFileSystem(), str(tmp_path), read_only=True)
    with pytest.raises(ObjectDBPermissionError):
        odb.put_if_absent("abcd1234aa", b"x")
    with pytest.raises(ObjectDBPermissionError):
        odb.compare_and_put("abcd1234aa", "rev", b"x")

    writable = ConditionalObjectDB(LocalFileSystem(), str(tmp_path))
    result = writable.put_if_absent("abcd1234aa", b"x")
    assert result.status is ConditionalStatus.CREATED
