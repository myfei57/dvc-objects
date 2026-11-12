import errno
import hashlib
import json
import os
import sqlite3
from collections import Counter
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from dvc_objects.db import ObjectDB
from dvc_objects.errors import (
    AtomicPublishUnsupportedError,
    LedgerMismatchError,
    ObjectDBPermissionError,
    ObjectFormatError,
    SessionLockedError,
    TransferCancelledError,
)
from dvc_objects.fs.local import LocalFileSystem
from dvc_objects.session import (
    ErrorKind,
    Ledger,
    MapVerifier,
    Sha256Verifier,
    Status,
    TransferSession,
)

if TYPE_CHECKING:
    from dvc_objects.fs.base import FileSystem


class ProcessCrash(BaseException):
    """Simulates a hard process death: it must bypass retry envelopes."""


# --------------------------------------------------------------------------- #
# Test filesystems and callbacks
# --------------------------------------------------------------------------- #


class SpyFS:
    """Wraps a dvc_objects FileSystem and records open()/ukey() activity."""

    CAN_ATOMIC_PUBLISH = True

    def __init__(self, fs):
        self._fs = fs
        self.opens: Counter[tuple[str, str]] = Counter()
        self.open_kwargs: list[tuple[str, str, dict]] = []

    def __getattr__(self, name):
        return getattr(self._fs, name)

    @property
    def fs(self):
        return self._fs.fs

    def open(self, path, mode="r", **kwargs):
        self.opens[(path, mode)] += 1
        self.open_kwargs.append((path, mode, dict(kwargs)))
        return self._fs.open(path, mode, **kwargs)


class _CrashWriter:
    def __init__(self, fobj, boom):
        self._fobj = fobj
        self._boom = boom

    def __enter__(self):
        self._fobj.__enter__()
        return self

    def __exit__(self, *exc):
        return self._fobj.__exit__(*exc)

    def write(self, data):
        if self._boom is not None:
            boom = self._boom
            self._boom = None
            self._fobj.write(data[: max(1, len(data) // 2)])
            self._fobj.close()
            boom()
        return self._fobj.write(data)

    def __getattr__(self, name):
        return getattr(self._fobj, name)


class PartialWriteFS(SpyFS):
    """Dies after a partial write into the tmp object of chosen OIDs."""

    def __init__(self, fs, crash_tags):
        super().__init__(fs)
        self._crash_tags = set(crash_tags)

    def open(self, path, mode="r", **kwargs):
        if "w" in mode and any(tag in path for tag in self._crash_tags):
            tag = next(tag for tag in self._crash_tags if tag in path)
            self._crash_tags.discard(tag)
            fobj = self._fs.open(path, mode, **kwargs)

            def boom():
                raise ProcessCrash(f"simulated process death while writing {path}")

            return _CrashWriter(fobj, boom)
        return super().open(path, mode, **kwargs)


class NonAtomicFS(SpyFS):
    CAN_ATOMIC_PUBLISH = False

    def __init__(self, fs):
        super().__init__(fs)
        self.fail_mv_once = set()

    def mv(self, src, dst, **kwargs):
        if dst in self.fail_mv_once:
            self.fail_mv_once.discard(dst)
            with self._fs.open(src, "rb") as fobj:
                data = fobj.read()
            # Emulate a non-atomic copy+rm that crashes mid-copy, leaving a
            # half-written final path.
            self._fs.pipe_file(dst, data[: max(1, len(data) // 2)])
            raise OSError("simulated non-atomic publish failure")
        return self._fs.mv(src, dst, **kwargs)


class DriftFS(SpyFS):
    """Scripted ukey() responses to emulate an unversioned source drifting."""

    def __init__(self, fs):
        super().__init__(fs)
        self.script = {}

    def ukey(self, path):
        values = self.script.get(path)
        if values:
            return values.pop(0)
        return self._fs.ukey(path)


class VersionedMemFS(SpyFS):
    def __init__(self, fs):
        super().__init__(fs)
        self.versions = {}
        self.current = {}
        self.on_first_info = None
        self._info_seen = set()

    @property
    def version_aware(self):
        return True

    def info(self, path, **kwargs):
        info = self._fs.info(path)
        info["version_id"] = self.current[path]
        if self.on_first_info is not None and path not in self._info_seen:
            self._info_seen.add(path)
            self.on_first_info(path)
        return info

    def add_version(self, path, version, data):
        self.versions.setdefault(path, {})[version] = data
        self.current[path] = version
        self._fs.pipe_file(path, data)

    def rotate(self, path, version, data):
        self.versions[path][version] = data
        self.current[path] = version
        self._fs.pipe_file(path, data)

    def open(self, path, mode="r", **kwargs):
        version = kwargs.get("version_id")
        if "r" in mode and version is not None:
            self.opens[(path, mode)] += 1
            self.open_kwargs.append((path, mode, dict(kwargs)))
            return BytesIO(self.versions[path][str(version)])
        return super().open(path, mode, **kwargs)


class FlakyFS(SpyFS):
    def __init__(self, fs, failures):
        super().__init__(fs)
        self.failures = dict(failures)

    def open(self, path, mode="r", **kwargs):
        if "r" in mode and self.failures.get(path, 0) > 0:
            self.failures[path] -= 1
            raise OSError(errno.EIO, "simulated transient IO error")
        return super().open(path, mode, **kwargs)


class EMFileFS(SpyFS):
    def open(self, path, mode="r", **kwargs):
        if "r" in mode:
            raise OSError(errno.EMFILE, "too many open files")
        return super().open(path, mode, **kwargs)


class PermissionFS(SpyFS):
    def open(self, path, mode="r", **kwargs):
        if "w" in mode:
            raise PermissionError("simulated permission denial")
        return super().open(path, mode, **kwargs)


class CountingCallback:
    def __init__(self, cancel_after=None, session=None):
        self.events = []
        self.count = 0
        self._cancel_after = cancel_after
        self._session = session

    def on_state(self, oid, status):
        self.events.append((oid, status.value))

    def relative_update(self):
        self.count += 1
        if (
            self._cancel_after
            and self._session is not None
            and self.count >= self._cancel_after
        ):
            self._session.cancel()


class BrokenVerifier:
    __slots__ = ()

    def verifier_id(self):
        return "broken-verifier"

    def digest(self, oid):
        return None

    def verify(self, oid, fs, path):
        raise ObjectFormatError(f"verifier always rejects {oid!r}")


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


def make_data(n, prefix="oid"):
    return {f"{prefix}{i:06d}": f"content-{i}-".encode() * (i + 4) for i in range(n)}


def sha_map(data):
    return {oid: hashlib.sha256(blob).hexdigest() for oid, blob in data.items()}


def fill(odb, data):
    for oid, blob in data.items():
        odb.add_bytes(oid, blob)


def assert_bytes(odb, data):
    for oid, blob in data.items():
        assert odb.fs.cat_file(odb.oid_to_path(oid)) == blob


def find_all(odb):
    try:
        return list(odb.fs.find(odb.path))
    except Exception:  # noqa: BLE001 - an empty/non-existent root has no files
        return []


def assert_no_tmp(odb):
    assert not [path for path in find_all(odb) if path.endswith(".tmp")]


def tmp_tag(oid):
    return hashlib.sha256(oid.encode("utf-8")).hexdigest()[:16]


def make_odb(fs, path, read_only=False):
    # The recording wrappers satisfy the FileSystem contract structurally.
    return ObjectDB(cast("FileSystem", fs), path, read_only=read_only)


def ledger_of(session):
    ledger = session._ledger
    assert ledger is not None
    return ledger


@pytest.fixture
def memory_ledger():
    ledger = Ledger.memory()
    yield ledger
    ledger.close()


@pytest.fixture(params=["memory", "local"])
def odb_pair(request, tmp_path, memfs):
    kind = request.param
    ledger_path = os.path.join(str(tmp_path), "ledger.db")
    if kind == "memory":
        src = make_odb(SpyFS(memfs), "/odb1")
        dest = make_odb(SpyFS(memfs), "/odb2")
    else:
        src = make_odb(SpyFS(LocalFileSystem()), str(tmp_path / "src"))
        dest = make_odb(SpyFS(LocalFileSystem()), str(tmp_path / "dest"))
    return SimpleNamespace(
        kind=kind,
        src=src,
        dest=dest,
        ledger_path=ledger_path,
        tmp_path=str(tmp_path),
    )


# --------------------------------------------------------------------------- #
# Basic behaviour
# --------------------------------------------------------------------------- #


def test_transfer_session_basic(odb_pair):
    data = make_data(4)
    fill(odb_pair.src, data)
    callback = CountingCallback()

    session = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        verifier=MapVerifier(sha_map(data)),
        jobs=2,
        callback=callback,
    )
    result = session.run()
    assert result.ok
    assert result.verified == set(data)
    assert result.retransferred == set(data)
    assert result.reused == set()
    assert result.degraded is False
    assert session.states() == dict.fromkeys(data, Status.VERIFIED)
    assert callback.count == len(data)
    assert dict(callback.events) == dict.fromkeys(data, "verified")
    assert_bytes(odb_pair.dest, data)
    assert_no_tmp(odb_pair.dest)
    session.close()


def test_resume_is_noop_and_reuses_valid_objects(odb_pair):
    data = make_data(3)
    fill(odb_pair.src, data)
    verifier = MapVerifier(sha_map(data))

    session = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        verifier=verifier,
    )
    session.run()
    session.close()

    session2 = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        verifier=verifier,
    )
    result = session2.run()
    assert result.ok
    assert result.reused == set(data)
    assert result.retransferred == set()
    # The source must not be read again when the destination is already valid.
    for oid in data:
        assert odb_pair.src.fs.opens[(odb_pair.src.oid_to_path(oid), "rb")] == 1
    session2.close()


# --------------------------------------------------------------------------- #
# Crash windows and recovery
# --------------------------------------------------------------------------- #


def test_resume_after_crash_during_copy(odb_pair):
    data = make_data(5)
    fill(odb_pair.src, data)
    target = sorted(data)[2]

    # Replace the destination FS with one that dies while writing the target.
    raw_dest = odb_pair.dest.fs._fs
    crashing = PartialWriteFS(raw_dest, {tmp_tag(target)})
    odb_pair.dest.fs = crashing

    # One shared progress counter across both processes: every confirmation
    # must be counted exactly once over the whole interrupted lifecycle.
    callback = CountingCallback()
    session = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        jobs=1,
        callback=callback,
    )
    with pytest.raises(ProcessCrash):
        session.run()
    assert session.states()[target] is Status.IN_FLIGHT
    # Plant another orphan tmp object, as a real crash might leave behind.
    orphan = odb_pair.dest.fs.join(
        odb_pair.dest.fs.parent(odb_pair.dest.oid_to_path(target)),
        f".xfer.{session.token}.deadbeef.tmp",
    )
    odb_pair.dest.fs.pipe_file(orphan, b"partial")
    session.close()

    # Rebuild the process: a brand new session over the same durable ledger.
    odb_pair.dest.fs = SpyFS(raw_dest)
    resumed = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        jobs=1,
        callback=callback,
    )
    result = resumed.run()
    assert result.ok
    # The two objects confirmed before the crash are reused; the crashed one
    # is copied again together with the objects that never started.
    confirmed_before = set(sorted(data)[:2])
    assert result.reused == confirmed_before
    assert result.retransferred == set(data) - confirmed_before
    assert target in result.retransferred
    assert_bytes(odb_pair.dest, data)
    assert_no_tmp(odb_pair.dest)

    # Minimal retransmission: only the crashed object is read from the source
    # a second time; every other object was confirmed before the crash.
    for oid in data:
        path = odb_pair.src.oid_to_path(oid)
        expected_reads = 2 if oid == target else 1
        assert odb_pair.src.fs.opens[(path, "rb")] == expected_reads
    assert callback.count == len(data)
    resumed.close()


def test_resume_after_publish_before_ledger_commit(odb_pair, mocker):
    data = make_data(3)
    fill(odb_pair.src, data)
    target = sorted(data)[1]
    real_update = Ledger.update
    crash_state = {"done": False}

    def patched_update(self, session_id, oid, **fields):
        if (
            fields.get("status") is Status.VERIFIED
            and oid == target
            and not crash_state["done"]
        ):
            crash_state["done"] = True
            raise ProcessCrash("checkpoint write failed after publish")
        return real_update(self, session_id, oid, **fields)

    mocker.patch.object(Ledger, "update", patched_update)

    session = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        jobs=1,
    )
    with pytest.raises(ProcessCrash):
        session.run()
    # The object is visible at the final path but its row never got verified.
    assert odb_pair.dest.exists(target)
    assert session.states()[target] is Status.IN_FLIGHT
    source_reads_after_crash = odb_pair.src.fs.opens[
        (odb_pair.src.oid_to_path(target), "rb")
    ]
    session.close()

    resumed = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        jobs=1,
    )
    result = resumed.run()
    assert result.ok
    # The published target is recovered purely by verifying the destination.
    assert target in result.reused
    assert target not in result.retransferred
    assert_bytes(odb_pair.dest, data)
    assert_no_tmp(odb_pair.dest)
    # Recovery verifies the real destination instead of blindly retransmitting
    # or deleting the already published, valid object.
    assert (
        odb_pair.src.fs.opens[(odb_pair.src.oid_to_path(target), "rb")]
        == source_reads_after_crash
    )
    resumed.close()


def test_pre_existing_bad_object_is_replaced_good_one_is_kept(odb_pair):
    data = make_data(3)
    fill(odb_pair.src, data)
    verifier = MapVerifier(sha_map(data))
    bad_oid, good_oid = sorted(data)[0], sorted(data)[1]

    odb_pair.dest.makedirs(odb_pair.dest.fs.parent(odb_pair.dest.oid_to_path(bad_oid)))
    odb_pair.dest.fs.pipe_file(odb_pair.dest.oid_to_path(bad_oid), b"corrupt")
    odb_pair.dest.fs.pipe_file(odb_pair.dest.oid_to_path(good_oid), data[good_oid])

    session = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        verifier=verifier,
        jobs=1,
    )
    result = session.run()
    assert result.ok
    assert result.reused == {good_oid}
    assert bad_oid in result.retransferred
    assert_bytes(odb_pair.dest, data)
    assert odb_pair.src.fs.opens[(odb_pair.src.oid_to_path(good_oid), "rb")] == 0
    assert_no_tmp(odb_pair.dest)
    session.close()


# --------------------------------------------------------------------------- #
# Source identity: unversioned drift and versioned binding
# --------------------------------------------------------------------------- #


def test_unversioned_source_change_fails_then_recovers(odb_pair):
    data = make_data(3)
    fill(odb_pair.src, data)
    target = sorted(data)[0]
    target_path = odb_pair.src.oid_to_path(target)

    drift = DriftFS(odb_pair.src.fs._fs)
    odb_pair.src.fs = drift
    # First ukey() call is the captured identity; all later calls report drift.
    real_fingerprint = drift._fs.ukey(target_path)
    drift.script[target_path] = [real_fingerprint] + ["drifted-fingerprint"] * 8

    session = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        jobs=1,
        max_retries=3,
    )
    result = session.run()
    assert not result.ok
    assert target in result.failed
    state = ledger_of(session).get(session.session_id, target)
    assert state.status is Status.FAILED
    assert state.error_kind is ErrorKind.SOURCE_CHANGED
    assert state.attempts == 1  # drift is never retried within a run
    assert not odb_pair.dest.exists(target)
    session.close()

    # The source is restored to its original identity before rebuilding.
    drift.script.clear()
    resumed = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        jobs=1,
    )
    resumed_result = resumed.run()
    assert resumed_result.ok
    assert_bytes(odb_pair.dest, data)
    assert_no_tmp(odb_pair.dest)
    assert isinstance(result.failed[target], str)
    resumed.close()


def test_versioned_source_binds_exact_version(memfs, memory_ledger):
    fs = VersionedMemFS(memfs)
    src = make_odb(fs, "/odb1")
    dest = make_odb(SpyFS(memfs), "/odb2")
    oid = "oid000001"
    path = src.oid_to_path(oid)
    v1 = b"version-one-content"
    v2 = b"version-two-content-very-different"
    fs.add_version(path, "v1", v1)
    # The head version rotates right after the session captures the source
    # identity; the bound v1 must still be what gets transferred.
    fs.on_first_info = lambda rotated_path: fs.rotate(rotated_path, "v2", v2)

    ledger = memory_ledger
    session = TransferSession(
        src,
        dest,
        {oid},
        ledger=ledger,
        jobs=1,
    )
    result = session.run()
    assert result.ok
    assert dest.fs.cat_file(dest.oid_to_path(oid)) == v1
    row = ledger.get(session.session_id, oid)
    assert row.src_version == "v1"
    assert row.status is Status.VERIFIED
    assert any(
        kwargs.get("version_id") == "v1"
        for _, mode, kwargs in fs.open_kwargs
        if mode == "rb"
    )
    session.close()

    # A second run reuses the verified destination and keeps the binding.
    resumed = TransferSession(src, dest, {oid}, ledger=ledger, jobs=1)
    assert resumed.run().reused == {oid}
    resumed.close()


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #


def test_cancel_and_resume_counts_each_confirmation_once(odb_pair):
    data = make_data(6)
    fill(odb_pair.src, data)
    callback = CountingCallback()

    session = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        jobs=2,
        callback=callback,
    )
    callback._session = session
    callback._cancel_after = 3
    with pytest.raises(TransferCancelledError) as exc:
        session.run()
    first_result = exc.value.result
    assert first_result.cancelled is True
    assert first_result.verified
    assert first_result.pending
    session.close()

    resumed = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        jobs=2,
        callback=callback,
    )
    result = resumed.run()
    assert result.ok
    assert_bytes(odb_pair.dest, data)
    assert_no_tmp(odb_pair.dest)
    # Cumulative progress: exactly one confirmation per object, no duplicates.
    assert callback.count == len(data)
    confirmed = [oid for oid, status in callback.events if status == "verified"]
    assert sorted(confirmed) == sorted(data)
    assert len(set(callback.events)) == len(data)
    resumed.close()


# --------------------------------------------------------------------------- #
# Opaque OIDs and verification
# --------------------------------------------------------------------------- #


def test_opaque_oids_are_never_interpreted_as_md5(memfs, memory_ledger):
    src = make_odb(memfs, "/odb1")
    dest = make_odb(memfs, "/odb2")
    data = {
        "opaque-alpha": b"alpha",
        "NOT-HEX-9": b"beta-beta",
        "oid_42": b"gamma" * 100,
    }
    fill(src, data)
    verifier = MapVerifier(sha_map(data))

    ledger = memory_ledger
    session = TransferSession(src, dest, set(data), ledger=ledger, verifier=verifier)
    result = session.run()
    assert result.ok
    assert_bytes(dest, data)
    session.close()


def test_hex_looking_oid_does_not_bypass_verification(memfs, memory_ledger):
    src = make_odb(memfs, "/odb1")
    dest = make_odb(memfs, "/odb2")
    # The well-known MD5 of the empty string; the content is anything but.
    oid = "d41d8cd98f00b204e9800998ecf8427e"
    blob = b"not the empty string"
    src.add_bytes(oid, blob)
    verifier = MapVerifier({oid: hashlib.sha256(b"something else").hexdigest()})

    ledger = memory_ledger
    session = TransferSession(
        src, dest, {oid}, ledger=ledger, verifier=verifier, max_retries=2
    )
    result = session.run()
    assert set(result.failed) == {oid}
    row = ledger.get(session.session_id, oid)
    assert row.status is Status.FAILED
    assert row.error_kind is ErrorKind.PERMANENT
    assert row.attempts == 1
    assert not dest.exists(oid)
    session.close()


def test_default_sha256_transport_verifier_detects_corrupt_tmp(
    memfs, mocker, memory_ledger
):
    src = make_odb(memfs, "/odb1")
    dest = make_odb(memfs, "/odb2")
    oid = "oid000001"
    src.add_bytes(oid, b"hello world")

    session = TransferSession(
        src, dest, {oid}, ledger=memory_ledger, verifier=Sha256Verifier()
    )
    # Corrupt the tmp object between the streaming copy and the verification.
    real_verify = Sha256Verifier.verify
    calls = {"n": 0}

    def corrupt_verify(self, oid, fs, path, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            fs.pipe_file(path, b"tampered bytes")
        return real_verify(self, oid, fs, path)

    mocker.patch.object(Sha256Verifier, "verify", corrupt_verify)
    result = session.run()
    assert oid in result.failed
    assert not dest.exists(oid)
    assert_no_tmp(dest)
    session.close()


# --------------------------------------------------------------------------- #
# Atomic publish capability
# --------------------------------------------------------------------------- #


def test_strict_mode_rejects_non_atomic_filesystem(memfs, memory_ledger):
    src = make_odb(memfs, "/odb1")
    dest = make_odb(NonAtomicFS(memfs), "/odb2")
    src.add_bytes("oid000001", b"x")

    session = TransferSession(
        src, dest, {"oid000001"}, ledger=memory_ledger, strict=True
    )
    with pytest.raises(AtomicPublishUnsupportedError):
        session.run()
    assert find_all(dest) == []
    session.close()


def test_non_atomic_filesystem_degrades_and_still_verifies_final(memfs, memory_ledger):
    src = make_odb(memfs, "/odb1")
    dest_fs = NonAtomicFS(memfs)
    dest = make_odb(dest_fs, "/odb2")
    oid = "oid000001"
    src.add_bytes(oid, b"degraded-but-correct")
    dest_fs.fail_mv_once = {dest.oid_to_path(oid)}

    session = TransferSession(
        src, dest, {oid}, ledger=memory_ledger, strict=False, max_retries=2
    )
    result = session.run()
    assert result.ok
    assert result.degraded is True
    assert dest.fs.cat_file(dest.oid_to_path(oid)) == b"degraded-but-correct"
    assert session.states()[oid] is Status.VERIFIED
    assert_no_tmp(dest)
    session.close()


# --------------------------------------------------------------------------- #
# Retry / error classification
# --------------------------------------------------------------------------- #


def test_transient_errors_retry_then_succeed_and_exhaust(odb_pair):
    data = make_data(2)
    fill(odb_pair.src, data)
    flaky = FlakyFS(
        odb_pair.src.fs._fs,
        {odb_pair.src.oid_to_path(sorted(data)[0]): 2},
    )
    odb_pair.src.fs = flaky
    verifier = MapVerifier(sha_map(data))

    session = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        verifier=verifier,
        jobs=1,
        max_retries=3,
    )
    result = session.run()
    assert result.ok
    assert ledger_of(session).get(session.session_id, sorted(data)[0]).attempts == 3
    session.close()

    # Exhausting the retry budget marks the object transient-failed.
    second = make_data(1, prefix="new")
    fill(odb_pair.src, second)
    oid = next(iter(second))
    flaky.failures[odb_pair.src.oid_to_path(oid)] = 99
    session2 = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(second),
        ledger=odb_pair.ledger_path + "2",
        verifier=MapVerifier(sha_map(second)),
        jobs=1,
        max_retries=1,
    )
    result2 = session2.run()
    row = ledger_of(session2).get(session2.session_id, oid)
    assert row.status is Status.FAILED
    assert row.error_kind is ErrorKind.TRANSIENT
    assert row.attempts == 2
    assert oid in result2.failed
    session2.close()


def test_permanent_permission_and_format_errors_are_not_retried(odb_pair):
    data = make_data(2)
    fill(odb_pair.src, data)
    odb_pair.dest.fs = PermissionFS(odb_pair.dest.fs._fs)

    session = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        jobs=1,
        max_retries=4,
    )
    result = session.run()
    assert set(result.failed) == set(data)
    for oid in data:
        row = ledger_of(session).get(session.session_id, oid)
        assert row.status is Status.FAILED
        assert row.error_kind is ErrorKind.PERMANENT
        assert row.attempts == 1
    assert_no_tmp(odb_pair.dest)
    session.close()


def test_broken_verifier_is_permanent(memfs, memory_ledger):
    src = make_odb(memfs, "/odb1")
    dest = make_odb(memfs, "/odb2")
    data = make_data(2)
    fill(src, data)

    session = TransferSession(
        src,
        dest,
        set(data),
        ledger=memory_ledger,
        verifier=BrokenVerifier(),
        jobs=1,
        max_retries=3,
    )
    result = session.run()
    assert set(result.failed) == set(data)
    for oid in data:
        assert (
            ledger_of(session).get(session.session_id, oid).error_kind
            is ErrorKind.PERMANENT
        )
    session.close()


def test_read_only_destination_is_rejected_upfront(memfs, memory_ledger):
    src = make_odb(memfs, "/odb1")
    dest = make_odb(memfs, "/odb2", read_only=True)
    src.add_bytes("oid000001", b"x")

    session = TransferSession(src, dest, {"oid000001"}, ledger=memory_ledger)
    with pytest.raises(ObjectDBPermissionError):
        session.run()
    session.close()


def test_missing_source_object_is_permanent(odb_pair):
    data = make_data(2)
    fill(odb_pair.src, {oid: data[oid] for oid in list(data)[:1]})

    session = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        jobs=1,
    )
    result = session.run()
    missing = sorted(data)[1]
    assert missing in result.failed
    row = ledger_of(session).get(session.session_id, missing)
    assert row.error_kind is ErrorKind.PERMANENT
    assert row.attempts == 1
    session.close()


def test_emfile_is_fatal(odb_pair):
    data = make_data(3)
    fill(odb_pair.src, data)
    odb_pair.src.fs = EMFileFS(odb_pair.src.fs._fs)

    session = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        jobs=1,
    )
    with pytest.raises(OSError, match="too many open files") as exc:
        session.run()
    assert exc.value.errno == errno.EMFILE
    first = sorted(data)[0]
    row = ledger_of(session).get(session.session_id, first)
    assert row.status is Status.FAILED
    assert row.error_kind is ErrorKind.FATAL
    # Scheduling stopped immediately; the other objects were never attempted.
    states = session.states()
    assert all(states[oid] is Status.PENDING for oid in data if oid != first)
    session.close()


# --------------------------------------------------------------------------- #
# Concurrency limits
# --------------------------------------------------------------------------- #


def test_inflight_bytes_are_bounded(odb_pair):
    # Equal-sized objects that all fit inside the budget keep the bound
    # enforceable at every scheduling decision.
    data = {f"oid{i:06d}": bytes([i]) * 256 for i in range(6)}
    fill(odb_pair.src, data)
    smallest = min(len(blob) for blob in data.values())
    limit = smallest * 2

    session = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        verifier=MapVerifier(sha_map(data)),
        jobs=6,
        max_inflight_bytes=limit,
    )
    result = session.run()
    assert result.ok
    assert session._inflight.max_seen <= limit
    assert_bytes(odb_pair.dest, data)
    session.close()

    # An object larger than the whole budget must not deadlock: it takes the
    # full capacity once the other in-flight transfers have drained.
    oversized = make_data(1, prefix="big")
    fill(odb_pair.src, oversized)
    big = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(oversized),
        ledger=odb_pair.ledger_path + "2",
        max_inflight_bytes=1,
        jobs=2,
    )
    assert big.run().ok
    big.close()


# --------------------------------------------------------------------------- #
# Ledger namespacing, locking and durability
# --------------------------------------------------------------------------- #


def test_ledger_rejects_different_namespace_and_oid_set(odb_pair):
    data = make_data(3)
    fill(odb_pair.src, data)
    verifier = MapVerifier(sha_map(data))
    session = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        verifier=verifier,
    )
    session.run()
    session.close()

    def assert_mismatch(**kwargs):
        session = TransferSession(
            kwargs.pop("src", odb_pair.src),
            kwargs.pop("dest", odb_pair.dest),
            kwargs.pop("oids", set(data)),
            ledger=odb_pair.ledger_path,
            verifier=kwargs.pop("verifier", verifier),
        )
        try:
            with pytest.raises(LedgerMismatchError):
                session.run()
        finally:
            session.close()

    # Different destination namespace.
    other_dest = make_odb(
        odb_pair.dest.fs._fs, odb_pair.dest.fs.join(odb_pair.dest.path, "other")
    )
    assert_mismatch(dest=other_dest)

    # Different OID set.
    assert_mismatch(oids=set(data) | {"oid999999"})

    # Different verifier configuration.
    assert_mismatch(verifier=MapVerifier(dict.fromkeys(data, "00" * 32)))

    # Different source namespace.
    other_src = make_odb(
        odb_pair.src.fs._fs, odb_pair.src.fs.join(odb_pair.src.path, "other")
    )
    assert_mismatch(src=other_src)


def test_file_ledger_single_session_lock(odb_pair):
    data = make_data(2)
    fill(odb_pair.src, data)
    session = TransferSession(
        odb_pair.src, odb_pair.dest, set(data), ledger=odb_pair.ledger_path
    )
    session.run()

    second = TransferSession(
        odb_pair.src, odb_pair.dest, set(data), ledger=odb_pair.ledger_path
    )
    with pytest.raises(SessionLockedError):
        second.run()

    session.close()
    # Once released, the same ledger can be reopened and resumes cleanly.
    result = second.run()
    assert result.ok
    second.close()
    assert not Path(odb_pair.ledger_path + ".lock").exists()


def test_stale_lock_from_dead_process_is_taken_over(odb_pair):
    data = make_data(1)
    fill(odb_pair.src, data)
    Path(odb_pair.ledger_path + ".lock").write_text(
        json.dumps({"token": "stale", "pid": 999999, "host": "nt", "created": 0}),
        encoding="utf-8",
    )

    session = TransferSession(
        odb_pair.src, odb_pair.dest, set(data), ledger=odb_pair.ledger_path
    )
    result = session.run()
    assert result.ok
    session.close()
    assert not Path(odb_pair.ledger_path + ".lock").exists()


def test_force_takes_over_existing_lock(odb_pair):
    data = make_data(1)
    fill(odb_pair.src, data)
    session = TransferSession(
        odb_pair.src, odb_pair.dest, set(data), ledger=odb_pair.ledger_path
    )
    session.run()

    forced = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        force=True,
    )
    result = forced.run()
    assert result.ok
    forced.close()
    # The superseded owner closes its own connection but must not strip the
    # new owner's lock on teardown.
    session.close()
    assert not Path(odb_pair.ledger_path + ".lock").exists()


def test_memory_ledger_owner_lock_and_resume(memfs, memory_ledger):
    src = make_odb(memfs, "/odb1")
    dest = make_odb(memfs, "/odb2")
    data = make_data(2)
    fill(src, data)
    ledger = memory_ledger

    session = TransferSession(src, dest, set(data), ledger=ledger)
    session.run()

    second = TransferSession(src, dest, set(data), ledger=ledger)
    with pytest.raises(SessionLockedError):
        second.run()
    session.close()

    resumed = TransferSession(src, dest, set(data), ledger=ledger)
    assert resumed.run().reused == set(data)
    resumed.close()


def test_ledger_is_durable_sqlite_with_expected_rows(odb_pair):
    data = make_data(3)
    fill(odb_pair.src, data)
    session = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        verifier=MapVerifier(sha_map(data)),
    )
    session.run()
    session.close()

    conn = sqlite3.connect(odb_pair.ledger_path)
    try:
        statuses = dict(
            conn.execute(
                "SELECT status, COUNT(*) FROM object GROUP BY status"
            ).fetchall()
        )
        assert statuses == {Status.VERIFIED.value: len(data)}
        session_row = conn.execute(
            "SELECT verifier_id, token, schema_version FROM session"
        ).fetchone()
        assert session_row[0].startswith("map:sha256:")
        assert session_row[1]
        assert session_row[2] == Ledger.SCHEMA_VERSION
    finally:
        conn.close()


def test_retryable_failures_reset_on_resume_but_permanent_do_not(odb_pair):
    data = make_data(2)
    transient_oid, permanent_oid = sorted(data)
    fill(odb_pair.src, {transient_oid: data[transient_oid]})
    flaky = FlakyFS(
        odb_pair.src.fs._fs,
        {odb_pair.src.oid_to_path(transient_oid): 99},
    )
    odb_pair.src.fs = flaky

    session = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        jobs=1,
        max_retries=0,
    )
    result = session.run()
    # Transient exhaustion for one OID, a permanent missing source for the
    # other one.
    assert transient_oid in result.failed
    assert permanent_oid in result.failed
    ledger = ledger_of(session)
    assert (
        ledger.get(session.session_id, transient_oid).error_kind is ErrorKind.TRANSIENT
    )
    assert (
        ledger.get(session.session_id, permanent_oid).error_kind is ErrorKind.PERMANENT
    )
    session.close()

    # The transient source recovers and the missing object is added; only the
    # transient failure is reopened, the permanent decision stays terminal.
    flaky.failures.clear()
    fill(odb_pair.src, {permanent_oid: data[permanent_oid]})
    resumed = TransferSession(
        odb_pair.src,
        odb_pair.dest,
        set(data),
        ledger=odb_pair.ledger_path,
        jobs=1,
        max_retries=0,
    )
    result2 = resumed.run()
    assert transient_oid in result2.verified
    assert permanent_oid in result2.failed
    assert (
        ledger_of(resumed).get(resumed.session_id, permanent_oid).status
        is Status.FAILED
    )
    assert_bytes(odb_pair.dest, {transient_oid: data[transient_oid]})
    resumed.close()
