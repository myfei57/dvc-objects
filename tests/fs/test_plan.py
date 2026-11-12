import errno
import hashlib
import os

import pytest
from fsspec import Callback

from dvc_objects.fs import plan as plan_mod
from dvc_objects.fs.errors import (
    PlanConflictError,
    PlanCrossDeviceError,
    PlanJournalError,
    PlanPrecheckError,
    PlanStagingSpaceError,
    PlanUnsupportedFileError,
)
from dvc_objects.fs.generic import transfer as generic_transfer
from dvc_objects.fs.local import LocalFileSystem
from dvc_objects.fs.memory import MemoryFileSystem
from dvc_objects.fs.plan import JOURNAL_NAME, PlanOverlappingTransfer


def _md5(path: str) -> str:
    digest = hashlib.md5()  # noqa: S324
    with open(path, "rb") as fobj:
        for chunk in iter(lambda: fobj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: str, data: bytes) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fobj:
        fobj.write(data)
    return path


def _read(path: str) -> bytes:
    with open(path, "rb") as fobj:
        return fobj.read()


def _snapshot(root: str) -> dict[str, str]:
    result = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            path = os.path.abspath(os.path.join(dirpath, name))
            result[path] = _md5(path)
    return result


def _expected_final(
    before: dict[str, str], pairs: list[tuple[str, str]]
) -> dict[str, str]:
    expected = dict(before)
    for src, dst in pairs:
        expected[os.path.abspath(dst)] = before[os.path.abspath(src)]
    return expected


def _make_plan(
    tmp_path,
    pairs: list[tuple[str, str]],
    *,
    callback: "Callback | None" = None,
    max_staging_bytes=None,
) -> PlanOverlappingTransfer:
    staging = os.path.join(str(tmp_path), "staging")
    return PlanOverlappingTransfer(
        LocalFileSystem(),
        pairs,
        staging_dir=staging,
        max_staging_bytes=max_staging_bytes,
        callback=callback or Callback(),
    )


def _assert_no_stray_temps(root: str) -> None:
    for _dirpath, _dirs, files in os.walk(root):
        for name in files:
            assert not name.endswith(".tmp"), f"stray temp file left: {name}"


def test_swap_preserves_cut_point_sources(tmp_path):
    root = str(tmp_path)
    a = _write(os.path.join(root, "a"), os.urandom(2048))
    b = _write(os.path.join(root, "b"), os.urandom(4096))
    other = _write(os.path.join(root, "sub", "other"), b"uninvolved")
    before = _snapshot(root)
    original_a, original_b = _read(a), _read(b)

    callback = Callback()
    plan = _make_plan(tmp_path, [(a, b), (b, a)], callback=callback)
    # precheck must not create the staging directory or anything else
    assert not os.path.exists(os.path.join(root, "staging"))
    assert plan.required_staging_bytes == os.path.getsize(a) + os.path.getsize(b)

    plan.run()

    assert _snapshot(root) == _expected_final(before, [(a, b), (b, a)])
    assert _read(a) == original_b
    assert _read(b) == original_a
    assert _read(other) == b"uninvolved"
    assert not os.path.exists(os.path.join(root, "staging"))
    assert callback.value == 2


def test_long_cycle(tmp_path):
    root = str(tmp_path)
    names = [f"f{i}" for i in range(5)]
    paths = [os.path.join(root, name) for name in names]
    contents = [b"0" * 10, b"1" * 100, os.urandom(2 * 1024 * 1024), b"3", b"4" * 7]
    for path, data in zip(paths, contents):
        _write(path, data)
    before = _snapshot(root)

    pairs = [(paths[i], paths[(i + 1) % len(paths)]) for i in range(len(paths))]
    plan = _make_plan(tmp_path, pairs)
    assert plan.cyclic_components == set(range(len(paths)))
    plan.run()

    assert _snapshot(root) == _expected_final(before, pairs)
    for i, path in enumerate(paths):
        assert _read(path) == contents[i - 1]


def test_fanout_one_source_many_targets(tmp_path):
    root = str(tmp_path)
    src = _write(os.path.join(root, "src"), b"source-bytes")
    targets = [
        _write(os.path.join(root, name), f"old-{name}".encode())
        for name in ("t1", "t2", "t3")
    ]
    before = _snapshot(root)

    pairs = [(src, target) for target in targets]
    plan = _make_plan(tmp_path, pairs)
    # nothing is cyclic: no staging bytes needed, sources are read live
    assert plan.required_staging_bytes == 0
    assert not plan.cyclic_components
    plan.run()

    assert _snapshot(root) == _expected_final(before, pairs)
    assert all(_read(target) == b"source-bytes" for target in targets)
    assert _read(src) == b"source-bytes"


def test_chain_a_b_b_c_runs_in_topological_order(tmp_path):
    root = str(tmp_path)
    a = _write(os.path.join(root, "a"), b"aaa")
    b = _write(os.path.join(root, "b"), b"bbb")
    c = _write(os.path.join(root, "c"), b"ccc")
    before = _snapshot(root)

    pairs = [(a, b), (b, c)]
    plan = _make_plan(tmp_path, pairs)
    assert plan.required_staging_bytes == 0
    plan.run()

    assert _snapshot(root) == _expected_final(before, pairs)
    assert _read(b) == b"aaa"
    assert _read(c) == b"bbb"
    assert _read(a) == b"aaa"


def test_same_inode_different_paths_are_aliases(tmp_path):
    root = str(tmp_path)
    x = _write(os.path.join(root, "x"), b"shared-inode-content")
    y = os.path.join(root, "y")
    os.link(x, y)  # y is a hardlink alias of x
    p = _write(os.path.join(root, "p"), b"p-old")
    q = _write(os.path.join(root, "q"), b"q-old")
    before = _snapshot(root)

    pairs = [(x, p), (y, q)]
    plan = _make_plan(tmp_path, pairs)
    assert plan.required_staging_bytes == 0
    plan.run()

    assert _snapshot(root) == _expected_final(before, pairs)
    assert _read(p) == _read(q) == b"shared-inode-content"
    # aliased source itself stays untouched
    assert _read(x) == _read(y) == b"shared-inode-content"


def test_hardlinked_source_and_destination_is_self_loop(tmp_path):
    root = str(tmp_path)
    h1 = _write(os.path.join(root, "h1"), b"hardlinked-payload")
    h2 = os.path.join(root, "h2")
    os.link(h1, h2)
    assert os.stat(h1).st_ino == os.stat(h2).st_ino
    before = _snapshot(root)

    plan = _make_plan(tmp_path, [(h1, h2)])
    # inode identity (not the string path) makes this a cyclic transfer
    assert plan.cyclic_components == {0}
    assert plan.required_staging_bytes == len(b"hardlinked-payload")
    plan.run()

    assert _snapshot(root) == _expected_final(before, [(h1, h2)])
    assert _read(h1) == _read(h2) == b"hardlinked-payload"
    # atomic replace severs the hardlink: the paths now own distinct inodes
    assert os.stat(h1).st_ino != os.stat(h2).st_ino


def test_duplicate_destination_path_rejected(tmp_path):
    root = str(tmp_path)
    s1 = _write(os.path.join(root, "s1"), b"1")
    s2 = _write(os.path.join(root, "s2"), b"2")
    dst = os.path.join(root, "dst")
    _write(dst, b"old")
    aliased_dst = os.path.join(root, "sub", "..", "dst")

    with pytest.raises(PlanPrecheckError, match="duplicate destination"):
        _make_plan(tmp_path, [(s1, dst), (s2, aliased_dst)])
    # precheck writes nothing
    assert not os.path.exists(os.path.join(root, "staging"))


def test_duplicate_destination_inode_rejected(tmp_path):
    root = str(tmp_path)
    s1 = _write(os.path.join(root, "s1"), b"1")
    s2 = _write(os.path.join(root, "s2"), b"2")
    d1 = _write(os.path.join(root, "d1"), b"d")
    d2 = os.path.join(root, "d2")
    os.link(d1, d2)

    with pytest.raises(PlanPrecheckError, match="duplicate destination"):
        _make_plan(tmp_path, [(s1, d1), (s2, d2)])


def test_symlink_is_rejected(tmp_path):
    root = str(tmp_path)
    target = _write(os.path.join(root, "target"), b"t")
    link = os.path.join(root, "link")
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not permitted on this platform")
    dst = _write(os.path.join(root, "dst"), b"old")

    with pytest.raises(PlanUnsupportedFileError, match="symlink"):
        _make_plan(tmp_path, [(link, dst)])
    with pytest.raises(PlanUnsupportedFileError, match="symlink"):
        _make_plan(tmp_path, [(target, link)])


def test_directory_and_missing_parent_rejected(tmp_path):
    root = str(tmp_path)
    directory = os.path.join(root, "dir")
    os.makedirs(directory)
    regular = _write(os.path.join(root, "file"), b"f")

    with pytest.raises(PlanUnsupportedFileError, match="regular file"):
        _make_plan(tmp_path, [(directory, regular)])
    with pytest.raises(PlanUnsupportedFileError, match="regular file"):
        _make_plan(tmp_path, [(regular, directory)])

    missing_dst = os.path.join(root, "no-such-dir", "dst")
    with pytest.raises(PlanPrecheckError, match="does not exist"):
        _make_plan(tmp_path, [(regular, missing_dst)])


def test_cross_device_staging_rejected(tmp_path, monkeypatch):
    root = str(tmp_path)
    a = _write(os.path.join(root, "a"), b"a")
    b = _write(os.path.join(root, "b"), b"b")

    monkeypatch.setattr(
        PlanOverlappingTransfer, "_staging_device", lambda self: 1 << 30
    )
    with pytest.raises(PlanCrossDeviceError, match="same device"):
        _make_plan(tmp_path, [(a, b), (b, a)])
    assert not os.path.exists(os.path.join(root, "staging"))


def test_non_local_filesystem_rejected(tmp_path):
    with pytest.raises(PlanPrecheckError, match="LocalFileSystem"):
        PlanOverlappingTransfer(
            MemoryFileSystem(global_store=False),  # type: ignore[arg-type]
            [("/a", "/b")],
            staging_dir=str(tmp_path / "staging"),
        )


def test_staging_budget_exceeded_fails_before_writes(tmp_path):
    root = str(tmp_path)
    a = _write(os.path.join(root, "a"), b"a" * 10)
    b = _write(os.path.join(root, "b"), b"b" * 10)
    before = _snapshot(root)

    with pytest.raises(PlanStagingSpaceError, match="budget"):
        _make_plan(tmp_path, [(a, b), (b, a)], max_staging_bytes=5)
    assert not os.path.exists(os.path.join(root, "staging"))
    assert _snapshot(root) == before


def test_staging_enospac_fails_before_overwriting(tmp_path, monkeypatch):
    root = str(tmp_path)
    a = _write(os.path.join(root, "a"), b"a" * 10)
    b = _write(os.path.join(root, "b"), b"b" * 10)
    before = _snapshot(root)

    def raise_enospac(src, dst):
        raise OSError(errno.ENOSPC, "no space left on device")

    monkeypatch.setattr(plan_mod, "_copy_file", raise_enospac)
    plan = _make_plan(tmp_path, [(a, b), (b, a)])
    with pytest.raises(PlanStagingSpaceError, match="space"):
        plan.run()

    # no destination was overwritten; journal is retained for recovery
    assert _md5(a) == before[os.path.abspath(a)]
    assert _md5(b) == before[os.path.abspath(b)]
    assert os.path.exists(plan.journal_path)
    # no publish temp files leaked next to the targets
    assert set(os.listdir(root)) == {"a", "b", "staging"}


def test_external_modification_conflicts_and_recovers(tmp_path, monkeypatch):
    root = str(tmp_path)
    a = _write(os.path.join(root, "a"), b"aaa")
    b = _write(os.path.join(root, "b"), b"bbb")
    original_a, original_b = _read(a), _read(b)

    real_copy = plan_mod._copy_file
    mutated = {"done": False}

    def copy_then_mutate(src, dst):
        real_copy(src, dst)
        if not mutated["done"] and os.path.abspath(src) == os.path.abspath(a):
            mutated["done"] = True
            with open(a, "ab") as fobj:
                fobj.write(b"external-change")

    monkeypatch.setattr(plan_mod, "_copy_file", copy_then_mutate)
    plan = _make_plan(tmp_path, [(a, b), (b, a)])
    with pytest.raises(PlanConflictError, match="modified externally"):
        plan.run()

    # destinations were not touched; the external change is on the source
    assert _read(b) == original_b
    assert _read(a) == original_a + b"external-change"
    assert os.path.exists(plan.journal_path)

    monkeypatch.undo()
    # a new process/object recovers from the journal using staged original bytes
    recover = _make_plan(tmp_path, [(a, b), (b, a)])
    recover.run()

    assert _read(b) == original_a
    assert _read(a) == original_b
    assert not os.path.exists(os.path.join(root, "staging"))
    _assert_no_stray_temps(root)


def test_publish_interruption_is_recoverable(tmp_path, monkeypatch):
    root = str(tmp_path)
    names = [f"f{i}" for i in range(3)]
    paths = [
        _write(os.path.join(root, name), f"payload-{name}".encode()) for name in names
    ]
    pairs = [
        (paths[0], paths[1]),
        (paths[1], paths[2]),
        (paths[2], paths[0]),
    ]
    before = _snapshot(root)

    plan = _make_plan(tmp_path, pairs)
    target_dsts = {entry.dst for entry in plan.entries}
    real_replace = plan_mod._atomic_replace
    attempts = {"count": 0}

    def fail_second_publish(src, dst):
        if dst in target_dsts:
            attempts["count"] += 1
            if attempts["count"] == 2:
                raise PermissionError("simulated crash during publish")
        return real_replace(src, dst)

    monkeypatch.setattr(plan_mod, "_atomic_replace", fail_second_publish)
    with pytest.raises(PermissionError, match="simulated crash"):
        plan.run()
    monkeypatch.undo()

    # exactly one target was published and journaled before the crash
    journal_file = os.path.join(root, "staging", JOURNAL_NAME)
    assert os.path.exists(journal_file)
    confirmed = [
        entry
        for entry in plan.entries
        if os.path.lexists(entry.dst) and _md5(entry.dst) == entry.digest
    ]
    assert len(confirmed) == 1

    callback = Callback()
    recovered = _make_plan(tmp_path, pairs, callback=callback)
    recovered.run()  # enters journaled recovery automatically

    assert _snapshot(root) == _expected_final(before, pairs)
    # each destination is confirmed exactly once across the recovered run
    assert callback.value == len(pairs)
    assert not os.path.exists(os.path.join(root, "staging"))
    _assert_no_stray_temps(root)


def test_recovery_refuses_foreign_journal(tmp_path):
    root = str(tmp_path)
    a = _write(os.path.join(root, "a"), b"aaa")
    b = _write(os.path.join(root, "b"), b"bbb")
    c = _write(os.path.join(root, "c"), b"ccc")

    plan = _make_plan(tmp_path, [(a, b), (b, a)])
    plan._prepare_staging()
    plan._write_journal()
    foreign_journal = plan.journal_path

    # same entry count, but different source/destination pairs
    other = PlanOverlappingTransfer(
        LocalFileSystem(),
        [(a, c), (c, a)],
        staging_dir=os.path.join(root, "staging"),
    )
    with pytest.raises(PlanJournalError, match="different plan"):
        other.run()
    # foreign destination is untouched
    assert _read(c) == b"ccc"
    assert os.path.exists(foreign_journal)


def test_callback_relative_update_once_per_target(tmp_path):
    root = str(tmp_path)
    a = _write(os.path.join(root, "a"), b"a")
    b = _write(os.path.join(root, "b"), b"b")

    callback = Callback()
    plan = _make_plan(tmp_path, [(a, b), (b, a)], callback=callback)
    plan.run()
    assert callback.value == 2
    # re-confirming must not double count within one process
    for entry in plan.entries:
        if entry.dst not in plan._confirmed:
            pytest.fail("destination was not confirmed")


def test_generic_transfer_default_behavior_unchanged(tmp_path):
    # the new planner must not change generic.transfer semantics for plain,
    # non-overlapping copies
    root = str(tmp_path)
    src_dir = os.path.join(root, "src")
    dst_dir = os.path.join(root, "dest")
    os.makedirs(src_dir)
    os.makedirs(dst_dir)
    src_files = [
        _write(os.path.join(src_dir, name), f"{name}-data".encode())
        for name in ("foo", "bar")
    ]
    dst_files = [os.path.join(dst_dir, name) for name in ("foo", "bar")]

    callback = Callback()
    generic_transfer(
        LocalFileSystem(),
        src_files,
        LocalFileSystem(),
        dst_files,
        callback=callback,
    )
    for src, dst in zip(src_files, dst_files):
        assert _read(dst) == _read(src)
    assert callback.value == 2
