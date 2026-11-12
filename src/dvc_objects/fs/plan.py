"""Planned, overlap-safe batch copies on a single local filesystem.

:class:`PlanOverlappingTransfer` executes a finite batch of
``source -> destination`` regular-file copies where sources and destinations
may overlap (e.g. ``a -> b`` together with ``b -> a``). The naive "copy in
order" strategy corrupts such batches, because an earlier copy can overwrite a
file that a later copy still needs to read.

Guarantees
----------
* Precheck is strictly read-only: paths are normalized to absolute paths and
  compared by real ``(device, inode)`` identity, not just strings. Symlinks,
  non-regular files, duplicate destinations, source/target directory conflicts
  and cross-device atomic strategies are rejected before anything is written.
* A real read/write dependency graph is built over inode identities. Acyclic
  parts run in topological order; each strongly connected component (cycle)
  first saves only the strictly necessary distinct source bytes into an owned
  staging directory (real independent copies, never hardlinks) and only then
  publishes. Staging stays inside the configured byte budget and the plan fails
  before overwriting anything when the budget or free space is insufficient.
* Every destination is replaced atomically (temp file in the destination's own
  parent directory followed by ``os.replace``). The batch as a whole is *not*
  atomic: only journal-recorded, recoverable partial progress is promised.
* Source identity ``(device, inode, size, mtime)`` is re-verified before
  staging, after staging and before every acyclic publish; external
  modification of a planned source raises :class:`PlanConflictError`. Within a
  cycle the staged, digest-verified copies are authoritative so our own
  publishes can legitimately replace source inodes.
* Recovery only ever touches files recorded in the plan journal and never
  deletes or modifies files that do not belong to the plan.
"""

from __future__ import annotations

import errno
import hashlib
import json
import logging
import os
import shutil
import stat
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fsspec.callbacks import DEFAULT_CALLBACK

from .errors import (
    PlanConflictError,
    PlanCrossDeviceError,
    PlanJournalError,
    PlanPrecheckError,
    PlanStagingSpaceError,
    PlanUnsupportedFileError,
)
from .local import LocalFileSystem
from .utils import tmp_fname

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fsspec import Callback

    from .base import AnyFSPath

logger = logging.getLogger(__name__)

JOURNAL_NAME = "journal.json"
_JOURNAL_VERSION = 1
_CHUNK_SIZE = 1024 * 1024  # 1 MB


# ---------------------------------------------------------------------------
# OS-level seams (module-level functions so tests can monkeypatch them)
# ---------------------------------------------------------------------------
def _lstat(path: str) -> os.stat_result:
    return os.lstat(path)


def _md5_file(path: str) -> str:
    # md5 is used as a content digest, not for any security purpose.
    digest = hashlib.md5()  # noqa: S324
    with open(path, "rb") as fobj:
        for chunk in iter(lambda: fobj.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(src: str, dst: str) -> None:
    """Copy real bytes from ``src`` to ``dst``.

    This deliberately never uses hardlinks or reflinks: a staged copy must be
    an independently owned backup that cannot change when the source changes.
    """
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        shutil.copyfileobj(fsrc, fdst, length=_CHUNK_SIZE)
        fdst.flush()
        os.fsync(fdst.fileno())


def _atomic_replace(src: str, dst: str) -> None:
    os.replace(src, dst)


def _free_bytes(path: str) -> int | None:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return None


def _remove_quiet(path: str) -> None:
    with suppress(FileNotFoundError):
        os.remove(path)


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------
def _strongly_connected_components(  # noqa: C901
    n_nodes: int, edges: Sequence[tuple[int, int]]
) -> list[list[int]]:
    """Iterative Kosaraju SCC (avoids recursion limits on long cycles)."""
    adjacency: list[list[int]] = [[] for _ in range(n_nodes)]
    reverse: list[list[int]] = [[] for _ in range(n_nodes)]
    for src, dst in edges:
        adjacency[src].append(dst)
        reverse[dst].append(src)

    visited = [False] * n_nodes
    finish_order: list[int] = []
    for start in range(n_nodes):
        if visited[start]:
            continue
        visited[start] = True
        stack: list[tuple[int, int]] = [(start, 0)]
        while stack:
            node, next_edge = stack[-1]
            if next_edge < len(adjacency[node]):
                nxt = adjacency[node][next_edge]
                stack[-1] = (node, next_edge + 1)
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append((nxt, 0))
            else:
                finish_order.append(node)
                stack.pop()

    assigned = [False] * n_nodes
    components: list[list[int]] = []
    for start in reversed(finish_order):
        if assigned[start]:
            continue
        assigned[start] = True
        node_stack: list[int] = [start]
        component: list[int] = []
        while node_stack:
            node = node_stack.pop()
            component.append(node)
            for nxt in reverse[node]:
                if not assigned[nxt]:
                    assigned[nxt] = True
                    node_stack.append(nxt)
        components.append(sorted(component))
    return components


def _topological_components(
    n_nodes: int,
    edges: Sequence[tuple[int, int]],
    components: Sequence[Sequence[int]],
) -> list[list[int]]:
    """Return SCCs in a deterministic topological (runnable) order."""
    node_component = [0] * n_nodes
    for component_id, component in enumerate(components):
        for node in component:
            node_component[node] = component_id

    dependencies: list[set[int]] = [set() for _ in components]
    for src, dst in edges:
        src_component = node_component[src]
        dst_component = node_component[dst]
        if src_component != dst_component:
            # dst component depends on src component (src runs first)
            dependencies[dst_component].add(src_component)

    ready = sorted(
        component_id for component_id, deps in enumerate(dependencies) if not deps
    )
    ordered: list[list[int]] = []
    remaining = [set(deps) for deps in dependencies]
    while ready:
        component_id = ready.pop(0)
        ordered.append(list(components[component_id]))
        for other_id, deps in enumerate(remaining):
            if component_id in deps:
                deps.discard(component_id)
                if not deps:
                    ready.append(other_id)
        ready.sort()
    return ordered


# ---------------------------------------------------------------------------
# Plan data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Identity:
    dev: int
    ino: int
    size: int
    mtime_ns: int
    mode: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.dev, self.ino)


@dataclass
class _Entry:
    index: int
    src: str
    dst: str
    src_identity: _Identity
    dst_identity: _Identity
    size: int
    digest: str
    stage_path: str | None = None
    state: str = "planned"


class PlanOverlappingTransfer:
    """Overlap-safe planned batch copy on one :class:`LocalFileSystem`.

    Args:
        fs: the single local filesystem holding every source and destination.
        transfers: ordered ``(source, destination)`` path pairs.
        staging_dir: dedicated, plan-owned directory for cycle staging and the
            recovery journal. It must reside on the same device as the targets.
        max_staging_bytes: optional hard budget for staged bytes. When the
            planned staging need exceeds the budget (or the reported free
            space), construction fails before anything is written.
        callback: progress callback; ``relative_update`` is called exactly
            once per destination when its completion is confirmed.

    Construction performs the full read-only precheck. Call :meth:`run` to
    execute the plan; if a journal from an interrupted run is present in
    ``staging_dir``, :meth:`run` performs journaled recovery instead.
    """

    def __init__(
        self,
        fs: LocalFileSystem,
        transfers: Sequence[tuple[AnyFSPath, AnyFSPath]],
        *,
        staging_dir: AnyFSPath,
        max_staging_bytes: int | None = None,
        callback: Callback = DEFAULT_CALLBACK,
    ) -> None:
        if not isinstance(fs, LocalFileSystem):
            raise PlanPrecheckError(
                "PlanOverlappingTransfer requires a single LocalFileSystem"
            )
        if not transfers:
            raise PlanPrecheckError("at least one source/destination pair is required")
        if max_staging_bytes is not None and max_staging_bytes < 0:
            raise PlanPrecheckError("max_staging_bytes must be non-negative")

        self.fs = fs
        self.callback = callback
        self.staging_dir = fs.abspath(staging_dir)
        self.max_staging_bytes = max_staging_bytes
        self.plan_id = uuid.uuid4().hex
        self._confirmed: set[str] = set()
        self._temp_files: list[str] = []

        self.entries = self._precheck(transfers)
        self.ordered_components, self.cyclic_components = self._build_graph()
        self.stage_paths: dict[tuple[int, int], str] = {}
        self.required_staging_bytes = self._plan_staging()
        self._check_staging_budget()

    # ------------------------------------------------------------------
    # Read-only precheck
    # ------------------------------------------------------------------
    def _read_identity(self, path: str) -> _Identity:
        try:
            st = _lstat(path)
        except FileNotFoundError as exc:
            raise PlanPrecheckError(f"planned path does not exist: {path}") from exc
        if stat.S_ISLNK(st.st_mode):
            raise PlanUnsupportedFileError(f"symlinks are not allowed: {path}")
        if not stat.S_ISREG(st.st_mode):
            raise PlanUnsupportedFileError(f"only regular files are supported: {path}")
        return _Identity(st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_mode)

    def _staging_device(self) -> int:
        path = self.staging_dir
        while True:
            try:
                st = _lstat(path)
            except FileNotFoundError:
                parent = self.fs.parent(path)
                if parent == path:
                    raise PlanPrecheckError(
                        f"cannot resolve device of staging directory: {path}"
                    ) from None
                path = parent
                continue
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
                raise PlanPrecheckError(f"staging path is not a directory: {path}")
            return st.st_dev

    def _precheck(
        self, transfers: Sequence[tuple[AnyFSPath, AnyFSPath]]
    ) -> list[_Entry]:
        normalized = [
            (self.fs.abspath(src), self.fs.abspath(dst)) for src, dst in transfers
        ]

        identities: dict[str, _Identity] = {}
        for path in sorted({p for pair in normalized for p in pair}):
            identities[path] = self._read_identity(path)

        # Duplicate destinations: same normalized path or same inode identity.
        seen_dst_paths: set[str] = set()
        seen_dst_keys: set[tuple[int, int]] = set()
        for _src, dst in normalized:
            if dst in seen_dst_paths:
                raise PlanPrecheckError(f"duplicate destination path: {dst}")
            seen_dst_paths.add(dst)
            dst_key = identities[dst].key
            if dst_key in seen_dst_keys:
                raise PlanPrecheckError(
                    f"duplicate destination: {dst} aliases another target inode"
                )
            seen_dst_keys.add(dst_key)

        # Source/target directory conflicts: every destination parent must be
        # an existing, non-symlink directory (precheck never creates them).
        for _src, dst in normalized:
            parent = self.fs.parent(dst)
            try:
                parent_st = _lstat(parent)
            except FileNotFoundError as exc:
                raise PlanPrecheckError(
                    f"destination parent directory does not exist: {parent}"
                ) from exc
            if stat.S_ISLNK(parent_st.st_mode) or not stat.S_ISDIR(parent_st.st_mode):
                raise PlanPrecheckError(
                    f"destination parent is not a directory: {parent}"
                )

        devices = {identity.dev for identity in identities.values()}
        devices.add(self._staging_device())
        if len(devices) != 1:
            raise PlanCrossDeviceError(
                "all sources, destinations and the staging directory must be on "
                "the same device; cross-device atomic replace is not supported"
            )
        self.device = next(iter(devices))

        # Plan-start reference read: digest every distinct source inode once.
        digests: dict[tuple[int, int], str] = {}
        entries: list[_Entry] = []
        for index, (src, dst) in enumerate(normalized):
            src_identity = identities[src]
            digest = digests.setdefault(src_identity.key, _md5_file(src))
            entries.append(
                _Entry(
                    index=index,
                    src=src,
                    dst=dst,
                    src_identity=src_identity,
                    dst_identity=identities[dst],
                    size=src_identity.size,
                    digest=digest,
                )
            )
        return entries

    # ------------------------------------------------------------------
    # Dependency graph
    # ------------------------------------------------------------------
    def _build_graph(self) -> tuple[list[list[int]], set[int]]:
        n_nodes = len(self.entries)
        readers_by_inode: dict[tuple[int, int], list[int]] = {}
        for entry in self.entries:
            readers_by_inode.setdefault(entry.src_identity.key, []).append(entry.index)

        edges: list[tuple[int, int]] = []
        edge_set: set[tuple[int, int]] = set()
        for entry in self.entries:
            for reader in readers_by_inode.get(entry.dst_identity.key, ()):
                # reader must consume the inode before entry overwrites it
                edge = (reader, entry.index)
                if edge not in edge_set:
                    edge_set.add(edge)
                    edges.append(edge)

        components = _strongly_connected_components(n_nodes, edges)
        ordered = _topological_components(n_nodes, edges, components)

        cyclic: set[int] = set()
        for component in components:
            if len(component) > 1:
                cyclic.update(component)
            elif (component[0], component[0]) in edge_set:
                # self loop: destination inode is the same as source inode
                cyclic.add(component[0])
        return ordered, cyclic

    def _plan_staging(self) -> int:
        """Sum distinct source inodes that cyclic components must preserve."""
        staged_keys: set[tuple[int, int]] = set()
        for component in self.ordered_components:
            if all(index in self.cyclic_components for index in component):
                for index in component:
                    staged_keys.add(self.entries[index].src_identity.key)
        total = 0
        for key in staged_keys:
            representative = next(
                entry for entry in self.entries if entry.src_identity.key == key
            )
            total += representative.size
        return total

    def _check_staging_budget(self) -> None:
        if (
            self.max_staging_bytes is not None
            and self.required_staging_bytes > self.max_staging_bytes
        ):
            raise PlanStagingSpaceError(
                f"staging requires {self.required_staging_bytes} bytes but the "
                f"budget is {self.max_staging_bytes} bytes"
            )
        free = _free_bytes(self.staging_dir)
        if free is not None and self.required_staging_bytes > free:
            raise PlanStagingSpaceError(
                f"staging requires {self.required_staging_bytes} bytes but only "
                f"{free} bytes are available"
            )

    # ------------------------------------------------------------------
    # Journal
    # ------------------------------------------------------------------
    @property
    def journal_path(self) -> str:
        return os.path.join(self.staging_dir, JOURNAL_NAME)

    def _journal_payload(self) -> dict[str, Any]:
        return {
            "version": _JOURNAL_VERSION,
            "plan_id": self.plan_id,
            "device": self.device,
            "staging_dir": self.staging_dir,
            "entries": [
                {
                    "index": entry.index,
                    "src": entry.src,
                    "dst": entry.dst,
                    "dev": entry.src_identity.dev,
                    "ino": entry.src_identity.ino,
                    "size": entry.size,
                    "digest": entry.digest,
                    "stage_path": entry.stage_path,
                    "state": entry.state,
                }
                for entry in self.entries
            ],
        }

    def _write_journal(self) -> None:
        tmp_path = os.path.join(self.staging_dir, tmp_fname("journal"))
        with open(tmp_path, "w", encoding="utf-8") as fobj:
            json.dump(self._journal_payload(), fobj, sort_keys=True)
            fobj.flush()
            os.fsync(fobj.fileno())
        _atomic_replace(tmp_path, self.journal_path)

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------
    def _reverify(self, entry: _Entry) -> None:
        """Re-check source identity; raise on any external modification."""
        try:
            st = _lstat(entry.src)
        except FileNotFoundError as exc:
            raise PlanConflictError(
                f"source vanished while the plan was running: {entry.src}"
            ) from exc
        if (st.st_dev, st.st_ino) != entry.src_identity.key:
            raise PlanConflictError(
                f"source identity changed while the plan was running: {entry.src}"
            )
        if (
            st.st_size != entry.src_identity.size
            or st.st_mtime_ns != entry.src_identity.mtime_ns
        ):
            raise PlanConflictError(
                f"source was modified externally while the plan was running: "
                f"{entry.src}"
            )

    def _reverify_all_sources(self) -> None:
        seen: set[tuple[int, int]] = set()
        for entry in self.entries:
            if entry.src_identity.key in seen:
                continue
            seen.add(entry.src_identity.key)
            self._reverify(entry)

    def _check_temp(self, tmp_path: str, entry: _Entry) -> None:
        st = _lstat(tmp_path)
        if st.st_size != entry.size:
            raise PlanConflictError(
                f"staged content size mismatch for {entry.dst}: "
                f"{st.st_size} != {entry.size}"
            )
        if _md5_file(tmp_path) != entry.digest:
            raise PlanConflictError(
                f"content digest mismatch for {entry.dst}; the source changed "
                "or produced corrupt bytes"
            )

    def _is_owned_by_staging(self, path: str) -> bool:
        path = os.path.abspath(path)
        if path == self.staging_dir:
            return True
        return path.startswith(self.staging_dir + os.sep)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def _prepare_staging(self) -> None:
        if os.path.lexists(self.staging_dir):
            st = _lstat(self.staging_dir)
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
                raise PlanPrecheckError(
                    f"staging path is not a directory: {self.staging_dir}"
                )
            if os.listdir(self.staging_dir):
                raise PlanPrecheckError(
                    f"staging directory must be empty and plan-owned: "
                    f"{self.staging_dir}"
                )
        else:
            os.makedirs(self.staging_dir, exist_ok=False)

    def _stage_cycles(self) -> None:
        inode_entries: dict[tuple[int, int], list[_Entry]] = {}
        for entry in self.entries:
            if entry.index in self.cyclic_components:
                inode_entries.setdefault(entry.src_identity.key, []).append(entry)

        for key in sorted(inode_entries, key=lambda item: (item[0], item[1])):
            group = inode_entries[key]
            representative = group[0]
            self._reverify(representative)
            stage_path = os.path.join(
                self.staging_dir,
                tmp_fname(f"stage-{representative.index}"),
            )
            try:
                _copy_file(representative.src, stage_path)
            except OSError as exc:
                _remove_quiet(stage_path)
                if exc.errno == errno.ENOSPC:
                    raise PlanStagingSpaceError(
                        "ran out of space while staging source content"
                    ) from exc
                raise
            # NOTE: staged files are deliberately not tracked as disposable
            # temp files: they must survive failures for journaled recovery.
            try:
                self._check_temp(stage_path, representative)
            except BaseException:
                _remove_quiet(stage_path)
                raise
            self.stage_paths[key] = stage_path
            for entry in group:
                entry.stage_path = stage_path
                entry.state = "staged"
            self._write_journal()

    def _make_publish_temp(self, entry: _Entry) -> str:
        tmp_path = os.path.join(
            self.fs.parent(entry.dst), tmp_fname(f".{self.plan_id}")
        )
        self._temp_files.append(tmp_path)
        return tmp_path

    def _publish(self, entry: _Entry, content_path: str) -> None:
        tmp_path = self._make_publish_temp(entry)
        try:
            _copy_file(content_path, tmp_path)
            self._check_temp(tmp_path, entry)
            _atomic_replace(tmp_path, entry.dst)
        except BaseException:
            _remove_quiet(tmp_path)
            raise
        if tmp_path in self._temp_files:
            self._temp_files.remove(tmp_path)
        self._confirm(entry)

    def _publish_from_source(self, entry: _Entry) -> None:
        self._reverify(entry)
        self._publish(entry, entry.src)

    def _publish_from_stage(self, entry: _Entry) -> None:
        # NOTE: source identity is not re-verified here: within a cyclic SCC
        # our own publishes legitimately replace source inodes, and the staged
        # copies are the authoritative content. External modification is
        # guarded by the bulk re-verification checkpoints before staging and
        # again after staging, before any publish starts.
        stage_path = entry.stage_path
        if not stage_path or not os.path.lexists(stage_path):
            raise PlanJournalError(
                f"staged content for {entry.dst} is missing: {stage_path}"
            )
        st = _lstat(stage_path)
        if st.st_size != entry.size or _md5_file(stage_path) != entry.digest:
            raise PlanJournalError(
                f"staged content for {entry.dst} failed identity/digest check"
            )
        self._publish(entry, stage_path)

    def _confirm(self, entry: _Entry) -> None:
        st = _lstat(entry.dst)
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
            raise PlanConflictError(
                f"destination is not a regular file after publish: {entry.dst}"
            )
        if _md5_file(entry.dst) != entry.digest:
            raise PlanConflictError(
                f"destination digest mismatch after publish: {entry.dst}"
            )
        entry.state = "published"
        self._write_journal()
        if entry.dst not in self._confirmed:
            self._confirmed.add(entry.dst)
            self.callback.relative_update()

    def _cleanup_temp_files(self) -> None:
        for path in list(self._temp_files):
            _remove_quiet(path)

    def _finish(self) -> None:
        """Remove the owned staging area; cleanup failures never mask success."""
        if not os.path.lexists(self.staging_dir):
            return
        for name in os.listdir(self.staging_dir):
            path = os.path.join(self.staging_dir, name)
            try:
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except OSError:
                logger.warning(
                    "failed to remove staged file %s; leaving staging area in "
                    "place for manual cleanup: %s",
                    path,
                    self.staging_dir,
                    exc_info=True,
                )
                return
        with suppress(OSError):
            os.rmdir(self.staging_dir)

    def run(self) -> None:
        """Execute the plan, or recover journaled progress if a journal exists."""
        if os.path.lexists(self.journal_path):
            self._recover()
            return

        self._prepare_staging()
        self._reverify_all_sources()
        self._write_journal()
        self.callback.set_size(len(self.entries))
        try:
            self._stage_cycles()
            self._reverify_all_sources()
            for component in self.ordered_components:
                if all(index in self.cyclic_components for index in component):
                    for index in component:
                        self._publish_from_stage(self.entries[index])
                else:
                    # acyclic SCCs are always single transfers
                    self._publish_from_source(self.entries[component[0]])
        except BaseException:
            # Keep journal + staged content for recovery; only remove stray
            # temp files we created, preserving the primary exception.
            self._cleanup_temp_files()
            raise
        self._finish()

    # ------------------------------------------------------------------
    # Crash recovery
    # ------------------------------------------------------------------
    def _recover(self) -> None:  # noqa: C901, PLR0912
        try:
            with open(self.journal_path, encoding="utf-8") as fobj:
                payload = json.load(fobj)
        except (OSError, json.JSONDecodeError) as exc:
            raise PlanJournalError(
                f"cannot read recovery journal: {self.journal_path}"
            ) from exc

        if payload.get("version") != _JOURNAL_VERSION:
            raise PlanJournalError("unsupported journal version")
        if payload.get("staging_dir") != self.staging_dir:
            raise PlanJournalError("journal does not belong to this staging directory")
        records = payload.get("entries")
        if not isinstance(records, list) or len(records) != len(self.entries):
            raise PlanJournalError("journal entry count does not match the plan")

        planned_pairs = {(entry.src, entry.dst) for entry in self.entries}
        journal_pairs = {(record.get("src"), record.get("dst")) for record in records}
        if journal_pairs != planned_pairs:
            raise PlanJournalError(
                "journal describes a different plan; refusing to touch files "
                "that do not belong to this plan"
            )

        required_keys = {"index", "src", "dst", "size", "digest", "stage_path"}
        if any(not required_keys.issubset(record) for record in records):
            raise PlanJournalError("journal records are missing required fields")

        self.callback.set_size(len(self.entries))
        entries_by_index = {entry.index: entry for entry in self.entries}
        for record in sorted(records, key=lambda item: item["index"]):
            entry = entries_by_index[record["index"]]
            expected_digest = record["digest"]
            expected_size = record["size"]
            stage_path = record.get("stage_path")

            if os.path.lexists(entry.dst):
                st = _lstat(entry.dst)
                if (
                    not stat.S_ISLNK(st.st_mode)
                    and stat.S_ISREG(st.st_mode)
                    and st.st_size == expected_size
                    and _md5_file(entry.dst) == expected_digest
                ):
                    entry.state = "published"
                    entry.stage_path = stage_path
                    if entry.dst not in self._confirmed:
                        self._confirmed.add(entry.dst)
                        self.callback.relative_update()
                    continue

            if not stage_path or not self._is_owned_by_staging(stage_path):
                raise PlanJournalError(
                    f"cannot recover {entry.dst}: no owned staged content"
                )
            if not os.path.lexists(stage_path):
                raise PlanJournalError(
                    f"cannot recover {entry.dst}: staged content vanished"
                )
            st = _lstat(stage_path)
            if (
                stat.S_ISLNK(st.st_mode)
                or not stat.S_ISREG(st.st_mode)
                or st.st_size != expected_size
                or _md5_file(stage_path) != expected_digest
            ):
                raise PlanJournalError(
                    f"cannot recover {entry.dst}: staged content failed "
                    "identity/digest verification"
                )
            entry.size = expected_size
            entry.digest = expected_digest
            entry.stage_path = stage_path
            entry.state = "staged"
            self._publish_from_stage(entry)

        self._finish()
