"""Optional persistent sharded Bloom membership index for an ObjectDB.

The index is a purely *accelerative* layer on top of
:class:`~dvc_objects.db.ObjectDB`:

* a negative answer from a Bloom shard is authoritative -- the oid cannot
  exist in the ObjectDB, so no HEAD/LIST request has to be made;
* a positive answer is always treated as *uncertain* (a Bloom filter can
  only probabilistically claim presence) and is confirmed against the real
  ObjectDB, so a probabilistic hit is never reported as existence;
* deletes never clear shared bits -- they can only add false positives
  until the next full rebuild, never false negatives.

Correctness is anchored on a mutation generation and an append-only
journal.  A shard is allowed to answer negatives only while it can prove
that it covers every ``add`` published up to the current generation.
Anything that breaks that proof (unclean shutdown, truncated/corrupt
data, an unknown writer, a namespace mismatch or a backend that cannot
provide the required atomic/replace semantics) makes the index fall back
to the plain ObjectDB query path instead of risking a false negative.

The module never interprets oid contents: oids are opaque strings encoded
as UTF-8.  Raw oid values are never written to the Python log; only
shard ids, generations, counts and sizes are logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import struct
import threading
import uuid
import zlib
from collections import OrderedDict
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from .fs.utils import tmp_fname

if TYPE_CHECKING:
    from .db import ObjectDB
    from .fs.base import FileSystem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format constants and hard limits
# ---------------------------------------------------------------------------

MAGIC = b"DVCMBF"
SHARD_VERSION = 1
MANIFEST_VERSION = 1
OID_ENCODING = "utf-8-opaque"
HASH_ALGO = "blake2b-128/km-v1"
SHARD_PERSON = b"dvcmbf-v1"
SHARD_ID_PERSON = b"dvcmbf-sid"

UINT16_MAX = 0xFFFF
UINT32_MAX = 0xFFFFFFFF
UINT64_MAX = 0xFFFFFFFFFFFFFFFF

MAX_OID_BYTES = 1024
MAX_NUM_HASHES = 32
MAX_NUM_SHARDS = 4096
# A single shard bitmap can never grow beyond this many bytes.
DEFAULT_MAX_SHARD_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_CACHE_SHARDS = 8
DEFAULT_MAX_TEMP_BYTES = 64 * 1024 * 1024
DEFAULT_NUM_SHARDS = 64
DEFAULT_FPP = 1e-3
DEFAULT_CAPACITY_PER_SHARD = 10_000

_MANIFEST_NAME = "manifest.json"
_JOURNAL_NAME = "journal.log"
_DIRTY_NAME = "dirty.lock"
_SHARDS_DIRNAME = "shards"
_TMP_DIRNAME = "tmp"

# Fixed-size shard header (see ``_Sketch.serialize``): 6 magic + 4 u16 +
# 2 u64 + 2 u32 + 1 header crc u32, then payload + payload crc u32.
_HEADER_LEN = 6 + 2 * 4 + 8 * 2 + 4 * 3
_HEADER_STRUCT = struct.Struct("<6s H H H H Q Q I I")
assert _HEADER_STRUCT.size == _HEADER_LEN - 4
_CRC_STRUCT = struct.Struct("<I")
_FRAME_HEAD = struct.Struct("<IQH")  # body_len, gen, oid_len
_FRAME_CRC = struct.Struct("<I")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MembershipIndexError(message)


class MembershipIndexError(Exception):
    """Base class for membership index failures."""


class MembershipIndexDisabledError(MembershipIndexError):
    """Raised when the index cannot be trusted on this backend/namespace."""


class MembershipIndexLimitError(MembershipIndexError):
    """Raised when a configured hard limit would be exceeded."""


# ---------------------------------------------------------------------------
# Hashing helpers (pure stdlib, no extra dependencies)
# ---------------------------------------------------------------------------


def _shard_id(oid_bytes: bytes, seed: int, num_shards: int) -> int:
    digest = hashlib.blake2b(
        oid_bytes,
        digest_size=8,
        key=seed.to_bytes(16, "little"),
        person=SHARD_ID_PERSON,
    ).digest()
    # num_shards is guaranteed to be a power of two.
    return int.from_bytes(digest, "little") & (num_shards - 1)


def _hash_positions(oid_bytes: bytes, seed: int, num_bits: int, num_hashes: int):
    digest = hashlib.blake2b(
        oid_bytes,
        digest_size=16,
        key=seed.to_bytes(16, "little"),
        person=SHARD_PERSON,
    ).digest()
    h1 = int.from_bytes(digest[:8], "little")
    # An odd second hash gives the Kirsch-Mitzenmacher sequence full
    # period whenever num_bits is a power of two.
    h2 = int.from_bytes(digest[8:], "little") | 1
    for i in range(num_hashes):
        yield (h1 + i * h2) % num_bits


def _size_sketch(capacity: int, fpp: float, max_shard_bytes: int) -> tuple[int, int]:
    """Return ``(num_bits, num_hashes)`` for one shard.

    The bitmap is sized once and never grows, so memory use is bounded by
    construction.  Over-capacity inserts only degrade the false-positive
    rate; they never trigger a reallocation.
    """
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if not 0.0 < fpp < 0.5:
        raise ValueError("fpp must be between 0 and 0.5")
    raw = -capacity * math.log(fpp) / (math.log(2) ** 2)
    num_bits = max(8, math.ceil(raw))
    num_bits = ((num_bits + 7) // 8) * 8
    max_bits = max_shard_bytes * 8
    num_bits = min(num_bits, max_bits)
    num_hashes = max(1, min(MAX_NUM_HASHES, round(num_bits / capacity * math.log(2))))
    return num_bits, num_hashes


# ---------------------------------------------------------------------------
# Single Bloom shard
# ---------------------------------------------------------------------------


class _Sketch:
    __slots__ = (
        "bits",
        "count",
        "generation",
        "num_bits",
        "num_hashes",
        "num_shards",
        "seed",
        "shard_id",
    )

    def __init__(
        self,
        shard_id: int,
        num_shards: int,
        num_bits: int,
        num_hashes: int,
        seed: int,
        generation: int = 0,
        count: int = 0,
        bits: bytearray | None = None,
    ) -> None:
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.num_bits = num_bits
        self.num_hashes = num_hashes
        self.seed = seed
        self.generation = generation
        self.count = count
        self.bits = bits if bits is not None else bytearray(num_bits // 8)

    def add(self, oid_bytes: bytes) -> None:
        for pos in _hash_positions(
            oid_bytes, self.seed, self.num_bits, self.num_hashes
        ):
            self.bits[pos >> 3] |= 1 << (pos & 7)

    def contains(self, oid_bytes: bytes) -> bool:
        for pos in _hash_positions(
            oid_bytes, self.seed, self.num_bits, self.num_hashes
        ):
            if not (self.bits[pos >> 3] & (1 << (pos & 7))):
                return False
        return True

    def serialize(self) -> bytes:
        if self.count > UINT32_MAX:
            raise MembershipIndexError("shard count overflow")
        if self.num_bits > UINT32_MAX or self.num_bits % 8:
            raise MembershipIndexError("invalid shard bit length")
        header = _HEADER_STRUCT.pack(
            MAGIC,
            SHARD_VERSION,
            self.shard_id,
            self.num_shards,
            self.num_hashes,
            self.seed,
            self.generation,
            self.count,
            self.num_bits,
        )
        header += _CRC_STRUCT.pack(zlib.crc32(header))
        payload = bytes(self.bits)
        return header + payload + _CRC_STRUCT.pack(zlib.crc32(payload))

    @classmethod
    def deserialize(
        cls,
        data: bytes,
        *,
        expected_shard: int,
        expected_num_shards: int,
        expected_num_bits: int,
        max_shard_bytes: int,
        max_generation: int,
    ) -> _Sketch:
        fields = cls._parse_header(data)
        cls._validate_header(
            fields,
            expected_shard=expected_shard,
            expected_num_shards=expected_num_shards,
            expected_num_bits=expected_num_bits,
            max_shard_bytes=max_shard_bytes,
            max_generation=max_generation,
        )
        (
            _magic,
            _version,
            shard_id,
            num_shards,
            num_hashes,
            seed,
            generation,
            count,
            num_bits,
        ) = fields
        payload = cls._extract_payload(data, num_bits)
        return cls(
            shard_id,
            num_shards,
            num_bits,
            num_hashes,
            seed,
            generation=generation,
            count=count,
            bits=bytearray(payload),
        )

    @staticmethod
    def _parse_header(data: bytes):
        if len(data) < _HEADER_LEN + 4:
            raise MembershipIndexError("shard payload truncated")
        header = data[: _HEADER_STRUCT.size]
        fields = _HEADER_STRUCT.unpack(header)
        (header_crc,) = _CRC_STRUCT.unpack(
            data[_HEADER_STRUCT.size : _HEADER_STRUCT.size + 4]
        )
        if zlib.crc32(header) != header_crc:
            raise MembershipIndexError("shard header checksum mismatch")
        return fields

    @staticmethod
    def _validate_header(
        fields,
        *,
        expected_shard: int,
        expected_num_shards: int,
        expected_num_bits: int,
        max_shard_bytes: int,
        max_generation: int,
    ) -> None:
        (
            magic,
            version,
            shard_id,
            num_shards,
            num_hashes,
            _seed,
            generation,
            count,
            num_bits,
        ) = fields
        _require(magic == MAGIC, "bad shard magic")
        _require(version == SHARD_VERSION, "unsupported shard version")
        _require(
            shard_id == expected_shard and 0 <= shard_id < num_shards,
            "shard id mismatch",
        )
        _require(num_shards == expected_num_shards, "shard count mismatch")
        _require(1 <= num_hashes <= MAX_NUM_HASHES, "shard hash count out of range")
        _require(num_bits > 0 and num_bits % 8 == 0, "invalid shard bit length")
        _require(num_bits == expected_num_bits, "shard bit length mismatch")
        _require(num_bits // 8 <= max_shard_bytes, "shard exceeds size hard limit")
        _require(count <= UINT32_MAX, "shard count overflow")
        _require(
            generation <= max_generation,
            "shard generation ahead of manifest",
        )

    @staticmethod
    def _extract_payload(data: bytes, num_bits: int) -> bytes:
        payload_len = num_bits // 8
        payload_start = _HEADER_LEN
        payload_end = payload_start + payload_len
        if len(data) != payload_end + 4:
            raise MembershipIndexError("shard payload length mismatch")
        payload = data[payload_start:payload_end]
        (payload_crc,) = _CRC_STRUCT.unpack(data[payload_end : payload_end + 4])
        if zlib.crc32(payload) != payload_crc:
            raise MembershipIndexError("shard payload checksum mismatch")
        return payload


# ---------------------------------------------------------------------------
# Journal framing
# ---------------------------------------------------------------------------


def _pack_frame(generation: int, oid_bytes: bytes) -> bytes:
    if generation > UINT64_MAX:
        raise MembershipIndexError("generation overflow")
    if len(oid_bytes) > UINT16_MAX:
        raise MembershipIndexError("oid too long")
    body = struct.pack("<Q", generation) + struct.pack("<H", len(oid_bytes)) + oid_bytes
    return (
        _FRAME_HEAD.pack(len(body), generation, len(oid_bytes))
        + body
        + _FRAME_CRC.pack(zlib.crc32(body))
    )


def _parse_journal(data: bytes, first_generation: int) -> list[tuple[int, bytes]]:
    """Parse journal frames.

    Frames must be contiguous starting at ``first_generation``.  A
    truncated or corrupt tail (or any generation gap) raises so the caller
    can stop trusting the index until a rebuild.
    """
    frames: list[tuple[int, bytes]] = []
    offset = 0
    expected = first_generation
    while offset < len(data):
        if len(data) - offset < _FRAME_HEAD.size + _FRAME_CRC.size:
            raise MembershipIndexError("journal truncated")
        body_len, generation, oid_len = _FRAME_HEAD.unpack_from(data, offset)
        total = _FRAME_HEAD.size + body_len + _FRAME_CRC.size
        if body_len != 8 + 2 + oid_len or offset + total > len(data):
            raise MembershipIndexError("journal frame out of bounds")
        body = data[offset + _FRAME_HEAD.size : offset + _FRAME_HEAD.size + body_len]
        (crc,) = _FRAME_CRC.unpack_from(data, offset + _FRAME_HEAD.size + body_len)
        if zlib.crc32(body) != crc:
            raise MembershipIndexError("journal frame checksum mismatch")
        if generation != expected:
            raise MembershipIndexError("journal generation gap")
        frames.append((generation, body[10:]))
        expected += 1
        offset += total
    return frames


# ---------------------------------------------------------------------------
# The index itself
# ---------------------------------------------------------------------------


class MembershipIndex:
    """Persistent sharded Bloom membership index bound to one ObjectDB.

    The index is optional and best-effort: every public entry point fails
    safe by falling back to the unaccelerated ObjectDB behaviour instead
    of risking a false negative.
    """

    def __init__(
        self,
        odb: ObjectDB,
        *,
        path: str | None = None,
        writer_id: str | None = None,
        num_shards: int = DEFAULT_NUM_SHARDS,
        capacity_per_shard: int = DEFAULT_CAPACITY_PER_SHARD,
        fpp: float = DEFAULT_FPP,
        seed: int | None = None,
        max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
        max_cache_shards: int = DEFAULT_MAX_CACHE_SHARDS,
        max_temp_bytes: int = DEFAULT_MAX_TEMP_BYTES,
    ) -> None:
        self._odb = odb
        self._fs: FileSystem = odb.fs
        if path is None:
            # Keep the index outside of the ODB tree so ordinary ODB
            # listings never observe index files.
            path = f"{odb.path}.membership"
        self._path = path
        self._shards_dir = self._fs.join(path, _SHARDS_DIRNAME)
        self._tmp_dir = self._fs.join(path, _TMP_DIRNAME)
        self._manifest_path = self._fs.join(path, _MANIFEST_NAME)
        self._journal_path = self._fs.join(path, _JOURNAL_NAME)
        self._dirty_path = self._fs.join(path, _DIRTY_NAME)

        self._lock = threading.RLock()
        self._cache: OrderedDict[int, _Sketch] = OrderedDict()
        self._frames: list[tuple[int, bytes]] = []
        self._manifest: dict[str, Any] | None = None
        self._current_gen = 0
        self._materialized = False
        self._needs_rebuild = False
        self._disabled_reason: str | None = None
        self._stats = {
            "negatives": 0,
            "uncertain": 0,
            "fallbacks": 0,
            "rebuilds": 0,
            "deletes": 0,
            "skipped_adds": 0,
        }

        # Creation parameters (only used until a manifest is adopted).
        self._requested_num_shards = num_shards
        self._requested_capacity = capacity_per_shard
        self._requested_fpp = fpp
        self._requested_seed = (
            seed if seed is not None else uuid.uuid4().int & UINT64_MAX
        )
        self._max_shard_bytes = max_shard_bytes
        self._max_cache_shards = max_cache_shards
        self._max_temp_bytes = max_temp_bytes
        self._writer_id = writer_id or uuid.uuid4().hex

        self._num_shards = num_shards
        self._shard_bits, self._num_hashes = _size_sketch(
            capacity_per_shard, fpp, max_shard_bytes
        )
        self._seed = self._requested_seed

        self._open()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """True only when negatives may be served authoritatively."""
        with self._lock:
            return (
                self._disabled_reason is None
                and not self._needs_rebuild
                and self._materialized
                and self._manifest is not None
            )

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    @property
    def generation(self) -> int:
        return self._current_gen

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def check(self, oids) -> tuple[set, set]:
        """Partition oids into ``(negative, uncertain)``.

        ``negative`` oids are provably absent from the ObjectDB;
        ``uncertain`` oids must be confirmed with a real query.  Any loss
        of coverage proof makes every oid uncertain.
        """
        oids = set(oids)
        with self._lock:
            if not self.ready:
                self._stats["fallbacks"] += 1
                return set(), set(oids)

            grouped: dict[int, list[tuple[Any, bytes]]] = {}
            uncertain: set = set()
            for oid in oids:
                try:
                    oid_bytes = self._encode(oid)
                except ValueError:
                    uncertain.add(oid)
                    continue
                sid = _shard_id(oid_bytes, self._seed, self._num_shards)
                grouped.setdefault(sid, []).append((oid, oid_bytes))

            negative: set = set()
            for sid, items in grouped.items():
                sketch = self._load_sketch(sid)
                if sketch is None or sketch.generation != self._current_gen:
                    # Shard cannot prove coverage of the current generation.
                    self._needs_rebuild = True
                    self._stats["fallbacks"] += 1
                    uncertain.update(oid for oid, _ in items)
                    continue
                for oid, oid_bytes in items:
                    if sketch.contains(oid_bytes):
                        uncertain.add(oid)
                    else:
                        negative.add(oid)
            self._stats["negatives"] += len(negative)
            self._stats["uncertain"] += len(uncertain)
            return negative, uncertain

    def add_oids(self, oids) -> None:
        """Record successfully published adds (controlled writer only).

        The ObjectDB must call this *after* the object bytes are visible.
        Encoding problems raise :class:`ValueError`; backend problems
        invalidate the index instead of propagating.
        """
        encoded = [(oid, self._encode(oid)) for oid in oids]
        with self._lock:
            if self._disabled_reason is not None:
                return
            for _, oid_bytes in encoded:
                self._add_locked(oid_bytes)

    def on_delete(self, oid: str) -> None:
        """Deletes never clear shared bits.

        They can only introduce false positives (filtered by the real
        query) until the next rebuild rebuilds sketches from a listing.
        """
        with self._lock:
            self._stats["deletes"] += 1
            logger.debug(
                "membership index: delete recorded as false-positive-only "
                "(generation=%s)",
                self._current_gen,
            )

    def rebuild(self, *, adopt: bool = False) -> None:
        """Rebuild every shard from an ODB listing plus the journal.

        Concurrent adds are merged after the cut point.  The manifest is
        the commit point: a crash at any earlier boundary leaves the old
        (or no) index in place, which is detected as a fallback condition.
        """
        with self._lock:
            if self._disabled_reason is not None and not adopt:
                raise MembershipIndexDisabledError(self._disabled_reason)
            if adopt:
                logger.warning(
                    "membership index: adopting rebuilt state (previous reason=%s)",
                    self._disabled_reason,
                )
                self._disabled_reason = None
            cut_gen = self._current_gen
            self._prepare_tmp_locked()
            num_shards = self._num_shards
            shard_bits = self._shard_bits
            num_hashes = self._num_hashes
            seed = self._seed

        # The (potentially slow) listing runs without the index lock so
        # published adds keep flowing; they are journaled and merged below.
        # The listing is just a snapshot, not a claim about the complete
        # current object set.
        sketches = {
            sid: _Sketch(sid, num_shards, shard_bits, num_hashes, seed)
            for sid in range(num_shards)
        }
        for oid in self._odb.all():
            try:
                oid_bytes = self._encode(oid)
            except ValueError:
                self._stats["skipped_adds"] += 1
                continue
            sketches[_shard_id(oid_bytes, seed, num_shards)].add(oid_bytes)

        with self._lock:
            self._commit_rebuild_locked(cut_gen, sketches)

    def invalidate(self, reason: str) -> None:
        """Mark the index untrusted until the next rebuild."""
        with self._lock:
            logger.warning("membership index invalidated: %s", reason)
            self._needs_rebuild = True
            self._cache.clear()

    def close(self) -> None:
        """Clean shutdown; removes the unclean-session marker."""
        with self._lock:
            with suppress(FileNotFoundError, OSError):
                self._fs.rm(self._dirty_path)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _encode(oid: Any) -> bytes:
        if not isinstance(oid, str) or not oid:
            raise ValueError("oid must be a non-empty string")
        oid_bytes = oid.encode("utf-8")
        if len(oid_bytes) > MAX_OID_BYTES:
            raise ValueError("oid exceeds maximum encoded length")
        return oid_bytes

    def _namespace(self) -> dict[str, str]:
        return {
            "protocol": self._fs.protocol,
            "path": self._fs.normpath(self._odb.path),
        }

    def _shard_path(self, sid: int) -> str:
        return self._fs.join(self._shards_dir, f"{sid:04x}.bf")

    def _pipe_atomic_locked(self, path: str, data: bytes) -> None:
        parent = self._fs.parent(path)
        tmp_path = self._fs.join(parent, tmp_fname())
        self._fs.pipe_file(tmp_path, data)
        self._fs.mv(tmp_path, path)

    def _supports_backend(self) -> str | None:
        for name in ("mv", "rm", "pipe_file", "open", "exists", "makedirs"):
            attr = getattr(self._fs, name, None)
            if not callable(attr):
                return f"backend lacks required operation: {name}"
        return None

    def _open(self) -> None:
        reason = self._supports_backend()
        if reason is not None:
            self._disable(reason)
            return
        if not self._is_valid_shard_count(self._requested_num_shards):
            self._disable("num_shards must be a power of two in range")
            return
        try:
            self._fs.makedirs(self._path, exist_ok=True)
            if self._fs.exists(self._dirty_path):
                self._needs_rebuild = True
                logger.warning(
                    "membership index: unclean previous session detected, "
                    "rebuild required"
                )
            else:
                self._fs.pipe_file(self._dirty_path, self._writer_id.encode())

            if self._fs.exists(self._manifest_path):
                self._load_manifest()
            else:
                self._manifest = self._build_manifest_locked(0, [])
                self._write_manifest_locked(self._manifest)
            self._load_journal()
            if self._materialized:
                self._verify_shards_locked()
        except MembershipIndexDisabledError as exc:
            # Unknown writer / namespace mismatch: stay constructable but
            # refuse to serve negatives until an adopted rebuild.
            self._disable(str(exc))
        except Exception as exc:  # noqa: BLE001 - fail safe, never crash ODB
            self._disable(f"failed to open index: {type(exc).__name__}")

    @staticmethod
    def _is_valid_shard_count(num_shards: int) -> bool:
        return 1 <= num_shards <= MAX_NUM_SHARDS and num_shards & (num_shards - 1) == 0

    def _disable(self, reason: str) -> None:
        self._disabled_reason = reason
        self._needs_rebuild = True
        self._cache.clear()
        logger.warning("membership index disabled: %s", reason)

    def _load_manifest(self) -> None:
        data = self._fs.cat_file(self._manifest_path)
        try:
            manifest = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MembershipIndexError("manifest is not valid JSON") from exc
        self._validate_manifest(manifest)
        if manifest["namespace"] != self._namespace():
            raise MembershipIndexDisabledError("namespace mismatch")
        if manifest["writer_id"] != self._writer_id:
            raise MembershipIndexDisabledError("unknown index writer")
        self._manifest = manifest
        self._materialized = bool(manifest["materialized"])
        self._num_shards = int(manifest["num_shards"])
        self._shard_bits = int(manifest["shard_bits"])
        self._num_hashes = int(manifest["num_hashes"])
        self._seed = int(manifest["seed"])
        self._current_gen = int(manifest["generation"])

    def _validate_manifest(self, manifest: Any) -> None:
        _require(isinstance(manifest, dict), "manifest must be an object")
        required = {
            "version",
            "materialized",
            "namespace",
            "oid_encoding",
            "hash_algo",
            "seed",
            "num_shards",
            "shard_bits",
            "num_hashes",
            "capacity_per_shard",
            "fpp",
            "writer_id",
            "generation",
            "shards",
        }
        _require(required.issubset(manifest), "manifest missing fields")
        _require(
            manifest["version"] == MANIFEST_VERSION,
            "unsupported manifest version",
        )
        _require(
            manifest["oid_encoding"] == OID_ENCODING,
            "oid encoding mismatch",
        )
        _require(
            manifest["hash_algo"] == HASH_ALGO,
            "hash algorithm mismatch",
        )
        _require(isinstance(manifest["namespace"], dict), "bad namespace binding")
        _require(
            isinstance(manifest["seed"], int) and 0 <= manifest["seed"] <= UINT64_MAX,
            "bad hash seed",
        )
        _require(
            self._is_valid_shard_count(manifest["num_shards"]),
            "bad shard count",
        )
        shard_bits = manifest["shard_bits"]
        _require(
            isinstance(shard_bits, int)
            and shard_bits > 0
            and shard_bits % 8 == 0
            and shard_bits // 8 <= self._max_shard_bytes,
            "bad shard bit length",
        )
        num_hashes = manifest["num_hashes"]
        _require(
            isinstance(num_hashes, int) and 1 <= num_hashes <= MAX_NUM_HASHES,
            "bad hash count",
        )
        generation = manifest["generation"]
        _require(
            isinstance(generation, int) and 0 <= generation <= UINT64_MAX,
            "bad generation",
        )
        _require(
            isinstance(manifest["writer_id"], str) and bool(manifest["writer_id"]),
            "bad writer id",
        )
        shards = manifest["shards"]
        _require(
            isinstance(shards, list) and len(shards) == manifest["num_shards"],
            "bad shard metadata",
        )
        for i, meta in enumerate(shards):
            self._validate_shard_meta(i, meta, generation)

    def _validate_shard_meta(self, index: int, meta: Any, generation: int) -> None:
        _require(isinstance(meta, dict), "bad shard metadata entry")
        _require(meta.get("id") == index, "bad shard metadata entry")
        _require(
            isinstance(meta.get("generation"), int)
            and 0 <= meta["generation"] <= generation,
            "bad shard metadata entry",
        )
        _require(
            isinstance(meta.get("count"), int) and 0 <= meta["count"] <= UINT32_MAX,
            "bad shard metadata entry",
        )
        _require(
            isinstance(meta.get("crc32"), int)
            and isinstance(meta.get("bytes"), int)
            and meta["bytes"] > 0,
            "bad shard metadata entry",
        )

    def _build_manifest_locked(
        self, generation: int, shards: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "version": MANIFEST_VERSION,
            "materialized": generation > 0 or self._materialized,
            "namespace": self._namespace(),
            "oid_encoding": OID_ENCODING,
            "hash_algo": HASH_ALGO,
            "seed": self._seed,
            "num_shards": self._num_shards,
            "shard_bits": self._shard_bits,
            "num_hashes": self._num_hashes,
            "capacity_per_shard": self._requested_capacity,
            "fpp": self._requested_fpp,
            "writer_id": self._writer_id,
            "generation": generation,
            "shards": shards,
        }

    def _write_manifest_locked(self, manifest: dict[str, Any]) -> None:
        data = json.dumps(manifest, sort_keys=True).encode("utf-8")
        self._pipe_atomic_locked(self._manifest_path, data)

    def _load_journal(self) -> None:
        if not self._fs.exists(self._journal_path):
            self._frames = []
            return
        data = self._fs.cat_file(self._journal_path)
        try:
            frames = _parse_journal(data, self._current_gen + 1)
        except MembershipIndexError:
            self._needs_rebuild = True
            logger.warning("membership index: journal unreadable, rebuild required")
            self._frames = []
            if self._manifest is not None:
                self._current_gen = self._manifest["generation"]
            else:
                self._current_gen = 0
            return
        self._frames = frames
        if frames:
            self._current_gen = frames[-1][0]

    def _verify_shards_locked(self) -> None:
        """Eagerly validate every committed shard against the manifest."""
        assert self._manifest is not None
        for meta in self._manifest["shards"]:
            sid = int(meta["id"])
            try:
                blob = self._fs.cat_file(self._shard_path(sid))
                if len(blob) != meta["bytes"] or zlib.crc32(blob) != meta["crc32"]:
                    raise MembershipIndexError("shard checksum mismatch")
                _Sketch.deserialize(
                    blob,
                    expected_shard=sid,
                    expected_num_shards=self._num_shards,
                    expected_num_bits=self._shard_bits,
                    max_shard_bytes=self._max_shard_bytes,
                    max_generation=self._current_gen,
                )
            except (FileNotFoundError, OSError, MembershipIndexError) as exc:
                self._needs_rebuild = True
                logger.warning(
                    "membership index: shard %s failed verification (%s), "
                    "rebuild required",
                    sid,
                    type(exc).__name__,
                )
                return

    def _add_locked(self, oid_bytes: bytes) -> None:
        generation = self._current_gen + 1
        frame = _pack_frame(generation, oid_bytes)
        try:
            # fsspec backends support binary append even though the typed
            # FileSystem.open overloads only list rb/br/wb.
            with self._fs.open(  # type: ignore[call-overload]
                self._journal_path, "ab"
            ) as fobj:
                fobj.write(frame)
        except OSError as exc:
            self._needs_rebuild = True
            logger.warning(
                "membership index: journal append failed (%s), "
                "falling back until rebuild",
                type(exc).__name__,
            )
            return
        self._frames.append((generation, oid_bytes))
        self._current_gen = generation
        if self._materialized:
            sid = _shard_id(oid_bytes, self._seed, self._num_shards)
            sketch = self._load_sketch(sid)
            if sketch is not None:
                sketch.add(oid_bytes)
                sketch.count += 1
                sketch.generation = generation

    def _load_sketch(self, sid: int) -> _Sketch | None:
        cached = self._cache.get(sid)
        if cached is not None:
            self._cache.move_to_end(sid)
            self._catch_up_locked(cached, sid)
            return cached
        try:
            data = self._fs.cat_file(self._shard_path(sid))
            sketch = _Sketch.deserialize(
                data,
                expected_shard=sid,
                expected_num_shards=self._num_shards,
                expected_num_bits=self._shard_bits,
                max_shard_bytes=self._max_shard_bytes,
                max_generation=self._current_gen,
            )
        except FileNotFoundError:
            self._needs_rebuild = True
            return None
        except (OSError, MembershipIndexError) as exc:
            self._needs_rebuild = True
            logger.warning(
                "membership index: shard %s unreadable (%s), "
                "falling back until rebuild",
                sid,
                type(exc).__name__,
            )
            return None
        self._catch_up_locked(sketch, sid)
        self._cache[sid] = sketch
        self._cache.move_to_end(sid)
        while len(self._cache) > self._max_cache_shards:
            self._cache.popitem(last=False)
        return sketch

    def _catch_up_locked(self, sketch: _Sketch, sid: int) -> None:
        """Replay journal frames published after the shard's generation."""
        if sketch.generation >= self._current_gen:
            return
        for gen, oid_bytes in self._frames:
            if gen <= sketch.generation:
                continue
            if _shard_id(oid_bytes, self._seed, self._num_shards) == sid:
                sketch.add(oid_bytes)
                sketch.count += 1
        sketch.generation = self._current_gen

    def _commit_rebuild_locked(
        self, cut_gen: int, sketches: dict[int, _Sketch]
    ) -> None:
        # Finalize under the lock so no add can land between reading the
        # final generation and committing the manifest.
        final_gen = self._current_gen
        seed = self._seed
        num_shards = self._num_shards
        temp_bytes = 0
        serialized: dict[int, bytes] = {}
        for gen, oid_bytes in self._frames:
            if gen <= cut_gen:
                continue
            sid = _shard_id(oid_bytes, seed, num_shards)
            sketches[sid].add(oid_bytes)
            sketches[sid].count += 1
        for sid, sketch in sketches.items():
            sketch.generation = final_gen
            blob = sketch.serialize()
            temp_bytes += len(blob)
            if temp_bytes > self._max_temp_bytes:
                self._cleanup_tmp_locked()
                raise MembershipIndexLimitError(
                    "rebuild temporary space hard limit exceeded"
                )
            serialized[sid] = blob

        self._fs.makedirs(self._shards_dir, exist_ok=True)
        shard_meta = []
        for sid, blob in serialized.items():
            tmp_path = self._fs.join(self._tmp_dir, f"{sid:04x}.bf.tmp")
            self._fs.pipe_file(tmp_path, blob)
            self._fs.mv(tmp_path, self._shard_path(sid))
            shard_meta.append(
                {
                    "id": sid,
                    "generation": final_gen,
                    "count": sketches[sid].count,
                    "crc32": zlib.crc32(blob),
                    "bytes": len(blob),
                }
            )

        # Rotate the journal before committing the manifest.  A crash in
        # between is detected on open as a generation gap and triggers
        # another rebuild instead of serving stale coverage.
        self._pipe_atomic_locked(self._journal_path, b"")
        self._frames = []
        self._current_gen = final_gen
        self._materialized = True
        self._needs_rebuild = False
        self._cache.clear()
        self._manifest = self._build_manifest_locked(final_gen, shard_meta)
        self._write_manifest_locked(self._manifest)
        self._cleanup_tmp_locked()
        self._stats["rebuilds"] += 1
        logger.info(
            "membership index rebuilt: shards=%d generation=%d",
            num_shards,
            final_gen,
        )

    def _prepare_tmp_locked(self) -> None:
        with suppress(FileNotFoundError, OSError):
            self._fs.rm(self._tmp_dir, recursive=True)
        self._fs.makedirs(self._tmp_dir, exist_ok=True)
        per_shard = _HEADER_LEN + self._shard_bits // 8 + 4
        if per_shard * self._num_shards > self._max_temp_bytes:
            self._cleanup_tmp_locked()
            raise MembershipIndexLimitError(
                "rebuild temporary space hard limit exceeded"
            )

    def _cleanup_tmp_locked(self) -> None:
        with suppress(FileNotFoundError, OSError):
            self._fs.rm(self._tmp_dir, recursive=True)
