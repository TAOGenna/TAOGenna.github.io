"""
Storage layer: in-memory KV store with the same serialization format
that would be used with RocksDB in production.

Key format: [type_prefix: 1 byte][node_id: 4 bytes big-endian]
Value format for KEY_TYPE_BASE: [float32 vector: DIM*4 bytes][count: 4 bytes][neighbor IDs: n*4 bytes]
Value format for KEY_TYPE_DELTA: [neighbor IDs: n*4 bytes] (no count prefix; length inferred)
Value format for KEY_TYPE_META: raw bytes (caller interprets)
"""

import struct
import numpy as np
from typing import Optional
from config import DIM, KEY_TYPE_BASE, KEY_TYPE_DELTA, KEY_TYPE_META


# ─── Key construction ─────────────────────────────────────────────────────────

def make_base_key(node_id: int) -> bytes:
    """Binary key for co-located vector+neighbors entry."""
    return struct.pack(">BI", KEY_TYPE_BASE, node_id)

def make_delta_key(node_id: int) -> bytes:
    """Binary key for backedge delta neighbors."""
    return struct.pack(">BI", KEY_TYPE_DELTA, node_id)


# ─── Value serialization ──────────────────────────────────────────────────────

def serialize_node(vector: np.ndarray, neighbors: list[int]) -> bytes:
    """
    Pack [float32 vector | neighbor_count | neighbor_ids] into bytes.
    This is the co-located format: vector and neighbors in one entry.
    """
    vec_bytes = vector.astype(np.float32).tobytes()
    n = len(neighbors)
    nbr_bytes = struct.pack(f"<I{n}I", n, *neighbors) if n > 0 else struct.pack("<I", 0)
    return vec_bytes + nbr_bytes


def deserialize_node(data: bytes) -> tuple[np.ndarray, list[int]]:
    """Unpack co-located entry into (vector, neighbors)."""
    vec_end = DIM * 4
    vector = np.frombuffer(data[:vec_end], dtype=np.float32).copy()
    n = struct.unpack_from("<I", data, vec_end)[0]
    if n > 0:
        neighbors = list(struct.unpack_from(f"<{n}I", data, vec_end + 4))
    else:
        neighbors = []
    return vector, neighbors


def serialize_delta(neighbor_ids: list[int]) -> bytes:
    """Pack a list of additional neighbor IDs (4 bytes each, no length prefix)."""
    if not neighbor_ids:
        return b''
    return struct.pack(f"<{len(neighbor_ids)}I", *neighbor_ids)


def deserialize_delta(data: Optional[bytes]) -> list[int]:
    """Unpack delta bytes into list of node IDs."""
    if not data:
        return []
    n = len(data) // 4
    return list(struct.unpack(f"<{n}I", data))


# ─── Storage backend ──────────────────────────────────────────────────────────

class KVStore:
    """
    Simple in-memory key-value store with the same interface as a RocksDB wrapper.

    In CoreNN, this would be backed by RocksDB on disk. Here we use a Python
    dict to simulate the same API and data format without requiring an actual
    database installation.
    """

    def __init__(self):
        self._data: dict[bytes, bytes] = {}

    def put(self, key: bytes, value: bytes) -> None:
        self._data[key] = value

    def get(self, key: bytes) -> Optional[bytes]:
        return self._data.get(key)

    def delete(self, key: bytes) -> None:
        self._data.pop(key, None)

    def contains(self, key: bytes) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)


# ─── High-level node I/O ──────────────────────────────────────────────────────

def put_node(db: KVStore, node_id: int, vector: np.ndarray, neighbors: list[int]) -> None:
    """Write co-located vector+neighbors entry to storage."""
    db.put(make_base_key(node_id), serialize_node(vector, neighbors))


def get_node(db: KVStore, node_id: int) -> Optional[tuple[np.ndarray, list[int]]]:
    """
    Read co-located entry. Returns (vector, neighbors) or None if not found.
    """
    data = db.get(make_base_key(node_id))
    if data is None:
        return None
    return deserialize_node(data)


def put_delta(db: KVStore, node_id: int, additional_neighbors: list[int]) -> None:
    """Write delta neighbor list to its separate key."""
    db.put(make_delta_key(node_id), serialize_delta(additional_neighbors))


def get_delta(db: KVStore, node_id: int) -> list[int]:
    """Read delta neighbors. Returns empty list if no delta exists."""
    return deserialize_delta(db.get(make_delta_key(node_id)))


def get_all_neighbors(db: KVStore, node_id: int) -> Optional[tuple[np.ndarray, list[int]]]:
    """
    Merge base neighbors and delta neighbors for query-time traversal.
    Returns (vector, base_neighbors + delta_neighbors) or None if node not found.
    """
    result = get_node(db, node_id)
    if result is None:
        return None
    vector, base_neighbors = result
    delta_neighbors = get_delta(db, node_id)
    return vector, base_neighbors + delta_neighbors


def delete_node(db: KVStore, node_id: int) -> None:
    """Remove a node's base entry and delta from storage."""
    db.delete(make_base_key(node_id))
    db.delete(make_delta_key(node_id))


def put_meta(db: KVStore, key: bytes, value: bytes) -> None:
    db.put(key, value)


def get_meta(db: KVStore, key: bytes) -> Optional[bytes]:
    return db.get(key)
