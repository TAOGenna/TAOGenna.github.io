"""
Exercise 3: Mini Vector Database — SOLUTION
============================================
"""

import struct
import numpy as np
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    DIM, DEGREE_BOUND, MAX_DEGREE_BOUND, ALPHA,
    N_SUBSPACES, N_CENTROIDS,
    META_ENTRY_POINT, META_NODE_COUNT,
)
from vectordb.distance import l2_squared
from vectordb.quantization import ProductQuantizer
from vectordb.storage import (
    KVStore, put_node, get_node, put_delta, get_delta, get_all_neighbors,
    delete_node, put_meta, get_meta, make_delta_key,
)


# ─── RobustPrune ──────────────────────────────────────────────────────────────

def robust_prune(center_vec: np.ndarray,
                 candidates: list[int],
                 get_vec_fn,
                 degree_bound: int = DEGREE_BOUND,
                 alpha: float = ALPHA) -> list[int]:
    if len(candidates) <= degree_bound:
        return candidates[:]

    valid = []
    dist_map = {}
    for v in candidates:
        vec = get_vec_fn(v)
        if vec is not None:
            dist_map[v] = l2_squared(center_vec, vec)
            valid.append(v)

    sorted_cands = sorted(valid, key=lambda v: dist_map[v])
    selected = []

    for v in sorted_cands:
        if len(selected) >= degree_bound:
            break
        dominated = False
        for vp in selected:
            vp_vec = get_vec_fn(vp)
            v_vec = get_vec_fn(v)
            if vp_vec is not None and v_vec is not None:
                dist_vp_v = l2_squared(vp_vec, v_vec)
                if dist_map.get(v, float('inf')) >= alpha * dist_vp_v:
                    dominated = True
                    break
        if not dominated:
            selected.append(v)

    return selected


# ─── Greedy Search ────────────────────────────────────────────────────────────

def greedy_search(db: KVStore,
                  entry_point: int,
                  query: np.ndarray,
                  beam_width: int,
                  get_vec_fn,
                  get_neighbors_fn) -> tuple[list[int], list[int]]:
    ep_vec = get_vec_fn(entry_point)
    if ep_vec is None:
        return [], []

    frontier = {entry_point: l2_squared(query, ep_vec)}
    visited = {}

    while True:
        unexplored = {nid: d for nid, d in frontier.items() if nid not in visited}
        if not unexplored:
            break
        current = min(unexplored, key=unexplored.get)
        visited[current] = frontier[current]

        for nbr in get_neighbors_fn(current):
            if nbr not in frontier and nbr not in visited:
                nbr_vec = get_vec_fn(nbr)
                if nbr_vec is not None:
                    frontier[nbr] = l2_squared(query, nbr_vec)

        if len(frontier) > beam_width:
            sorted_f = sorted(frontier.items(), key=lambda x: x[1])
            frontier = dict(sorted_f[:beam_width])

    best = sorted(visited.items(), key=lambda x: x[1])[:beam_width]
    return list(visited.keys()), [nid for nid, _ in best]


# ─── VectorDB ─────────────────────────────────────────────────────────────────

class VectorDB:
    """
    Mini vector database combining CoreNN's storage innovations.
    """

    def __init__(self, mode: str = "in_memory"):
        # Storage backend (simulates RocksDB)
        self.db = KVStore()
        self.mode = mode
        # In-memory cache for full-precision vectors (in_memory mode)
        self._memory_cache: dict[int, np.ndarray] = {}
        # Incoming edge tracking for deletion (maps node_id → set of incoming node IDs)
        self._incoming: dict[int, set] = {}
        # Graph entry point
        self._entry_point: Optional[int] = None
        # Node count (includes deleted)
        self._node_count: int = 0
        # Product Quantizer (set on transition to on_disk)
        self._pq: Optional[ProductQuantizer] = None
        # PQ codes (node_id → uint8 array of shape (N_SUBSPACES,))
        self._pq_codes: dict[int, np.ndarray] = {}
        # Deleted node IDs (soft-filtered immediately, hard-deleted gradually)
        self._deleted: set = set()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_vec(self, node_id: int) -> Optional[np.ndarray]:
        if node_id in self._deleted:
            return None
        if self.mode == "in_memory":
            return self._memory_cache.get(node_id)
        else:
            result = get_node(self.db, node_id)
            return result[0] if result is not None else None

    def _get_neighbors(self, node_id: int) -> list[int]:
        result = get_all_neighbors(self.db, node_id)
        if result is None:
            return []
        _, neighbors = result
        return [n for n in neighbors if n not in self._deleted]

    def _maybe_compact(self, node_id: int) -> None:
        result = get_all_neighbors(self.db, node_id)
        if result is None:
            return
        vector, all_nbrs = result
        all_nbrs = [n for n in all_nbrs if n not in self._deleted]
        if len(all_nbrs) <= MAX_DEGREE_BOUND:
            return
        pruned = robust_prune(vector, all_nbrs, self._get_vec, DEGREE_BOUND, ALPHA)
        put_node(self.db, node_id, vector, pruned)
        self.db.delete(make_delta_key(node_id))

    # ── Public API ────────────────────────────────────────────────────────────

    def insert(self, node_id: int, vector: np.ndarray) -> None:
        vector = vector.astype(np.float32)

        # First node: just store and set entry point
        if self._entry_point is None:
            put_node(self.db, node_id, vector, [])
            self._memory_cache[node_id] = vector
            self._incoming[node_id] = set()
            self._entry_point = node_id
            self._node_count += 1
            if self.mode == "on_disk" and self._pq is not None:
                self._pq_codes[node_id] = self._pq.encode(vector)
            return

        # Greedy search to find candidate neighbors
        visited, best_candidates = greedy_search(
            self.db, self._entry_point, vector,
            beam_width=DEGREE_BOUND * 2,
            get_vec_fn=self._get_vec,
            get_neighbors_fn=self._get_neighbors,
        )

        # Select outgoing neighbors via RobustPrune
        candidates = best_candidates if best_candidates else visited[:DEGREE_BOUND*2]
        out_neighbors = robust_prune(vector, candidates, self._get_vec, DEGREE_BOUND, ALPHA)

        # Store the new node (co-located vector + neighbors)
        put_node(self.db, node_id, vector, out_neighbors)

        # Cache in memory (in_memory mode) or encode to PQ (on_disk mode)
        if self.mode == "in_memory":
            self._memory_cache[node_id] = vector
        elif self._pq is not None:
            self._pq_codes[node_id] = self._pq.encode(vector)

        # Initialize incoming tracking for new node
        if node_id not in self._incoming:
            self._incoming[node_id] = set()

        # Add backedges to each selected neighbor
        for v in out_neighbors:
            # Track incoming edge: node_id → v and v ← node_id
            self._incoming.setdefault(v, set())
            self._incoming[v].add(node_id)
            self._incoming[node_id].add(v)

            # Write backedge as a tiny delta (CoreNN approach)
            existing_delta = get_delta(self.db, v)
            put_delta(self.db, v, existing_delta + [node_id])

            # Compact if delta list grew too large
            self._maybe_compact(v)

        self._node_count += 1

    def search(self, query: np.ndarray, k: int = 10,
               beam_width: int = None) -> list[tuple[int, float]]:
        if self._entry_point is None:
            return []
        if beam_width is None:
            beam_width = max(k * 2, DEGREE_BOUND * 2)

        query = query.astype(np.float32)

        # Run greedy search
        visited, best_candidates = greedy_search(
            self.db, self._entry_point, query,
            beam_width=beam_width,
            get_vec_fn=self._get_vec,
            get_neighbors_fn=self._get_neighbors,
        )

        # Rank candidates by exact distance and return top-k
        results = []
        seen = set()
        for node_id in best_candidates + visited:
            if node_id in seen or node_id in self._deleted:
                continue
            seen.add(node_id)
            vec = self._get_vec(node_id)
            if vec is not None:
                dist = l2_squared(query, vec)
                results.append((node_id, dist))

        results.sort(key=lambda x: x[1])
        return results[:k]

    def delete(self, node_id: int) -> None:
        # Step 1: mark as deleted (immediately filtered from search)
        self._deleted.add(node_id)

        # Step 2: find incoming and outgoing neighbors
        incoming = self._incoming.get(node_id, set())
        result = get_all_neighbors(self.db, node_id)
        outgoing = []
        if result is not None:
            _, outgoing = result
            outgoing = [n for n in outgoing if n not in self._deleted]

        # Step 3: reconnect incoming nodes around deleted node
        for u in incoming:
            if u in self._deleted:
                continue
            result_u = get_node(self.db, u)
            if result_u is None:
                continue
            vec_u, u_base_nbrs = result_u
            delta_u = get_delta(self.db, u)
            u_all_nbrs = u_base_nbrs + delta_u
            # Remove deleted node, add outgoing as candidates
            u_all_nbrs = [n for n in u_all_nbrs if n != node_id and n not in self._deleted]
            candidates = list(set(u_all_nbrs + outgoing))
            pruned = robust_prune(vec_u, candidates, self._get_vec, DEGREE_BOUND, ALPHA)
            put_node(self.db, u, vec_u, pruned)
            self.db.delete(make_delta_key(u))

        # Step 4: remove from storage
        delete_node(self.db, node_id)
        if node_id in self._memory_cache:
            del self._memory_cache[node_id]
        if node_id in self._pq_codes:
            del self._pq_codes[node_id]

        # Step 5: update incoming tracking
        if node_id in self._incoming:
            del self._incoming[node_id]
        for u in list(self._incoming.keys()):
            self._incoming[u].discard(node_id)

        # Step 6: update entry point if needed
        if node_id == self._entry_point:
            if outgoing:
                self._entry_point = outgoing[0]
            else:
                # Find any remaining node
                remaining = [nid for nid in self._incoming if nid not in self._deleted]
                self._entry_point = remaining[0] if remaining else None

    def transition_to_disk(self) -> None:
        if not self._memory_cache:
            self.mode = "on_disk"
            return

        # Collect all vectors
        node_ids = list(self._memory_cache.keys())
        vectors = np.array([self._memory_cache[nid] for nid in node_ids])

        # Fit PQ codebook
        pq = ProductQuantizer(N_SUBSPACES, N_CENTROIDS)
        pq.fit(vectors)

        # Encode all vectors to PQ codes
        for nid, vec in zip(node_ids, vectors):
            self._pq_codes[nid] = pq.encode(vec)

        # Evict full-precision vectors from RAM
        self._memory_cache.clear()

        # Flip mode
        self._pq = pq
        self.mode = "on_disk"

    @property
    def n_nodes(self) -> int:
        return self._node_count - len(self._deleted)

    @property
    def compression_ratio(self) -> float:
        if self._pq is None:
            return 1.0
        full_bytes = DIM * 4
        compressed_bytes = N_SUBSPACES
        return full_bytes / compressed_bytes
