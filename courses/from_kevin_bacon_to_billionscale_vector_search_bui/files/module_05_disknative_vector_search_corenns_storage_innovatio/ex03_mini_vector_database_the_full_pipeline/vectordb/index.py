"""
Exercise 3: Mini Vector Database — The Full Pipeline
====================================================
Implement the VectorDB class. See README.md for the full spec.

PROVIDED (do not modify):
  - robust_prune() — from Module 3
  - greedy_search() — Vamana's GreedySearch
  - VectorDB._get_vec() — mode-aware vector retrieval
  - VectorDB._get_neighbors() — merged base+delta neighbors
  - VectorDB._maybe_compact() — lazy RobustPrune trigger
  - VectorDB.n_nodes, compression_ratio properties

YOU IMPLEMENT (5 methods):
  - VectorDB.__init__()          (~10 lines) — initialize all attributes
  - VectorDB.insert()            (~30 lines) — Vamana insert + backedge deltas
  - VectorDB.search()            (~25 lines) — greedy graph search
  - VectorDB.delete()            (~20 lines) — graph contraction deletion
  - VectorDB.transition_to_disk() (~15 lines) — fit PQ, clear cache
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


# ─── RobustPrune (provided — from Module 3) ───────────────────────────────────

def robust_prune(center_vec, candidates, get_vec_fn,
                 degree_bound=DEGREE_BOUND, alpha=ALPHA):
    """Prune candidate list to at most degree_bound edges using RobustPrune."""
    if len(candidates) <= degree_bound:
        return candidates[:]
    valid, dist_map = [], {}
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
            vp_vec, v_vec = get_vec_fn(vp), get_vec_fn(v)
            if vp_vec is not None and v_vec is not None:
                if dist_map.get(v, float("inf")) >= alpha * l2_squared(vp_vec, v_vec):
                    dominated = True
                    break
        if not dominated:
            selected.append(v)
    return selected


# ─── Greedy Search (provided — Vamana's GreedySearch) ─────────────────────────

def greedy_search(db, entry_point, query, beam_width, get_vec_fn, get_neighbors_fn):
    """Traverse graph greedily from entry_point toward query. Returns (visited, best)."""
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
            frontier = dict(sorted(frontier.items(), key=lambda x: x[1])[:beam_width])
    best = sorted(visited.items(), key=lambda x: x[1])[:beam_width]
    return list(visited.keys()), [nid for nid, _ in best]


# ─── VectorDB ─────────────────────────────────────────────────────────────────

class VectorDB:
    """
    Mini vector database. Implement the 5 methods marked YOUR CODE HERE.

    Modes:
      "in_memory" — all vectors cached in RAM, zero disk reads
      "on_disk"   — PQ codes in RAM; full vectors read from KVStore (simulates disk)

    Attributes you must initialize in __init__:
      self.db             : KVStore()
      self.mode           : str
      self._memory_cache  : dict[int, np.ndarray]
      self._incoming      : dict[int, set]   ← tracks incoming edges for deletion
      self._entry_point   : Optional[int] = None
      self._node_count    : int = 0
      self._pq            : Optional[ProductQuantizer] = None
      self._pq_codes      : dict[int, np.ndarray] = {}
      self._deleted       : set = set()
    """

    def __init__(self, mode: str = "in_memory"):
        ###########################################################
        # YOUR CODE HERE - ~10 lines                              #
        # Initialize all 9 attributes listed in the docstring     #
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################

    # ── Provided helpers (do not modify) ──────────────────────────────────────

    def _get_vec(self, node_id):
        if node_id in self._deleted:
            return None
        if self.mode == "in_memory":
            return self._memory_cache.get(node_id)
        result = get_node(self.db, node_id)
        return result[0] if result else None

    def _get_neighbors(self, node_id):
        result = get_all_neighbors(self.db, node_id)
        if result is None:
            return []
        _, neighbors = result
        return [n for n in neighbors if n not in self._deleted]

    def _maybe_compact(self, node_id):
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

    # ── YOUR CODE ─────────────────────────────────────────────────────────────

    def insert(self, node_id: int, vector: np.ndarray) -> None:
        """
        Insert vector using Vamana algorithm + backedge deltas.

        Steps:
          1. First node edge case: put_node, set entry point, cache, return.
          2. greedy_search(self.db, self._entry_point, vector,
               DEGREE_BOUND*2, self._get_vec, self._get_neighbors)
          3. robust_prune(vector, best_candidates, self._get_vec) → N_out
          4. put_node(self.db, node_id, vector, N_out)
             If in_memory: self._memory_cache[node_id] = vector
          5. For each v in N_out:
             - Update self._incoming[node_id] and self._incoming[v]
             - delta = get_delta(self.db, v); put_delta(self.db, v, delta + [node_id])
             - self._maybe_compact(v)
          6. self._node_count += 1
        """
        ###########################################################
        # YOUR CODE HERE - ~30 lines                              #
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################

    def search(self, query: np.ndarray, k: int = 10,
               beam_width: int = None) -> list:
        """
        Find k nearest neighbors.

        Steps:
          1. Return [] if no entry point.
          2. beam_width = max(k*2, DEGREE_BOUND*2) if None.
          3. greedy_search(...) → visited, best_candidates
          4. Compute exact distances for all visited nodes (skip deleted).
          5. Sort ascending, return top-k as [(node_id, distance)].
        """
        ###########################################################
        # YOUR CODE HERE - ~25 lines                              #
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################

    def delete(self, node_id: int) -> None:
        """
        True deletion via graph contraction.

        Steps:
          1. self._deleted.add(node_id)
          2. incoming = self._incoming.get(node_id, set())
          3. Get node_id's outgoing neighbors from get_all_neighbors(self.db, node_id)
          4. For each u in incoming (skip deleted):
             a. get_node + get_delta → vec_u, all_nbrs
             b. Filter deleted, add outgoing as candidates
             c. robust_prune(vec_u, candidates, self._get_vec) → pruned
             d. put_node(self.db, u, vec_u, pruned)
             e. self.db.delete(make_delta_key(u))
          5. delete_node(self.db, node_id); clean caches.
          6. Update self._incoming.
          7. If entry point deleted: update to outgoing[0] or None.
        """
        ###########################################################
        # YOUR CODE HERE - ~20 lines                              #
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################

    def transition_to_disk(self) -> None:
        """
        Switch from in-memory to on-disk mode. No rewrites needed!

        Steps:
          1. node_ids = list(self._memory_cache.keys())
          2. vectors = np.array([self._memory_cache[nid] for nid in node_ids])
          3. pq = ProductQuantizer(N_SUBSPACES, N_CENTROIDS); pq.fit(vectors)
          4. Encode each vector: self._pq_codes[nid] = pq.encode(vec)
          5. self._memory_cache.clear()
          6. self._pq = pq; self.mode = "on_disk"

        The KVStore entries were already written in the correct format during
        insert() — no mass rewrites needed (this is the whole point!).
        """
        ###########################################################
        # YOUR CODE HERE - ~15 lines                              #
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################

    @property
    def n_nodes(self):
        return self._node_count - len(self._deleted)

    @property
    def compression_ratio(self):
        if self._pq is None:
            return 1.0
        return (DIM * 4) / N_SUBSPACES
