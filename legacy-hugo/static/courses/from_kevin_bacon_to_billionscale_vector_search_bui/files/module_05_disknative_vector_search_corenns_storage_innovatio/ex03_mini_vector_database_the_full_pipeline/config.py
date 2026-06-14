"""
Configuration for the Mini Vector Database.
These constants mirror CoreNN's production defaults scaled for exercise use.

# TODO: Study these constants before implementing vectordb/index.py.
#       Each one maps directly to a CoreNN design choice:
#         - DEGREE_BOUND / MAX_DEGREE_BOUND control graph fan-out and compaction.
#         - ALPHA is the RobustPrune slack parameter (try 1.0 vs 1.2 and compare recall).
#         - N_SUBSPACES / N_CENTROIDS determine PQ compression quality vs ratio.
"""

# Vector dimensionality
DIM = 64                  # 64-dim for fast simulation (full: 768)

# Graph parameters
DEGREE_BOUND = 12         # M: max edges per node after RobustPrune
MAX_DEGREE_BOUND = 18     # M_max: compact when total neighbors exceed this
ALPHA = 1.1               # RobustPrune slack parameter

# PQ compression parameters
N_SUBSPACES = 8           # number of PQ subspaces (DIM must be divisible)
N_CENTROIDS = 16          # centroids per subspace (full: 256, but small for test)
PQ_SUBSPACE_DIM = DIM // N_SUBSPACES  # dimensions per subspace = 8

# Storage key type prefixes (1 byte each)
KEY_TYPE_BASE = 0x01      # co-located vector + neighbor list
KEY_TYPE_DELTA = 0x02     # backedge delta neighbors
KEY_TYPE_META = 0x03      # metadata (entry point, node count, etc.)

# Metadata keys
META_ENTRY_POINT = b'\x03\x00'   # stores the integer entry point node ID
META_NODE_COUNT = b'\x03\x01'    # stores total node count

# Memory vs. disk threshold (for transitions)
MEMORY_LIMIT_NODES = 200  # switch to on-disk after this many nodes (full: much larger)
