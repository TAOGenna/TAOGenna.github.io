"""
Test fixtures for the Mini Vector Database.
"""
import os
import sys

# Ensure the project root is on the path so imports work
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import pytest
from config import DIM


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def small_vectors(rng):
    """50 random vectors for quick tests."""
    return rng.standard_normal((50, DIM)).astype(np.float32)


@pytest.fixture
def medium_vectors(rng):
    """200 random vectors for larger tests."""
    return rng.standard_normal((200, DIM)).astype(np.float32)


@pytest.fixture
def clustered_vectors(rng):
    """
    300 vectors in 3 clusters, useful for testing recall.
    Cluster centers are well-separated so nearest-neighbor queries are unambiguous.
    """
    centers = rng.standard_normal((3, DIM)).astype(np.float32) * 5.0
    vecs = []
    labels = []
    for cluster_id, center in enumerate(centers):
        cluster_vecs = center + rng.standard_normal((100, DIM)).astype(np.float32) * 0.5
        vecs.append(cluster_vecs)
        labels.extend([cluster_id] * 100)
    return np.vstack(vecs).astype(np.float32), np.array(labels)


@pytest.fixture
def empty_db():
    from vectordb import VectorDB
    return VectorDB(mode="in_memory")


@pytest.fixture
def populated_db(small_vectors):
    """VectorDB with 50 vectors inserted."""
    from vectordb import VectorDB
    db = VectorDB(mode="in_memory")
    for i, vec in enumerate(small_vectors):
        db.insert(i, vec)
    return db, small_vectors
