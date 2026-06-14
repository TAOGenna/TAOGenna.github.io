"""
Distance computation utilities.
Provides L2 (squared Euclidean) distance, used throughout the vector DB.
"""

import numpy as np


def l2_squared(a: np.ndarray, b: np.ndarray) -> float:
    """
    Squared L2 (Euclidean) distance between two vectors.

    Parameters
    ----------
    a, b : np.ndarray of same shape and dtype

    Returns
    -------
    float: ||a - b||^2
    """
    diff = a.astype(np.float32) - b.astype(np.float32)
    return float(np.dot(diff, diff))


def batch_l2_squared(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    Squared L2 distance from query to each row in candidates.

    Parameters
    ----------
    query      : shape (d,)
    candidates : shape (n, d)

    Returns
    -------
    np.ndarray of shape (n,) with distances
    """
    diff = candidates.astype(np.float32) - query.astype(np.float32)
    return np.sum(diff ** 2, axis=1)
