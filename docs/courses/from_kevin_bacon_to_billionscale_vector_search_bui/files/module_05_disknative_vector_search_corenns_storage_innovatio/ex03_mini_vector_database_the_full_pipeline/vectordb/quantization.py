"""
Product Quantization (PQ) for vector compression.
Implements training (codebook fitting) and encoding/decoding.

From Module 4: PQ divides a d-dimensional vector into n_subspaces subvectors,
clusters each subspace into n_centroids clusters via k-means, and represents
each chunk with its 1-byte centroid index. Achieves ~32x compression.
"""

import numpy as np
from config import N_SUBSPACES, N_CENTROIDS, DIM, PQ_SUBSPACE_DIM


class ProductQuantizer:
    """
    Product Quantizer: compress vectors to PQ codes and compute ADC distances.

    Attributes
    ----------
    n_subspaces  : number of subvector segments
    n_centroids  : centroids per subspace (≤ 256 for uint8 codes)
    subspace_dim : dimensions per subspace
    codebooks    : shape (n_subspaces, n_centroids, subspace_dim) — fitted centroids
    is_fitted    : bool
    """

    def __init__(self, n_subspaces: int = N_SUBSPACES,
                 n_centroids: int = N_CENTROIDS):
        self.n_subspaces = n_subspaces
        self.n_centroids = n_centroids
        self.subspace_dim = DIM // n_subspaces
        self.codebooks: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, vectors: np.ndarray, n_iter: int = 20) -> "ProductQuantizer":
        """
        Fit PQ codebooks via k-means on training vectors.

        Parameters
        ----------
        vectors : shape (n_train, DIM) — training data
        n_iter  : k-means iterations

        Returns
        -------
        self
        """
        n_train = len(vectors)
        self.codebooks = np.zeros(
            (self.n_subspaces, self.n_centroids, self.subspace_dim), dtype=np.float32)

        for s in range(self.n_subspaces):
            start = s * self.subspace_dim
            end = start + self.subspace_dim
            subvecs = vectors[:, start:end].astype(np.float32)

            # K-means: randomly initialize centroids
            rng = np.random.default_rng(s)
            indices = rng.choice(n_train, size=min(self.n_centroids, n_train),
                                 replace=False)
            centroids = subvecs[indices].copy()

            for _ in range(n_iter):
                # Assign each subvector to nearest centroid
                diffs = subvecs[:, None, :] - centroids[None, :, :]  # (n, k, d_sub)
                dists = np.sum(diffs ** 2, axis=2)                   # (n, k)
                assignments = np.argmin(dists, axis=1)               # (n,)

                # Update centroids
                new_centroids = np.zeros_like(centroids)
                for k in range(self.n_centroids):
                    mask = assignments == k
                    if mask.sum() > 0:
                        new_centroids[k] = subvecs[mask].mean(axis=0)
                    else:
                        new_centroids[k] = centroids[k]  # keep if empty
                centroids = new_centroids

            self.codebooks[s] = centroids

        self.is_fitted = True
        return self

    def encode(self, vector: np.ndarray) -> np.ndarray:
        """
        Encode a single vector to PQ code.

        Parameters
        ----------
        vector : shape (DIM,)

        Returns
        -------
        np.ndarray of shape (n_subspaces,), dtype uint8
        """
        assert self.is_fitted, "Must call fit() before encode()"
        code = np.zeros(self.n_subspaces, dtype=np.uint8)
        v = vector.astype(np.float32)
        for s in range(self.n_subspaces):
            start = s * self.subspace_dim
            end = start + self.subspace_dim
            subvec = v[start:end]
            diffs = subvec[None, :] - self.codebooks[s]  # (k, d_sub)
            dists = np.sum(diffs ** 2, axis=1)           # (k,)
            code[s] = np.argmin(dists)
        return code

    def decode(self, code: np.ndarray) -> np.ndarray:
        """
        Reconstruct approximate vector from PQ code.

        Parameters
        ----------
        code : shape (n_subspaces,), dtype uint8

        Returns
        -------
        np.ndarray of shape (DIM,), dtype float32 — approximate reconstruction
        """
        assert self.is_fitted
        parts = []
        for s in range(self.n_subspaces):
            parts.append(self.codebooks[s, code[s]])
        return np.concatenate(parts)

    def build_adc_table(self, query: np.ndarray) -> np.ndarray:
        """
        Build Asymmetric Distance Computation (ADC) lookup table for a query.

        Precomputes distance from query to every centroid in every subspace.
        Then, distance from query to any encoded vector is just n_subspaces
        table lookups — much faster than computing full L2.

        Parameters
        ----------
        query : shape (DIM,)

        Returns
        -------
        np.ndarray of shape (n_subspaces, n_centroids)
            table[s, k] = squared L2 distance from query subvec s to centroid k
        """
        assert self.is_fitted
        q = query.astype(np.float32)
        table = np.zeros((self.n_subspaces, self.n_centroids), dtype=np.float32)
        for s in range(self.n_subspaces):
            start = s * self.subspace_dim
            end = start + self.subspace_dim
            q_sub = q[start:end]
            diffs = q_sub[None, :] - self.codebooks[s]  # (k, d_sub)
            table[s] = np.sum(diffs ** 2, axis=1)        # (k,)
        return table

    def adc_distance(self, adc_table: np.ndarray, code: np.ndarray) -> float:
        """
        Compute approximate distance using pre-built ADC table.

        Parameters
        ----------
        adc_table : shape (n_subspaces, n_centroids) from build_adc_table()
        code      : shape (n_subspaces,), dtype uint8

        Returns
        -------
        float: approximate squared L2 distance
        """
        return float(sum(adc_table[s, code[s]] for s in range(self.n_subspaces)))
