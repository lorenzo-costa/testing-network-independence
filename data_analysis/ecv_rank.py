"""
ecv_rank.py

Python implementation of Edge Cross-Validation (ECV) rank selection
translated from the supplied R implementation.

Dependencies
------------
numpy
scikit-learn

Install with:
    pip install numpy scikit-learn
"""

import numpy as np
from dataclasses import dataclass
from typing import Iterator, Optional

from sklearn.utils.extmath import randomized_svd
from sklearn.metrics import roc_auc_score


@dataclass
class ECVRankResult:
    """Container for ECV rank-selection results."""

    sse_rank: int
    auc_rank: Optional[int]
    sse: np.ndarray
    auc: np.ndarray


def _iter_rank_reconstructions(
    A_missing,
    max_k,
    p_sample,
    random_state=None,
) -> Iterator[np.ndarray]:
    """
    Yield successive truncated-SVD reconstructions using one work array.

    Missing entries are replaced by zero and the observed matrix is
    rescaled by 1 / p_sample, matching the fast ECV reconstruction
    logic in the supplied R code.

    Parameters
    ----------
    A_missing : array-like, shape (n, m)
        Matrix containing NaN at held-out entries.

    max_k : int
        Maximum rank.

    p_sample : float
        Probability that an entry is observed.

    random_state : int or None
        Seed passed to randomized_svd.

    Yields
    ------
    np.ndarray
        The raw rank-k reconstruction for k=1,...,max_k. The same
        work array is updated and yielded at every rank, so callers
        must copy it if they need to retain a reconstruction.

    Notes
    -----
    Streaming the reconstructions keeps memory use O(n * m) instead
    of retaining two n-by-m arrays for every candidate rank.
    """
    X = np.asarray(A_missing, dtype=float).copy()

    # Replace held-out entries by zero.
    X[np.isnan(X)] = 0.0

    # Correct for edge sampling probability.
    X /= p_sample

    n, m = X.shape
    max_k = min(max_k, n, m)

    U, s, Vt = randomized_svd(
        X,
        n_components=max_k,
        random_state=random_state,
    )

    Xk = np.zeros_like(X)
    del X

    for k in range(max_k):
        Xk += s[k] * np.outer(U[:, k], Vt[k, :])
        yield Xk


def _evaluate_directed(
    A,
    holdout_indices,
    max_k,
    p_sample,
    weighted=True,
    random_state=None,
):
    """Evaluate one ECV replication for a directed network."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    A_train = A.copy()

    flat = A_train.ravel()
    flat[holdout_indices] = np.nan
    A_train = flat.reshape(n, n)

    response = A.ravel()[holdout_indices]
    compute_auc = not weighted and np.unique(response).size == 2

    sse = np.zeros(max_k)
    auc = np.full(max_k, np.nan)

    reconstructions = _iter_rank_reconstructions(
        A_train,
        max_k=max_k,
        p_sample=p_sample,
        random_state=random_state,
    )

    for k, A_hat in enumerate(reconstructions):
        # Clipping only the held-out predictions is equivalent to clipping
        # the full matrix and avoids another n-by-n temporary array.
        predictions = np.clip(
            A_hat.ravel()[holdout_indices],
            0.0,
            1.0,
        )

        sse[k] = np.mean((response - predictions) ** 2)

        if compute_auc:
            auc[k] = roc_auc_score(response, predictions)

    return sse, auc


def _evaluate_undirected(
    A,
    holdout_pairs,
    max_k,
    p_sample,
    weighted=True,
    random_state=None,
):
    """Evaluate one ECV replication for an undirected network."""
    A = np.asarray(A, dtype=float)

    A_train = A.copy()
    i, j = holdout_pairs

    # Hide symmetric edge entries.
    A_train[i, j] = np.nan
    A_train[j, i] = np.nan

    # Preserve diagonal.
    np.fill_diagonal(A_train, np.diag(A))

    # Evaluate each undirected edge only once.
    response = A[i, j]
    compute_auc = not weighted and np.unique(response).size == 2

    sse = np.zeros(max_k)
    auc = np.full(max_k, np.nan)

    reconstructions = _iter_rank_reconstructions(
        A_train,
        max_k=max_k,
        p_sample=p_sample,
        random_state=random_state,
    )

    for k, A_hat in enumerate(reconstructions):
        # Undirected R implementation evaluates the raw reconstruction.
        predictions = A_hat[i, j]

        sse[k] = np.mean((response - predictions) ** 2)

        if compute_auc:
            auc[k] = roc_auc_score(response, predictions)

    return sse, auc


def ecv_rank(
    A,
    max_k,
    B=3,
    holdout_p=0.1,
    weighted=True,
    mode="directed",
    random_state=None,
):
    """
    Edge Cross-Validation (ECV) for selecting matrix/network rank.

    Parameters
    ----------
    A : array-like, shape (n, n)
        Network adjacency matrix.

    max_k : int
        Maximum candidate rank.

    B : int, default=3
        Number of independent ECV holdout replications.

    holdout_p : float, default=0.1
        Fraction of candidate edges/entries held out.

    weighted : bool, default=True
        If True, rank is selected using held-out squared error only.
        If False, AUC is also computed for binary networks.

    mode : {"directed", "undirected"}, default="directed"
        Type of network.

    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    ECVRankResult
        sse_rank :
            Rank minimizing mean held-out squared error.

        auc_rank :
            Rank maximizing mean held-out AUC, or None when
            weighted=True or AUC cannot be computed.

        sse :
            Mean held-out squared error for ranks 1,...,max_k.

        auc :
            Mean held-out AUC for ranks 1,...,max_k.
    """
    A = np.asarray(A, dtype=float)

    if A.ndim != 2:
        raise ValueError("A must be a matrix.")

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square.")

    if max_k < 1:
        raise ValueError("max_k must be >= 1.")

    if B < 1:
        raise ValueError("B must be >= 1.")

    if not 0 < holdout_p < 1:
        raise ValueError("holdout_p must satisfy 0 < holdout_p < 1.")

    if mode not in {"directed", "undirected"}:
        raise ValueError("mode must be 'directed' or 'undirected'.")

    if mode == "undirected" and not np.allclose(A, A.T):
        raise ValueError("Undirected adjacency matrix must be symmetric.")

    n = A.shape[0]
    max_k = min(max_k, n)

    rng = np.random.default_rng(random_state)

    sse_matrix = np.zeros((B, max_k))
    auc_matrix = np.full((B, max_k), np.nan)

    if mode == "directed":
        # Matches supplied R code: all n^2 entries are candidate holdouts.
        edge_n = n * n
        holdout_n = int(np.floor(holdout_p * edge_n))

        if holdout_n < 1:
            raise ValueError(
                "holdout_p is too small; no entries are held out."
            )

        for b in range(B):
            holdout = rng.choice(
                edge_n,
                size=holdout_n,
                replace=False,
            )

            seed = int(rng.integers(0, 2**32 - 1))

            sse, auc = _evaluate_directed(
                A=A,
                holdout_indices=holdout,
                max_k=max_k,
                p_sample=1.0 - holdout_p,
                weighted=weighted,
                random_state=seed,
            )

            sse_matrix[b, :] = sse
            auc_matrix[b, :] = auc

    else:
        # Candidate undirected edges are the strict upper triangle.
        row_idx, col_idx = np.triu_indices(n, k=1)
        edge_n = len(row_idx)
        holdout_n = int(np.floor(holdout_p * edge_n))

        if holdout_n < 1:
            raise ValueError(
                "holdout_p is too small; no edges are held out."
            )

        for b in range(B):
            selected = rng.choice(
                edge_n,
                size=holdout_n,
                replace=False,
            )

            holdout_pairs = (
                row_idx[selected],
                col_idx[selected],
            )

            seed = int(rng.integers(0, 2**32 - 1))

            sse, auc = _evaluate_undirected(
                A=A,
                holdout_pairs=holdout_pairs,
                max_k=max_k,
                p_sample=1.0 - holdout_p,
                weighted=weighted,
                random_state=seed,
            )

            sse_matrix[b, :] = sse
            auc_matrix[b, :] = auc

    sse_sequence = np.mean(sse_matrix, axis=0)

    # +1 because ranks are 1,...,max_k while Python indices are 0-based.
    sse_rank = int(np.argmin(sse_sequence) + 1)

    if weighted:
        auc_sequence = np.full(max_k, np.nan)
        auc_rank = None
    else:
        with np.errstate(invalid="ignore"):
            auc_sequence = np.nanmean(auc_matrix, axis=0)

        if np.all(np.isnan(auc_sequence)):
            auc_rank = None
        else:
            auc_rank = int(np.nanargmax(auc_sequence) + 1)

    return ECVRankResult(
        sse_rank=sse_rank,
        auc_rank=auc_rank,
        sse=sse_sequence,
        auc=auc_sequence,
    )


if __name__ == "__main__":
    # Small reproducible example.
    rng = np.random.default_rng(123)

    n = 80
    K_true = 3

    # Generate a simple symmetric low-rank probability matrix.
    Z = rng.normal(size=(n, K_true))
    P = Z @ Z.T

    # Rescale to probabilities.
    P -= P.min()
    if P.max() > 0:
        P /= P.max()
    P *= 0.25

    # Sample an undirected binary adjacency matrix.
    U = rng.random((n, n))
    A = (U < P).astype(float)
    A = np.triu(A, k=1)
    A = A + A.T

    result = ecv_rank(
        A,
        max_k=6,
        B=3,
        holdout_p=0.1,
        weighted=False,
        mode="undirected",
        random_state=123,
    )

    print("SSE-selected rank:", result.sse_rank)
    print("AUC-selected rank:", result.auc_rank)
    print("SSE sequence:", result.sse)
    print("AUC sequence:", result.auc)
