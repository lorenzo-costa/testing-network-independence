import operator

import numpy as np
from scipy.spatial import cKDTree

from ._ac_helpers_old import _orthant_counts as _direct_orthant_counts
from ._orthant_ranks import _dominance_counts

def _orthant_counts(
    sample: np.ndarray,
    thresholds: np.ndarray,
    *,
    relation: str,
    block_size: int,
) -> np.ndarray:
    """Count coordinatewise ranks exactly using recursive sorted blocks.

    ``block_size`` bounds direct-comparison leaf batches. Non-finite,
    non-real, or mixed-dtype inputs retain the original NumPy comparisons.
    """
    if block_size < 1:
        raise ValueError("block_size must be positive.")

    thresholds = np.asarray(thresholds)
    if sample.ndim != 2 or thresholds.shape[-1] != sample.shape[1]:
        raise ValueError("Incompatible sample and threshold shapes.")

    flat = thresholds.reshape(-1, sample.shape[1])
    # Match range(..., block_size)'s integer requirement in the old routine.
    block_size = operator.index(block_size)
    if (
        sample.dtype.kind not in "biuf"
        or flat.dtype.kind not in "biuf"
        # Mixed integer/float searches can promote differently from NumPy's
        # elementwise comparisons (notably int64/uint64 near their limits).
        or sample.dtype != flat.dtype
        or not np.isfinite(sample).all()
        or not np.isfinite(flat).all()
    ):
        return _direct_orthant_counts(
            sample, thresholds, relation=relation, block_size=block_size
        )

    counts = _dominance_counts(
        sample, flat, lower=relation == "le", block_size=block_size
    )
    return counts.reshape(thresholds.shape[:-1])

# neighbour maps
def _neighbor_maps(
    z,
    x,
    *,
    n: int,
    M: int,
    rng=None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return neighbors in Z/(X,Z), and in X when conditioning is used."""
    if x is None:
        m_idx = _knn_indices(z, M, rng=rng)
        return m_idx, None
    else:
        x = _as_2d(x, name="X", n=n)
        m_idx = _knn_indices(np.hstack((x, z)), M, rng=rng)
        n_idx = _knn_indices(x, M, rng=rng)
        return m_idx, n_idx

# NN-indices

def _cutoff_radii(
    tree: cKDTree,
    points: np.ndarray,
    distances: np.ndarray,
    indices: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the original code's cutoff values and radii for every query point.

    This batches only the rare full-tree fallback used when duplicate points
    push ``self`` outside the initial ``k + 1`` query results.  The selected
    cutoff in each row is exactly the one obtained by the original
    ``d[indices != i][k - 1]`` logic.
    """
    n = len(points)
    nonself = indices != np.arange(n)[:, None]
    count = nonself.sum(axis=1)

    cutoffs = np.empty(n, dtype=float)
    regular = count >= k

    # The k-th non-self item in each initial cKDTree query, preserving the
    # original query ordering.  This also handles duplicate rows for which
    # self is absent and there are k + 1 valid non-self entries.
    rank = np.cumsum(nonself, axis=1)
    regular_rows = np.flatnonzero(regular)
    if regular_rows.size:
        regular_cols = np.argmax(rank[regular] == k, axis=1)
        cutoffs[regular_rows] = distances[regular_rows, regular_cols]

    # This is rare: query all points for affected duplicate rows in one
    # batched cKDTree call instead of one Python-level call per row.
    short_rows = np.flatnonzero(~regular)
    if short_rows.size:
        full_distances, full_indices = tree.query(points[short_rows], k=n)
        full_distances = np.atleast_2d(full_distances)
        full_indices = np.atleast_2d(full_indices)

        full_nonself = full_indices != short_rows[:, None]
        full_rank = np.cumsum(full_nonself, axis=1)
        full_cols = np.argmax(full_rank == k, axis=1)
        cutoffs[short_rows] = full_distances[
            np.arange(short_rows.size), full_cols
        ]

    return cutoffs, np.nextafter(cutoffs, np.inf)

# faster than scipy from Huang's paper
def _knn_indices(
    points: np.ndarray,
    M: int,
    *,
    rng=None,
) -> np.ndarray:
    """
    Return the M nearest non-self neighbors for every row.

    This is behaviorally identical to the reference implementation for a
    fixed SciPy version and RNG state.  It keeps the reference tie-breaking
    and candidate-processing logic, but performs all cutoff-radius searches
    in one batched ``query_ball_point`` call.

    ``return_sorted=False`` is essential: the original makes single-point
    queries, whose default output is unsorted.  Preserving that order also
    preserves the exact seeded tie outcomes and output layout.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)

    if not 1 <= M < n:
        raise ValueError("Require 1 <= M < n.")

    generator = (
        rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    )

    tree = cKDTree(points)

    # Same initial query as the reference implementation.
    distances, indices = tree.query(points, k=M + 1)
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)

    cutoffs, radii = _cutoff_radii(tree, points, distances, indices, M)

    # cKDTree treats batched queries differently under its default
    # return_sorted=None, so request the single-query behavior explicitly.
    candidate_lists = tree.query_ball_point(
        points,
        radii,
        return_sorted=False,
    )

    neighbors = np.empty((n, M), dtype=np.int64)
    eps = np.finfo(float).eps

    # This loop intentionally mirrors the reference code so RNG calls,
    # candidate order, tolerance checks, and fallback behavior are unchanged.
    for i, candidate_list in enumerate(candidate_lists):
        candidates = np.asarray(candidate_list, dtype=np.int64)
        candidates = candidates[candidates != i]

        candidate_distances = np.linalg.norm(
            points[candidates] - points[i],
            axis=1,
        )

        cutoff = cutoffs[i]

        atol = eps * max(1.0, cutoff) * 16
        strictly_closer = candidates[candidate_distances < cutoff - atol]
        tied = candidates[np.abs(candidate_distances - cutoff) <= atol]

        needed = M - len(strictly_closer)

        if needed < 0:
            jitter = generator.random(len(candidates))
            order = np.lexsort((jitter, candidate_distances))
            neighbors[i] = candidates[order[:M]]
            continue

        if len(tied) < needed:
            jitter = generator.random(len(candidates))
            order = np.lexsort((jitter, candidate_distances))
            neighbors[i] = candidates[order[:M]]
            continue

        if needed == 0:
            neighbors[i] = strictly_closer[:M]
        else:
            chosen_ties = generator.choice(tied, size=needed, replace=False)
            neighbors[i] = np.concatenate((strictly_closer, chosen_ties))

    return neighbors

# validation functions
def _validate_m(M: int | float, n: int) -> int:
    """Resolve an integer neighbor count or a sublinear exponent of ``n``."""
    if isinstance(M, (bool, np.bool_)):
        raise ValueError(
            "M must be a positive integer or a finite float smaller than 1."
        )

    if isinstance(M, (int, np.integer)):
        resolved_m = int(M)
    elif isinstance(M, (float, np.floating)):
        exponent = float(M)
        if not np.isfinite(exponent) or exponent >= 1:
            raise ValueError("A float M must be finite and smaller than 1.")
        resolved_m = round(n**exponent)
    else:
        raise ValueError(
            "M must be a positive integer or a finite float smaller than 1, "
            f"got type {type(M)}."
        )

    if resolved_m < 1:
        raise ValueError("M must resolve to a positive integer.")
    if resolved_m >= n:
        raise ValueError("M must resolve to a value smaller than n.")

    return resolved_m


# misc helpers
def _as_2d(values, *, name: str, n: int | None = None) -> np.ndarray:
    """Return a finite array with shape ``(n, d)``."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    elif values.ndim != 2:
        raise ValueError(f"{name} must have shape (n,) or (n, d).")

    if values.shape[1] == 0:
        raise ValueError(f"{name} must have at least one column.")
    if n is not None and values.shape[0] != n:
        raise ValueError(f"{name} must have {n} rows.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values.")
    return values

def _make_permutations(n: int, d_y: int, rng=None) -> np.ndarray:
    """Make coordinate permutations with distinct source rows per observation."""
    if d_y > n:
        raise ValueError("The multivariate permutation construction requires d_Y <= n.")

    generator = (
        rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    )
    base = generator.permutation(n)
    shifts = generator.choice(n, size=d_y, replace=False)
    return np.array([np.roll(base, -shift) for shift in shifts], dtype=np.int64)
