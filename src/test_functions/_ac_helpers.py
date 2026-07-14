import numpy as np
from scipy.spatial import cKDTree

def _orthant_counts(
    sample: np.ndarray,
    thresholds: np.ndarray,
    *,
    relation: str,
    block_size: int,
) -> np.ndarray:
    """Count sample rows lying below or above each coordinatewise threshold."""
    if block_size < 1:
        raise ValueError("block_size must be positive.")

    thresholds = np.asarray(thresholds)
    if sample.ndim != 2 or thresholds.shape[-1] != sample.shape[1]:
        raise ValueError("Incompatible sample and threshold shapes.")

    compare = np.less_equal if relation == "le" else np.greater_equal
    flat = thresholds.reshape(-1, sample.shape[1])
    counts = np.empty(len(flat), dtype=np.int64)

    for start in range(0, len(flat), block_size):
        stop = min(start + block_size, len(flat))
        counts[start:stop] = np.sum(
            np.all(compare(sample[:, None, :], flat[None, start:stop, :]), axis=-1),
            axis=0,
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
    right_neighbor=False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return neighbors in Z/(X,Z), and in X when conditioning is used."""
    if right_neighbor is True:
        if x is not None:
            raise ValueError("right_neighbor is only used when X is None.")
        m_idx = _right_neighbor_indices(z, M, rng=rng)
        return m_idx, None
    if x is None:
        m_idx = _knn_indices(z, M, rng=rng)
        return m_idx, None
    else:
        x = _as_2d(x, name="X", n=n)
        m_idx = _knn_indices(np.hstack((x, z)), M, rng=rng)
        n_idx = _knn_indices(x, M, rng=rng)
        return m_idx, n_idx

def _neighbor_prefix_maps(
    z: np.ndarray,
    x,
    *,
    n: int,
    max_m: int,
    rng=None,
    right_neighbor=False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ordered neighbor maps whose first M columns define each M-NN map."""
    if right_neighbor is True:
        if x is not None:
            raise ValueError("right_neighbor is only used when X is None.")
        return _right_neighbor_indices(z, max_m, rng=rng), None

    if x is None:
        return _knn_prefix_indices(z, max_m, rng=rng), None

    x = _as_2d(x, name="X", n=n)
    return (
        _knn_prefix_indices(np.hstack((x, z)), max_m, rng=rng),
        _knn_prefix_indices(x, max_m, rng=rng),
    )


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

def _right_neighbor_indices(
    z: np.ndarray,
    M: int,
    rng=None,
) -> np.ndarray:
    """
    Return the m-th right neighbor of each scalar z_i.

    Ties in z are broken uniformly at random, reproducibly when rng is given.
    """

    if z.ndim != 2 or z.shape[1] != 1:
        raise ValueError("Right neighbors require a scalar ordering variable.")

    values = z[:, 0]
    n = len(values)

    generator = (
        rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    )

    # Primary sort key: values.
    # Secondary sort key: random numbers, used only within ties.
    tie_breaker = generator.random(n)
    order = np.lexsort((tie_breaker, values))

    position = np.empty(n, dtype=np.int64)
    position[order] = np.arange(n)

    targets = position[:, None] + np.arange(1, M + 1)

    # Boundary convention: if the m-th right neighbor does not exist,
    # map the observation to itself.
    neighbors = np.broadcast_to(
        np.arange(n)[:, None],
        (n, M),
    ).copy()

    valid = targets < n
    neighbors[valid] = order[targets[valid]]
    
    return neighbors


def _knn_prefix_indices(
    points: np.ndarray,
    max_m: int,
    *,
    rng=None,
) -> np.ndarray:
    """Return an ordered K-NN map; column m is the (m+1)-th neighbor.

    This retains the reference implementation's exact ordering and RNG
    consumption, while batching the cutoff-radius searches across all rows.
    Thus the first M columns match the reference output for every M.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if not 1 <= max_m < n:
        raise ValueError("Require 1 <= max_m < n.")

    generator = (
        rng
        if isinstance(rng, np.random.Generator)
        else np.random.default_rng(rng)
    )
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=max_m + 1)
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)

    _, radii = _cutoff_radii(tree, points, distances, indices, max_m)
    candidate_lists = tree.query_ball_point(
        points,
        radii,
        return_sorted=False,
    )

    neighbors = np.empty((n, max_m), dtype=np.int64)
    all_indices = np.arange(n, dtype=np.int64)

    for i, candidate_list in enumerate(candidate_lists):
        candidates = np.asarray(candidate_list, dtype=np.int64)
        candidates = candidates[candidates != i]

        # Preserve the reference numerical guard.
        if candidates.size < max_m:
            candidates = np.delete(all_indices, i)

        candidate_distances = np.linalg.norm(
            points[candidates] - points[i],
            axis=1,
        )
        tie_breaker = generator.random(candidates.size)
        order = np.lexsort((tie_breaker, candidate_distances))
        neighbors[i] = candidates[order[:max_m]]

    return neighbors

# validation functions 
def _validate_m(M: int, n: int) -> int:
    if isinstance(M, str):
        if M.lower() == "sqrt":
            M = int(np.sqrt(n))
        elif M.lower() == "log":
            M = int(np.log(n))
        elif M.lower() == "half":
            M = int(n / 2)
        elif M.lower() == "third":
            M = int(n / 3)
        elif M.lower() == "quarter":
            M = int(n / 4)
        else:
            raise ValueError("M must be a positive integer or 'sqrt' or 'log'.")
    elif isinstance(M, (int, np.integer)):
        if M < 1:
            raise ValueError("M must be a positive integer.")
        elif M >= n:
            raise ValueError("M must be smaller than n.")
        else:
            M = int(M)
    else:
        raise ValueError("M must be a positive integer or 'sqrt' or 'log', got type {}".format(type(M)))
      
    return int(M)

def _validate_aggregate(aggregate: str) -> str:
    if not isinstance(aggregate, str):
        raise TypeError("aggregate must be either 'avg' or 'max'.")

    aggregate = aggregate.lower()
    if aggregate not in {"avg", "max"}:
        raise ValueError("aggregate must be either 'avg' or 'max'.")
    return aggregate

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

def _aggregate_coefficients(coefficients: np.ndarray, aggregate: str) -> float:
    return float(coefficients.max() if aggregate == "max" else coefficients.mean())

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