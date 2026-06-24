
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
def _knn_indices(
    points: np.ndarray,
    M: int,
    *,
    rng=None,
) -> np.ndarray:
    """
    Return the M nearest non-self neighbors for every row.

    Distance ties at the M-th cutoff are broken uniformly at random.
    Passing the same seed gives reproducible results, independent of the
    order returned by cKDTree for tied points.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)

    if not 1 <= M < n:
        raise ValueError("Require 1 <= M < n.")

    generator = (
        rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    )

    tree = cKDTree(points)

    # Query M+1 because one returned point is normally self.
    distances, indices = tree.query(points, k=M + 1)
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)

    neighbors = np.empty((n, M), dtype=np.int64)

    for i in range(n):
        # Remove self, regardless of where cKDTree placed it.
        mask = indices[i] != i
        d = distances[i][mask]
        idx = indices[i][mask]

        # In duplicate-point cases, self might not be included among M+1
        # returned entries. Re-query more broadly in that rare case.
        if len(idx) < M:
            d, idx = tree.query(points[i], k=n)
            d = np.asarray(d)
            idx = np.asarray(idx)

            mask = idx != i
            d = d[mask]
            idx = idx[mask]

        cutoff = d[M - 1]

        # Radius is nudged upward so points exactly at cutoff are included.
        radius = np.nextafter(cutoff, np.inf)
        candidates = np.asarray(tree.query_ball_point(points[i], radius))

        candidates = candidates[candidates != i]

        candidate_distances = np.linalg.norm(
            points[candidates] - points[i],
            axis=1,
        )

        # Tolerance is needed because floating-point distance computations
        # can differ slightly between KD-tree traversal and direct norm.
        atol = np.finfo(float).eps * max(1.0, cutoff) * 16

        strictly_closer = candidates[candidate_distances < cutoff - atol]
        tied = candidates[np.abs(candidate_distances - cutoff) <= atol]

        needed = M - len(strictly_closer)

        if needed < 0:
            # Numerical guard: retain exactly the M closest, with seeded
            # random ordering only within nearly equal distances.
            jitter = generator.random(len(candidates))
            order = np.lexsort((jitter, candidate_distances))
            neighbors[i] = candidates[order[:M]]
            continue

        if len(tied) < needed:
            # Extremely unusual numerical mismatch; recover with an exact
            # local sort, still avoiding a full n-by-n matrix.
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

    The ordering is by distance, with uniform random ordering within exact
    distance ties. Therefore, the first M columns form a valid M-nearest-
    neighbor map for every M <= max_m.
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

    neighbors = np.empty((n, max_m), dtype=np.int64)

    for i in range(n):
        d = distances[i]
        idx = indices[i]
        keep = idx != i
        d, idx = d[keep], idx[keep]

        # In duplicate-point cases, self may be absent from the first K+1
        # returned indices, or some returned entries may not be enough after
        # self-removal. A full query is only needed in that rare situation.
        if idx.size < max_m:
            d, idx = tree.query(points[i], k=n)
            d = np.asarray(d)
            idx = np.asarray(idx)
            keep = idx != i
            d, idx = d[keep], idx[keep]

        cutoff = d[max_m - 1]
        radius = np.nextafter(cutoff, np.inf)
        candidates = np.asarray(tree.query_ball_point(points[i], radius), dtype=np.int64)
        candidates = candidates[candidates != i]

        if candidates.size < max_m:
            # Guard against an exceptional KD-tree radius / floating-point
            # mismatch by using direct distances to every other observation.
            candidates = np.delete(np.arange(n, dtype=np.int64), i)

        candidate_distances = np.linalg.norm(points[candidates] - points[i], axis=1)
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

