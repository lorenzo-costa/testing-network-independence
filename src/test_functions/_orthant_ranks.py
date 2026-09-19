"""Exact multivariate ranks using recursive sorted-block decomposition.

Based on Algorithms 1--2 of Huang, Li and Wang, Section 5 / Theorem 8,
https://arxiv.org/abs/2512.07443v2. For fixed dimension d and N equal to
max(number of sample rows, number of queries), the bound is O(N log(N)**d).
Small subproblems use direct comparisons to avoid recursive sorting overhead.
"""

import numpy as np
from numba import njit

from ._ac_helpers_old import _orthant_counts as _direct_counts


@njit(cache=True)
def _two_dimensional_counts(second, query_second, lengths, lower):
    """Algorithm 1's sorted-block ranks, with no RNG or fast-math changes.

    ``lengths`` already encodes the inclusive first-coordinate prefix.
    Sorting second-coordinate blocks and binary-searching them avoids
    scanning sample points for each query. This kernel is single-threaded;
    permutation-level multiprocessing remains the only parallel layer.
    """
    n = len(second)
    counts = np.zeros(len(lengths), dtype=np.int64)
    for i in range(len(lengths)):
        last = lengths[i] - 1
        if last >= 0:
            if lower:
                counts[i] = second[last] <= query_second[i]
            else:
                counts[i] = second[last] >= query_second[i]

    blocks = second.copy()
    width = 1
    while width < n:
        # Independent block sorts have the paper's O(n log(n)^2) bound.
        for start in range(0, n, width):
            blocks[start:min(start + width, n)].sort()
        for i in range(len(lengths)):
            last = lengths[i] - 1
            if last < 1:
                continue
            block = last // width
            if block % 2 == 0:
                continue
            start = (block - 1) * width
            stop = start + width
            lo, hi = start, stop
            value = query_second[i]
            while lo < hi:
                mid = (lo + hi) // 2
                go_right = (blocks[mid] <= value) if lower else (blocks[mid] < value)
                if go_right:
                    lo = mid + 1
                else:
                    hi = mid
            counts[i] += (lo - start) if lower else (stop - lo)
        width *= 2
    return counts


def _dominance_counts(sample, queries, *, lower, block_size):
    """Count sample rows <= each query (or >= when lower is false).

    Counts and query order are exact, including coordinate ties. Descending
    prefixes handle >= without negating values, avoiding integer overflow.
    This routine receives two-dimensional, finite real arrays.
    """
    n, d = sample.shape
    q = len(queries)
    if n == 0 or q == 0:
        return np.zeros(q, dtype=np.int64)

    if d == 1:
        ordered = np.sort(sample[:, 0])
        if lower:
            counts = np.searchsorted(ordered, queries[:, 0], side="right")
        else:
            counts = n - np.searchsorted(ordered, queries[:, 0], side="left")
        return counts.astype(np.int64, copy=False)

    # Bounded-size leaves preserve the asymptotic bound and limit temporaries
    # to at most block_size queries, as in the original implementation.
    if n <= 8 or n * q <= 256:
        return _direct_counts(
            sample, queries, relation="le" if lower else "ge",
            block_size=block_size,
        )

    order = np.argsort(sample[:, 0], kind="stable")
    ordered_first = sample[order, 0]
    if lower:
        lengths = np.searchsorted(ordered_first, queries[:, 0], side="right")
        compare = np.less_equal
    else:
        lengths = n - np.searchsorted(ordered_first, queries[:, 0], side="left")
        order = order[::-1]
        compare = np.greater_equal
    ordered_sample = sample[order]

    # Algorithm 2 terminates at Algorithm 1. Other real dtypes retain the
    # NumPy recursion (Numba does not support e.g. float16 or longdouble).
    if d == 2 and sample.dtype in (
        np.dtype("float32"), np.dtype("float64"),
        np.dtype("int64"), np.dtype("uint64"),
    ):
        return _two_dimensional_counts(
            ordered_sample[:, 1], queries[:, 1], lengths, lower
        )

    counts = np.zeros(q, dtype=np.int64)
    active = np.flatnonzero(lengths)
    if active.size == 0:
        return counts
    # Queries with the same prefix block are contiguous; grouping them once
    # avoids scanning every query separately for every sample block.
    query_order = active[np.argsort(lengths[active], kind="stable")]
    last = lengths[query_order] - 1
    counts[query_order] = np.all(
        compare(ordered_sample[last, 1:], queries[query_order, 1:]), axis=1
    )

    # The last point plus the left-sibling blocks on its binary-tree path
    # partition each eligible prefix without omission or double counting.
    width = 1
    while width < n:
        middles = np.arange(width, n, 2 * width)
        ends = np.minimum(middles + width, n)
        starts_q = np.searchsorted(last, middles, side="left")
        ends_q = np.searchsorted(last, ends, side="left")
        for middle, start_q, end_q in zip(middles, starts_q, ends_q):
            if start_q == end_q:
                continue
            indices = query_order[start_q:end_q]
            counts[indices] += _dominance_counts(
                ordered_sample[middle - width:middle, 1:],
                queries[indices, 1:],
                lower=lower,
                block_size=block_size,
            )
        width *= 2

    return counts
