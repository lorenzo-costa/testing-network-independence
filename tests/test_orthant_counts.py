"""Exact regression comparisons against the saved pre-optimization code."""

import importlib

import numpy as np
import pytest

from src.methods.ac_test import MultivariateACTest
from src.test_functions._ac_helpers import _orthant_counts
from src.test_functions._ac_helpers_old import _orthant_counts as old_counts


@pytest.mark.parametrize("dimension", [1, 2, 3, 4, 5, 7])
@pytest.mark.parametrize("n", [0, 1, 8, 9, 16, 17, 31, 32, 33, 65, 129])
@pytest.mark.parametrize("relation", ["le", "ge"])
@pytest.mark.parametrize("tied", [False, True])
def test_counts_match_saved_version(dimension, n, relation, tied):
    rng = np.random.default_rng(10)
    sample = (
        rng.integers(-3, 4, size=(n, dimension)).astype(float)
        if tied else rng.normal(size=(n, dimension))
    )
    queries = np.concatenate((sample, rng.normal(size=(75, dimension))))
    before_sample, before_queries = sample.copy(), queries.copy()
    kwargs = dict(relation=relation, block_size=19)

    result = _orthant_counts(sample, queries, **kwargs)

    np.testing.assert_array_equal(result, old_counts(sample, queries, **kwargs))
    assert result.dtype == np.int64
    np.testing.assert_array_equal(sample, before_sample)
    np.testing.assert_array_equal(queries, before_queries)


@pytest.mark.parametrize("dimension", [1, 2, 3, 4, 5, 7])
@pytest.mark.parametrize("n", [9, 33, 65, 129])
@pytest.mark.parametrize("relation", ["le", "ge"])
def test_all_identical_samples_match_saved_version(dimension, n, relation):
    point = np.arange(dimension, dtype=float) - 3
    sample = np.tile(point, (n, 1))
    # Move one coordinate at a time to catch incorrect tie handling at any
    # recursion level, not just in the first-coordinate prefix search.
    queries = np.vstack((
        point, point - 1, point + 1,
        point - np.eye(dimension), point + np.eye(dimension),
    ))
    expected = (
        [n, 0, n] + [0] * dimension + [n] * dimension
        if relation == "le"
        else [n, n, 0] + [n] * dimension + [0] * dimension
    )
    # Repeated queries also keep n*q above the direct-counting cutoff.
    queries = np.repeat(queries, 4, axis=0)
    expected = np.repeat(np.asarray(expected, dtype=np.int64), 4)
    kwargs = dict(relation=relation, block_size=7)

    np.testing.assert_array_equal(old_counts(sample, queries, **kwargs), expected)
    np.testing.assert_array_equal(_orthant_counts(sample, queries, **kwargs), expected)


@pytest.mark.parametrize("dimension", [1, 2, 3, 4, 5, 7])
@pytest.mark.parametrize("relation", ["le", "ge"])
def test_all_or_none_dominated_queries_match_saved_version(dimension, relation):
    rng = np.random.default_rng(57)
    sample = rng.integers(-10, 11, size=(65, dimension)).astype(float)
    if relation == "le":
        inclusive_all = sample.max(axis=0)
        strictly_all = inclusive_all + 1
        strictly_none = sample.min(axis=0) - 1
    else:
        inclusive_all = sample.min(axis=0)
        strictly_all = inclusive_all - 1
        strictly_none = sample.max(axis=0) + 1
    fails_first, fails_last = strictly_all.copy(), strictly_all.copy()
    fails_first[0] = strictly_none[0]
    fails_last[-1] = strictly_none[-1]
    # fails_last admits every first-coordinate prefix point when d > 1,
    # but must still count zero after checking the remaining coordinates.
    queries = np.stack((
        inclusive_all, strictly_all, fails_first, fails_last, strictly_none,
    ))
    expected = np.array([65, 65, 0, 0, 0], dtype=np.int64)
    kwargs = dict(relation=relation, block_size=3)

    np.testing.assert_array_equal(old_counts(sample, queries, **kwargs), expected)
    np.testing.assert_array_equal(_orthant_counts(sample, queries, **kwargs), expected)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("dimension", [1, 2, 3, 4, 5, 7])
@pytest.mark.parametrize("relation", ["le", "ge"])
@pytest.mark.parametrize("distribution", ["continuous", "tied", "constant_coordinate"])
def test_randomized_counts_match_saved_version(seed, dimension, relation, distribution):
    rng = np.random.default_rng(seed)
    n, q = rng.integers(9, 258), rng.integers(33, 194)
    sample = rng.normal(size=(n, dimension))
    queries = rng.normal(size=(q, dimension))
    if distribution == "tied":
        sample = np.round(sample)
        queries = np.round(queries)
    elif distribution == "constant_coordinate":
        coordinate = seed % dimension
        sample[:, coordinate] = 0
        queries[:q // 2, coordinate] = 0

    anchors = sample[rng.choice(n, size=min(n, 16), replace=False)]
    queries = np.concatenate((
        queries, anchors,
        np.nextafter(anchors, -np.inf), np.nextafter(anchors, np.inf),
        sample.min(axis=0, keepdims=True) - 1,
        sample.max(axis=0, keepdims=True) + 1,
    ))
    # Vary sample and query order, and exercise reversed, noncontiguous views.
    sample = sample[rng.permutation(n)][::-1, ::-1]
    queries = queries[rng.permutation(len(queries))][::-1, ::-1]
    before_sample, before_queries = sample.copy(), queries.copy()
    kwargs = dict(relation=relation, block_size=int(rng.choice([1, 7, 32, 2048])))

    result = _orthant_counts(sample, queries, **kwargs)

    np.testing.assert_array_equal(result, old_counts(sample, queries, **kwargs))
    assert result.dtype == np.int64
    assert result.shape == (len(queries),)
    np.testing.assert_array_equal(sample, before_sample)
    np.testing.assert_array_equal(queries, before_queries)


@pytest.mark.parametrize("shape", [(3,), (0, 3), (9, 0, 3), (17, 11, 3)])
@pytest.mark.parametrize("block_size", [1, 7, 2048])
def test_query_shapes_and_noncontiguous_views(shape, block_size):
    rng = np.random.default_rng(12)
    sample = rng.normal(size=(66, 6))[::2, ::2]
    queries = rng.normal(size=(*shape[:-1], 6))[..., ::2]
    kwargs = dict(relation="le", block_size=block_size)
    result = _orthant_counts(sample, queries, **kwargs)

    assert result.shape == shape[:-1]
    np.testing.assert_array_equal(result, old_counts(sample, queries, **kwargs))


@pytest.mark.parametrize("dtype", [np.int64, np.uint64, np.float32, np.float64])
@pytest.mark.parametrize("relation", ["le", "ge"])
def test_numeric_extremes_and_duplicates(dtype, relation):
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        values = np.array([limits.min, 0, 1, limits.max], dtype=dtype)
    else:
        limits = np.finfo(dtype)
        values = np.array([-limits.max, -0.0, 0.0, limits.max], dtype=dtype)
    sample = np.array([[x, y] for x in values for y in values], dtype=dtype)
    queries = np.repeat(sample, 3, axis=0)
    kwargs = dict(relation=relation, block_size=11)
    np.testing.assert_array_equal(
        _orthant_counts(sample, queries, **kwargs),
        old_counts(sample, queries, **kwargs),
    )


@pytest.mark.parametrize("relation", ["le", "ge", "legacy-other-value"])
def test_nonfinite_comparisons_preserve_legacy_behavior(relation):
    sample = np.array([[np.nan, 0], [0, np.inf], [-np.inf, 2], [1, 1]])
    queries = np.array([[np.inf, np.inf], [np.nan, 1], [0, 2], [-np.inf, 2]])
    kwargs = dict(relation=relation, block_size=2)
    np.testing.assert_array_equal(
        _orthant_counts(sample, queries, **kwargs),
        old_counts(sample, queries, **kwargs),
    )


@pytest.mark.parametrize("sample_kind", ["signed", "unsigned", "float", "bool"])
@pytest.mark.parametrize("query_kind", ["signed", "unsigned", "float", "bool"])
@pytest.mark.parametrize("dimension", [1, 2, 3])
@pytest.mark.parametrize("relation", ["le", "ge"])
def test_mixed_dtypes_preserve_numpy_comparison_semantics(
    sample_kind, query_kind, dimension, relation
):
    values = {
        "signed": np.array(
            [-2**63, -2**53-1, -1, 0, 1, 2**53, 2**53+1, 2**63-1],
            dtype=np.int64,
        ),
        "unsigned": np.array(
            [0, 1, 2**53, 2**53+1, 2**63, 2**64-1], dtype=np.uint64,
        ),
        "float": np.array([-2.**63, -2.**53, -1, 0, 1, 2.**53, 2.**63, 2.**64]),
        "bool": np.array([True, False]),
    }
    rng = np.random.default_rng(4)
    sample = rng.choice(values[sample_kind], size=(33, dimension))
    queries = rng.choice(values[query_kind], size=(100, dimension))
    kwargs = dict(relation=relation, block_size=31)
    np.testing.assert_array_equal(
        _orthant_counts(sample, queries, **kwargs),
        old_counts(sample, queries, **kwargs),
    )


@pytest.mark.parametrize("block_size", [0, -1, 1.5])
def test_invalid_block_sizes_preserve_exception_type(block_size):
    sample = np.ones((20, 2))
    with pytest.raises((TypeError, ValueError)) as previous:
        old_counts(sample, sample, relation="le", block_size=block_size)
    with pytest.raises(type(previous.value)):
        _orthant_counts(sample, sample, relation="le", block_size=block_size)


@pytest.mark.parametrize("dimension", [2, 3, 5])
@pytest.mark.parametrize("M", [1, 5, 0.9])
@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("permutation", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_entire_coefficient_and_rng_are_identical(
    monkeypatch, dimension, M, conditional, permutation, tied
):
    module = importlib.import_module("src.test_functions.ac_coefficient")
    rng = np.random.default_rng(21)
    if tied:
        y = rng.integers(0, 4, size=(33, dimension)).astype(float)
        z = rng.integers(0, 3, size=(33, 2)).astype(float)
    else:
        y = rng.normal(size=(33, dimension))
        z = rng.normal(size=(33, 2))
    x = np.repeat(np.arange(3), 11) if conditional else None
    results, states = [], []
    for counter in (old_counts, _orthant_counts):
        monkeypatch.setattr(module, "_orthant_counts", counter)
        coefficient_rng = np.random.default_rng(22)
        results.append(module.ac_coefficient(
            y, z, X=x, M=M, permutation=permutation, rng=coefficient_rng,
        ))
        states.append(coefficient_rng.bit_generator.state)
    assert results[0] == results[1]
    assert states[0] == states[1]


def _solver(matrix, k, rng=None):
    return np.asarray(matrix)[:, :k].copy(), np.ones(k)


def _run_method(counter, monkeypatch, data, **kwargs):
    module = importlib.import_module("src.test_functions.ac_coefficient")
    monkeypatch.setattr(module, "_orthant_counts", counter)
    method = MultivariateACTest(
        M=5, npermutations=3, solver=_solver, k=2,
        rng=np.random.default_rng(41), **kwargs,
    )
    method.fit(data)
    return method


def _assert_identical_methods(old, new):
    for name in ("test_stat_estimate", "pvalue", "reject_null"):
        assert getattr(old, name) == getattr(new, name)
    for name in ("permutation_indices", "permutation_distribution",
                 "observed_statistics", "permuted_statistics"):
        np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
    if old.adaptive_m:
        for name in ("adaptive_m_values", "adaptive_s_statistics",
                     "adaptive_m_means", "adaptive_m_stds", "adaptive_z_statistics"):
            np.testing.assert_array_equal(getattr(old, name), getattr(new, name))
    assert old.rng.bit_generator.state == new.rng.bit_generator.state


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("permutation", [False, True])
@pytest.mark.parametrize("permutation_type", ["covariate", "latent", "observed"])
def test_complete_permutation_tests_are_identical(
    monkeypatch, adaptive, conditional, permutation, permutation_type
):
    rng = np.random.default_rng(40)
    data = {
        "Y": rng.integers(0, 5, size=(24, 3)).astype(float),
        "Z": rng.normal(size=(24, 2)),
        "A": rng.normal(size=(24, 24)),
    }
    if conditional:
        data["X"] = np.repeat(np.arange(3), 8)
    kwargs = dict(
        adaptive_m=adaptive, use_permutation_coeff=permutation,
        permutation_type=permutation_type,
        use_true_latent=permutation_type != "observed",
    )
    old = _run_method(old_counts, monkeypatch, data, **kwargs)
    new = _run_method(_orthant_counts, monkeypatch, data, **kwargs)
    _assert_identical_methods(old, new)


@pytest.mark.parametrize("adaptive", [False, True])
def test_parallel_test_matches_saved_serial_version(monkeypatch, adaptive):
    rng = np.random.default_rng(42)
    data = {"Y": rng.normal(size=(24, 2)), "Z": rng.normal(size=(24, 2))}
    kwargs = dict(adaptive_m=adaptive, use_permutation_coeff=True, use_true_latent=True)
    old = _run_method(old_counts, monkeypatch, data, **kwargs)
    new = _run_method(_orthant_counts, monkeypatch, data, n_jobs=2, **kwargs)
    _assert_identical_methods(old, new)
