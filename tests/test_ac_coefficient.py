import numpy as np
import pytest

from src.test_functions._ac_helpers import _validate_m
from src.test_functions.ac_coefficient import ac_coefficient, asymptotic_unconditional_ac


def test_integer_m_is_used_directly():
    assert _validate_m(7, 100) == 7
    assert _validate_m(np.int64(8), 100) == 8


def test_float_m_is_an_exponent_rounded_to_nearest_integer():
    assert _validate_m(0.5, 10) == 3
    assert _validate_m(np.float64(0.4), 20) == 3


def test_public_coefficient_resolves_float_m_as_an_exponent():
    y = np.arange(10, dtype=float)
    z = np.square(y)

    exponent_result = ac_coefficient(y, z, M=0.5, rng=0)
    integer_result = ac_coefficient(y, z, M=3, rng=0)

    assert exponent_result == pytest.approx(integer_result)


@pytest.mark.parametrize("M", [1, 3])
def test_scalar_asymptotic_statistic_matches_rank_coefficient_with_ties(M):
    y = np.array([0, 1, 1, 3, 2, 3, 4, 2, 0, 4], dtype=float)
    z = np.arange(len(y), dtype=float)[:, None]

    coefficient, variance = asymptotic_unconditional_ac(
        y, z, M=M, rng=np.random.default_rng(3)
    )
    expected = ac_coefficient(y, z, M=M, rng=np.random.default_rng(3))

    assert coefficient == pytest.approx(expected)
    assert np.isfinite(variance) and variance > 0


@pytest.mark.parametrize("M", ["sqrt", "half", "third", "quarter", "log"])
def test_string_m_aliases_are_rejected(M):
    with pytest.raises(ValueError, match="positive integer or a finite float"):
        _validate_m(M, 100)


@pytest.mark.parametrize("M", [1.0, np.inf, np.nan])
def test_invalid_float_m_is_rejected(M):
    with pytest.raises(ValueError, match="finite and smaller than 1"):
        _validate_m(M, 100)


def test_multivariate_permutation_is_used_to_construct_y_tilde(monkeypatch):
    import importlib

    ac_module = importlib.import_module("src.test_functions.ac_coefficient")
    y = np.array(
        [
            [10.0, 11.0],
            [20.0, 21.0],
            [30.0, 31.0],
            [40.0, 41.0],
        ]
    )
    z = np.arange(4, dtype=float).reshape(-1, 1)
    permutations = np.array(
        [
            [3, 2, 1, 0],
            [1, 0, 3, 2],
        ]
    )
    captured = {}

    def fixed_permutations(n, d_y, rng=None):
        assert (n, d_y) == y.shape
        return permutations

    def recording_coefficient(
        passed_y,
        y_tilde,
        m_idx,
        n_idx,
        *,
        block_size,
    ):
        captured["y"] = passed_y.copy()
        captured["y_tilde"] = y_tilde.copy()
        return 0.375

    monkeypatch.setattr(ac_module, "_make_permutations", fixed_permutations)
    monkeypatch.setattr(
        ac_module,
        "_multivariate_coefficient",
        recording_coefficient,
    )

    result = ac_coefficient(y, z, M=1, permutation=True, rng=0)

    expected_y_tilde = np.take_along_axis(y, permutations.T, axis=0)
    assert result == pytest.approx(0.375)
    np.testing.assert_array_equal(captured["y"], y)
    np.testing.assert_array_equal(captured["y_tilde"], expected_y_tilde)
    assert not np.array_equal(captured["y_tilde"], y)


@pytest.mark.parametrize("neighbors", [
    np.array([[1], [0], [1], [2], [3], [4]]),
    np.array([[1, 2], [0, 2], [1, 3], [2, 4], [3, 5], [4, 3]]),
])
def test_asymptotic_variance_matches_direct_kernel_and_graph_counts(monkeypatch, neighbors):
    import importlib

    module = importlib.import_module("src.test_functions.ac_coefficient")
    y = np.array([[0, 1], [1, 0], [1, 2], [2, 1], [2, 2], [3, 0]], dtype=float)
    z = np.arange(6, dtype=float)[:, None]
    permutations = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
    K = neighbors.shape[1]
    monkeypatch.setattr(module, "_make_permutations", lambda *args, **kwargs: permutations)
    monkeypatch.setattr(module, "_neighbor_maps", lambda *args, **kwargs: (neighbors, None))

    coefficient, variance = asymptotic_unconditional_ac(y, z, M=K)
    y_tilde = np.take_along_axis(y, permutations.T, axis=0)
    n = len(y)

    def h(i, j):
        return np.mean(np.all(y_tilde <= np.minimum(y[i], y[j]), axis=1))

    adjacent = np.array([h(i, i + 1) for i in range(n - 1)])
    skip_one = np.array([h(i, i + 2) for i in range(n - 2)])
    theta = adjacent.mean()
    gamma_2 = max(np.mean(adjacent[:-1] * skip_one) - theta**2, 0)
    gamma_1 = max(
        np.mean(adjacent**2) - 2 * np.mean(adjacent[:-1] * skip_one) + theta**2,
        0,
    )
    reciprocal = sum(i in neighbors[j] for i, row in enumerate(neighbors) for j in row)
    indegrees = np.bincount(neighbors.ravel(), minlength=n)
    b_n = 1 / K + reciprocal / (n * K**2)
    q_n = np.mean((indegrees / K - 1) ** 2)
    g_n = np.array([np.mean(np.all(y >= row, axis=1)) for row in y_tilde])
    s_hat = np.mean(g_n * (1 - g_n))
    expected_variance = (gamma_1 * b_n + gamma_2 * q_n) / s_hat**2

    assert coefficient == pytest.approx(ac_coefficient(y, z, M=K, permutation=True))
    assert variance == pytest.approx(expected_variance)
