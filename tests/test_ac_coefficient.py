import numpy as np
import pytest

from src.test_functions._ac_helpers import _validate_m
from src.test_functions.ac_coefficient import ac_coefficient


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
