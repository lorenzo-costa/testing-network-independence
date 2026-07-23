import importlib

import numpy as np
import pytest

from src.test_functions._ac_helpers import _validate_m, _validate_m_bounds
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


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ((2, 5), (2, 5)),
        ([0.2, 0.5], (3, 10)),
        ((0.2, 8), (3, 8)),
    ],
)
def test_aggregate_bounds_accept_independent_integer_or_float_values(
    bounds, expected
):
    assert _validate_m_bounds(bounds, 100) == expected


def test_aggregate_bounds_must_be_ordered_as_numbers():
    with pytest.raises(ValueError, match="greater than the lower"):
        _validate_m_bounds((2, 0.9), 100)


def test_aggregate_bounds_must_remain_ordered_after_exponent_resolution():
    with pytest.raises(ValueError, match="after resolving float exponents"):
        _validate_m_bounds((0.2, 0.21), 100)


def test_aggregate_bounds_are_inclusive(monkeypatch):
    n = 16
    y = np.arange(n, dtype=float)
    z = np.arange(n, dtype=float)

    def coefficients_over_m(y, m_idx, n_idx):
        return np.arange(1, m_idx.shape[1] + 1, dtype=float)

    ac_module = importlib.import_module("src.test_functions.ac_coefficient")
    monkeypatch.setattr(ac_module, "_scalar_coefficients_over_m", coefficients_over_m)

    # round(16**0.2) = 2, so scalar M=4 aggregates M=2, 3, 4.
    assert ac_coefficient(y, z, M=4, aggregate="avg", rng=0) == 3.0
    # Explicit mixed bounds include both round(16**0.2)=2 and M=4.
    assert ac_coefficient(y, z, M=(0.2, 4), aggregate="max", rng=0) == 4.0
