import importlib

import numpy as np
import pytest

from src.methods.fit_independent import FitIndependent


def dummy_solver(matrix, k, rng=None):
    values = np.asarray(matrix, dtype=float)
    return values[:, :k].copy(), np.arange(k, dtype=float)


def test_module_imports_fit_independent():
    module = importlib.import_module("src.methods.fit_independent")

    assert module.FitIndependent is FitIndependent


def test_fit_independent_runs_solver_for_each_network():
    matrix_a = np.arange(36, dtype=float).reshape(6, 6)
    matrix_b = matrix_a + 100
    method = FitIndependent(
        solver=dummy_solver,
        k=2,
        rng=np.random.default_rng(6),
    )

    method.fit({"A": matrix_a, "B": matrix_b})
    result = method.get_estimated()

    np.testing.assert_array_equal(method.Zhat, matrix_a[:, :2])
    np.testing.assert_array_equal(method.Xhat, matrix_b[:, :2])
    assert method.get_name() == "FitIndependent"
    assert result["p-value"] is None
    assert result["reject_null"] is None
    assert result["test_stat"] is None


def test_fit_independent_requires_solver():
    with pytest.raises(ValueError, match="Solver must be provided"):
        FitIndependent(k=2)

