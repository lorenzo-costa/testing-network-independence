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


def test_fit_independent_runs_solver_for_all_networks():
    matrices = [
        np.arange(36, dtype=float).reshape(6, 6) + offset for offset in (0, 100, 200)
    ]
    method = FitIndependent(
        solver=dummy_solver,
        d_y=2,
        d_x=1,
        rng=np.random.default_rng(6),
    )

    method.fit({"A_Y": matrices[0], "A_X": matrices[1:]})
    result = method.get_estimated()

    np.testing.assert_array_equal(method.Yhat, matrices[0][:, :2])
    np.testing.assert_array_equal(method.Xhat_blocks[0], matrices[1][:, :1])
    np.testing.assert_array_equal(method.Xhat_blocks[1], matrices[2][:, :1])
    np.testing.assert_array_equal(
        method.Xhat, np.concatenate(method.Xhat_blocks, axis=1)
    )
    assert result["estimated_latent"]["Y"] is method.Yhat
    assert result["estimated_latent"]["X"] is method.Xhat
    assert result["estimated_latent"]["X_blocks"] is method.Xhat_blocks
    assert result["true_latent"] == {"Y": None, "X": None, "X_blocks": None}
    assert method.get_name() == "FitIndependent"
    assert result["p-value"] is None
    assert result["reject_null"] is None
    assert result["test_stat"] is None


def test_fit_independent_requires_solver():
    with pytest.raises(ValueError, match="Solver must be provided"):
        FitIndependent(d_y=2, d_x=2)
