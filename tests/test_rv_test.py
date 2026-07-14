import importlib

import numpy as np
import pytest

from src.methods.rv_test import EstimateRV, RVTest


def dummy_solver(matrix, k, rng=None):
    values = np.asarray(matrix, dtype=float)
    return values[:, :k].copy(), np.ones(k)


def latent_data(seed=70):
    rng = np.random.default_rng(seed)
    return {
        "X": rng.normal(size=(14, 2)),
        "Z": rng.normal(size=(14, 2)),
    }


def test_module_imports_rv_methods():
    module = importlib.import_module("src.methods.rv_test")

    assert module.EstimateRV is EstimateRV
    assert module.RVTest is RVTest


def test_estimate_rv_runs_with_true_latent_positions():
    method = EstimateRV(
        solver=dummy_solver,
        use_true_latent=True,
        rng=np.random.default_rng(9),
    )

    method.fit(latent_data())
    result = method.get_estimated()

    assert method.get_name() == "EstimateRV"
    assert np.isfinite(result["test_stat"])
    assert result["p-value"] is None
    assert result["reject_null"] is None


def test_rv_permutation_test_runs_by_itself():
    method = RVTest(
        solver=dummy_solver,
        use_true_latent_x=True,
        use_true_latent_z=True,
        approximation="permutation",
        npermutations=4,
        rng=np.random.default_rng(10),
    )

    method.fit(latent_data())
    result = method.get_estimated()

    assert method.get_name() == "RV_PermutationTest_latent"
    assert len(method.permutation_distribution) == 4
    assert np.isfinite(result["test_stat"])
    assert 0.0 <= result["p-value"] <= 1.0
    assert isinstance(result["reject_null"], bool)


def test_rv_test_rejects_unknown_approximation():
    method = RVTest(
        solver=dummy_solver,
        use_true_latent_x=True,
        use_true_latent_z=True,
        approximation="invalid",
        npermutations=1,
    )

    with pytest.raises(ValueError, match="Invalid approximation method"):
        method.fit(latent_data())

