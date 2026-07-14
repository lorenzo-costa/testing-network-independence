import importlib

import numpy as np
import pytest

from src.methods.ac_test import EstimateAC, MultivariateACTest


def dummy_solver(matrix, k, rng=None):
    values = np.asarray(matrix, dtype=float)
    return values[:, :k].copy(), np.ones(k)


def latent_data(seed=10):
    rng = np.random.default_rng(seed)
    return {
        "Y": rng.normal(size=(14, 2)),
        "Z": rng.normal(size=(14, 2)),
    }


def test_module_imports_ac_methods():
    module = importlib.import_module("src.methods.ac_test")

    assert module.EstimateAC is EstimateAC
    assert module.MultivariateACTest is MultivariateACTest


def test_estimate_ac_runs_with_true_latent_positions():
    method = EstimateAC(
        solver=dummy_solver,
        use_true_latent=True,
        M=1,
        rng=np.random.default_rng(1),
    )

    method.fit(latent_data())
    result = method.get_estimated()

    assert method.get_name() == "EstimateAC"
    assert np.isfinite(result["test_stat"])
    assert result["p-value"] is None
    assert result["reject_null"] is None


def test_multivariate_ac_test_runs_permutations_by_itself():
    method = MultivariateACTest(
        solver=dummy_solver,
        use_true_latent=True,
        M=1,
        npermutations=3,
        rng=np.random.default_rng(2),
    )

    method.fit(latent_data())
    result = method.get_estimated()

    assert len(method.permutation_distribution) == 3
    assert np.isfinite(result["test_stat"])
    assert 0.0 <= result["p-value"] <= 1.0
    assert isinstance(result["reject_null"], bool)


def test_multivariate_ac_requires_observed_y():
    method = MultivariateACTest(
        solver=dummy_solver,
        k=2,
        npermutations=1,
    )

    with pytest.raises(ValueError, match="Observed covariates Y must be provided"):
        method.fit({"A": np.eye(4)})
