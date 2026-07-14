import importlib

import numpy as np
import pytest

from src.methods.distance_correlation_test import DistanceCorrelationTest


def dummy_solver(matrix, k, rng=None):
    values = np.asarray(matrix, dtype=float)
    return values[:, :k].copy(), np.ones(k)


def latent_data(seed=40):
    rng = np.random.default_rng(seed)
    return {
        "Y": rng.normal(size=(14, 2)),
        "Z": rng.normal(size=(14, 2)),
    }


def test_module_imports_distance_correlation_test():
    module = importlib.import_module("src.methods.distance_correlation_test")

    assert module.DistanceCorrelationTest is DistanceCorrelationTest


def test_distance_matrix_helper_runs_by_itself():
    method = DistanceCorrelationTest(solver=dummy_solver)
    values = np.array([[0.0, 0.0], [3.0, 4.0]])

    distances = method.compute_distance_matrix(values)

    np.testing.assert_allclose(distances, [[0.0, 5.0], [5.0, 0.0]])


def test_distance_correlation_runs_mgc_by_itself():
    method = DistanceCorrelationTest(
        solver=dummy_solver,
        use_true_latent=True,
        npermutations=3,
        rng=np.random.default_rng(5),
    )

    method.fit(latent_data())
    result = method.get_estimated()

    assert method.get_name() == "DistanceCorrelation_covariate"
    assert np.isfinite(result["test_stat"])
    assert 0.0 <= result["p-value"] <= 1.0
    assert isinstance(result["reject_null"], bool)


def test_distance_correlation_rejects_unknown_method():
    with pytest.raises(ValueError, match="test_method must be"):
        DistanceCorrelationTest(
            solver=dummy_solver,
            test_method="invalid",
            use_true_latent=True,
            npermutations=1,
        )
