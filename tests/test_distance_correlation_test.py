import importlib

import numpy as np
import pytest

from src.methods.distance_correlation_test import DistanceCorrelationTest


def global_network_data(seed=40, n=14):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(n, 2))
    X = [rng.normal(size=(n, 1)), rng.normal(size=(n, 2))]
    return {
        "Ay": Y @ Y.T,
        "Ax": [block @ block.T for block in X],
        "Y": Y,
        "X": X,
    }


def test_module_imports_distance_correlation_test():
    module = importlib.import_module("src.methods.distance_correlation_test")
    assert module.DistanceCorrelationTest is DistanceCorrelationTest


def test_distance_matrix_helper_runs_by_itself():
    method = DistanceCorrelationTest(use_true_latent=True)
    values = np.array([[0.0, 0.0], [3.0, 4.0]])

    distances = method.compute_distance_matrix(values)

    np.testing.assert_allclose(distances, [[0.0, 5.0], [5.0, 0.0]])


def test_distance_correlation_global_test_runs():
    method = DistanceCorrelationTest(
        use_true_latent=True,
        test_method="dcorr",
        npermutations=3,
        rng=np.random.default_rng(5),
    )

    method.fit(global_network_data())
    result = method.get_estimated()

    assert method.Xhat.shape == (14, 3)
    assert method.get_name() == "DistanceCorrelation_latent"
    assert np.isfinite(result["test_stat"])
    assert 0.0 <= result["p-value"] <= 1.0
    assert isinstance(result["reject_null"], bool)


def test_distance_correlation_rejects_unknown_method():
    with pytest.raises(ValueError, match="test_method must be"):
        DistanceCorrelationTest(
            test_method="invalid",
            use_true_latent=True,
            npermutations=1,
        )
