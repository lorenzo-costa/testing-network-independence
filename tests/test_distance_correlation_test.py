import numpy as np
import pytest

from src.load_config import _resolve_method
from src.methods import DistanceCorrelationTest


def dummy_solver(A, k, rng=None):
    return np.asarray(A)[:, :k].copy(), np.ones(k)


def latent_data(seed=40):
    rng = np.random.default_rng(seed)
    return {
        "Y": rng.normal(size=(14, 2)),
        "Z": rng.normal(size=(14, 2)),
    }


@pytest.mark.parametrize("test_method", ["dcorr", "mgc"])
def test_distance_correlation_permutation_test(test_method):
    method = DistanceCorrelationTest(
        test_method=test_method,
        use_true_latent=True,
        npermutations=3,
        rng=np.random.default_rng(5),
    )

    method.fit(latent_data())
    result = method.get_estimated()

    assert method.get_name() == "DistanceCorrelation_covariate"
    assert np.isfinite(result["test_stat"])
    assert len(method.null_distribution) == 3
    assert 0.0 <= result["p-value"] <= 1.0
    assert isinstance(result["reject_null"], bool)


def test_distance_correlation_uses_estimated_latent_positions():
    data = latent_data()
    data["A"] = data["Z"] @ data["Z"].T
    method = DistanceCorrelationTest(
        test_method="dcorr",
        solver=dummy_solver,
        k=2,
        npermutations=3,
        rng=np.random.default_rng(6),
    )

    method.fit(data)

    np.testing.assert_array_equal(method.Zhat, data["A"][:, :2])
    assert np.isfinite(method.test_stat_estimate)


def test_distance_correlation_is_registered_for_yaml_methods():
    resolved = _resolve_method(
        {"name": "DistanceCorrelationTest", "kwargs": {"test_method": "dcorr"}}
    )
    assert resolved.func is DistanceCorrelationTest
    assert resolved.keywords == {"test_method": "dcorr"}
    assert _resolve_method({"name": "DiffusionCorrelation"}) is DistanceCorrelationTest


def test_unknown_distance_correlation_statistic_is_rejected():
    with pytest.raises(ValueError, match="test_method must be"):
        DistanceCorrelationTest(test_method="invalid", use_true_latent=True)
