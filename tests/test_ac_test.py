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


@pytest.mark.parametrize("method_type", [EstimateAC, MultivariateACTest])
def test_ac_method_passes_x_as_conditioning_variable(monkeypatch, method_type):
    recorded = []

    def recording_ac(*, Y, Z, X=None, **kwargs):
        recorded.append(X)
        return 0.25

    monkeypatch.setattr("src.methods.ac_test.ac_coefficient", recording_ac)
    data = latent_data()
    data["X"] = np.repeat([0, 1], 7).reshape(-1, 1)
    kwargs = {"npermutations": 2} if method_type is MultivariateACTest else {}
    method = method_type(use_true_latent=True, M=1, **kwargs)
    method.fit(data)

    assert recorded[0] is method.X
    np.testing.assert_array_equal(recorded[0], data["X"])
    if method_type is MultivariateACTest:
        assert len(recorded) == 3
        assert all(conditioning is method.X for conditioning in recorded)


@pytest.mark.parametrize("permutation_type", ["covariate", "latent", "observed"])
def test_ac_permutations_are_restricted_within_x_strata(
    monkeypatch, permutation_type
):
    monkeypatch.setattr(
        "src.methods.ac_test.ac_coefficient", lambda **kwargs: 0.25
    )
    rng = np.random.default_rng(20)
    data = {
        "A": rng.normal(size=(12, 12)),
        "Z": rng.normal(size=(12, 2)),
        "Y": rng.normal(size=(12, 2)),
        "X": np.repeat([0, 1, 2], 4).reshape(-1, 1),
    }
    method = MultivariateACTest(
        k=2,
        solver=dummy_solver,
        use_true_latent=permutation_type != "observed",
        permutation_type=permutation_type,
        M=1,
        npermutations=5,
        rng=np.random.default_rng(21),
    )
    method.fit(data)

    assert len(method.permutation_indices) == 5
    for permutation in method.permutation_indices:
        np.testing.assert_array_equal(data["X"][permutation], data["X"])


def test_ac_permutations_remain_global_when_x_is_none(monkeypatch):
    monkeypatch.setattr(
        "src.methods.ac_test.ac_coefficient", lambda **kwargs: 0.25
    )
    seed = 22
    data = latent_data()
    method = MultivariateACTest(
        use_true_latent=True,
        M=1,
        npermutations=1,
        rng=np.random.default_rng(seed),
    )
    method.fit(data)

    expected = np.random.default_rng(seed).permutation(data["Y"].shape[0])
    np.testing.assert_array_equal(method.permutation_indices[0], expected)
