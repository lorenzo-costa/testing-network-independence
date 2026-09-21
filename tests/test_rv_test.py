import importlib

import numpy as np
import pytest
from scipy.stats import chi2

from src.methods.rv_test import EstimateRV, RVTest
from src.test_functions.rv_cca_coefficients import (
    _rv_coefficient_from_x_cache,
    _rv_squared_gram_norm,
    _rv_x_cache,
    rv_coefficient,
)


def dummy_solver(matrix, k, rng=None):
    values = np.asarray(matrix, dtype=float)
    return values[:, :k].copy(), np.ones(k)


def multiple_latent_data(seed=70):
    rng = np.random.default_rng(seed)
    return {
        "Y": rng.normal(size=(14, 2)),
        "X": [rng.normal(size=(14, 2)), rng.normal(size=(14, 2))],
    }


def uncached_rv_coefficient(y, x):
    return rv_coefficient(y, x)


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

    data = multiple_latent_data()
    method.fit(data)
    result = method.get_estimated()

    assert method.get_name() == "EstimateRV"
    assert result["test_stat"] == pytest.approx(
        method.test_function(data["Y"], np.concatenate(data["X"], axis=1))
    )
    assert result["estimated_latent"]["Y"].shape == (14, 2)
    assert result["estimated_latent"]["X"].shape == (14, 4)
    assert np.isfinite(result["test_stat"])
    assert result["p-value"] is None
    assert result["reject_null"] is None


def test_rv_permutation_test_runs_by_itself():
    method = RVTest(
        solver=dummy_solver,
        use_true_latent=True,
        approximation="permutation",
        npermutations=4,
        rng=np.random.default_rng(10),
    )

    method.fit(multiple_latent_data())
    result = method.get_estimated()

    assert method.get_name() == "RV_PermutationTest_latent"
    assert len(method.permutation_distribution) == 4
    assert np.isfinite(result["test_stat"])
    assert 0.0 <= result["p-value"] <= 1.0
    assert isinstance(result["reject_null"], bool)


def test_cached_rv_matches_direct_coefficient():
    rng = np.random.default_rng(71)
    y = rng.normal(size=(25, 3))
    x = rng.normal(size=(25, 40))
    centered_x, x_squared_gram_norm = _rv_x_cache(x)
    centered_y = y - y.mean(axis=0, keepdims=True)

    cached = _rv_coefficient_from_x_cache(
        y,
        centered_x,
        x_squared_gram_norm,
        _rv_squared_gram_norm(centered_y),
    )

    assert cached == pytest.approx(rv_coefficient(y, x))


def test_latent_permutation_cache_preserves_rv_results():
    data = multiple_latent_data(seed=72)
    common = {
        "use_true_latent": True,
        "approximation": "permutation",
        "npermutations": 6,
    }
    cached = RVTest(rng=np.random.default_rng(73), **common)
    uncached = RVTest(
        rng=np.random.default_rng(73),
        test_function=uncached_rv_coefficient,
        **common,
    )

    cached.fit(data)
    uncached.fit(data)

    assert cached._rv_cache_enabled is True
    assert uncached._rv_cache_enabled is False
    assert cached.test_stat_estimate == pytest.approx(uncached.test_stat_estimate)
    np.testing.assert_allclose(
        cached.permutation_distribution,
        uncached.permutation_distribution,
    )
    assert cached.pvalue == uncached.pvalue


def test_adjacency_permutation_cache_preserves_rv_results():
    rng = np.random.default_rng(74)
    data = {
        "A_Y": rng.normal(size=(12, 12)),
        "A_X": [rng.normal(size=(12, 12)), rng.normal(size=(12, 12))],
    }
    common = {
        "solver": dummy_solver,
        "d_y": 2,
        "d_x": 2,
        "approximation": "permutation",
        "permutation_type": "adjacency",
        "npermutations": 5,
    }
    cached = RVTest(rng=np.random.default_rng(75), **common)
    uncached = RVTest(
        rng=np.random.default_rng(75),
        test_function=uncached_rv_coefficient,
        **common,
    )

    cached.fit(data)
    uncached.fit(data)

    assert cached._rv_y_squared_gram_norm is None
    assert cached.test_stat_estimate == pytest.approx(uncached.test_stat_estimate)
    np.testing.assert_allclose(
        cached.permutation_distribution,
        uncached.permutation_distribution,
    )
    assert cached.pvalue == uncached.pvalue


@pytest.mark.parametrize("permutation_type", ["latent", "adjacency"])
def test_parallel_cached_rv_matches_serial(permutation_type):
    rng = np.random.default_rng(76)
    data = {
        "A_Y": rng.normal(size=(12, 12)),
        "A_X": [rng.normal(size=(12, 12)), rng.normal(size=(12, 12))],
    }
    methods = []
    for n_jobs in (1, 2):
        method = RVTest(
            solver=dummy_solver,
            d_y=2,
            d_x=2,
            approximation="permutation",
            permutation_type=permutation_type,
            npermutations=6,
            n_jobs=n_jobs,
            rng=np.random.default_rng(77),
        )
        method.fit(data)
        methods.append(method)

    serial, parallel = methods
    assert serial.test_stat_estimate == pytest.approx(parallel.test_stat_estimate)
    np.testing.assert_allclose(
        serial.permutation_distribution,
        parallel.permutation_distribution,
    )
    assert serial.pvalue == parallel.pvalue


def test_asymptotic_rv_null_calibration_with_identity_covariance():
    """The default RV statistic should follow its Gaussian identity-null limit."""
    rng = np.random.default_rng(20260726)
    n = 300
    k = 3
    repetitions = 2_000
    alpha = 0.05

    method = RVTest(
        use_true_latent=True,
        approximation="asymptotic",
        alpha=alpha,
    )
    critical_value = chi2.ppf(1.0 - alpha, df=k) / np.sqrt(k)

    rejection_count = 0
    for _ in range(repetitions):
        Z = rng.normal(size=(n, k))
        Y = rng.normal(size=(n, 1))
        rejection_count += n * method.test_function(Z, Y) > critical_value

    rejection_rate = rejection_count / repetitions
    monte_carlo_se = np.sqrt(alpha * (1.0 - alpha) / repetitions)

    assert abs(rejection_rate - alpha) <= 4.0 * monte_carlo_se, (
        f"Gaussian identity-null rejection rate {rejection_rate:.4f} is not "
        f"calibrated at alpha={alpha:.2f}"
    )


def test_asymptotic_rv_uses_centered_cross_product_covariance_and_rv_denominator(
    monkeypatch,
):
    Y = np.array(
        [
            [1.0, -1.0],
            [2.0, 0.0],
            [-1.0, 2.0],
            [0.0, 1.0],
            [3.0, -2.0],
        ]
    )
    X_blocks = [
        np.array([[0.0], [1.0], [3.0], [-1.0], [2.0]]),
        np.array([[2.0], [-2.0], [1.0], [4.0], [0.0]]),
    ]
    captured = {}

    def capture_imhof(q, weights):
        captured["q"] = q
        captured["weights"] = weights.copy()
        return {"Qq": 0.25}

    monkeypatch.setattr("src.methods.rv_test.imhof", capture_imhof)
    method = RVTest(use_true_latent=True, approximation="asymptotic")
    method.fit({"Y": Y, "X": X_blocks})

    centered_y = Y - Y.mean(axis=0, keepdims=True)
    X = np.concatenate(X_blocks, axis=1)
    centered_x = X - X.mean(axis=0, keepdims=True)
    W = np.einsum("ni,nj->nij", centered_y, centered_x).reshape(len(Y), -1)
    centered_W = W - W.mean(axis=0, keepdims=True)
    Omega = centered_W.T @ centered_W / len(Y)
    eigenvalues = np.linalg.eigvalsh(Omega)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-10 * eigenvalues[0]]
    S_Y = centered_y.T @ centered_y / len(Y)
    S_X = centered_x.T @ centered_x / len(Y)
    denominator = np.sqrt(np.trace(S_Y @ S_Y) * np.trace(S_X @ S_X))

    np.testing.assert_allclose(captured["weights"], eigenvalues / denominator)
    assert captured["q"] == pytest.approx(
        len(Y) * rv_coefficient(centered_y, centered_x)
    )
    assert method.pvalue == 0.25


def test_asymptotic_rv_rejects_degenerate_sample_covariance():
    method = RVTest(use_true_latent=True, approximation="asymptotic")

    with pytest.raises(ValueError, match="asymptotic RV.*Frobenius norms"):
        method.fit(
            {
                "Y": np.ones((5, 1)),
                "X": [np.arange(5.0).reshape(-1, 1)],
            }
        )


def test_rv_test_rejects_unknown_approximation():
    method = RVTest(
        solver=dummy_solver,
        use_true_latent=True,
        approximation="invalid",
        npermutations=1,
    )

    with pytest.raises(ValueError, match="Invalid approximation method"):
        method.fit(multiple_latent_data())


def test_rv_test_forwards_parallel_options():
    method = RVTest(
        use_true_latent=True,
        n_jobs=2,
        batch_size=3,
        verbose=True,
    )

    assert (method.n_jobs, method.batch_size, method.verbose) == (2, 3, True)
