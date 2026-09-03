import importlib

import numpy as np
import pytest
from scipy.stats import chi2

from src.methods.rv_test import EstimateRV, RVTest


def dummy_solver(matrix, k, rng=None):
    values = np.asarray(matrix, dtype=float)
    return values[:, :k].copy(), np.ones(k)


def latent_data(seed=70):
    rng = np.random.default_rng(seed)
    return {
        "Y": rng.normal(size=(14, 2)),
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
        use_true_latent=True,
        approximation="permutation",
        npermutations=4,
        rng=np.random.default_rng(10),
    )

    method.fit(latent_data())
    result = method.get_estimated()

    assert method.get_name() == "RV_PermutationTest_covariate"
    assert len(method.permutation_distribution) == 4
    assert np.isfinite(result["test_stat"])
    assert 0.0 <= result["p-value"] <= 1.0
    assert isinstance(result["reject_null"], bool)


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


def test_rv_test_rejects_unknown_approximation():
    method = RVTest(
        solver=dummy_solver,
        use_true_latent=True,
        approximation="invalid",
        npermutations=1,
    )

    with pytest.raises(ValueError, match="Invalid approximation method"):
        method.fit(latent_data())


def test_rv_test_forwards_parallel_options():
    method = RVTest(
        use_true_latent=True,
        n_jobs=2,
        batch_size=3,
        verbose=True,
    )

    assert (method.n_jobs, method.batch_size, method.verbose) == (2, 3, True)
