import importlib

import numpy as np
import pytest

from src.methods.cca_test import CanonicalCorrelationTest
from src.test_functions.rv_cca_coefficients import first_cca_component


def dummy_solver(matrix, k, rng=None):
    values = np.asarray(matrix, dtype=float)
    return values[:, :k].copy(), np.ones(k)


def test_module_imports_canonical_correlation_test():
    module = importlib.import_module("src.methods.cca_test")

    assert module.CanonicalCorrelationTest is CanonicalCorrelationTest


def test_canonical_correlation_test_runs_permutations_by_itself():
    rng = np.random.default_rng(20)
    data = {
        "Y": rng.normal(size=(14, 2)),
        "X": [rng.normal(size=(14, 2)), rng.normal(size=(14, 2))],
    }
    method = CanonicalCorrelationTest(
        solver=dummy_solver,
        use_true_latent=True,
        npermutations=3,
        rng=np.random.default_rng(3),
    )

    method.fit(data)
    result = method.get_estimated()

    assert method.get_name() == "CCA_PermutationTest_latent"
    assert len(method.permutation_distribution) == 3
    assert 0.0 <= result["test_stat"] <= 1.0
    assert 0.0 <= result["p-value"] <= 1.0
    assert method.effective_gamma == pytest.approx(np.sqrt(14))
    expected = first_cca_component(
        data["Y"],
        np.concatenate(data["X"], axis=1),
        gamma=method.effective_gamma,
    )
    assert method.test_stat_estimate == pytest.approx(expected)


def test_canonical_correlation_test_requires_solver():
    with pytest.raises(ValueError, match="Solver must be provided"):
        CanonicalCorrelationTest()


def test_canonical_correlation_forwards_parallel_options():
    method = CanonicalCorrelationTest(
        use_true_latent=True,
        n_jobs=2,
        batch_size=3,
        verbose=True,
    )

    assert (method.n_jobs, method.batch_size, method.verbose) == (2, 3, True)


def test_regularized_cca_matches_direct_covariance_calculation():
    rng = np.random.default_rng(91)
    y = rng.normal(size=(30, 3))
    x = rng.normal(size=(30, 8))
    gamma = 0.7

    y_centered = y - y.mean(axis=0)
    x_centered = x - x.mean(axis=0)
    covariance_y = y_centered.T @ y_centered / len(y)
    covariance_x = x_centered.T @ x_centered / len(x)
    cross_covariance = y_centered.T @ x_centered / len(y)

    def inverse_square_root(matrix):
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        return (eigenvectors * (1 / np.sqrt(eigenvalues))) @ eigenvectors.T

    whitened = (
        inverse_square_root(covariance_y)
        @ cross_covariance
        @ inverse_square_root(covariance_x + gamma * np.eye(x.shape[1]))
    )
    expected = np.linalg.svd(whitened, compute_uv=False)[0]

    assert first_cca_component(y, x, gamma=gamma) == pytest.approx(expected)


def test_default_gamma_equals_explicit_sqrt_n():
    rng = np.random.default_rng(92)
    y = rng.normal(size=(20, 2))
    x = rng.normal(size=(20, 30))

    assert first_cca_component(y, x) == pytest.approx(
        first_cca_component(y, x, gamma=np.sqrt(len(y)))
    )


def test_canonical_correlation_uses_explicit_gamma():
    rng = np.random.default_rng(93)
    data = {
        "Y": rng.normal(size=(12, 2)),
        "X": [rng.normal(size=(12, 2))],
    }
    method = CanonicalCorrelationTest(
        use_true_latent=True,
        gamma=0.25,
        npermutations=2,
        rng=np.random.default_rng(94),
    )

    method.fit(data)

    assert method.gamma == 0.25
    assert method.effective_gamma == 0.25


@pytest.mark.parametrize("gamma", [-1, np.nan, np.inf, True, "ridge"])
def test_canonical_correlation_rejects_invalid_gamma(gamma):
    with pytest.raises(ValueError, match="gamma"):
        CanonicalCorrelationTest(use_true_latent=True, gamma=gamma)
