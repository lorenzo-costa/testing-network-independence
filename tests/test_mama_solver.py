import importlib

import numpy as np
import pytest

from src.solvers.MaMa_uuuuu import (
    compute_theta,
    pgd_fit,
    pgd_fit_wrapper,
    project_Z,
    svt_init,
)


def binary_adjacency(seed=100, n=6):
    rng = np.random.default_rng(seed)
    matrix = (rng.random((n, n)) < 0.4).astype(float)
    matrix = np.triu(matrix, k=1)
    return matrix + matrix.T


def test_module_imports_mama_solver_components():
    module = importlib.import_module("src.solvers.MaMa_uuuuu")

    assert module.compute_theta is compute_theta
    assert module.project_Z is project_Z
    assert module.svt_init is svt_init
    assert module.pgd_fit is pgd_fit
    assert module.pgd_fit_wrapper is pgd_fit_wrapper


def test_compute_theta_combines_latent_node_and_covariate_effects():
    latent = np.array([[1.0], [2.0]])
    alpha = np.array([0.5, -0.5])
    covariate = np.array([[0.0, 1.0], [1.0, 0.0]])

    theta = compute_theta(latent, alpha=alpha, beta=2.0, X=covariate)

    expected = latent @ latent.T + np.add.outer(alpha, alpha) + 2.0 * covariate
    np.testing.assert_allclose(theta, expected)


def test_project_z_centers_columns_and_applies_radius_bound():
    latent = np.array([[3.0, 0.0], [0.0, 3.0], [-3.0, -3.0]])

    projected = project_Z(latent, M=1.0)

    assert np.max(np.linalg.norm(projected, axis=1)) <= 1.0 + 1e-12
    assert np.isfinite(projected).all()


def test_svt_initialization_runs_by_itself():
    alpha, latent, beta = svt_init(
        binary_adjacency(),
        k=2,
        tau=0.01,
        M1=1.0,
    )

    assert alpha.shape == (6,)
    assert latent.shape == (6, 2)
    assert np.isscalar(beta)
    assert np.isfinite(alpha).all()
    assert np.isfinite(latent).all()
    assert np.isfinite(beta)


def test_projected_gradient_solver_runs_short_numpy_optimization():
    latent, alpha, beta, history = pgd_fit(
        binary_adjacency(),
        k=2,
        num_iters=2,
        init="random",
        backend="numpy",
        return_history=True,
        rng=np.random.default_rng(14),
    )

    assert latent.shape == (6, 2)
    assert alpha.shape == (6,)
    assert np.isscalar(beta)
    assert len(history) == 3
    assert np.isfinite(latent).all()
    assert np.isfinite(alpha).all()
    np.testing.assert_allclose(latent.mean(axis=0), 0.0, atol=1e-12)


def test_projected_gradient_wrapper_runs_by_itself():
    latent, beta = pgd_fit_wrapper(
        binary_adjacency(),
        k=2,
        num_iters=2,
        init="random",
        backend="numpy",
        rng=np.random.default_rng(15),
    )

    assert latent.shape == (6, 2)
    assert np.isscalar(beta)
    assert np.isfinite(latent).all()
    assert np.isfinite(beta)


def test_projected_gradient_solver_rejects_unknown_backend():
    with pytest.raises(KeyError, match="invalid"):
        pgd_fit(
            binary_adjacency(),
            k=2,
            num_iters=1,
            init="random",
            backend="invalid",
            rng=np.random.default_rng(16),
        )

