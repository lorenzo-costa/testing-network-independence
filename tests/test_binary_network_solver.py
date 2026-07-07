import importlib

import numpy as np

from src.solvers.binary_network import (
    MLE_logistic,
    logistic_grad,
    logistic_grad_fixed_mu,
    solve_logistic_scipy,
)


def logistic_data():
    features = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    target = np.array([0.0, 0.0, 1.0, 1.0])
    return features, target


def binary_adjacency(seed=90, n=8):
    rng = np.random.default_rng(seed)
    matrix = (rng.random((n, n)) < 0.35).astype(float)
    matrix = np.triu(matrix, k=1)
    return matrix + matrix.T


def test_module_imports_binary_network_solvers():
    module = importlib.import_module("src.solvers.binary_network")

    assert module.logistic_grad is logistic_grad
    assert module.logistic_grad_fixed_mu is logistic_grad_fixed_mu
    assert module.solve_logistic_scipy is solve_logistic_scipy
    assert module.MLE_logistic is MLE_logistic


def test_logistic_objectives_run_by_themselves():
    features, target = logistic_data()

    loss, gradient = logistic_grad(np.zeros(3), features, target)
    fixed_loss, fixed_gradient = logistic_grad_fixed_mu(
        np.zeros(2),
        features,
        target,
        0.0,
    )

    assert np.isfinite(loss)
    assert np.isfinite(fixed_loss)
    assert gradient.shape == (2,)
    assert fixed_gradient.shape == (2,)
    assert np.isfinite(gradient).all()
    assert np.isfinite(fixed_gradient).all()


def test_fixed_intercept_logistic_solver_runs_by_itself():
    features, target = logistic_data()

    coefficients, intercept = solve_logistic_scipy(
        features,
        target,
        mu=0.0,
    )

    assert coefficients.shape == (2,)
    assert np.isfinite(coefficients).all()
    assert np.all(coefficients >= 0)
    assert intercept == 0.0


def test_logistic_mle_runs_by_itself():
    latent, eigenvalues = MLE_logistic(
        binary_adjacency(),
        k=2,
        rng=np.random.default_rng(13),
    )

    assert latent.shape == (8, 2)
    assert eigenvalues.shape == (2,)
    assert np.isfinite(latent).all()
    assert np.isfinite(eigenvalues).all()

