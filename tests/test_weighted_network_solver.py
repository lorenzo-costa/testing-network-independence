import importlib

import numpy as np

from src.solvers.weighted_network import ASE, MLE_gaussian


def symmetric_matrix(seed=80, n=8):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, n))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    return matrix


def test_module_imports_weighted_network_solvers():
    module = importlib.import_module("src.solvers.weighted_network")

    assert module.ASE is ASE
    assert module.MLE_gaussian is MLE_gaussian


def test_ase_runs_by_itself():
    latent, eigenvalues = ASE(
        symmetric_matrix(),
        k=2,
        rng=np.random.default_rng(11),
    )

    assert latent.shape == (8, 2)
    assert eigenvalues.shape == (2,)
    assert np.isfinite(latent).all()
    assert np.isfinite(eigenvalues).all()
    assert np.all(eigenvalues >= 0)


def test_gaussian_mle_runs_by_itself():
    latent, eigenvalues = MLE_gaussian(
        symmetric_matrix(),
        k=2,
        shrink=0.25,
        rng=np.random.default_rng(12),
    )

    assert latent.shape == (8, 2)
    assert eigenvalues.shape == (2,)
    assert np.isfinite(latent).all()
    assert np.isfinite(eigenvalues).all()
    assert np.all(eigenvalues >= 0)

