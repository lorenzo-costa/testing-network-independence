import importlib

import numpy as np

from src.solvers.placeholder import placeholder_solver


def test_module_imports_placeholder_solver():
    module = importlib.import_module("src.solvers.placeholder")

    assert module.placeholder_solver is placeholder_solver


def test_placeholder_solver_returns_expected_shapes():
    latent, eigenvalues = placeholder_solver(np.eye(6), k=3)

    assert latent.shape == (6, 3)
    assert eigenvalues.shape == (3,)
    assert np.isfinite(latent).all()
    assert np.isfinite(eigenvalues).all()
