import importlib

import numpy as np

from src.solvers.passtthrough import placeholder_method


def test_module_imports_placeholder_solver():
    module = importlib.import_module("src.solvers.passtthrough")

    assert module.placeholder_method is placeholder_method


def test_placeholder_solver_returns_expected_shapes():
    latent, eigenvalues = placeholder_method(np.eye(6), k=3)

    assert latent.shape == (6, 3)
    assert eigenvalues.shape == (3,)
    assert np.isfinite(latent).all()
    assert np.isfinite(eigenvalues).all()

