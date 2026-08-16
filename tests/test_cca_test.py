import importlib

import numpy as np
import pytest

from src.methods.cca_test import CanonicalCorrelationTest


def global_network_data(seed=20, n=14):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(n, 2))
    X = [rng.normal(size=(n, 1)), rng.normal(size=(n, 2))]
    return {
        "Ay": Y @ Y.T,
        "Ax": [block @ block.T for block in X],
        "Y": Y,
        "X": X,
    }


def test_module_imports_canonical_correlation_test():
    module = importlib.import_module("src.methods.cca_test")
    assert module.CanonicalCorrelationTest is CanonicalCorrelationTest


def test_canonical_correlation_global_test_runs():
    method = CanonicalCorrelationTest(
        use_true_latent=True,
        npermutations=3,
        rng=np.random.default_rng(3),
    )

    method.fit(global_network_data())
    result = method.get_estimated()

    assert method.Xhat.shape == (14, 3)
    assert method.get_name() == "CCA_PermutationTest_latent"
    assert len(method.permutation_distribution) == 3
    assert 0.0 <= result["test_stat"] <= 1.0
    assert 0.0 <= result["p-value"] <= 1.0


def test_canonical_correlation_requires_dimensions_and_solver():
    with pytest.raises(ValueError, match="k must be provided"):
        CanonicalCorrelationTest()
