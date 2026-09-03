import importlib

import numpy as np
import pytest

from src.methods.cca_test import CanonicalCorrelationTest


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
        "Z": rng.normal(size=(14, 2)),
    }
    method = CanonicalCorrelationTest(
        solver=dummy_solver,
        use_true_latent=True,
        npermutations=3,
        rng=np.random.default_rng(3),
    )

    method.fit(data)
    result = method.get_estimated()

    assert method.get_name() == "CCA_PermutationTest_covariate"
    assert len(method.permutation_distribution) == 3
    assert 0.0 <= result["test_stat"] <= 1.0
    assert 0.0 <= result["p-value"] <= 1.0


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
