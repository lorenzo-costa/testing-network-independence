import importlib

import numpy as np
import pytest

from src.methods.llk_ratio_test import LLKRatioTest


def dummy_solver(matrix, k, rng=None):
    values = np.asarray(matrix, dtype=float)
    return values[:, :k].copy(), np.ones(k)


def estimated_latent_data(seed=50):
    rng = np.random.default_rng(seed)
    return {
        "estimated_X": rng.normal(size=(20, 2)),
        "estimated_Z": rng.normal(size=(20, 2)),
    }


def test_module_imports_likelihood_ratio_test():
    module = importlib.import_module("src.methods.llk_ratio_test")

    assert module.LLKRatioTest is LLKRatioTest


@pytest.mark.parametrize("approximation", ["chi-sq", "F-distr"])
def test_likelihood_ratio_approximations_run_by_themselves(approximation):
    method = LLKRatioTest(
        solver=dummy_solver,
        approximation=approximation,
        rng=np.random.default_rng(7),
    )

    method.fit(estimated_latent_data())

    assert method.get_name() == "LLKRatioTest"
    assert 0.0 <= method.pvalue <= 1.0
    assert isinstance(method.reject_null, bool)


def test_likelihood_ratio_requires_solver():
    with pytest.raises(ValueError, match="Solver must be provided"):
        LLKRatioTest(approximation="chi-sq")


def test_beta_approximation_reports_that_it_is_not_implemented():
    method = LLKRatioTest(solver=dummy_solver, approximation="beta")

    with pytest.raises(NotImplementedError, match="Beta approximation not implemented"):
        method.fit(estimated_latent_data())
