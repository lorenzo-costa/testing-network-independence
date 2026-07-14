import importlib

import numpy as np
import pytest

from src.latent_samplers.sbm_cov_sampler import SBMCovariateGenerator


def test_module_imports_sbm_covariate_generator():
    module = importlib.import_module("src.latent_samplers.sbm_cov_sampler")

    assert module.SBMCovariateGenerator is SBMCovariateGenerator


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"x_upper_bound": 1, "sampling": "invalid"}, "sampling must be one of"),
        ({"x_upper_bound": 1, "rho": 1.1}, "rho must lie in"),
        ({"x_upper_bound": 1, "assortativity": 1.1}, "assortativity must lie in"),
        ({"x_upper_bound": 1, "sparsity_bias": -0.1}, "sparsity_bias must lie in"),
        ({}, "Specify x_distribution for continuous X or x_upper_bound"),
        (
            {"x_upper_bound": 1, "x_distribution": "gaussian"},
            "Specify either x_distribution.*or x_upper_bound",
        ),
    ],
)
def test_constructor_validates_basic_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SBMCovariateGenerator(n=10, k=2, **kwargs)


def test_probabilities_are_rejected_for_continuous_covariates():
    with pytest.raises(ValueError, match="only valid for a discrete covariate"):
        SBMCovariateGenerator(
            n=10,
            k=2,
            x_distribution="gaussian",
            x_probabilities=[0.5, 0.5],
        )


def test_block_probability_matrix_must_match_number_of_communities():
    with pytest.raises(ValueError, match=r"block_probs must have shape \(2, 2\)"):
        SBMCovariateGenerator(
            n=10,
            k=2,
            x_distribution="gaussian",
            block_probs=np.eye(3),
        )
