import importlib

import numpy as np
import pytest

from src.latent_samplers.copula_sampler import CopulaGenerator


def test_module_imports_copula_generator():
    module = importlib.import_module("src.latent_samplers.copula_sampler")

    assert module.CopulaGenerator is CopulaGenerator


def test_student_t_copula_requires_degrees_of_freedom():
    with pytest.raises(ValueError, match="df parameter must be provided"):
        CopulaGenerator(
            n=10,
            k=2,
            copula_model="student_t",
            marginals="gaussian",
        )


@pytest.mark.parametrize(
    "copula_params, message",
    [
        ({}, "weights and correlations must be provided"),
        (
            {"weights": [1.0], "correlations": [0.0, 0.5]},
            "weights and correlations must have the same length",
        ),
        (
            {"weights": [0.2, 0.2], "correlations": [0.0, 0.5]},
            "weights must sum to 1",
        ),
    ],
)
def test_mixture_uniform_validates_component_parameters(copula_params, message):
    with pytest.raises(ValueError, match=message):
        CopulaGenerator(
            n=10,
            k=2,
            copula_model="mixture_uniform",
            copula_params=copula_params,
            marginals="gaussian",
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"column_covariance": np.eye(3)}, "column_covariance_z must be a 2x2"),
        ({"column_covariance_y": np.eye(3)}, "column_covariance_y must be a 2x2"),
    ],
)
def test_covariance_dimensions_are_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CopulaGenerator(
            n=10,
            k=2,
            ky=2,
            copula_model="gaussian",
            marginals="gaussian",
            **kwargs,
        )


def test_unknown_marginal_is_rejected():
    with pytest.raises(ValueError, match="Unknown distribution: invalid"):
        CopulaGenerator(
            n=10,
            k=2,
            copula_model="gaussian",
            marginals="invalid",
        )


def test_one_sided_pareto_marginal_supports_shape_and_scale():
    sampler = CopulaGenerator(
        n=100,
        k=2,
        copula_model="gaussian",
        rho=0,
        marginals={"z": "pareto 2.5 3", "y": "pareto 3"},
        center_latent=False,
        rng=np.random.default_rng(42),
    )

    Z, Y = sampler._sample_latent_copula()

    assert Z.shape == (100, 2)
    assert Y.shape == (100, 1)
    assert (Z >= 3).all()
    assert (Y >= 1).all()


@pytest.mark.parametrize("marginal", ["pareto", "pareto 0", "pareto 2 -1"])
def test_pareto_marginal_validates_parameters(marginal):
    with pytest.raises(ValueError, match="pareto"):
        CopulaGenerator(
            n=10,
            k=2,
            copula_model="gaussian",
            marginals=marginal,
        )
