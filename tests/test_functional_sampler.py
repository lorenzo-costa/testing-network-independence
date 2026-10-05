import importlib

import numpy as np
import pytest

from src.dgp import BernoulliNetwork
from src.latent_samplers.functional_sampler import FunctionalGenerator


def test_module_imports_functional_generator():
    module = importlib.import_module("src.latent_samplers.functional_sampler")

    assert module.FunctionalGenerator is FunctionalGenerator


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n": 0, "k": 2, "ky": 1}, "n must be a positive integer"),
        ({"n": 5, "k": 0, "ky": 1}, "k must be a positive integer"),
        ({"n": 5, "k": 2, "ky": 2}, "only supports scalar responses"),
        (
            {"n": 5, "k": 2, "ky": 1, "noise_scale": -0.1},
            "noise_scale must be non-negative",
        ),
        (
            {"n": 5, "k": 2, "ky": 1, "noise_type": "invalid"},
            "noise_type must be 'additive' or 'multiplicative'",
        ),
        (
            {"n": 5, "k": 2, "ky": 1, "predictor_distribution": "invalid"},
            "predictor_distribution must be 'gaussian', 'student_t', or 'uniform_rdpg'",
        ),
    ],
)
def test_constructor_validates_basic_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        FunctionalGenerator(**kwargs)


def test_unknown_functional_is_rejected():
    with pytest.raises(ValueError, match="Unknown functional"):
        FunctionalGenerator(n=5, k=2, ky=1, functional_form="invalid")


@pytest.mark.parametrize(
    "covariance, message",
    [
        (np.eye(3), "column_covariance must have shape"),
        (np.array([[1.0, 1.0], [0.0, 1.0]]), "must be symmetric"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "must be positive semidefinite"),
    ],
)
def test_column_covariance_is_validated(covariance, message):
    with pytest.raises(ValueError, match=message):
        FunctionalGenerator(
            n=5,
            k=2,
            ky=1,
            column_covariance=covariance,
        )


def test_uniform_rdpg_predictors_produce_valid_inner_products():
    generator = FunctionalGenerator(
        n=100,
        k=4,
        ky=1,
        predictor_distribution="uniform_rdpg",
        rng=np.random.default_rng(42),
    )

    z, _ = generator.sample_latent()
    probabilities = z @ z.T

    assert np.all(z >= 0)
    assert np.all(np.linalg.norm(z, axis=1) <= 1)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_uniform_rdpg_predictors_generate_binary_rdpg_network():
    result = BernoulliNetwork(
        n=30,
        k=3,
        ky=1,
        rdpg=True,
        functional_form="linear",
        predictor_distribution="uniform_rdpg",
        rng=np.random.default_rng(43),
    ).generate()

    probabilities = result["Z"] @ result["Z"].T
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert set(np.unique(result["A"])).issubset({0, 1})


def test_uniform_rdpg_rejects_centering():
    with pytest.raises(ValueError, match="center_latent must be False"):
        FunctionalGenerator(
            n=5,
            k=2,
            predictor_distribution="uniform_rdpg",
            center_latent=True,
        )
