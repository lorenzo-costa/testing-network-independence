import importlib

import numpy as np
import pytest

from src.latent_samplers.functional_sampler import FunctionalGenerator


def test_module_imports_functional_generator():
    module = importlib.import_module("src.latent_samplers.functional_sampler")

    assert module.FunctionalGenerator is FunctionalGenerator


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n": 0, "k": 2, "kx": 1}, "n must be a positive integer"),
        ({"n": 5, "k": 0, "kx": 1}, "k must be a positive integer"),
        ({"n": 5, "k": 2, "kx": 2}, "only supports scalar responses"),
        (
            {"n": 5, "k": 2, "kx": 1, "noise_scale": -0.1},
            "noise_scale must be non-negative",
        ),
        (
            {"n": 5, "k": 2, "kx": 1, "noise_type": "invalid"},
            "noise_type must be 'additive' or 'multiplicative'",
        ),
        (
            {"n": 5, "k": 2, "kx": 1, "predictor_distribution": "invalid"},
            "predictor_distribution must be 'gaussian' or 'student_t'",
        ),
    ],
)
def test_constructor_validates_basic_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        FunctionalGenerator(**kwargs)


def test_unknown_functional_is_rejected():
    with pytest.raises(ValueError, match="Unknown functional"):
        FunctionalGenerator(n=5, k=2, kx=1, functional_form="invalid")


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
            kx=1,
            column_covariance=covariance,
        )

