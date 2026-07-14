import importlib

import numpy as np
import pytest

from src.load_config import _resolve_copula_setup
from src.latent_samplers.conditional_independence_copula_sampler import (
    ConditionalIndependenceCopulaSampler,
)


MARGINALS = {"z": "gaussian", "y": "uniform -1 1"}


def test_module_imports_sampler():
    module = importlib.import_module(
        "src.latent_samplers.conditional_independence_copula_sampler"
    )
    assert (
        module.ConditionalIndependenceCopulaSampler
        is ConditionalIndependenceCopulaSampler
    )


@pytest.mark.parametrize(
    "copula_model, copula_params",
    [
        ("gaussian", None),
        ("student_t", {"df": 5}),
        ("clayton", None),
        ("gumbel", None),
    ],
)
def test_supported_copulas_return_expected_shapes(copula_model, copula_params):
    sampler = ConditionalIndependenceCopulaSampler(
        n=120,
        k=3,
        ky=2,
        C=3,
        rho=[0.0, 0.2, 0.4],
        marginals=MARGINALS,
        copula_model=copula_model,
        copula_params=copula_params,
        rng=np.random.default_rng(10),
    )

    Z, Y, X = sampler.sample_latent()

    assert Z.shape == (120, 3)
    assert Y.shape == (120, 2)
    assert X.shape == (120, 1)
    assert np.issubdtype(X.dtype, np.integer)
    assert set(np.unique(X)).issubset({0, 1, 2})
    assert np.isfinite(Z).all()
    assert np.isfinite(Y).all()


def test_scalar_rho_is_repeated_for_every_stratum():
    sampler = ConditionalIndependenceCopulaSampler(
        n=10,
        k=2,
        C=4,
        rho=0.35,
        marginals=MARGINALS,
    )
    np.testing.assert_allclose(sampler.rho, np.full(4, 0.35))


def test_gaussian_uses_the_matching_stratum_correlation():
    sampler = ConditionalIndependenceCopulaSampler(
        n=4000,
        k=1,
        ky=1,
        C=2,
        rho=[0.0, 0.8],
        marginals={"z": "gaussian", "y": "gaussian"},
        copula_model="gaussian",
        center_latent=False,
        rng=np.random.default_rng(20),
    )
    Z, Y, X = sampler.sample_latent()

    correlations = []
    for stratum in range(2):
        mask = X[:, 0] == stratum
        correlations.append(np.corrcoef(Z[mask, 0], Y[mask, 0])[0, 1])

    assert abs(correlations[0]) < 0.08
    assert correlations[1] > 0.72


def test_centering_is_applied_separately_within_each_stratum():
    sampler = ConditionalIndependenceCopulaSampler(
        n=300,
        k=2,
        ky=2,
        C=3,
        rho=[0.1, 0.3, 0.5],
        marginals={"z": "exponential", "y": "uniform 2 5"},
        copula_model="gaussian",
        center_latent=True,
        rng=np.random.default_rng(30),
    )
    Z, Y, X = sampler.sample_latent()

    assert set(np.unique(X)) == {0, 1, 2}
    for stratum in range(3):
        mask = X[:, 0] == stratum
        np.testing.assert_allclose(Z[mask].mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(Y[mask].mean(axis=0), 0, atol=1e-12)


def test_custom_class_probabilities_are_used():
    sampler = ConditionalIndependenceCopulaSampler(
        n=40,
        k=1,
        C=3,
        rho=0,
        marginals=MARGINALS,
        class_probabilities=[0, 1, 0],
        rng=np.random.default_rng(40),
    )
    _, _, X = sampler.sample_latent()
    np.testing.assert_array_equal(X, np.ones((40, 1), dtype=int))


def test_p_is_an_alias_for_class_probabilities():
    sampler = ConditionalIndependenceCopulaSampler(
        n=20,
        k=1,
        C=2,
        rho=0,
        marginals=MARGINALS,
        p=[1, 0],
        rng=np.random.default_rng(41),
    )
    _, _, X = sampler.sample()
    np.testing.assert_array_equal(X, np.zeros((20, 1), dtype=int))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"C": 0}, "C must be a positive integer"),
        ({"C": 2, "rho": [0.1]}, "vector of length C=2"),
        ({"C": 2, "rho": [0.1, 1.1]}, "must lie in"),
        (
            {"C": 2, "class_probabilities": [0.2, 0.2]},
            "must sum to 1",
        ),
        (
            {"C": 2, "class_probabilities": [0.5, -0.5]},
            "nonnegative",
        ),
    ],
)
def test_common_parameters_are_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ConditionalIndependenceCopulaSampler(
            n=10,
            k=2,
            marginals=MARGINALS,
            **kwargs,
        )


@pytest.mark.parametrize("copula_model", ["clayton", "gumbel"])
def test_archimedean_copulas_reject_negative_rho(copula_model):
    with pytest.raises(ValueError, match="does not support negative rho"):
        ConditionalIndependenceCopulaSampler(
            n=10,
            k=1,
            C=2,
            rho=[0.2, -0.1],
            marginals=MARGINALS,
            copula_model=copula_model,
        )


def test_student_t_requires_positive_df():
    with pytest.raises(ValueError, match="positive 'df'"):
        ConditionalIndependenceCopulaSampler(
            n=10,
            k=1,
            C=2,
            marginals=MARGINALS,
            copula_model="student_t",
            copula_params={"df": 0},
        )


@pytest.mark.parametrize(
    "marginals, error",
    [
        ("gaussian", TypeError),
        ({"z": "gaussian"}, ValueError),
        ({"z": "gaussian", "y": "gaussian", "x": "uniform"}, ValueError),
    ],
)
def test_marginals_require_new_dictionary_format(marginals, error):
    with pytest.raises(error, match="keys 'z' and 'y'"):
        ConditionalIndependenceCopulaSampler(
            n=10,
            k=1,
            C=2,
            marginals=marginals,
        )


def test_unknown_marginal_is_rejected_during_initialization():
    with pytest.raises(ValueError, match="Unknown distribution"):
        ConditionalIndependenceCopulaSampler(
            n=10,
            k=1,
            C=2,
            marginals={"z": "invalid", "y": "gaussian"},
        )


def test_unused_keyword_arguments_are_accepted():
    sampler = ConditionalIndependenceCopulaSampler(
        n=10,
        k=1,
        C=2,
        irrelevant_option="ignored",
        rdpg=False,
    )
    Z, Y, X = sampler.sample_latent()
    assert Z.shape == (10, 1)
    assert Y.shape == (10, 1)
    assert X.shape == (10, 1)


def test_gaussian_marginals_are_the_default():
    sampler = ConditionalIndependenceCopulaSampler(
        n=20,
        k=2,
        ky=1,
        C=2,
        rho=0.3,
        rng=np.random.default_rng(61),
    )
    Z, Y, X = sampler.sample_latent()

    assert sampler.marginals == {"z": "gaussian", "y": "gaussian"}
    assert Z.shape == (20, 2)
    assert Y.shape == (20, 1)
    assert X.shape == (20, 1)


def test_config_resolver_forwards_conditional_sampler_arguments():
    dgp_factory, _ = _resolve_copula_setup(
        {
            "dgp": "GaussianNetwork",
            "solver": "ASE",
            "conditional_copula": "gaussian",
            "C": 2,
            "class_probabilities": [0.25, 0.75],
            "center_latent": False,
        }
    )
    dgp = dgp_factory(
        n=20,
        k=1,
        rho=[0, 0.4],
        marginals={"z": "gaussian", "y": "gaussian"},
        rng=np.random.default_rng(60),
    )

    assert isinstance(dgp.latent_sampler, ConditionalIndependenceCopulaSampler)
    np.testing.assert_allclose(
        dgp.latent_sampler.class_probabilities,
        [0.25, 0.75],
    )
    assert dgp.latent_sampler.center_latent is False
