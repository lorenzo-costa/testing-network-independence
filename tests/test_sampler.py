import importlib

import pytest

from src.latent_samplers.copula_sampler import CopulaGenerator
from src.latent_samplers.conditional_independence_copula_sampler import (
    ConditionalIndependenceCopulaSampler,
)
from src.latent_samplers.functional_sampler import FunctionalGenerator
from src.latent_samplers.orthogonal_subspace import OrthogonalSubspaceSampler
from src.latent_samplers.post_nonlinear_noise_sampler import (
    PostNonLinearNoiseSampler,
)
from src.latent_samplers.rdpg_sampler import RDPGGenerator
from src.latent_samplers.sampler import LatentSampler
from src.latent_samplers.sbm_cov_sampler import SBMCovariateGenerator


def test_module_imports_latent_sampler():
    module = importlib.import_module("src.latent_samplers.sampler")

    assert module.LatentSampler is LatentSampler


@pytest.mark.parametrize(
    "selector, extra_kwargs, expected_type",
    [
        (
            {"post_nonlinear_noise": True},
            {"C": 2, "rho": [0, 0.5]},
            PostNonLinearNoiseSampler,
        ),
        (
            {"conditional_copula": "gaussian"},
            {
                "C": 2,
                "rho": [0, 0.5],
                "marginals": {"z": "gaussian", "y": "gaussian"},
            },
            ConditionalIndependenceCopulaSampler,
        ),
        ({"copula_model": "gaussian"}, {"marginals": "gaussian"}, CopulaGenerator),
        ({"rdpg_distr": "dirichlet"}, {}, RDPGGenerator),
        (
            {"dim_common": 1},
            {"shared_latent_type": "gaussian"},
            OrthogonalSubspaceSampler,
        ),
        ({"functional_form": "linear"}, {"ky": 1}, FunctionalGenerator),
        (
            {"sbm_covariate_sampling": "random"},
            {"y_distribution": "gaussian"},
            SBMCovariateGenerator,
        ),
    ],
)
def test_dispatches_to_selected_sampler(selector, extra_kwargs, expected_type):
    sampler = LatentSampler(n=5, k=2, **selector, **extra_kwargs)

    assert isinstance(sampler.latent_sampler, expected_type)


def test_sampling_without_a_generation_method_is_rejected():
    sampler = LatentSampler(n=5, k=2, marginals="gaussian")

    with pytest.raises(ValueError, match="No valid latent generation method specified"):
        sampler._sample_latent()


def test_conditional_selector_retains_x_on_wrapper():
    sampler = LatentSampler(
        n=30,
        k=2,
        ky=1,
        conditional_copula="gaussian",
        C=2,
        rho=[0, 0.5],
        marginals={"z": "gaussian", "y": "gaussian"},
        rng=__import__("numpy").random.default_rng(50),
    )

    Z, Y, X = sampler._sample_latent()

    assert Z.shape == (30, 2)
    assert Y.shape == (30, 1)
    assert X.shape == (30, 1)
    assert sampler.X is X


def test_nonconditioning_sampler_returns_none_x():
    sampler = LatentSampler(
        n=10,
        k=2,
        copula_model="gaussian",
        rho=0,
        marginals="gaussian",
    )
    Z, Y, X = sampler.sample_latent()
    assert Z.shape == (10, 2)
    assert Y.shape == (10, 1)
    assert X is None


def test_conditional_selector_can_use_true_activation_and_copula_model():
    sampler = LatentSampler(
        n=10,
        k=1,
        conditional_copula=True,
        copula_model="student_t",
        C=2,
        rho=0.2,
        marginals={"z": "gaussian", "y": "gaussian"},
        copula_params={"df": 4},
    )
    assert sampler.latent_sampler.copula_model == "student_t"
