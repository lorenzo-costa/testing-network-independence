import importlib

import pytest

from src.latent_samplers.copula_sampler import CopulaGenerator
from src.latent_samplers.functional_sampler import FunctionalGenerator
from src.latent_samplers.hyppo_sampler import HyppoSimSampler
from src.latent_samplers.orthogonal_subspace import OrthogonalSubspaceSampler
from src.latent_samplers.rdpg_sampler import RDPGGenerator
from src.latent_samplers.sampler import LatentSampler
from src.latent_samplers.sbm_cov_sampler import SBMCovariateGenerator
from src.latent_samplers.sbm_sampler import SBMGenerator


def test_module_imports_latent_sampler():
    module = importlib.import_module("src.latent_samplers.sampler")

    assert module.LatentSampler is LatentSampler


@pytest.mark.parametrize(
    "selector, extra_kwargs, expected_type",
    [
        ({"copula_model": "gaussian"}, {"marginals": "gaussian"}, CopulaGenerator),
        ({"rdpg_distr": "dirichlet"}, {}, RDPGGenerator),
        ({"block_probs_type": "random"}, {}, SBMGenerator),
        (
            {"dim_common": 1},
            {"shared_latent_type": "gaussian"},
            OrthogonalSubspaceSampler,
        ),
        ({"latent_sim": "linear"}, {}, HyppoSimSampler),
        ({"functional_form": "linear"}, {"kx": 1}, FunctionalGenerator),
        (
            {"sbm_covariate_sampling": "random"},
            {"x_distribution": "gaussian"},
            SBMCovariateGenerator,
        ),
    ],
)
def test_dispatches_to_selected_sampler(selector, extra_kwargs, expected_type):
    sampler = LatentSampler(n=5, k=2, **selector, **extra_kwargs)

    assert isinstance(sampler.latent_sampler, expected_type)


def test_higher_priority_selector_warns_about_ignored_selector():
    with pytest.warns(UserWarning, match="latent_sim was specified; ignoring: copula_model"):
        sampler = LatentSampler(
            n=5,
            k=2,
            latent_sim="linear",
            copula_model="gaussian",
            marginals="gaussian",
        )

    assert isinstance(sampler.latent_sampler, HyppoSimSampler)


def test_sampling_without_a_generation_method_is_rejected():
    sampler = LatentSampler(n=5, k=2, marginals="gaussian")

    with pytest.raises(ValueError, match="No valid latent generation method specified"):
        sampler._sample_latent()
