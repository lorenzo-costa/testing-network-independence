import importlib

import pytest

from src.latent_samplers.copula_sampler import CopulaGenerator
from src.latent_samplers.functional_sampler import FunctionalGenerator
from src.latent_samplers.orthogonal_subspace import OrthogonalSubspaceSampler
from src.latent_samplers.rdpg_sampler import RDPGGenerator
from src.latent_samplers.sampler import LatentSampler
from src.latent_samplers.sbm_cov_sampler import SBMCovariateGenerator


def test_module_imports_latent_sampler():
    module = importlib.import_module("src.latent_samplers.sampler")

    assert module.LatentSampler is LatentSampler


@pytest.mark.parametrize(
    "selector, extra_kwargs, expected_type",
    [
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
