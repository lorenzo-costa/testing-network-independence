import numpy as np
import pytest

from src.dgp import BernoulliNetwork, GaussianNetwork
from src.latent_samplers.copula_sampler import CopulaGenerator
from src.latent_samplers.functional_sampler import FunctionalGenerator
from src.latent_samplers.orthogonal_subspace import OrthogonalSubspaceSampler
from src.latent_samplers.rdpg_sampler import RDPGGenerator
from src.latent_samplers.sbm_cov_sampler import SBMCovariateGenerator


LATENT_POSITION_CONFIGS = [
    pytest.param(
        {"copula_model": "gaussian", "marginals": "gaussian", "rho": 0.25},
        CopulaGenerator,
        2,
        1,
        id="gaussian-copula",
    ),
    pytest.param(
        {"rdpg_distr": "dirichlet", "rho": 0.25},
        RDPGGenerator,
        2,
        1,
        id="dirichlet-rdpg",
    ),
    pytest.param(
        {"dim_common": 1, "shared_latent_type": "gaussian"},
        OrthogonalSubspaceSampler,
        2,
        1,
        id="orthogonal-gaussian",
    ),
    pytest.param(
        {"functional_form": "linear", "ky": 1},
        FunctionalGenerator,
        2,
        1,
        id="functional-linear",
    ),
    pytest.param(
        {
            "sbm_covariate_sampling": "step",
            "y_distribution": "gaussian",
        },
        SBMCovariateGenerator,
        2,
        1,
        id="sbm-continuous-covariate",
    ),
]


@pytest.mark.parametrize("network_type", [GaussianNetwork, BernoulliNetwork])
@pytest.mark.parametrize(
    "latent_kwargs, expected_sampler_type, z_dimension, x_dimension",
    LATENT_POSITION_CONFIGS,
)
def test_user_can_select_latent_positions_and_generate_networks(
    network_type,
    latent_kwargs,
    expected_sampler_type,
    z_dimension,
    x_dimension,
):
    n = 12
    kwargs = dict(latent_kwargs)
    if network_type is GaussianNetwork:
        kwargs["edge_var"] = 0.25
    elif kwargs.get("rdpg_distr") is not None:
        kwargs["rdpg"] = True

    network = network_type(
        n=n,
        k=2,
        symmetric=True,
        self_loops=False,
        rng=np.random.default_rng(42),
        **kwargs,
    )
    result = network.generate()

    assert isinstance(network.latent_sampler, expected_sampler_type)
    assert set(result) == {"A", "Z", "Y"}
    assert result["A"].shape == (n, n)
    assert result["Z"].shape == (n, z_dimension)
    assert result["Y"].shape == (n, x_dimension)
    assert np.isfinite(result["A"]).all()
    assert np.isfinite(result["Z"]).all()
    assert np.isfinite(result["Y"]).all()
    np.testing.assert_allclose(result["A"], result["A"].T)
    np.testing.assert_array_equal(np.diag(result["A"]), 0)

    if network_type is BernoulliNetwork:
        assert set(np.unique(result["A"])).issubset({0, 1})
