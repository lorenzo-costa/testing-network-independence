import numpy as np
import pytest

from src.dgp import GaussianNetwork
from src.latent_samplers import CopulaSampler


@pytest.mark.parametrize(
    "copula_model,copula_params",
    [
        ("gaussian", {}),
        ("student_t", {"df": 5}),
        (
            "mixture_uniform",
            {"weights": [0.5, 0.5], "correlations": [0.4, -0.4]},
        ),
    ],
)
def test_supported_copulas_split_concatenated_latents_into_p_networks(
    copula_model, copula_params
):
    sampler = CopulaSampler(
        n=20,
        p=3,
        d_x=2,
        d_y=1,
        rho=0.1,
        marginals="gaussian",
        copula_model=copula_model,
        copula_params=copula_params,
        column_covariance=np.eye(6),
        rng=np.random.default_rng(41),
    )

    latent = sampler.sample_latent()

    assert latent["Y"].shape == (20, 1)
    assert len(latent["X"]) == 3
    assert all(x.shape == (20, 2) for x in latent["X"])


def test_gaussian_copula_column_covariance_correlates_x_networks():
    sampler = CopulaSampler(
        n=30_000,
        p=2,
        d_x=1,
        d_y=1,
        rho=0,
        marginals="gaussian",
        copula_model="gaussian",
        column_covariance=np.array([[1.0, 0.7], [0.7, 1.0]]),
        rng=np.random.default_rng(42),
    )

    latent = sampler.sample_latent()
    observed = np.corrcoef(latent["X"][0][:, 0], latent["X"][1][:, 0])[0, 1]

    assert observed == pytest.approx(0.7, abs=0.02)


def test_gaussian_network_generates_one_adjacency_per_copula_x_block():
    network = GaussianNetwork(
        n=12,
        p=2,
        d_x=2,
        d_y=1,
        latent_sampler=CopulaSampler,
        copula_model="gaussian",
        rho=0.2,
        marginals="gaussian",
        column_covariance=np.eye(4),
        rng=np.random.default_rng(43),
    )

    data = network.generate()

    assert len(data["X"]) == 2
    assert len(data["A_X"]) == 2
    assert all(x.shape == (12, 2) for x in data["X"])
    assert all(adjacency.shape == (12, 12) for adjacency in data["A_X"])


def test_copula_rejects_column_covariance_without_all_x_network_blocks():
    with pytest.raises(ValueError, match=r"column_covariance_x.*\(4, 4\)"):
        CopulaSampler(
            n=5,
            p=2,
            d_x=2,
            d_y=1,
            rho=0,
            marginals="gaussian",
            copula_model="gaussian",
            column_covariance=np.eye(2),
        )


def test_copula_rejects_p_greater_than_one_for_unsupported_models():
    with pytest.raises(ValueError, match="p > 1 is supported only"):
        CopulaSampler(
            n=5,
            p=2,
            d_x=1,
            d_y=1,
            rho=0.2,
            marginals="gaussian",
            copula_model="clayton",
        )
