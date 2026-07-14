import numpy as np
import pytest

from src.dgp import BernoulliNetwork, GaussianNetwork
from src.methods import (
    CanonicalCorrelationTest,
    DistanceCorrelationTest,
    MultivariateACTest,
    RVTest,
)
from src.solvers.weighted_network import ASE


@pytest.mark.parametrize("network_type", [GaussianNetwork, BernoulliNetwork])
@pytest.mark.parametrize(
    "method_type, kwargs",
    [
        (RVTest, {}),
        (CanonicalCorrelationTest, {}),
        (DistanceCorrelationTest, {}),
        (MultivariateACTest, {"M": 1}),
    ],
)
def test_one_network_covariate_pipeline(network_type, method_type, kwargs):
    data = network_type(
        n=14,
        k=2,
        ky=1,
        copula_model="gaussian",
        marginals="gaussian",
        rho=0.25,
        rng=np.random.default_rng(101),
    ).generate()
    method = method_type(
        k=2,
        solver=ASE,
        npermutations=2,
        rng=np.random.default_rng(202),
        **kwargs,
    )
    method.fit(data)
    result = method.get_estimated()

    assert set(data) == {"A", "Z", "Y"}
    assert result["estimated_latent"].shape == (14, 2)
    assert result["true_latent"].shape == (14, 2)
    assert result["observed_Y"].shape == (14, 1)
    assert 0 <= result["p-value"] <= 1
