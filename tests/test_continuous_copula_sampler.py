import numpy as np

from src.dgp import GaussianNetwork
from src.latent_samplers import ContinuousConditionalCopulaSampler


def test_continuous_copula_network_generation_and_conditional_sampling():
    dgp = GaussianNetwork(
        n=20,
        k=2,
        ky=1,
        continuous_conditional_copula="gaussian",
        correlation_zx=0.5,
        correlation_yx=0.4,
        conditional_correlation_zy=0,
        rng=np.random.default_rng(7),
    )

    assert isinstance(dgp.latent_sampler, ContinuousConditionalCopulaSampler)
    data = dgp.generate()
    assert data["Z"].shape == (20, 2)
    assert data["Y"].shape == (20, 1)
    assert data["X"].shape == (20, 1)
    assert dgp.is_null

    fresh_y = dgp.latent_sampler.sample(data["X"])
    assert fresh_y.shape == data["Y"].shape
    assert np.isfinite(fresh_y).all()
    np.testing.assert_array_equal(dgp.latent_sampler.X, data["X"])
