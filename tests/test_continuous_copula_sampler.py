import numpy as np
import pytest

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


def test_multivariate_x_network_and_conditional_sampling():
    loadings = 0.3 * np.eye(3)
    sampler = ContinuousConditionalCopulaSampler(
        n=30, k=3, ky=3, px=3,
        correlation_zx=loadings,
        correlation_yx=loadings,
        conditional_correlation_zy=-0.3**2 / (1 - 0.3**2) * np.eye(3),
        rng=np.random.default_rng(12),
    )

    Z, Y, X = sampler.sample_latent()
    assert Z.shape == Y.shape == X.shape == (30, 3)
    assert sampler.correlation_matrix.shape == (9, 9)
    np.testing.assert_allclose(sampler.correlation_matrix[:3, 3:6], 0, atol=1e-12)
    fresh_y = sampler.sample(X)
    assert fresh_y.shape == (30, 3)
    assert np.isfinite(fresh_y).all()
    with pytest.raises(ValueError, match="shape"):
        sampler.sample(X[:, :1])
