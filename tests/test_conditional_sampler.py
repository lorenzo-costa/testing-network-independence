import numpy as np
import pytest

from src.latent_samplers import ConditionalSampler
from src.latent_samplers.sampler import LatentSampler


def test_modulo_sampler_conditional_draws_and_wrapper():
    rng = np.random.default_rng(17)
    sampler = LatentSampler(
        n=1000, k=2, ky=2, conditional_modulo=True,
        x_distribution="gaussian", rng=rng,
    )
    z, y, x = sampler.sample_latent()
    assert z.shape == y.shape == (1000, 2)
    assert x.shape == (1000, 1)
    np.testing.assert_allclose(y, np.mod(z + x, 1))
    assert sampler.X is x

    fixed_x = np.full((1000, 1), 2.25)
    z_fixed, y_fixed = sampler.latent_sampler.sample(fixed_x)
    np.testing.assert_allclose(y_fixed, np.mod(z_fixed + fixed_x, 1))
    assert np.all((y_fixed >= 0) & (y_fixed < 1))
    assert abs(np.mean(y_fixed[:, 0]) - 0.5) < 0.04


def test_scalar_response_uses_first_predictor_coordinate():
    sampler = ConditionalSampler(n=30, k=3, ky=1, rng=np.random.default_rng(8))
    z, y, x = sampler.sample_latent()
    np.testing.assert_allclose(y, np.mod(z[:, :1] + x, 1))
    with pytest.raises(ValueError, match="x must have finite shape"):
        sampler.sample(np.zeros((30, 2)))
