import numpy as np
import pytest

from src.latent_samplers import ConditionalSampler
from src.latent_samplers.sampler import LatentSampler
from src.load_config import build_factorial_design, load_config


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
    y_fixed = sampler.latent_sampler.sample(fixed_x)
    assert y_fixed.shape == (1000, 2)
    assert np.all((y_fixed >= 0) & (y_fixed < 1))
    assert abs(np.mean(y_fixed[:, 0]) - 0.5) < 0.04


def test_independent_response_null_preserves_conditional_law():
    sampler = ConditionalSampler(
        n=2000, k=3, ky=3, z_distribution="uniform_rdpg",
        independent_response=True, rng=np.random.default_rng(7),
    )
    z, y, x = sampler.sample_latent()
    assert sampler.is_null
    assert np.all(np.linalg.norm(z, axis=1) <= 1 + 1e-12)
    assert np.all(z >= 0)
    assert np.all((y >= 0) & (y < 1))
    assert abs(np.corrcoef(z[:, 0], y[:, 0])[0, 1]) < 0.07
    conditional_y = sampler.sample(x)
    assert conditional_y.shape == y.shape


def test_scalar_response_uses_first_predictor_coordinate():
    sampler = ConditionalSampler(n=30, k=3, ky=1, rng=np.random.default_rng(8))
    z, y, x = sampler.sample_latent()
    np.testing.assert_allclose(y, np.mod(z[:, :1] + x, 1))
    with pytest.raises(ValueError, match="x must have finite shape"):
        sampler.sample(np.zeros((30, 2)))


def test_conditional_config_has_matching_null_and_alternative():
    config = load_config("config_conditional.yaml")
    rows = build_factorial_design(config)
    assert config["simulation"]["nsim"] == 200
    assert len(rows) == 6
    assert {row["setup"][0].keywords["independent_response"] for row in rows} == {True, False}
    sampler = rows[0]["setup"][0](n=100, k=3, ky=3, rng=np.random.default_rng(5))
    z, y, x = sampler.sample_latent()
    assert np.all(z >= 0)
    assert np.all(np.linalg.norm(z, axis=1) <= 1 + 1e-12)
    assert y.shape == (100, 3)
    assert x.shape == (100, 1)
