import re

import numpy as np
import pytest

from src.dgp import GaussianNetwork
from src.latent_samplers.linear_model_sampler import LinearModelGenerator


def test_target_r2_rescales_explicit_heterogeneous_coefficients():
    coefficients = [np.array([[1.0], [2.0]]), np.ones((2, 3))]
    sampler = LinearModelGenerator(
        n=20,
        k=[2, 1, 3],
        B=coefficients,
        marginals={
            "x": ["uniform 0 2", "gaussian"],
            "epsilon": "gamma 2 3",
        },
        noise_scale=0.5,
        target_r2=0.4,
        rng=np.random.default_rng(4),
    )

    # Var[Uniform(0, 2)] = 1/3, Var[N(0, 1)] = 1, and
    # Var[Gamma(shape=2, scale=3)] = 18.
    unscaled_signal = (1 / 3) * np.sum(coefficients[0] ** 2) + np.sum(
        coefficients[1] ** 2
    )
    noise = 2 * 0.5**2 * 18
    expected_scale = np.sqrt((0.4 / 0.6) * noise / unscaled_signal)

    assert sampler.coefficient_scale == pytest.approx(expected_scale)
    assert sampler.population_signal_variance == pytest.approx(
        expected_scale**2 * unscaled_signal
    )
    assert sampler.population_noise_variance == pytest.approx(noise)
    assert sampler.population_r2 == pytest.approx(0.4)
    for actual, original in zip(sampler.B, coefficients):
        np.testing.assert_allclose(actual, expected_scale * original)


def test_target_r2_zero_creates_the_null_model():
    sampler = LinearModelGenerator(
        n=10,
        k=[2, 1, 3],
        target_r2=0,
        rng=np.random.default_rng(5),
    )

    assert sampler.is_null
    assert sampler.coefficient_scale == 0
    assert sampler.population_r2 == 0
    assert all(np.count_nonzero(Bj) == 0 for Bj in sampler.B)


def test_target_r2_none_preserves_explicit_coefficients():
    coefficients = [np.array([[1.0, -2.0]])]
    sampler = LinearModelGenerator(n=10, k=[1, 2], B=coefficients)

    np.testing.assert_array_equal(sampler.B[0], coefficients[0])
    assert sampler.coefficient_scale == 1
    assert sampler.population_r2 is None


def test_target_r2_is_reproducible_and_coefficients_are_sampled_once():
    first = LinearModelGenerator(
        n=12,
        k=[2, 1, 3],
        target_r2=0.25,
        rng=np.random.default_rng(8),
    )
    second = LinearModelGenerator(
        n=12,
        k=[2, 1, 3],
        target_r2=0.25,
        rng=np.random.default_rng(8),
    )
    saved = [Bj.copy() for Bj in first.B]

    first._sample_latent_linear()
    first._sample_latent_linear()

    for first_B, second_B, saved_B in zip(first.B, second.B, saved):
        np.testing.assert_array_equal(first_B, second_B)
        np.testing.assert_array_equal(first_B, saved_B)


def test_sample_variance_ratio_approaches_target_r2():
    sampler = LinearModelGenerator(
        n=60_000,
        k=[2, 1, 3],
        target_r2=0.35,
        rng=np.random.default_rng(11),
    )

    Y, _, epsilon = sampler._sample_latent_linear(return_noise=True)
    signal = Y - epsilon
    empirical_signal = np.var(signal, axis=0).sum()
    empirical_noise = np.var(epsilon, axis=0).sum()
    empirical_r2 = empirical_signal / (empirical_signal + empirical_noise)

    assert empirical_r2 == pytest.approx(0.35, abs=0.01)


@pytest.mark.parametrize(
    "target_r2",
    [-0.1, 1.0, np.inf, np.nan],
)
def test_invalid_target_r2_is_rejected(target_r2):
    with pytest.raises(ValueError, match="0 <= target_r2 < 1"):
        LinearModelGenerator(n=5, k=[1, 1], target_r2=target_r2)


def test_boolean_target_r2_is_rejected():
    with pytest.raises(TypeError, match="real scalar"):
        LinearModelGenerator(n=5, k=[1, 1], target_r2=True)


def test_positive_target_r2_rejects_zero_coefficients():
    with pytest.raises(ValueError, match="incompatible"):
        LinearModelGenerator(n=5, k=[1, 1], B="zero", target_r2=0.2)


@pytest.mark.parametrize(
    "marginals, message",
    [
        ({"x": "cauchy", "epsilon": "gaussian"}, 'marginals["x"][0]'),
        ({"x": "gaussian", "epsilon": "cauchy"}, 'marginals["epsilon"]'),
    ],
)
def test_positive_target_r2_requires_finite_variances(marginals, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        LinearModelGenerator(
            n=5,
            k=[1, 1],
            marginals=marginals,
            target_r2=0.2,
        )


def test_positive_target_r2_requires_positive_noise_variance():
    with pytest.raises(ValueError, match="positive finite noise variance"):
        LinearModelGenerator(
            n=5,
            k=[1, 1],
            noise_scale=0,
            target_r2=0.2,
        )


def test_gaussian_network_forwards_target_r2_to_linear_sampler():
    network = GaussianNetwork(
        n=12,
        k=[2, 1, 3],
        target_r2=0.6,
        rng=np.random.default_rng(19),
    )

    assert network.latent_sampler.target_r2 == 0.6
    assert network.latent_sampler.population_r2 == pytest.approx(0.6)
