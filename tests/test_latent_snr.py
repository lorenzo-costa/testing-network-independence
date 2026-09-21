"""Population SNR calibration changes coefficients, not X or the error law."""

from itertools import product

import numpy as np
import pytest

from src.dgp import BernoulliNetwork, GaussianNetwork
from src.latent_samplers import MultipleNetworksSampler


def covariance(dimension):
    indices = np.arange(dimension)
    return 0.5 ** np.abs(indices[:, None] - indices[None, :])


@pytest.mark.parametrize("target", [0, 0.1, 1, 5, 50])
@pytest.mark.parametrize("fixed", [False, True])
def test_population_snr_matches_target_and_preserves_error_draws(target, fixed):
    B = np.arange(1.0, 13.0).reshape(2, 6) if fixed else None
    kwargs = dict(
        n=40,
        p=3,
        d_x=2,
        d_y=2,
        B=B,
        x_mean=np.arange(2.0, 8.0),
        x_variance=covariance(6),
        eps_variance=[[2.0, 0.3], [0.3, 0.5]],
    )
    sampler = MultipleNetworksSampler(
        **kwargs, snr=target, rng=np.random.default_rng(300)
    )
    reference = MultipleNetworksSampler(**kwargs, rng=np.random.default_rng(300))
    original_B = None if B is None else B.copy()
    original_noise = sampler.eps_covariance.copy()
    for _ in range(2):
        actual, raw = sampler.sample_latent(), reference.sample_latent()
        actual_power = np.trace(actual["B"] @ sampler.x_covariance @ actual["B"].T)
        noise_power = np.trace(sampler.eps_covariance)
        assert actual_power / noise_power == pytest.approx(target, rel=1e-12, abs=1e-14)
        raw_power = np.trace(raw["B"] @ sampler.x_covariance @ raw["B"].T)
        expected_B = raw["B"] * np.sqrt(target * noise_power / raw_power)
        np.testing.assert_allclose(actual["B"], expected_B, rtol=1e-12, atol=1e-14)
        x = np.concatenate(actual["X"], axis=1)
        np.testing.assert_array_equal(x, np.concatenate(raw["X"], axis=1))
        np.testing.assert_allclose(
            actual["Y"] - x @ actual["B"].T,
            raw["Y"] - x @ raw["B"].T,
            atol=1e-12,
        )
        assert sampler.rng.bit_generator.state == reference.rng.bit_generator.state
        np.testing.assert_array_equal(sampler.eps_covariance, original_noise)
    if fixed:
        np.testing.assert_array_equal(B, original_B)


@pytest.mark.parametrize("B", [None, 0, [[1.0, 2.0], [3.0, 4.0]]])
def test_snr_none_preserves_previous_results_exactly(B):
    kwargs = dict(n=9, p=2, d_x=1, d_y=2, B=B)
    default = MultipleNetworksSampler(**kwargs, rng=np.random.default_rng(301))
    explicit = MultipleNetworksSampler(
        **kwargs, snr=None, rng=np.random.default_rng(301)
    )
    for _ in range(2):
        a, b = default.sample_latent(), explicit.sample_latent()
        for key in ("Y", "B"):
            np.testing.assert_array_equal(a[key], b[key])
        for x, expected in zip(a["X"], b["X"]):
            np.testing.assert_array_equal(x, expected)
        assert default.rng.bit_generator.state == explicit.rng.bit_generator.state


@pytest.mark.parametrize(
    "x_dist,eps_dist,b_dist", product(["gaussian", "multivariate_gaussian"], repeat=3)
)
def test_snr_uses_population_covariance_even_for_one_node(x_dist, eps_dist, b_dist):
    sampler = MultipleNetworksSampler(
        1,
        2,
        2,
        3,
        snr=2.5,
        x_variance=3,
        eps_variance=2,
        x_distribution=x_dist,
        eps_distribution=eps_dist,
        b_distribution=b_dist,
        rng=np.random.default_rng(302),
    )
    result = sampler.sample_latent()
    assert result["Y"].shape == (1, 3)
    ratio = np.trace(result["B"] @ sampler.x_covariance @ result["B"].T) / 6
    assert ratio == pytest.approx(2.5)


def test_student_t_3_error_variance_preserves_population_snr_calibration():
    sampler = MultipleNetworksSampler(
        5,
        3,
        2,
        4,
        snr=2.5,
        x_variance=2,
        x_network_correlation=0.5,
        eps_variance=3,
        eps_distribution="student_t_3",
        rng=np.random.default_rng(308),
    )

    result = sampler.sample_latent()

    signal_power = np.trace(result["B"] @ sampler.x_covariance @ result["B"].T)
    noise_power = np.trace(sampler.eps_covariance)
    assert signal_power / noise_power == pytest.approx(2.5)


def test_empirical_variance_ratio_converges_to_requested_population_snr():
    target = 2.5
    noise_covariance = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 0.5]])
    sampler = MultipleNetworksSampler(
        50_000,
        2,
        2,
        3,
        snr=target,
        x_mean=10,
        x_variance=covariance(4),
        eps_variance=noise_covariance,
        rng=np.random.default_rng(303),
    )
    result = sampler.sample_latent()
    signal = np.concatenate(result["X"], axis=1) @ result["B"].T
    errors = result["Y"] - signal
    empirical_snr = signal.var(axis=0, ddof=1).sum() / errors.var(axis=0, ddof=1).sum()
    assert empirical_snr == pytest.approx(target, rel=0.03)
    np.testing.assert_allclose(
        np.cov(errors, rowvar=False), noise_covariance, atol=0.03
    )


def test_fixed_coefficients_are_scaled_once_and_returned_as_independent_copies():
    B = np.array([[1.0, 2.0], [3.0, 4.0]])
    sampler = MultipleNetworksSampler(4, 2, 1, 2, B=B, snr=0.25)
    first = sampler.sample_latent()["B"]
    second = sampler.sample_latent()["B"]
    np.testing.assert_array_equal(first, second)
    assert not np.shares_memory(first, sampler.B)
    first[:] = 0
    np.testing.assert_array_equal(sampler.sample_latent()["B"], second)
    np.testing.assert_array_equal(B, [[1.0, 2.0], [3.0, 4.0]])


def test_singular_covariances_are_valid_when_signal_and_noise_have_positive_trace():
    sampler = MultipleNetworksSampler(
        8,
        2,
        1,
        2,
        B=np.ones((2, 2)),
        snr=3,
        x_variance=np.ones((2, 2)),
        eps_variance=np.ones((2, 2)),
        rng=np.random.default_rng(307),
    )
    result = sampler.sample_latent()
    power = np.trace(result["B"] @ sampler.x_covariance @ result["B"].T)
    assert power / np.trace(sampler.eps_covariance) == pytest.approx(3)
    assert np.isfinite(result["Y"]).all()


def test_zero_snr_with_zero_B_matches_unscaled_null():
    kwargs = dict(n=10, p=3, d_x=2, d_y=4, B=0)
    scaled = MultipleNetworksSampler(**kwargs, snr=0, rng=np.random.default_rng(304))
    raw = MultipleNetworksSampler(**kwargs, rng=np.random.default_rng(304))
    a, b = scaled.sample_latent(), raw.sample_latent()
    np.testing.assert_array_equal(a["Y"], b["Y"])
    np.testing.assert_array_equal(a["B"], np.zeros((4, 6)))


@pytest.mark.parametrize(
    "value", [-1, np.nan, np.inf, -np.inf, True, np.bool_(False), 1j, [1], [[1]], "bad"]
)
def test_invalid_snr_is_rejected(value):
    with pytest.raises(ValueError, match="snr"):
        MultipleNetworksSampler(5, 2, 1, 2, snr=value)


@pytest.mark.parametrize("target", [0, 1])
def test_finite_snr_requires_nonzero_noise_variance(target):
    with pytest.raises(ValueError, match="snr.*eps_variance"):
        MultipleNetworksSampler(5, 2, 1, 2, snr=target, eps_variance=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"B": 0},
        {"B": np.zeros((2, 2))},
        {"B": np.eye(2), "x_variance": 0},
        {"B": [[1.0, -1.0], [-2.0, 2.0]], "x_variance": np.ones((2, 2))},
    ],
)
def test_positive_snr_rejects_fixed_zero_signal(kwargs):
    with pytest.raises(ValueError, match="Positive snr requires.*signal variance"):
        MultipleNetworksSampler(5, 2, 1, 2, snr=1, **kwargs)


def test_positive_snr_rejects_a_sampled_zero_signal():
    sampler = MultipleNetworksSampler(5, 2, 1, 2, snr=1, b_mean=0, b_variance=0)
    with pytest.raises(ValueError, match="Positive snr requires.*signal variance"):
        sampler.sample_latent()


@pytest.mark.parametrize("network_class", [GaussianNetwork, BernoulliNetwork])
@pytest.mark.parametrize("target", [0, 0.5, 4])
def test_network_generators_forward_snr_and_return_effective_coefficients(
    network_class, target
):
    kwargs = dict(n=14, p=3, d_x=2, d_y=4, snr=target, eps_variance=2)
    network = network_class(**kwargs, rng=np.random.default_rng(305))
    reference = MultipleNetworksSampler(**kwargs, rng=np.random.default_rng(305))
    actual, expected = network.generate(), reference.sample_latent()
    assert network.snr == target
    np.testing.assert_array_equal(actual["B"], expected["B"])
    np.testing.assert_array_equal(actual["Y"], expected["Y"])
    assert np.sum(actual["B"] ** 2) / 8 == pytest.approx(target)
    for adjacency in [actual["A_Y"], *actual["A_X"]]:
        assert np.isfinite(adjacency).all()
        np.testing.assert_array_equal(adjacency, adjacency.T)
