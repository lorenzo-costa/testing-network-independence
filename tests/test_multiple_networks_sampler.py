"""Contracts and distribution semantics for multiple-network latent positions."""

from itertools import product

import numpy as np
import pytest
from scipy import stats

from src.latent_samplers import MultipleNetworksSampler


@pytest.mark.parametrize(
    "n,p,d_x,d_y", [(12, 3, 2, 4), (1, 1, 1, 1), (1, 2, 1, 3), (8, 2, 3, 1)]
)
def test_latent_shapes_and_exact_keys(n, p, d_x, d_y):
    result = MultipleNetworksSampler(
        n, p, d_x, d_y, rng=np.random.default_rng(7)
    ).sample_latent()

    assert set(result) == {"Y", "X", "B"}
    assert isinstance(result["X"], list)
    assert len(result["X"]) == p
    assert all(isinstance(x, np.ndarray) and x.shape == (n, d_x) for x in result["X"])
    assert result["Y"].shape == (n, d_y)
    assert result["B"].shape == (d_y, p * d_x)


def test_fixed_B_and_zero_noise_preserve_block_order_exactly():
    mean = np.arange(1.0, 7.0)
    B = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [-1.0, 0.0, 2.0, 7.0, 0.0, 1.0]])
    result = MultipleNetworksSampler(
        5,
        3,
        2,
        2,
        B=B.tolist(),
        x_mean=mean,
        x_variance=0,
        eps_variance=0,
        rng=np.random.default_rng(8),
    ).sample_latent()

    for k, x in enumerate(result["X"]):
        np.testing.assert_array_equal(x, np.tile(mean[2 * k : 2 * (k + 1)], (5, 1)))
    np.testing.assert_array_equal(result["B"], B)
    np.testing.assert_array_equal(
        result["Y"], np.concatenate(result["X"], axis=1) @ B.T
    )


@pytest.mark.parametrize("distribution", ["gaussian", "multivariate_gaussian"])
def test_sampled_B_has_correct_shape_seed_and_constant_zero_variance(distribution):
    kwargs = dict(n=4, p=2, d_x=3, d_y=2, b_distribution=distribution)
    a = MultipleNetworksSampler(**kwargs, rng=np.random.default_rng(9)).sample_latent()
    b = MultipleNetworksSampler(**kwargs, rng=np.random.default_rng(9)).sample_latent()
    assert a["B"].shape == (2, 6)
    np.testing.assert_array_equal(a["B"], b["B"])

    constant = MultipleNetworksSampler(
        **kwargs, b_mean=-2.5, b_variance=0, rng=np.random.default_rng(10)
    ).sample_latent()
    np.testing.assert_array_equal(constant["B"], np.full((2, 6), -2.5))


def test_default_B_entries_have_standard_normal_moments():
    result = MultipleNetworksSampler(
        1, 1, 100, 100, rng=np.random.default_rng(11)
    ).sample_latent()
    assert abs(result["B"].mean()) < 0.04
    assert abs(result["B"].var() - 1) < 0.06
    assert abs(np.corrcoef(result["B"].reshape(-1, 2), rowvar=False)[0, 1]) < 0.04


def test_scalar_covariances_match_explicit_identity_matrices():
    kwargs = dict(n=10, p=2, d_x=2, d_y=3, B=np.zeros((3, 4)))
    scalar = MultipleNetworksSampler(
        **kwargs, x_variance=2.5, eps_variance=0.4, rng=np.random.default_rng(12)
    )
    matrix = MultipleNetworksSampler(
        **kwargs,
        x_variance=2.5 * np.eye(4),
        eps_variance=0.4 * np.eye(3),
        rng=np.random.default_rng(12),
    )
    np.testing.assert_array_equal(scalar.x_covariance, 2.5 * np.eye(4))
    np.testing.assert_array_equal(scalar.eps_covariance, 0.4 * np.eye(3))
    a, b = scalar.sample_latent(), matrix.sample_latent()
    np.testing.assert_array_equal(a["Y"], b["Y"])
    for x_a, x_b in zip(a["X"], b["X"]):
        np.testing.assert_array_equal(x_a, x_b)


def test_univariate_draws_use_square_root_of_variance_and_shared_rng():
    rng = np.random.default_rng(13)
    reference_rng = np.random.default_rng(13)
    sampler = MultipleNetworksSampler(
        4,
        2,
        1,
        1,
        x_mean=[1.0, -1.0],
        x_variance=4,
        eps_variance=9,
        b_mean=2,
        b_variance=16,
        x_distribution="gaussian",
        eps_distribution="gaussian",
        rng=rng,
    )
    expected_x = stats.norm.rvs(
        loc=[1.0, -1.0], scale=2, size=(4, 2), random_state=reference_rng
    )
    expected_b = stats.norm.rvs(loc=2, scale=4, size=(1, 2), random_state=reference_rng)
    expected_error = stats.norm.rvs(
        loc=0, scale=3, size=(4, 1), random_state=reference_rng
    )
    result = sampler.sample_latent()

    np.testing.assert_array_equal(np.concatenate(result["X"], axis=1), expected_x)
    np.testing.assert_array_equal(result["B"], expected_b)
    np.testing.assert_array_equal(
        result["Y"], expected_x @ expected_b.T + expected_error
    )
    assert rng.bit_generator.state == reference_rng.bit_generator.state


def test_full_covariances_control_cross_network_dependence_and_errors():
    x_covariance = np.array([[1.0, 0.75], [0.75, 1.0]])
    eps_covariance = np.array([[2.0, -0.8], [-0.8, 1.0]])
    result = MultipleNetworksSampler(
        30_000,
        2,
        1,
        2,
        B=np.zeros((2, 2)),
        x_mean=[1.0, -2.0],
        x_variance=x_covariance,
        eps_variance=eps_covariance,
        rng=np.random.default_rng(14),
    ).sample_latent()
    x = np.concatenate(result["X"], axis=1)
    np.testing.assert_allclose(x.mean(axis=0), [1.0, -2.0], atol=0.025)
    np.testing.assert_allclose(np.cov(x, rowvar=False), x_covariance, atol=0.04)
    np.testing.assert_allclose(result["Y"].mean(axis=0), 0, atol=0.025)
    np.testing.assert_allclose(
        np.cov(result["Y"], rowvar=False), eps_covariance, atol=0.04
    )
    joint_covariance = np.cov(np.column_stack((x, result["Y"])), rowvar=False)
    np.testing.assert_allclose(joint_covariance[:2, 2:], 0, atol=0.04)


@pytest.mark.parametrize("covariance", [np.zeros((2, 2)), np.ones((2, 2))])
def test_singular_covariances_are_supported(covariance):
    result = MultipleNetworksSampler(
        16,
        2,
        1,
        2,
        B=np.zeros((2, 2)),
        x_variance=covariance,
        eps_variance=covariance,
        rng=np.random.default_rng(15),
    ).sample_latent()
    np.testing.assert_allclose(result["X"][0], result["X"][1], atol=1e-7)
    np.testing.assert_allclose(result["Y"][:, 0], result["Y"][:, 1], atol=1e-7)
    if not covariance.any():
        np.testing.assert_array_equal(result["Y"], np.zeros((16, 2)))


def test_roundoff_in_covariance_is_tolerated_without_mutating_inputs():
    covariance = np.array([[1.0, 1.0 + 1e-12], [1.0, 1.0]])
    original = covariance.copy()
    sampler = MultipleNetworksSampler(
        3, 2, 1, 2, x_variance=covariance, eps_variance=covariance
    )
    assert np.isfinite(sampler.sample_latent()["Y"]).all()
    np.testing.assert_array_equal(covariance, original)


def test_sampling_does_not_mutate_or_alias_supplied_parameters():
    mean, B, covariance = np.arange(4.0), np.eye(4), np.eye(4)
    originals = [value.copy() for value in (mean, B, covariance)]
    sampler = MultipleNetworksSampler(
        3, 2, 2, 4, B=B, x_mean=mean, x_variance=covariance
    )
    result = sampler.sample_latent()
    result["B"][:] = 99
    result["X"][0][:] = 99
    for value, original in zip((mean, B, covariance), originals):
        np.testing.assert_array_equal(value, original)
    np.testing.assert_array_equal(sampler.sample_latent()["B"], originals[1])


def test_registry_has_exact_scipy_objects():
    assert MultipleNetworksSampler.distribution_registry == {
        "gaussian": stats.norm,
        "multivariate_gaussian": stats.multivariate_normal,
    }


@pytest.mark.parametrize(
    "x_dist,eps_dist,b_dist", product(["gaussian", "multivariate_gaussian"], repeat=3)
)
def test_all_distribution_combinations_preserve_singleton_shapes(
    x_dist, eps_dist, b_dist
):
    sampler = MultipleNetworksSampler(
        1,
        1,
        1,
        1,
        x_distribution=x_dist,
        eps_distribution=eps_dist,
        b_distribution=b_dist,
        rng=np.random.default_rng(16),
    )
    result = sampler.sample_latent()
    assert result["X"][0].shape == result["Y"].shape == result["B"].shape == (1, 1)


@pytest.mark.parametrize(
    "parameter", ["x_distribution", "eps_distribution", "b_distribution"]
)
@pytest.mark.parametrize("name", ["student_t", "invalid", None, []])
def test_unknown_distributions_are_validated_even_with_fixed_B(parameter, name):
    with pytest.raises(ValueError, match=parameter) as error:
        MultipleNetworksSampler(2, 1, 1, 1, B=[[1.0]], **{parameter: name})
    assert "gaussian" in str(error.value)
    assert "multivariate_gaussian" in str(error.value)


@pytest.mark.parametrize("kind", ["univariate", "multivariate"])
def test_registered_distribution_is_shared_by_all_draws(monkeypatch, kind):
    monkeypatch.setattr(
        MultipleNetworksSampler,
        "distribution_registry",
        MultipleNetworksSampler.distribution_registry.copy(),
    )
    monkeypatch.setattr(
        MultipleNetworksSampler,
        "_distribution_kinds",
        MultipleNetworksSampler._distribution_kinds.copy(),
    )
    rng = np.random.default_rng(17)
    calls = []
    delegate = stats.norm if kind == "univariate" else stats.multivariate_normal

    class RecordingDistribution:
        def rvs(self, **kwargs):
            calls.append(kwargs)
            return delegate.rvs(**kwargs)

    distribution = RecordingDistribution()
    MultipleNetworksSampler.register_distribution("recording", distribution, kind=kind)
    assert MultipleNetworksSampler.distribution_registry["recording"] is distribution
    sampler = MultipleNetworksSampler(
        4,
        2,
        2,
        3,
        x_variance=4,
        eps_variance=9,
        b_variance=16,
        x_distribution="recording",
        eps_distribution="recording",
        b_distribution="recording",
        rng=rng,
    )
    result = sampler.sample_latent()
    assert result["Y"].shape == (4, 3)
    assert len(calls) == 3
    assert all(call["random_state"] is rng for call in calls)
    for call, variance, dimension in zip(calls, [4, 16, 9], [4, 4, 3]):
        if kind == "multivariate":
            np.testing.assert_array_equal(call["cov"], variance * np.eye(dimension))
        else:
            np.testing.assert_array_equal(
                call["scale"], np.full(dimension, np.sqrt(variance))
            )


@pytest.mark.parametrize("parameter", ["n", "p", "d_x", "d_y"])
@pytest.mark.parametrize(
    "value", [True, np.bool_(False), 0, -1, 1.5, 2.0, np.nan, np.inf, "2"]
)
def test_dimensions_require_positive_integers(parameter, value):
    kwargs = dict(n=3, p=2, d_x=1, d_y=2)
    kwargs[parameter] = value
    with pytest.raises(ValueError, match=f"{parameter} must be a positive integer"):
        MultipleNetworksSampler(**kwargs)


def test_numpy_integer_dimensions_and_default_generator_are_supported():
    sampler = MultipleNetworksSampler(
        np.int64(3), np.int32(2), np.int64(1), np.int32(2)
    )
    assert isinstance(sampler.rng, np.random.Generator)
    assert sampler.sample_latent()["Y"].shape == (3, 2)


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("x_mean", np.nan),
        ("x_mean", np.inf),
        ("x_mean", [1.0]),
        ("x_mean", [[1.0, 2.0]]),
        ("x_mean", [1.0, np.nan]),
        ("x_mean", 1j),
        ("b_mean", np.nan),
        ("b_mean", np.inf),
        ("b_mean", [0.0]),
        ("b_variance", -1),
        ("b_variance", np.nan),
        ("b_variance", np.inf),
        ("b_variance", [1.0]),
        ("b_variance", np.eye(2)),
        ("B", [[1.0, 2.0]]),
        ("B", [1.0, 2.0, 3.0, 4.0]),
        ("B", [[np.inf, 0.0], [0.0, 1.0]]),
        ("B", [[np.nan, 0.0], [0.0, 1.0]]),
    ],
)
def test_invalid_means_variances_and_B_name_the_parameter(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        MultipleNetworksSampler(3, 2, 1, 2, **{parameter: value})


@pytest.mark.parametrize("parameter", ["x_variance", "eps_variance"])
@pytest.mark.parametrize(
    "value,message",
    [
        (-1, "nonnegative"),
        (np.nan, "finite"),
        (np.inf, "finite"),
        ([1.0, 1.0], "shape"),
        (np.eye(3), "shape"),
        ([[1.0, np.nan], [np.nan, 1.0]], "finite"),
        ([[1.0, np.inf], [np.inf, 1.0]], "finite"),
        ([[1.0, 0.5], [0.0, 1.0]], "symmetric"),
        ([[1.0, 2.0], [2.0, 1.0]], "positive semidefinite"),
    ],
)
def test_invalid_covariance_parameters(parameter, value, message):
    with pytest.raises(ValueError, match=f"{parameter}.*{message}"):
        MultipleNetworksSampler(3, 2, 1, 2, **{parameter: value})


@pytest.mark.parametrize(
    "parameter,distribution",
    [("x_variance", "x_distribution"), ("eps_variance", "eps_distribution")],
)
@pytest.mark.parametrize("covariance", [np.eye(2), [[1.0, 0.5], [0.5, 1.0]]])
def test_full_covariance_requires_multivariate_distribution(
    parameter, distribution, covariance
):
    with pytest.raises(
        ValueError, match=f"{parameter}.*choose a multivariate distribution"
    ):
        MultipleNetworksSampler(
            3, 2, 1, 2, **{parameter: covariance, distribution: "gaussian"}
        )


def test_unused_B_sampling_parameters_are_still_validated():
    with pytest.raises(ValueError, match="b_variance"):
        MultipleNetworksSampler(1, 1, 1, 1, B=[[1.0]], b_variance=-1)


def test_generator_advances_without_using_global_random_state():
    before = np.random.get_state()
    sampler = MultipleNetworksSampler(5, 2, 1, 2, rng=np.random.default_rng(18))
    first, second = sampler.sample_latent(), sampler.sample_latent()
    assert not np.array_equal(first["B"], second["B"])
    assert not np.array_equal(first["X"][0], second["X"][0])
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
