import numpy as np
import pytest
from scipy.stats import multivariate_normal

from src.dgp import BernoulliNetwork, GaussianNetwork
from src.latent_samplers import ConditionalGaussianSampler
from src.methods.ac_test import MultivariateACTest


def _parameters():
    return {
        "means": [[-1.0, 2.0, 3.0], [1.0, -2.0, 4.0]],
        "covariances": [
            [[2.0, 0.2, 0.0], [0.2, 1.5, 0.0], [0.0, 0.0, 0.5]],
            [[1.0, 0.0, 0.4], [0.0, 2.0, -0.2], [0.4, -0.2, 3.0]],
        ],
        "class_probabilities": [0.25, 0.75],
    }


def test_full_sample_and_conditional_law_match_specification():
    sampler = ConditionalGaussianSampler(
        n=20000, k=2, rng=np.random.default_rng(10), **_parameters()
    )
    Z, Y, X = sampler.sample_latent()

    assert Z.shape == (20000, 2)
    assert Y.shape == (20000, 1)
    assert X.shape == (20000, 1)
    assert sampler.is_null is False
    np.testing.assert_allclose(np.mean(X == 1), 0.75, atol=0.02)
    for category in range(2):
        mask = X[:, 0] == category
        joint = np.column_stack([Z[mask], Y[mask]])
        np.testing.assert_allclose(joint.mean(axis=0), sampler.means[category], atol=0.07)
        np.testing.assert_allclose(
            np.cov(joint, rowvar=False), sampler.covariances[category], atol=0.12
        )


def test_conditional_sampler_and_density_match_frozen_gaussians():
    sampler = ConditionalGaussianSampler(
        n=10, k=2, rng=np.random.default_rng(11), **_parameters()
    )
    X = np.array([[0], [1], [1], [0]])
    Y = np.array([[2.5], [4.2], [3.7], [1.9]])

    expected = np.array(
        [
            multivariate_normal.logpdf(
                Y[i],
                mean=sampler.means[category, 2:],
                cov=sampler.covariances[category, 2:, 2:],
            )
            for i, category in enumerate(X[:, 0])
        ]
    )
    np.testing.assert_allclose(sampler.conditional_logpdf(Y, X), expected)
    np.testing.assert_allclose(
        sampler.conditional_distribution(1).logpdf([4.2]), expected[1]
    )

    draws = sampler.sample(np.repeat([[1]], 20000, axis=0))
    assert draws.shape == (20000, 1)
    np.testing.assert_allclose(draws.mean(), 4.0, atol=0.06)
    np.testing.assert_allclose(draws.var(), 3.0, atol=0.12)


def test_zero_cross_covariance_is_conditional_null():
    parameters = _parameters()
    parameters["covariances"][1][0][2] = 0
    parameters["covariances"][1][2][0] = 0
    parameters["covariances"][1][1][2] = 0
    parameters["covariances"][1][2][1] = 0
    sampler = ConditionalGaussianSampler(n=10, k=2, **parameters)
    assert sampler.is_null is True


def test_multivariate_response_density_and_rng_replacement():
    parameters = _parameters()
    sampler = ConditionalGaussianSampler(n=10, k=1, ky=2, **parameters)
    X = np.array([0, 1, 0])
    Y = np.array([[2.0, 3.0], [-2.0, 4.0], [1.5, 2.5]])
    expected = [
        sampler.conditional_distribution(int(category)).logpdf(Y[i])
        for i, category in enumerate(X)
    ]
    np.testing.assert_allclose(sampler.conditional_logpdf(Y, X), expected)
    sampler.rng = np.random.default_rng(13)
    first = sampler.sample(X)
    sampler.rng = np.random.default_rng(13)
    np.testing.assert_array_equal(sampler.sample(X), first)


@pytest.mark.parametrize("network_type", [GaussianNetwork, BernoulliNetwork])
def test_network_pipeline_exposes_same_conditional_generator(network_type):
    dgp = network_type(
        n=40,
        k=2,
        conditional_gaussian=True,
        rng=np.random.default_rng(12),
        **_parameters(),
    )
    result = dgp.generate()
    assert result["A"].shape == (40, 40)
    assert result["Z"].shape == (40, 2)
    assert result["Y"].shape == (40, 1)
    assert result["X"].shape == (40, 1)
    assert isinstance(dgp.latent_sampler, ConditionalGaussianSampler)
    assert dgp.latent_sampler.sample(result["X"]).shape == result["Y"].shape


def test_generated_conditional_law_runs_model_x_test():
    dgp = GaussianNetwork(
        n=14,
        k=2,
        conditional_gaussian=True,
        rng=np.random.default_rng(14),
        **_parameters(),
    )
    data = dgp.generate()
    method = MultivariateACTest(
        use_true_latent=True,
        M=1,
        npermutations=3,
        test_type="model_X",
        conditional_generator=dgp.latent_sampler,
        rng=np.random.default_rng(15),
    )
    method.fit(data)
    assert len(method.null_distribution) == 3
    assert np.isfinite(method.null_distribution).all()


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"class_probabilities": [0.4, 0.4]}, "sum to 1"),
        ({"means": [[0, 0], [0, 0]]}, "means must have shape"),
        ({"covariances": [np.eye(3), np.zeros((3, 3))]}, "positive definite"),
        (
            {"covariances": [np.eye(3), [[1, 1, 0], [0, 1, 0], [0, 0, 1]]]},
            "symmetric",
        ),
    ],
)
def test_invalid_parameters_are_rejected(updates, message):
    parameters = _parameters()
    parameters.update(updates)
    with pytest.raises(ValueError, match=message):
        ConditionalGaussianSampler(n=10, k=2, **parameters)


def test_invalid_labels_and_response_shape_are_rejected():
    sampler = ConditionalGaussianSampler(n=10, k=2, **_parameters())
    with pytest.raises(ValueError, match="integer category labels"):
        sampler.sample([0.0, 1.0])
    with pytest.raises(ValueError, match="labels must lie"):
        sampler.sample([2])
    with pytest.raises(ValueError, match="Y must have shape"):
        sampler.conditional_logpdf(np.zeros((2, 2)), [0, 1])
