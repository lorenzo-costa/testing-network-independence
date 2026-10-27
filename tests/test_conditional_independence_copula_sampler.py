import importlib

import numpy as np
import pytest

from src.dgp import BernoulliNetwork, GaussianNetwork
from src.load_config import _resolve_copula_setup
from src.latent_samplers.conditional_independence_copula_sampler import (
    ConditionalIndependenceCopulaSampler,
)
from src.methods.ac_test import MultivariateACTest


MARGINALS = {"z": "gaussian", "y": "uniform -1 1"}


def test_module_imports_sampler():
    module = importlib.import_module(
        "src.latent_samplers.conditional_independence_copula_sampler"
    )
    assert (
        module.ConditionalIndependenceCopulaSampler
        is ConditionalIndependenceCopulaSampler
    )


@pytest.mark.parametrize(
    "copula_model, copula_params",
    [
        ("gaussian", None),
        ("student_t", {"df": 5}),
        ("clayton", None),
        ("gumbel", None),
    ],
)
def test_supported_copulas_return_expected_shapes(copula_model, copula_params):
    sampler = ConditionalIndependenceCopulaSampler(
        n=120,
        k=3,
        ky=2,
        C=3,
        rho=[0.0, 0.2, 0.4],
        marginals=MARGINALS,
        copula_model=copula_model,
        copula_params=copula_params,
        rng=np.random.default_rng(10),
    )

    Z, Y, X = sampler.sample_latent()

    assert Z.shape == (120, 3)
    assert Y.shape == (120, 2)
    assert X.shape == (120, 1)
    assert np.issubdtype(X.dtype, np.integer)
    assert set(np.unique(X)).issubset({0, 1, 2})
    assert np.isfinite(Z).all()
    assert np.isfinite(Y).all()


def test_scalar_rho_is_repeated_for_every_stratum():
    sampler = ConditionalIndependenceCopulaSampler(
        n=10,
        k=2,
        C=4,
        rho=0.35,
        marginals=MARGINALS,
    )
    np.testing.assert_allclose(sampler.rho, np.full(4, 0.35))


def test_gaussian_uses_the_matching_stratum_correlation():
    sampler = ConditionalIndependenceCopulaSampler(
        n=4000,
        k=1,
        ky=1,
        C=2,
        rho=[0.0, 0.8],
        marginals={"z": "gaussian", "y": "gaussian"},
        copula_model="gaussian",
        center_latent=False,
        rng=np.random.default_rng(20),
    )
    Z, Y, X = sampler.sample_latent()

    correlations = []
    for stratum in range(2):
        mask = X[:, 0] == stratum
        correlations.append(np.corrcoef(Z[mask, 0], Y[mask, 0])[0, 1])

    assert abs(correlations[0]) < 0.08
    assert correlations[1] > 0.72


STRATUM_MARGINALS = [
    {"z": "uniform 0 0.4", "y": "gaussian"},
    {"z": "uniform 0.6 1", "y": "pareto 2.5"},
]


def test_category_specific_marginals_generate_conditional_null_with_marginal_dependence():
    sampler = ConditionalIndependenceCopulaSampler(
        n=8000,
        k=1,
        C=2,
        rho=[0, 0],
        marginals_by_class=STRATUM_MARGINALS,
        rng=np.random.default_rng(28),
    )
    Z, Y, X = sampler.sample_latent()
    low = X[:, 0] == 0
    high = ~low

    assert sampler.center_latent is False
    assert sampler.is_null is True
    assert np.all((0 <= Z[low]) & (Z[low] <= 0.4))
    assert np.all((0.6 <= Z[high]) & (Z[high] <= 1))
    assert np.all(Y[high] >= 1)
    assert abs(Y[low].mean()) < 0.06
    assert np.corrcoef(Z[:, 0], Y[:, 0])[0, 1] > 0.3

    fixed_X = np.array([[1], [0], [1], [0]])
    sampled_Y = sampler.sample(fixed_X)
    assert sampled_Y.shape == (4, 1)
    assert np.all(sampled_Y[[0, 2]] >= 1)
    np.testing.assert_array_equal(sampler.X, X)


@pytest.mark.parametrize(
    "copula_model, copula_params",
    [
        ("gaussian", None),
        ("student_t", {"df": 5}),
        ("clayton", None),
        ("gumbel", None),
    ],
)
def test_category_specific_marginals_work_with_each_copula(copula_model, copula_params):
    sampler = ConditionalIndependenceCopulaSampler(
        n=100,
        k=1,
        C=2,
        rho=[0.1, 0.4],
        marginals_by_class=STRATUM_MARGINALS,
        copula_model=copula_model,
        copula_params=copula_params,
        rng=np.random.default_rng(29),
    )
    Z, Y, X = sampler.sample_latent()
    assert np.all((0 <= Z[X[:, 0] == 0]) & (Z[X[:, 0] == 0] <= 0.4))
    assert np.all((0.6 <= Z[X[:, 0] == 1]) & (Z[X[:, 0] == 1] <= 1))
    assert np.all(Y[X[:, 0] == 1] >= 1)
    assert np.isfinite(sampler.sample(X)).all()


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"marginals_by_class": STRATUM_MARGINALS[:1]}, "C=2"),
        ({"marginals_by_class": [STRATUM_MARGINALS[0], {"y": "gaussian"}]}, "marginals_by_class\\[1\\]"),
        ({"marginals_by_class": [STRATUM_MARGINALS[0], {"z": "invalid", "y": "gaussian"}]}, "Unknown distribution"),
        ({"marginals_by_class": STRATUM_MARGINALS, "marginals": MARGINALS}, "either marginals or marginals_by_class"),
        ({"marginals_by_class": STRATUM_MARGINALS, "means": [[0, 0], [1, 1]]}, "cannot be combined"),
    ],
)
def test_category_specific_marginal_validation(extra, message):
    with pytest.raises((TypeError, ValueError), match=message):
        ConditionalIndependenceCopulaSampler(n=10, k=1, C=2, **extra)


def test_bernoulli_rdpg_with_category_marginals_runs_model_x():
    dgp = BernoulliNetwork(
        n=18,
        k=1,
        conditional_copula="gaussian",
        rdpg=True,
        C=2,
        rho=[0, 0],
        marginals_by_class=STRATUM_MARGINALS,
        rng=np.random.default_rng(30),
    )
    data = dgp.generate()
    assert np.all((0 <= data["Z"]) & (data["Z"] <= 1))
    assert set(np.unique(data["A"])).issubset({0, 1})

    method = MultivariateACTest(
        use_true_latent=True,
        M=1,
        npermutations=3,
        test_type="model_X",
        conditional_generator=dgp.latent_sampler,
        rng=np.random.default_rng(31),
    )
    method.fit(data)
    assert len(method.null_distribution) == 3
    assert np.isfinite(method.null_distribution).all()


def test_gaussian_category_means_preserve_rho_and_create_marginal_dependence():
    sampler = ConditionalIndependenceCopulaSampler(
        n=12000,
        k=1,
        C=2,
        rho=[0, 0],
        means=[[-2, -2], [2, 2]],
        class_probabilities=[0.5, 0.5],
        rng=np.random.default_rng(21),
    )
    Z, Y, X = sampler.sample_latent()

    assert sampler.center_latent is False
    assert sampler.is_null is True
    assert np.corrcoef(Z[:, 0], Y[:, 0])[0, 1] > 0.75
    for category, expected_mean in enumerate(sampler.means):
        joint = np.column_stack((Z[X[:, 0] == category], Y[X[:, 0] == category]))
        np.testing.assert_allclose(joint.mean(axis=0), expected_mean, atol=0.08)
        assert abs(np.corrcoef(joint, rowvar=False)[0, 1]) < 0.06


def test_gaussian_category_covariances_control_full_joint_law():
    means = [[-1, 2, 3], [2, -1, -3]]
    covariances = [
        [[1.5, 0.3, 0], [0.3, 2, 0], [0, 0, 0.8]],
        [[2, 0.1, 0.6], [0.1, 1, -0.2], [0.6, -0.2, 1.5]],
    ]
    sampler = ConditionalIndependenceCopulaSampler(
        n=16000,
        k=2,
        C=2,
        means=means,
        covariances=covariances,
        rng=np.random.default_rng(22),
    )
    Z, Y, X = sampler.sample_latent()

    assert sampler.center_latent is False
    assert sampler.is_null is False
    for category in range(2):
        joint = np.column_stack((Z[X[:, 0] == category], Y[X[:, 0] == category]))
        np.testing.assert_allclose(joint.mean(axis=0), means[category], atol=0.08)
        np.testing.assert_allclose(np.cov(joint, rowvar=False), covariances[category], atol=0.1)


def test_sample_given_x_uses_gaussian_response_marginals():
    sampler = ConditionalIndependenceCopulaSampler(
        n=10,
        k=1,
        ky=2,
        C=2,
        means=[[0, 1, 2], [0, -2, 4]],
        covariances=[
            [[1, 0, 0], [0, 2, 0.5], [0, 0.5, 1]],
            [[1, 0.2, 0.3], [0.2, 1, -0.4], [0.3, -0.4, 3]],
        ],
        rng=np.random.default_rng(23),
    )
    X = np.repeat([[1]], 10000, axis=0)
    Y = sampler.sample(X)

    assert Y.shape == (10000, 2)
    np.testing.assert_allclose(Y.mean(axis=0), [-2, 4], atol=0.08)
    np.testing.assert_allclose(
        np.cov(Y, rowvar=False), [[1, -0.4], [-0.4, 3]], atol=0.1
    )
    assert sampler.X is None  # Conditional draws do not replace a stored data set.


def test_sample_given_x_reproduces_stratum_centering():
    sampler = ConditionalIndependenceCopulaSampler(
        n=10,
        k=1,
        C=2,
        rho=[0, 0.5],
        marginals={"z": "gaussian", "y": "exponential"},
        rng=np.random.default_rng(24),
    )
    X = np.array([0, 1, 1, 0, 1, 0])
    Y = sampler.sample(X)
    assert Y.shape == (6, 1)
    for category in range(2):
        np.testing.assert_allclose(Y[X == category].mean(), 0, atol=1e-12)


@pytest.mark.parametrize(
    "copula_model, copula_params",
    [
        ("gaussian", None),
        ("student_t", {"df": 5}),
        ("clayton", None),
        ("gumbel", None),
    ],
)
def test_sample_given_x_supports_every_copula_family(copula_model, copula_params):
    sampler = ConditionalIndependenceCopulaSampler(
        n=10,
        k=2,
        ky=2,
        C=2,
        rho=[0, 0.4],
        marginals={"z": "gaussian", "y": "uniform -1 1"},
        copula_model=copula_model,
        copula_params=copula_params,
        center_latent=False,
        rng=np.random.default_rng(27),
    )
    Y = sampler.sample(np.array([[0], [1], [1], [0], [1]]))
    assert Y.shape == (5, 2)
    assert np.isfinite(Y).all()


@pytest.mark.parametrize("X, message", [
    ([[0, 1]], "X must have shape"),
    ([0.0, 1.0], "integer category labels"),
    ([True, False], "integer category labels"),
    ([-1, 0], "labels must lie"),
    ([0, 2], "labels must lie"),
])
def test_sample_given_x_rejects_invalid_labels(X, message):
    sampler = ConditionalIndependenceCopulaSampler(n=10, k=1, C=2)
    with pytest.raises(ValueError, match=message):
        sampler.sample(X)


def test_conditional_copula_runs_model_x_test():
    dgp = GaussianNetwork(
        n=14,
        k=1,
        conditional_copula="gaussian",
        C=2,
        means=[[-2, -2], [2, 2]],
        rho=[0, 0],
        rng=np.random.default_rng(25),
    )
    data = dgp.generate()
    method = MultivariateACTest(
        use_true_latent=True,
        M=1,
        npermutations=3,
        test_type="model_X",
        conditional_generator=dgp.latent_sampler,
        rng=np.random.default_rng(26),
    )
    method.fit(data)
    assert len(method.null_distribution) == 3
    assert np.isfinite(method.null_distribution).all()


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"copula_model": "student_t", "means": [[0, 0]], "copula_params": {"df": 5}}, "gaussian copula"),
        ({"marginals": MARGINALS, "means": [[0, 0]]}, "gaussian marginals"),
        ({"means": [[0, 0]], "center_latent": True}, "center_latent must be False"),
        ({"means": [[0]]}, "means must have shape"),
        ({"covariances": [np.zeros((2, 2))]}, "positive definite"),
        ({"covariances": [np.eye(2)], "rho": 0.2}, "either covariances or rho"),
    ],
)
def test_explicit_gaussian_parameters_validate_configuration(extra, message):
    with pytest.raises(ValueError, match=message):
        ConditionalIndependenceCopulaSampler(n=10, k=1, C=1, **extra)


def test_centering_is_applied_separately_within_each_stratum():
    sampler = ConditionalIndependenceCopulaSampler(
        n=300,
        k=2,
        ky=2,
        C=3,
        rho=[0.1, 0.3, 0.5],
        marginals={"z": "exponential", "y": "uniform 2 5"},
        copula_model="gaussian",
        center_latent=True,
        rng=np.random.default_rng(30),
    )
    Z, Y, X = sampler.sample_latent()

    assert set(np.unique(X)) == {0, 1, 2}
    for stratum in range(3):
        mask = X[:, 0] == stratum
        np.testing.assert_allclose(Z[mask].mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(Y[mask].mean(axis=0), 0, atol=1e-12)


def test_custom_class_probabilities_are_used():
    sampler = ConditionalIndependenceCopulaSampler(
        n=40,
        k=1,
        C=3,
        rho=0,
        marginals=MARGINALS,
        class_probabilities=[0, 1, 0],
        rng=np.random.default_rng(40),
    )
    _, _, X = sampler.sample_latent()
    np.testing.assert_array_equal(X, np.ones((40, 1), dtype=int))


def test_p_is_an_alias_for_class_probabilities():
    sampler = ConditionalIndependenceCopulaSampler(
        n=20,
        k=1,
        C=2,
        rho=0,
        marginals=MARGINALS,
        p=[1, 0],
        rng=np.random.default_rng(41),
    )
    _, _, X = sampler.sample()
    np.testing.assert_array_equal(X, np.zeros((20, 1), dtype=int))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"C": 0}, "C must be a positive integer"),
        ({"C": 2, "rho": [0.1]}, "vector of length C=2"),
        ({"C": 2, "rho": [0.1, 1.1]}, "must lie in"),
        (
            {"C": 2, "class_probabilities": [0.2, 0.2]},
            "must sum to 1",
        ),
        (
            {"C": 2, "class_probabilities": [0.5, -0.5]},
            "nonnegative",
        ),
    ],
)
def test_common_parameters_are_validated(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ConditionalIndependenceCopulaSampler(
            n=10,
            k=2,
            marginals=MARGINALS,
            **kwargs,
        )


@pytest.mark.parametrize("copula_model", ["clayton", "gumbel"])
def test_archimedean_copulas_reject_negative_rho(copula_model):
    with pytest.raises(ValueError, match="does not support negative rho"):
        ConditionalIndependenceCopulaSampler(
            n=10,
            k=1,
            C=2,
            rho=[0.2, -0.1],
            marginals=MARGINALS,
            copula_model=copula_model,
        )


def test_student_t_requires_positive_df():
    with pytest.raises(ValueError, match="positive 'df'"):
        ConditionalIndependenceCopulaSampler(
            n=10,
            k=1,
            C=2,
            marginals=MARGINALS,
            copula_model="student_t",
            copula_params={"df": 0},
        )


@pytest.mark.parametrize(
    "marginals, error",
    [
        ("gaussian", TypeError),
        ({"z": "gaussian"}, ValueError),
        ({"z": "gaussian", "y": "gaussian", "x": "uniform"}, ValueError),
    ],
)
def test_marginals_require_new_dictionary_format(marginals, error):
    with pytest.raises(error, match="keys 'z' and 'y'"):
        ConditionalIndependenceCopulaSampler(
            n=10,
            k=1,
            C=2,
            marginals=marginals,
        )


def test_unknown_marginal_is_rejected_during_initialization():
    with pytest.raises(ValueError, match="Unknown distribution"):
        ConditionalIndependenceCopulaSampler(
            n=10,
            k=1,
            C=2,
            marginals={"z": "invalid", "y": "gaussian"},
        )


def test_unused_keyword_arguments_are_accepted():
    sampler = ConditionalIndependenceCopulaSampler(
        n=10,
        k=1,
        C=2,
        irrelevant_option="ignored",
        rdpg=False,
    )
    Z, Y, X = sampler.sample_latent()
    assert Z.shape == (10, 1)
    assert Y.shape == (10, 1)
    assert X.shape == (10, 1)


def test_gaussian_marginals_are_the_default():
    sampler = ConditionalIndependenceCopulaSampler(
        n=20,
        k=2,
        ky=1,
        C=2,
        rho=0.3,
        rng=np.random.default_rng(61),
    )
    Z, Y, X = sampler.sample_latent()

    assert sampler.marginals == {"z": "gaussian", "y": "gaussian"}
    assert Z.shape == (20, 2)
    assert Y.shape == (20, 1)
    assert X.shape == (20, 1)


def test_config_resolver_forwards_conditional_sampler_arguments():
    dgp_factory, _ = _resolve_copula_setup(
        {
            "dgp": "GaussianNetwork",
            "solver": "ASE",
            "conditional_copula": "gaussian",
            "C": 2,
            "class_probabilities": [0.25, 0.75],
            "center_latent": False,
        }
    )
    dgp = dgp_factory(
        n=20,
        k=1,
        rho=[0, 0.4],
        marginals={"z": "gaussian", "y": "gaussian"},
        rng=np.random.default_rng(60),
    )

    assert isinstance(dgp.latent_sampler, ConditionalIndependenceCopulaSampler)
    np.testing.assert_allclose(
        dgp.latent_sampler.class_probabilities,
        [0.25, 0.75],
    )
    assert dgp.latent_sampler.center_latent is False


def test_config_resolver_forwards_gaussian_category_parameters():
    means = [[-2, -2], [2, 2]]
    covariances = [np.eye(2).tolist(), np.eye(2).tolist()]
    dgp_factory, _ = _resolve_copula_setup(
        {
            "dgp": "GaussianNetwork",
            "solver": "ASE",
            "conditional_copula": "gaussian",
            "C": 2,
            "means": means,
            "covariances": covariances,
        }
    )
    dgp = dgp_factory(n=30, k=1, rng=np.random.default_rng(62))
    np.testing.assert_array_equal(dgp.latent_sampler.means, means)
    np.testing.assert_array_equal(dgp.latent_sampler.covariances, covariances)
    assert dgp.latent_sampler.is_null is True


def test_config_resolver_forwards_category_marginals():
    dgp_factory, _ = _resolve_copula_setup(
        {
            "dgp": "BernoulliNetwork",
            "solver": "ASE",
            "conditional_copula": "gaussian",
            "rdpg": True,
            "C": 2,
            "marginals_by_class": STRATUM_MARGINALS,
        }
    )
    dgp = dgp_factory(n=20, k=1, rho=[0, 0], rng=np.random.default_rng(32))
    assert dgp.latent_sampler.marginals_by_class == tuple(STRATUM_MARGINALS)
    assert dgp.latent_sampler.center_latent is False
