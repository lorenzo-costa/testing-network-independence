import numpy as np
import pytest

from src.load_config import _resolve_copula_setup
from src.latent_samplers.post_nonlinear_noise_sampler import (
    PostNonLinearNoiseSampler,
)


def test_sample_shapes_effects_functions_and_uniform_probabilities():
    sampler = PostNonLinearNoiseSampler(
        n=200,
        k=3,
        ky=2,
        C=4,
        rho=[-0.5, 0, 0.25, 0.75],
        center_latent=False,
        rng=np.random.default_rng(1),
    )
    Z, Y, X = sampler.sample_latent()

    assert Z.shape == (200, 3)
    assert Y.shape == (200, 2)
    assert X.shape == (200, 1)
    assert np.issubdtype(X.dtype, np.integer)
    assert sampler.a.shape == (4, 3)
    assert sampler.b.shape == (4, 2)
    assert sampler.stratum_effects.shape == (4, 5)
    assert sampler.nonlinear_functions_z.shape == (3,)
    assert sampler.nonlinear_functions_y.shape == (2,)
    assert set(sampler.nonlinear_functions_z).issubset(sampler._FUNCTION_NAMES)
    assert set(sampler.nonlinear_functions_y).issubset(sampler._FUNCTION_NAMES)
    np.testing.assert_allclose(sampler.class_probabilities, np.full(4, 0.25))
    assert np.isfinite(Z).all()
    assert np.isfinite(Y).all()


def test_unused_keyword_arguments_are_accepted():
    sampler = PostNonLinearNoiseSampler(
        n=10,
        k=1,
        C=2,
        irrelevant_option="ignored",
        copula_params={"df": 3},
    )
    Z, Y, X = sampler.sample_latent()
    assert Z.shape == (10, 1)
    assert Y.shape == (10, 1)
    assert X.shape == (10, 1)
    assert sampler.unused_kwargs["irrelevant_option"] == "ignored"


def test_scalar_rho_expands_and_constructs_copula_style_covariance():
    covariance_z = np.array([[2.0, 0.4], [0.4, 1.0]])
    covariance_y = np.array([[3.0]])
    sampler = PostNonLinearNoiseSampler(
        n=10,
        k=2,
        ky=1,
        C=3,
        rho=0.5,
        column_covariance_z=covariance_z,
        column_covariance_y=covariance_y,
    )

    np.testing.assert_allclose(sampler.rho, [0.5, 0.5, 0.5])
    expected_cross = (
        0.5
        * np.linalg.cholesky(covariance_z)
        @ np.array([[1.0], [0.0]])
        @ np.linalg.cholesky(covariance_y).T
    )
    np.testing.assert_allclose(
        sampler.error_covariances[0, :2, 2:], expected_cross
    )
    np.testing.assert_allclose(
        sampler.error_covariances[:, :2, :2],
        np.repeat(covariance_z[None, :, :], 3, axis=0),
    )


def test_custom_error_covariance_warns_and_takes_precedence_over_rho():
    covariance = np.array(
        [
            [1.0, 0.0, 0.3],
            [0.0, 1.0, -0.2],
            [0.3, -0.2, 1.0],
        ]
    )
    with pytest.warns(UserWarning, match="takes precedence"):
        sampler = PostNonLinearNoiseSampler(
            n=12,
            k=2,
            ky=1,
            C=2,
            rho=[0.9, -0.9],
            error_covariance=covariance,
        )

    np.testing.assert_allclose(sampler.error_covariances[0], covariance)
    np.testing.assert_allclose(sampler.error_covariances[1], covariance)
    assert sampler.is_null is False


def test_zero_cross_covariance_is_conditional_null():
    sampler = PostNonLinearNoiseSampler(
        n=20,
        k=2,
        ky=2,
        C=2,
        error_covariance=np.eye(4),
    )
    assert sampler.is_null is True


def test_centering_occurs_within_each_sampled_stratum_after_transform():
    sampler = PostNonLinearNoiseSampler(
        n=500,
        k=2,
        ky=2,
        C=3,
        rho=0.4,
        class_probabilities=[0.2, 0.3, 0.5],
        center_latent=True,
        rng=np.random.default_rng(4),
    )
    Z, Y, X = sampler.sample_latent()

    for stratum in range(3):
        mask = X[:, 0] == stratum
        assert mask.any()
        np.testing.assert_allclose(Z[mask].mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(Y[mask].mean(axis=0), 0, atol=1e-12)


def test_custom_probabilities_and_p_alias_are_supported():
    probabilities = [0.0, 0.2, 0.8]
    sampler = PostNonLinearNoiseSampler(
        n=100,
        k=1,
        C=3,
        p=probabilities,
        rng=np.random.default_rng(5),
    )
    _, _, X = sampler.sample()
    np.testing.assert_allclose(sampler.class_probabilities, probabilities)
    assert 0 not in X

    with pytest.raises(ValueError, match="Specify only one"):
        PostNonLinearNoiseSampler(
            n=10,
            k=1,
            C=2,
            p=[0.5, 0.5],
            class_probabilities=[0.5, 0.5],
        )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"rho": [0.1, 0.2]}, "length C=3"),
        ({"rho": 1.01}, r"\[-1, 1\]"),
        ({"class_probabilities": [0.2, 0.8]}, "length C=3"),
        ({"class_probabilities": [0.2, 0.2, 0.2]}, "sum to 1"),
        ({"stratum_covariance": np.eye(2)}, r"shape \(3, 3\)"),
        (
            {"error_covariance": np.array([[1, 2, 0], [2, 1, 0], [0, 0, 1]])},
            "positive semidefinite",
        ),
        (
            {"column_covariance_z": np.array([[1, 1], [1, 1]])},
            "positive definite",
        ),
        ({"cross_correlation_template": np.ones((1, 2))}, "must have shape"),
    ],
)
def test_invalid_parameters_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PostNonLinearNoiseSampler(n=10, k=2, ky=1, C=3, **kwargs)


def test_invalid_cross_template_is_rejected_when_joint_covariance_is_not_psd():
    with pytest.raises(ValueError, match="constructed error covariance"):
        PostNonLinearNoiseSampler(
            n=10,
            k=1,
            ky=1,
            C=2,
            rho=1,
            cross_correlation_template=np.array([[2.0]]),
        )


def test_config_resolver_selects_post_nonlinear_sampler():
    dgp_factory, _ = _resolve_copula_setup(
        {
            "dgp": "GaussianNetwork",
            "solver": "ASE",
            "post_nonlinear_noise": True,
            "C": 3,
            "rho": [0, 0.2, 0.4],
            "class_probabilities": [0.2, 0.3, 0.5],
            "center_latent": False,
        }
    )
    dgp = dgp_factory(n=30, k=2, ky=1, rng=np.random.default_rng(6))
    data = dgp.generate()

    assert isinstance(dgp.latent_sampler, PostNonLinearNoiseSampler)
    assert data["X"].shape == (30, 1)
    np.testing.assert_allclose(
        dgp.latent_sampler.class_probabilities, [0.2, 0.3, 0.5]
    )
    assert dgp.latent_sampler.center_latent is False
