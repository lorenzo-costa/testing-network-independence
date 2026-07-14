import numpy as np
import pytest

from src.dgp import BernoulliNetwork, GaussianNetwork


def test_gaussian_network_generates_only_a_from_z_and_keeps_y_observed():
    z = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    y = np.array([1.0, 2.0, 3.0])
    network = GaussianNetwork(
        n=3,
        k=2,
        ky=1,
        Y=y,
        Z=z,
        edge_var=0,
        copula_model="gaussian",
        marginals="gaussian",
        rng=np.random.default_rng(1),
    )

    result = network.generate()
    expected_a = z @ z.T
    np.fill_diagonal(expected_a, 0)

    assert set(result) == {"A", "Z", "Y", "X"}
    assert result["X"] is None
    np.testing.assert_allclose(result["A"], expected_a)
    np.testing.assert_array_equal(result["Z"], z)
    np.testing.assert_array_equal(result["Y"], y[:, None])


def test_bernoulli_network_generates_only_a_from_z():
    z = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    y = np.arange(6.0).reshape(3, 2)
    result = BernoulliNetwork(
        n=3,
        k=2,
        ky=2,
        Y=y,
        Z=z,
        rdpg=True,
        copula_model="gaussian",
        marginals="gaussian",
        rng=np.random.default_rng(2),
    ).generate()

    assert set(result) == {"A", "Z", "Y", "X"}
    assert result["X"] is None
    assert result["A"].shape == (3, 3)
    assert result["Y"].shape == (3, 2)
    assert set(np.unique(result["A"])).issubset({0, 1})


@pytest.mark.parametrize("network_type", [GaussianNetwork, BernoulliNetwork])
def test_default_covariate_dimension_is_one(network_type):
    result = network_type(
        n=8,
        k=3,
        copula_model="gaussian",
        marginals="gaussian",
        rng=np.random.default_rng(3),
    ).generate()
    assert result["Z"].shape == (8, 3)
    assert result["Y"].shape == (8, 1)


@pytest.mark.parametrize("network_type", [GaussianNetwork, BernoulliNetwork])
def test_generate_requires_a_latent_generation_method(network_type):
    network = network_type(n=5, k=2, marginals="gaussian")
    with pytest.raises(ValueError, match="No valid latent generation method specified"):
        network.generate()


@pytest.mark.parametrize("network_type", [GaussianNetwork, BernoulliNetwork])
def test_generate_rejects_unknown_rdpg_distribution(network_type):
    network = network_type(n=5, k=2, rdpg_distr="invalid")
    with pytest.raises(ValueError, match="Unknown rdpg_distr: invalid"):
        network.generate()
