import importlib

import numpy as np
import pytest

from src.dgp import BernoulliNetwork, GaussianNetwork


def test_module_imports_network_generators():
    module = importlib.import_module("src.dgp")

    assert module.GaussianNetwork is GaussianNetwork
    assert module.BernoulliNetwork is BernoulliNetwork


def test_gaussian_network_uses_supplied_latent_positions():
    z = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    x = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    network = GaussianNetwork(
        n=3,
        k=2,
        X=x,
        Z=z,
        edge_var=0,
        copula_model="gaussian",
        marginals="gaussian",
        rng=np.random.default_rng(1),
    )

    result = network.generate()

    expected_a = z @ z.T
    expected_b = x @ x.T
    np.fill_diagonal(expected_a, 0)
    np.fill_diagonal(expected_b, 0)
    assert result["X"] is x
    assert result["Z"] is z
    np.testing.assert_allclose(result["A"], expected_a)
    np.testing.assert_allclose(result["B"], expected_b)


def test_bernoulli_network_uses_supplied_latent_positions():
    z = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    x = np.array([[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]])
    network = BernoulliNetwork(
        n=3,
        k=2,
        X=x,
        Z=z,
        rdpg=True,
        copula_model="gaussian",
        marginals="gaussian",
        rng=np.random.default_rng(2),
    )

    result = network.generate()

    assert result["X"] is x
    assert result["Z"] is z
    assert result["A"].shape == (3, 3)
    assert result["B"].shape == (3, 3)
    assert set(np.unique(result["A"])).issubset({0, 1})
    assert set(np.unique(result["B"])).issubset({0, 1})


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


@pytest.mark.parametrize("network_type", [GaussianNetwork, BernoulliNetwork])
def test_network_name_and_representation_include_selected_sampler(network_type):
    network = network_type(
        n=5,
        k=2,
        rdpg_distr="dirichlet",
        rng=np.random.default_rng(3),
    )

    assert "RDPG_dirichlet" in network.get_name()
    assert "n=5" in repr(network)
    assert "k=2" in repr(network)
