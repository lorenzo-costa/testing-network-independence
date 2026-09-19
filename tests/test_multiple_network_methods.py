"""Multiple-network DGP-to-test integration, including both permutation modes."""

import numpy as np
import pytest

from src.dgp import BernoulliNetwork, GaussianNetwork
from src.methods import CanonicalCorrelationTest, DistanceCorrelationTest, RVTest
from src.methods._base_class import BasePermutationTest
from src.solvers.weighted_network import ASE


def stochastic_solver(adjacency, k, rng):
    return adjacency[:, :k] + rng.normal(size=(len(adjacency), k)), np.ones(k)


def vector_statistic(y, x):
    return np.array([y[:, 0] @ x[:, 0], y[:, -1] @ x[:, -1]])


@pytest.mark.parametrize("network_class", [GaussianNetwork, BernoulliNetwork])
@pytest.mark.parametrize(
    "method_class", [CanonicalCorrelationTest, DistanceCorrelationTest, RVTest]
)
@pytest.mark.parametrize("permutation_type", ["latent", "adjacency"])
def test_public_methods_fit_multiple_networks(
    network_class, method_class, permutation_type
):
    data = network_class(
        n=18, p=3, d_x=1, d_y=2, rng=np.random.default_rng(90)
    ).generate()
    original_y = data["A_Y"].copy()
    original_x = [a.copy() for a in data["A_X"]]
    method = method_class(
        solver=ASE,
        d_y=2,
        d_x=1,
        npermutations=3,
        permutation_type=permutation_type,
        rng=np.random.default_rng(91),
    )
    method.fit(data)
    result = method.get_estimated()

    assert method.Yhat.shape == (18, 2)
    assert method.Xhat.shape == (18, 3)
    assert len(method.Xhat_blocks) == 3
    np.testing.assert_array_equal(
        method.Xhat, np.concatenate(method.Xhat_blocks, axis=1)
    )
    expected_statistic = method.test_function(method.Yhat, method.Xhat)
    assert result["test_stat"] == pytest.approx(expected_statistic)
    assert result["p-value"] >= 1 / 4
    assert result["p-value"] <= 1
    assert len(method.permutation_distribution) == 3
    np.testing.assert_array_equal(data["A_Y"], original_y)
    for actual, original in zip(data["A_X"], original_x):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("permutation_type", ["latent", "adjacency"])
def test_parallel_vector_tests_match_serial_with_estimated_latents(permutation_type):
    data = GaussianNetwork(12, 2, 2, 3, rng=np.random.default_rng(92)).generate()
    methods = []
    for n_jobs in (1, 2):
        method = BasePermutationTest(
            solver=stochastic_solver,
            d_y=3,
            d_x=2,
            test_function=vector_statistic,
            npermutations=7,
            permutation_type=permutation_type,
            n_jobs=n_jobs,
            rng=np.random.default_rng(93),
        )
        method.fit(data)
        methods.append(method)
    serial, parallel = methods
    for attr in (
        "Yhat",
        "Xhat",
        "permutation_indices",
        "permuted_statistics",
        "standardized_statistics",
    ):
        np.testing.assert_array_equal(getattr(serial, attr), getattr(parallel, attr))
    assert serial.pvalue == parallel.pvalue
    assert serial.rng.bit_generator.state == parallel.rng.bit_generator.state


def test_rv_asymptotic_fit_uses_both_new_latent_matrices():
    data = GaussianNetwork(30, 2, 1, 2, rng=np.random.default_rng(94)).generate()
    method = RVTest(
        solver=ASE,
        d_y=2,
        d_x=1,
        approximation="asymptotic",
        rng=np.random.default_rng(95),
    )
    method.fit(data)
    assert method.test_stat_estimate == pytest.approx(
        method.test_function(method.Yhat, method.Xhat)
    )
    assert np.isfinite(method.pvalue)
