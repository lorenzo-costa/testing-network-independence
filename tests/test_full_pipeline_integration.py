import numpy as np
import pytest

from src.dgp import BernoulliNetwork, GaussianNetwork
from src.methods import (
    CanonicalCorrelationTest,
    DistanceCorrelationTest,
    RVTest,
)
from src.solvers.weighted_network import ASE


@pytest.mark.parametrize("network_type", [GaussianNetwork, BernoulliNetwork])
@pytest.mark.parametrize(
    "method_type, method_kwargs",
    [
        (RVTest, {}),
        (CanonicalCorrelationTest, {}),
        (DistanceCorrelationTest, {"test_method": "dcorr"}),
    ],
)
def test_multiple_network_global_independence_pipeline(
    network_type,
    method_type,
    method_kwargs,
):
    n = 20
    k = [2, 1, 3]
    data = network_type(
        n=n,
        k=k,
        rng=np.random.default_rng(101),
    ).generate()
    method = method_type(
        k=k,
        solver=ASE,
        npermutations=3,
        rng=np.random.default_rng(202),
        **method_kwargs,
    )

    method.fit(data)
    result = method.get_estimated()

    assert set(data) == {"Ay", "Ax", "Y", "X"}
    assert data["Ay"].shape == (n, n)
    assert [matrix.shape for matrix in data["Ax"]] == [(n, n), (n, n)]
    assert data["Y"].shape == (n, 2)
    assert [block.shape for block in data["X"]] == [(n, 1), (n, 3)]

    assert method.Yhat.shape == (n, 2)
    assert [block.shape for block in method.Xhat_blocks] == [(n, 1), (n, 3)]
    assert method.Xhat.shape == (n, 4)
    assert [block.shape for block in result["estimated_latent"]] == [
        (n, 2),
        (n, 1),
        (n, 3),
    ]
    assert len(method.permutation_distribution) == 3
    assert np.isfinite(result["test_stat"])
    assert 0.0 <= result["p-value"] <= 1.0
    assert isinstance(result["reject_null"], bool)
