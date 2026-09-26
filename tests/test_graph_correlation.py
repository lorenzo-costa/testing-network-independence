import importlib

import numpy as np
import pytest

from src.methods import GraphCorrelationTest, RVTest
from src.test_functions.graph_correlation import (
    _gcor_from_x_cache,
    _gcor_x_cache,
    gcor,
)


def _latent_data(seed=41, n=16):
    rng = np.random.default_rng(seed)
    return {
        "Y": rng.normal(size=(n, 3)),
        "X": [rng.normal(size=(n, 2)), rng.normal(size=(n, 2))],
    }


def test_cached_graph_correlation_matches_direct_statistic():
    data = _latent_data()
    x = np.concatenate(data["X"], axis=1)
    centered_x_gram, x_variance = _gcor_x_cache(x)

    assert _gcor_from_x_cache(data["Y"], centered_x_gram, x_variance) == pytest.approx(
        gcor(data["Y"], x)
    )


def test_graph_correlation_test_matches_uncached_permutation_test():
    data = _latent_data()
    options = {
        "use_true_latent": True,
        "npermutations": 12,
        "permutation_type": "latent",
        "n_jobs": 1,
    }
    cached = GraphCorrelationTest(rng=np.random.default_rng(82), **options)
    uncached = RVTest(
        test_function=gcor,
        approximation="permutation",
        rng=np.random.default_rng(82),
        **options,
    )

    cached.fit(data)
    uncached.fit(data)

    assert cached.test_stat_estimate == pytest.approx(uncached.test_stat_estimate)
    np.testing.assert_allclose(
        cached.permutation_distribution,
        uncached.permutation_distribution,
    )
    assert cached.pvalue == uncached.pvalue
    assert cached.reject_null == uncached.reject_null


def test_graph_correlation_test_centers_x_only_once(monkeypatch):
    graph_module = importlib.import_module("src.test_functions.graph_correlation")
    original = graph_module.u_center
    calls = 0

    def counting_u_center(kernel):
        nonlocal calls
        calls += 1
        return original(kernel)

    monkeypatch.setattr(graph_module, "u_center", counting_u_center)
    npermutations = 7
    method = GraphCorrelationTest(
        use_true_latent=True,
        npermutations=npermutations,
        rng=np.random.default_rng(83),
    )
    method.fit(_latent_data())

    assert calls == 1 + 1 + npermutations


def test_graph_correlation_is_available_to_configuration():
    from src.load_config import METHOD_REGISTRY

    assert METHOD_REGISTRY["GraphCorrelationTest"] is GraphCorrelationTest
