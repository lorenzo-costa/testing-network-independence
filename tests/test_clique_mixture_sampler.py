import numpy as np
import pytest

from src.dgp import BernoulliNetwork
from src.latent_samplers import CliqueMixtureSampler
from src.load_config import load_config, build_factorial_design
from src.helper_functions.simulation_functions import run_scenario
from src.metrics import ReturnMetric


@pytest.mark.parametrize('pi', [0, .5, 1])
def test_graph_and_regime(pi):
    dgp = BernoulliNetwork(n=40, k=2, clique_mixture=True, pi=pi,
                           rdpg=True, rng=np.random.default_rng(123))
    data = dgp.generate()
    sampler = dgp.latent_sampler
    z = sampler.community_labels
    expected = (z[:, None] == z[None, :]).astype(int)
    np.fill_diagonal(expected, 0)
    np.testing.assert_array_equal(data['A'], expected)
    np.testing.assert_array_equal(data['Z'].sum(axis=1), 1)
    assert dgp.is_null == (pi == 0)
    if pi == 1:
        np.testing.assert_array_equal(z, data['Y'].ravel())
    assert sampler.diagnostics['regime'] == sampler.B


def test_fresh_graph_regime_and_reproducibility():
    first = CliqueMixtureSampler(200, rng=18)
    second = CliqueMixtureSampler(200, rng=18)
    regimes = []
    for _ in range(30):
        a, b, _ = first.sample_latent()
        c, d, _ = second.sample_latent()
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        regimes.append(first.B)
    assert set(regimes) == {0, 1}


@pytest.mark.parametrize('kwargs', [{'k': 1}, {'ky': 2}, {'pi': -1}, {'pi': 2}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        CliqueMixtureSampler(20, **kwargs)


def test_config_uses_supplied_embedding_and_permutation_engine():
    cfg = load_config('config_clique_mixture.yaml')
    design = build_factorial_design(cfg)
    assert len(design) == 15
    calls = []

    def solver(A, k, rng=None):
        calls.append(A.copy())
        return np.column_stack((A.sum(axis=1), np.arange(len(A)))), None

    row = dict(design[0])
    row['setup'] = (row['setup'][0], solver)
    row['npermutations'] = 3
    result = run_scenario([ReturnMetric('p-value')], row, seed=1)
    assert len(calls) == 1
    assert result['ReturnMetric'] in [.25, .5, .75, 1.]
    assert result['diagnostics']['regime'] == 0
