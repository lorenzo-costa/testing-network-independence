import pytest
import sys
from pathlib import Path
import numpy as np
from src.dgp import GaussianNetwork, BernoulliNetwork
from scipy import stats
import sys
from pathlib import Path

# add parent directory to Python path
parent_dir = Path.cwd().parent
sys.path.append(str(parent_dir))


@pytest.mark.parametrize(
    "n, k, sigma, edge_var, marginals",
    [
        (100, 5, 0.5, 1, {'z': stats.norm, 'x': stats.norm}),
        (200, 10, 0.1, 0.5, {'z': stats.norm, 'x': stats.norm}),
        (300, 15, 0.2, 0.8, {'z': stats.norm, 'x': stats.norm}),
        (100, 5, 0.5, 1, {'z': stats.expon, 'x': stats.expon}),
        (100, 10, 0.1, 0.5, {'z': stats.chi2(df=2), 'x': stats.chi2(df=2)}),
        (100, 10, 0.1, 0.5, {'z': stats.norm, 'x': stats.chi2(df=2)}, {},),
        (100, 10, 0.1, 0.5, {'z': stats.chi2(df=2), 'x': stats.expon}),
        (100, 10, 0.1, 0.5, {'z': stats.expon, 'x': stats.norm}),
    ],
)
def test_GaussianNetwork(
    n, k, sigma, edge_var, marginals
):
    rng = np.random.default_rng(42)
    md = GaussianNetwork(
        n=n,
        k=k,
        sigma=sigma,
        marginals=marginals,
        rng=rng,
        edge_var=edge_var,
    )
    out = md.generate()
    A = out['A']
    B = out['B']
    X = out['X']
    Z = out['Z']
    assert A.shape == (n, n)
    assert B.shape == (n, n)
    assert X.shape == (n, k)
    assert Z.shape == (n, k)
    assert np.allclose(A, A.T), "Adjacency matrix A is not symmetric"
    assert np.allclose(B, B.T), "Adjacency matrix B is not symmetric"
    assert np.diag(A).sum() == 0, "Adjacency matrix A has non-zero diagonal"
    assert np.diag(B).sum() == 0, "Adjacency matrix B has non-zero diagonal"


@pytest.mark.parametrize(
    "n, k, sigma, edge_var, marginals",
    [
        (100, 5, 0.5, 1, {'z': stats.norm, 'x': stats.norm}),
        (200, 10, 0.1, 0.5, {'z': stats.norm, 'x': stats.norm}),
        (300, 15, 0.2, 0.8, {'z': stats.norm, 'x': stats.norm}),
        (100, 5, 0.5, 1, {'z': stats.expon, 'x': stats.expon}),
        (100, 10, 0.1, 0.5, {'z': stats.chi2(df=2), 'x': stats.chi2(df=2)}),
        (100, 10, 0.1, 0.5, {'z': stats.norm, 'x': stats.chi2(df=2)}, {},),
        (100, 10, 0.1, 0.5, {'z': stats.chi2(df=2), 'x': stats.expon}),
        (100, 10, 0.1, 0.5, {'z': stats.expon, 'x': stats.norm}),
    ],
)
def test_BernoulliNetwork(
    n, k, sigma, edge_var, marginals
):
    rng = np.random.default_rng(42)
    md = BernoulliNetwork(
        n=n,
        k=k,
        sigma=sigma,
        marginals=marginals,
        rng=rng,
        edge_var=edge_var,
    )
    out = md.generate()
    A = out['A']
    B = out['B']
    X = out['X']
    Z = out['Z']
    assert A.shape == (n, n)
    assert B.shape == (n, n)
    assert X.shape == (n, k)
    assert Z.shape == (n, k)
    assert np.allclose(A, A.T), "Adjacency matrix A is not symmetric"
    assert np.allclose(B, B.T), "Adjacency matrix B is not symmetric"
    assert set(np.unique(A)).issubset({0, 1}), "Adjacency matrix A is not binary"
    assert set(np.unique(B)).issubset({0, 1}), "Adjacency matrix B is not binary"
    assert np.diag(A).sum() == 0, "Adjacency matrix A has non-zero diagonal"
    assert np.diag(B).sum() == 0, "Adjacency matrix B has non-zero diagonal"