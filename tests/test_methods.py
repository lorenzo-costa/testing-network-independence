import pytest
import sys
from pathlib import Path
import numpy as np
from src.dgp import GaussianNetwork
from scipy import stats
import sys
from pathlib import Path
from src.methods import ASE
from scipy.sparse.linalg import eigsh

rng = np.random.default_rng(42)

def solve_independent_old(A, k=2, rng=None, **kwargs):
    if rng is None:
        rng = np.random.default_rng()
    v0 = rng.normal(size=A.shape[0])
    evals, evectors = eigsh(A, k=k, which='LM', v0=v0)
    evals = np.maximum(evals-0.5, 0)
    xhat = evectors @ np.diag(np.sqrt(evals))
    return [xhat], [evals]


@pytest.mark.parametrize(
    "n, k, sigma",
    [(100, 2, 1),
     (100, 2, 0),
     (100, 2, 0.5),
     (100, 5, 0.5),
     (50, 2, 0)]
)
def test_ASE(n, k, sigma, rng=rng):
    A, _, _, _ = GaussianNetwork(n=n, k=k, sigma=sigma, rng=rng).generate()

    rng = np.random.default_rng(42)
    xhat, evals = ASE(A, k=k, rng=rng)
    # check shapes
    assert len(xhat) == 1
    assert xhat[0].shape == (A.shape[0], k)
    assert len(evals) == 1
    assert evals[0].shape == (k,)
    
    rng = np.random.default_rng(42)
    xhat_old, evals_old = solve_independent_old(A, k=k, rng=rng)

    assert np.allclose(xhat[0], xhat_old[0], rtol=1e-5)
    assert np.allclose(evals[0], evals_old[0], rtol=1e-5)
