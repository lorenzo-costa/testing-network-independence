import pytest
import sys
from pathlib import Path
import numpy as np
from simulation_code.dgp import GaussianNetwork
from scipy import stats
import sys
from pathlib import Path
from simulation_code.methods import solve_independent
from scipy.sparse.linalg import eigsh

# add parent directory to Python path
parent_dir = Path.cwd().parent
sys.path.append(str(parent_dir))

rng = np.random.default_rng(42)

def solve_independent_old(A, k=2, rng=None, **kwargs):
    if rng is None:
        rng = np.random.default_rng()
    v0 = rng.normal(size=A.shape[0])
    evals, evectors = eigsh(A, k=k, which='LM', v0=v0)
    evals = np.maximum(evals-0.5, 0)
    xhat = evectors @ np.diag(np.sqrt(evals))
    return [xhat], [evals]


def test_solve_independent(A, k, rng=rng):
    
    rng = np.random.default_rng(42)
    xhat, evals = solve_independent(A, k=k, rng=rng)
    # check shapes
    assert len(xhat) == 1
    assert xhat[0].shape == (A.shape[0], k)
    assert len(evals) == 1
    assert evals[0].shape == (k,)
    
    rng = np.random.default_rng(42)
    xhat_old, evals_old = solve_independent_old(A, k=k, rng=rng)
    
    assert np.allclose(xhat[0], xhat_old[0])
    assert np.allclose(evals[0], evals_old[0])
