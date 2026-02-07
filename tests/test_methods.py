import pytest
import sys
from pathlib import Path
import numpy as np
from src.dgp import GaussianNetwork
from scipy import stats
import sys
from pathlib import Path
from src.methods import RVPermutationTest
from src.solvers import ASE, MLE_gaussian, MLE_logistic
from scipy.sparse.linalg import eigsh

rng = np.random.default_rng(42)


@pytest.mark.parametrize(
    "solver", [ASE, MLE_gaussian, MLE_logistic]
)
def test_rv_permutation_null(solver, rng=rng):
    data = {
        'A': np.random.rand(100, 100),
        'B': np.random.rand(100, 100),
        'Z': np.random.rand(100, 10),
        'X': np.random.rand(100, 10)
    }
    
    def constant_rv(X, Y): return 1.0

    tester = RVPermutationTest(
        sigma=0, 
        solver=solver, 
        test_function=constant_rv,
        npermutations=10,
        rng=rng
        )
    
    tester.fit(data)
    out = tester.get_estimated()
    
    assert out['p-value']==1.0
    assert out['reject_null']==False
        


