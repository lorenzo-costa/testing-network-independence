import pytest
import sys
from pathlib import Path
import numpy as np
from src.dgp import GaussianNetwork
from scipy import stats
import sys
from pathlib import Path
from src.methods import RVPermutationTest, QAP
from src.solvers import ASE, MLE_gaussian, MLE_logistic
from scipy.sparse.linalg import eigsh
from src.metrics import rv_coefficient
from src.dgp import GaussianNetwork, BernoulliNetwork

@pytest.fixture
def rng():
    return np.random.default_rng(42)



# A simple valid statistic function for the behavioral test
# (RV coefficient numerator proxy: Trace(XX'YY'))
def inner_product_statistic(X, Y):
    # Center the variables to ensure we are measuring covariance/correlation
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)
    # A simple proxy for alignment is the Frobenius norm of X^T Y
    return np.linalg.norm(X_c.T @ Y_c)


def mock_solver(A, k, rng=None):
    return A[:A.shape[0]//2, :k], np.ones(k)

@pytest.mark.parametrize(
    "solver, npermutations", [
        (ASE, 10), 
        (ASE, 20),
        (mock_solver, 10), # use this to make sure it's not a solver problem
        (MLE_gaussian, 10), 
        (MLE_logistic, 10)
        ]
)
def test_rv_permutation_basic(solver, npermutations, rng):
    """
    Checks if RVPermutationTest runs without crashing.
    Checks output keys and value ranges.
    """
    
    data = {
        'A': np.zeros((10, 10)), 
        'B': np.zeros((10, 10)), 
        'X': rng.random((20, 5)), 
        'Z': rng.random((20, 5))  
    }
    
    # placeholder stat
    def dummy_stat(X, Y): return 0.5

    tester = RVPermutationTest(
        sigma=0, 
        solver=solver, 
        test_function=dummy_stat,
        npermutations=npermutations,
        rng=rng
    )
    
    tester.fit(data)
    out = tester.get_estimated()
    
    assert 'p-value' in out
    assert 'reject_null' in out
    assert 0.0 <= out['p-value'] <= 1.0
    assert isinstance(out['reject_null'], (bool, np.bool_))


@pytest.mark.parametrize(
    "solver, dgp", [
        (ASE, GaussianNetwork),
        (MLE_gaussian, GaussianNetwork),
        (MLE_logistic, BernoulliNetwork)
        ]
)
def test_rv_permutation_correlated(solver, dgp, rng):
    """
    Checks if the test correctly REJECTS the null hypothesis when the latent 
    positions X and Z are perfectly correlated.
    """
    n = 50
    
    data = dgp(n=n, k=3, sigma=1, rng=rng).generate()

    tester = RVPermutationTest(
        sigma=0,
        solver=solver, 
        test_function=rv_coefficient,
        npermutations=100,
        rng=rng
    )
    
    tester.fit(data)
    out = tester.get_estimated()
    
    assert out['p-value'] < 0.05
    assert out['reject_null'] is True

@pytest.mark.parametrize(
    "solver", [ASE, MLE_gaussian, MLE_logistic]
)
def test_rv_permutation_null(solver, rng):
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

def test_qap_basic(rng):
    """
    Basic smoke test:
    1. Checks if the QAP class instantiates and fits without error.
    2. Verifies the output dictionary keys and value ranges (p-value).
    """
    # Generate random data
    n = 20
    data = {
        'A': rng.random((n, n)),
        'B': rng.random((n, n)),
    }

    # Initialize QAP
    qap = QAP(sigma=0, npermutations=50, rng=rng)
    
    # Run fit
    qap.fit(data)
    results = qap.get_estimated()

    assert 'p-value' in results
    assert 'reject_null' in results
    assert 'null' in results    
    assert 0.0 <= results['p-value'] <= 1.0    
    assert results['null'] is True  # passed sigma=0


def test_qap_perfect_correlation(rng):
    """
    Behavioral Test:
    Checks if QAP correctly rejects the null hypothesis when A and B 
    are identical (perfect structural correlation).
    """
    n = 20
    A = rng.random((n, n))    
    A = (A + A.T) / 2
    
    data = {
        'A': A,
        'B': A.copy(), # Identical matrix
        'X': np.zeros((n, 2)),
        'Z': np.zeros((n, 2))
    }

    
    qap = QAP(sigma=1, npermutations=100, alpha=0.05, rng=rng)
    
    qap.fit(data)
    results = qap.get_estimated()

    assert results['p-value'] < 0.05
    assert bool(results['reject_null']) is True
    assert results['null'] is False # Since we passed sigma=1
        


