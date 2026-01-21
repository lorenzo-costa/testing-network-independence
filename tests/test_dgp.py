import pytest
import sys
from pathlib import Path
import numpy as np
from simulation_code.dgp import GaussianNetwork
from scipy.stats import norm, chi2, expon

import sys
from pathlib import Path

# add parent directory to Python path
parent_dir = Path.cwd().parent
sys.path.append(str(parent_dir))


@pytest.mark.parametrize(
    "n, k, sigma, edge_var, x_marginal, z_marginal, additional_args_x, additional_args_z",
    [(100, 5, 0.5, 1, norm, norm, {}, {}),
     (200, 10, 0.1, 0.5, norm, norm, {}, {}),
     (300, 15, 0.2, 0.8, norm, norm, {}, {}),
     (100, 5, 0.5, 1, expon, expon, {}, {}),
     (100, 10, 0.1, 0.5, chi2, chi2, {'df': 2}, {'df': 2}),
     (100, 10, 0.1, 0.5, norm, chi2, {}, {'df': 2}),
     (100, 10, 0.1, 0.5, chi2, expon, {'df': 2}, {}),
     (100, 10, 0.1, 0.5, expon, norm, {}, {})
]
)
def test_GaussianNetwork(n, k, sigma, 
                         edge_var, 
                         x_marginal, 
                         z_marginal,
                         additional_args_x,
                         additional_args_z):
    
    rng = np.random.default_rng(42)
    md = GaussianNetwork(n=n, k=k, sigma=sigma, 
                         marginal_z=z_marginal, 
                         marginal_x=x_marginal,
                         rng=rng, 
                         edge_var=edge_var,
                         marginal_z_params=additional_args_z,
                         marginal_x_params=additional_args_x)
    A, B, X, Z = md.generate()
    assert A.shape == (n, n)
    assert B.shape == (n, n)
    assert X.shape == (n, k)
    assert Z.shape == (n, k)