import pytest
import sys
from pathlib import Path
import numpy as np
from src.dgp import GaussianNetwork
from scipy import stats
import sys
from pathlib import Path
from src.metrics import Rejection, TrueRejection, FalseRejection, RelativeFrobeniusNorm, ComputeAll
from src.simulation_functions import run_simulation
from src.solvers import ASE, MLE_gaussian, MLE_logistic
from src.dgp import GaussianNetwork, BernoulliNetwork
from src.methods import RVPermutationTest, LLKRatioTest, QAP
from itertools import product


def test_simulation_run():
    # test only that simulation function actually runs
    nsim = 2
    n = [10, 20]
    k = [2, 3]
    sigma = [0, 0.5]
    alpha = [0.05]
    marginals = [stats.norm]
    edge_var = [1, 2]
    methods = [RVPermutationTest, LLKRatioTest, QAP]

    setups = [
        (GaussianNetwork, ASE),
        (BernoulliNetwork, MLE_logistic),
        (GaussianNetwork, MLE_gaussian)
    ]
    
    metrics = [ComputeAll()]
    
    approximation = ['F-distr', 'chi-sq']

    rng = np.random.default_rng(1)
    
    param_names = [
        "setup",
        "methods",
        "n",
        "k",
        "sigma",
        "alpha",
        "marginals",
        "edge_var",
        "approximation"
    ]

    param_values = product(
        setups, methods, n, k, sigma, alpha, marginals , edge_var, approximation
    )

    factorial_design = [dict(zip(param_names, v)) for v in param_values]

    out = run_simulation(
        nsim=nsim,
        metrics=metrics,
        factorial_design=factorial_design,
        rng=rng,
        parallel=False,
    )
    assert out is not None