import pytest
import numpy as np
from src.dgp import GaussianNetwork
from scipy import stats
import sys
from src.metrics import ComputeAll
from src.helper_functions.simulation_functions import run_simulation
from src.solvers import ASE, MLE_gaussian, MLE_logistic
from src.dgp import GaussianNetwork, BernoulliNetwork
from src.methods import RVPermutationTest, LLKRatioTest, QAP
from itertools import product


@pytest.mark.parametrize("parallel", [True, False])
def test_simulation_run(parallel):
    # test only that simulation function actually runs
    nsim = 2
    n = [10, 20]
    k = [2, 3]
    rho = [0, 0.5]
    alpha = [0.05]
    marginals = [stats.norm]
    edge_var = [1, 2]
    method = [RVPermutationTest, LLKRatioTest, QAP]
    npermutations = [100]

    setup = [
        (GaussianNetwork, ASE),
        (BernoulliNetwork, MLE_logistic),
        (GaussianNetwork, MLE_gaussian),
    ]

    metrics = [ComputeAll()]

    approximation = ["F-distr", "chi-sq"]

    rng = np.random.default_rng(1)

    param_names = [
        "setup",
        "method",
        "n",
        "k",
        "rho",
        "alpha",
        "marginals",
        "edge_var",
        "approximation",
        "npermutations",
    ]

    param_values = product(
        setup,
        method,
        n,
        k,
        rho,
        alpha,
        marginals,
        edge_var,
        approximation,
        npermutations,
    )

    factorial_design = [dict(zip(param_names, v)) for v in param_values]

    out = run_simulation(
        nsim=nsim,
        metrics=metrics,
        factorial_design=factorial_design,
        rng=rng,
        parallel=parallel,
    )
    assert out is not None
