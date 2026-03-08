from src.dgp import GaussianNetwork, BernoulliNetwork
from src.metrics import (
    Rejection,
    FalseRejection,
    TrueRejection,
    FalseAcceptance,
    TrueAcceptance,
    RelativeFrobeniusNorm,
)
from src.metrics import ComputeAll
from src.methods import RVPermutationTest, PermutationTest, QAP, DiffusionCorrelation, CanonicalCorrelationTest, FitIndependent
from src.solvers.binary_network import MLE_logistic
from src.solvers.weighted_network import MLE_gaussian, ASE
from src.helper_functions.simulation_functions import run_simulation
from src.helper_functions.analyse_functions import aggregate_results
from src.metrics import rv_coefficient_adjusted
from src.solvers.MaMa_uuuuu import pgd_fit_wrapper
from src.helper_functions._metrics_helper import cvm_stat_multivariate


import numpy as np
import pandas as pd
from scipy import stats
from itertools import product
from datetime import datetime
from functools import partial



def get_dist_string(dist_obj):
    name = dist_obj.dist.name

    # Combine positional args and keyword args into one list of strings
    args_str = [str(a) for a in dist_obj.args]
    kwds_str = [f"{k}={v}" for k, v in dist_obj.kwds.items()]

    # Join them together with commas
    params_join = ", ".join(args_str + kwds_str)

    return f"{name}({params_join})"


if __name__ == "__main__":
    
    nsim = 2
    n = [100, 200]
    k = [3]
    alpha = [0.05]
    marginals = ['gaussian', 'uniform 0 10', 'cauchy']
    edge_var = [1]
    method = [
        partial(RVPermutationTest, permutation_type="latent"),
        # QAP,
        # DiffusionCorrelation,
        partial(PermutationTest, permutation_type="latent", test_function=cvm_stat_multivariate),
    ]

    npermutations = [100]
    metrics = [ComputeAll()]
    approximation = ["F-distr"]
    
    dgp = ['bernoulli', 'gaussian']
    rho = [.2]
    
    setup = [
        (partial(BernoulliNetwork, copula_model='gaussian', center_latent=True), pgd_fit_wrapper),
        (partial(BernoulliNetwork, copula_model='clayton', center_latent=True), pgd_fit_wrapper),
        (partial(BernoulliNetwork, copula_model='gumbel', center_latent=True), pgd_fit_wrapper),
        (partial(BernoulliNetwork, copula_model='mixture_uniform', weights=[0.5, 0.5], correlations=[98, -98], center_latent=True), pgd_fit_wrapper)
    ]

    rng = np.random.default_rng(2)    

    param_names = [
        "setup",
        "method",
        "n",
        "k",
        "alpha",
        "marginals",
        "rho",
        "edge_var",
        "approximation",
        "npermutations"
    ]

    param_values = product(
        setup, method, n, k, alpha, marginals, rho, edge_var, approximation, npermutations
    )

    factorial_design = [dict(zip(param_names, v)) for v in param_values]

    out = run_simulation(
        nsim=nsim,
        metrics=metrics,
        factorial_design=factorial_design,
        rng=rng,
        parallel=False,
    )
