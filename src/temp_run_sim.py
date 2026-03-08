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
    
    nsim = 50
    n = [100, 200, 300]
    k = [3]
    alpha = [0.05]
    marginals = ['gaussian', 'uniform -2 2', 'cauchy']
    edge_var = [1]
    method = [
        partial(RVPermutationTest, permutation_type="latent"),
        QAP,
        DiffusionCorrelation,
        partial(PermutationTest, permutation_type="latent", test_function=cvm_stat_multivariate),
    ]

    npermutations = [100]
    metrics = [ComputeAll()]
    approximation = ["F-distr"]
    
    dgp = ['bernoulli']
    rho = [.2, .5]
    scenarios = list(product(dgp, rho))
    scenarios = [('bernoulli', 0.5), ('bernoulli', 0.2)]
    for d, r in scenarios:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        file_name = f"results/{d}_{r}_{timestamp}.csv"
        print(file_name)
        
        if d == 'gaussian':
            d = GaussianNetwork
            solver = ASE
        elif d == 'bernoulli':
            d = BernoulliNetwork
            solver = pgd_fit_wrapper
        
        setup = [
            (partial(d, copula_model='gaussian'), solver),
            (partial(d, copula_model='clayton'), solver),
            (partial(d, copula_model='gumbel'), solver),
            (partial(d, copula_model='mixture_uniform', weights=[0.5, 0.5], correlations=[r, -r]), solver)
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
            setup, method, n, k, alpha, marginals, [r], edge_var, approximation, npermutations
        )

        factorial_design = [dict(zip(param_names, v)) for v in param_values]

        out = run_simulation(
            nsim=nsim,
            metrics=metrics,
            factorial_design=factorial_design,
            rng=rng,
            parallel=True,
        )

        out = pd.DataFrame(out)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        out["n"] = out["args"].apply(lambda x: x["n"])
        out["k"] = out["args"].apply(lambda x: x["k"])
        out["edge_var"] = out["args"].apply(lambda x: x.get("edge_var", "NA"))
        out["approximation"] = out["args"].apply(lambda x: x.get("approximation", "NA"))
        out["dgp"] = out["args"].apply(lambda x: x.get("dgp_name", "NA"))
        out["solver"] = out["args"].apply(lambda x: x["setup"][1].__name__)
        out['rho'] = out["args"].apply(lambda x: x.get("rho", "NA"))

        out["method"] = out["args"].apply(lambda x: x.get("method_name", "NA"))

        out["marginals"] = out["args"].apply(lambda x: x.get("marginals").name if hasattr(x.get("marginals"), "name") else "NA")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        out.to_csv(file_name, index=False)
