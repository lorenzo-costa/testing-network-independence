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
from src.methods import RVPermutationTest, LLKRatioTest, QAP, DiffusionCorrelation, CanonicalCorrelationTest, FitIndependent
from src.solvers import MLE_gaussian, MLE_logistic, ASE
from src.simulation_functions import run_simulation
from src.analyse_functions import aggregate_results
from src.metrics import rv_coefficient_adjusted

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
    nsim = 100
    n = [50, 100, 150, 200, 250]
    k = [3]
    rho = [0]
    alpha = [0.05]
    marginals = [stats.norm]
    edge_var = [1]
    method = [
        partial(RVPermutationTest, permutation_type="latent"),
        partial(RVPermutationTest, permutation_type="observed"),
        LLKRatioTest,
        QAP,
        DiffusionCorrelation,
        partial(CanonicalCorrelationTest, permutation_type="latent"),
        partial(CanonicalCorrelationTest, permutation_type="observed")
    ]
    
    npermutations = [100]
    metrics = [ComputeAll()]
    approximation = ["F-distr"]

    setup = [
        (partial(GaussianNetwork, copula_model='gaussian'), MLE_gaussian),
        # (partial(GaussianNetwork, copula_model='frank'), MLE_gaussian),
        # (partial(GaussianNetwork, copula_model='clayton'), MLE_gaussian),
        # (partial(GaussianNetwork, copula_model='mixture_uniform', weights=[0.5, 0.5], correlations=[0.98, -0.98]), MLE_gaussian),

        (partial(BernoulliNetwork, copula_model='gaussian'), MLE_logistic),
        # (partial(BernoulliNetwork, copula_model='frank'), MLE_logistic),
        # (partial(BernoulliNetwork, copula_model='clayton'), MLE_logistic),
        # (partial(BernoulliNetwork, copula_model='mixture_uniform', weights=[0.5, 0.5], correlations=[0.98, -0.98]), MLE_logistic),
    ]
    
    rng = np.random.default_rng(1)    

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
        parallel=True,
    )

    out = pd.DataFrame(out)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    file_name = f"results/simulation_results_{timestamp}.csv"

    out.to_csv(file_name, index=False)

    out["n"] = out["args"].apply(lambda x: x["n"])
    out["k"] = out["args"].apply(lambda x: x["k"])
    out["edge_var"] = out["args"].apply(lambda x: x.get("edge_var", "NA"))
    out["approximation"] = out["args"].apply(lambda x: x.get("approximation", "NA"))
    out["dgp"] = out["args"].apply(lambda x: x.get("dgp_name", "NA"))
    out["solver"] = out["args"].apply(lambda x: x["setup"][1].__name__)
    out['rho'] = out["args"].apply(lambda x: x.get("rho", "NA"))

    out["method"] = out["args"].apply(lambda x: x.get("method_name", "NA"))

    out["marginals"] = out["args"].apply(lambda x: x.get("marginals", "NA"))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    out.to_csv(file_name, index=False)
