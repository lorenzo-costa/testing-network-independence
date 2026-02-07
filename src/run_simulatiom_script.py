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
from src.methods import RVPermutationTest, LLKRatioTest, FitIndependent
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
    nsim = 500
    n = [10, 25, 50, 100, 150]
    k = [2, 5, 7]
    sigma = [0, 0.01, 0.1, 0.5]
    alpha = [0.05]
    marginal_z = [stats.norm(loc=0, scale=1), stats.laplace(loc=0, scale=1)]
    edge_var = [1, 3, 5]
    dgp = [GaussianNetwork, BernoulliNetwork]
    methods = [RVPermutationTest, LLKRatioTest]
    metrics = [ComputeAll()]
    rv_coefficient_function = [rv_coefficient_adjusted]
    approximation = ["F-distr"]
    npermutations = [1000]

    rng = np.random.default_rng(1)

    param_names = [
        "dgp",
        "method",
        "n",
        "k",
        "sigma",
        "alpha",
        "marginal_z",
        "edge_var",
        "approximation",
        "npermutations",
    ]

    param_values = product(
        dgp,
        methods,
        n,
        k,
        sigma,
        alpha,
        marginal_z,
        edge_var,
        approximation,
        npermutations,
    )

    # 3. Zip keys with values to create dictionaries
    factorial_design = [dict(zip(param_names, v)) for v in param_values]

    out = run_simulation(
        nsim=nsim,
        metrics=metrics,
        factorial_design=factorial_design,
        rng=rng,
        parallel=True,
    )

    out = pd.DataFrame(out)
    out["n"] = out["args"].apply(lambda x: x["n"])
    out["k"] = out["args"].apply(lambda x: x["k"])
    out["edge_var"] = out["args"].apply(lambda x: x.get("edge_var", "NA"))
    out["approximation"] = out["args"].apply(lambda x: x.get("approximation", "NA"))
    out["dgp"] = out["args"].apply(lambda x: x["dgp"].__name__)
    out["solver"] = out["args"].apply(lambda x: x.get("solver", "NA").__name__)
    out["marginal_z"] = out["args"].apply(
        lambda x: get_dist_string(x.get("marginal_z", "NA"))
    )
    out["method"] = out["args"].apply(lambda x: x["method"].__name__)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    out.to_csv(f"results/simulation_results_{timestamp}.csv", index=False)
