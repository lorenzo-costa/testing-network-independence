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
from src.methods import RVPermutationTest, LLKRatioTest, QAP, DiffusionCorrelation, PermutationTest, ObservedCVM
from src.solvers.binary_network import MLE_logistic
from src.solvers.weighted_network import MLE_gaussian, ASE
from src.solvers.MaMa_uuuuu import pgd_fit, pgd_fit_wrapper
from src.helper_functions.simulation_functions import run_simulation
from src.helper_functions.analyse_functions import aggregate_results
from src.metrics import rv_coefficient_adjusted
from src.helper_functions._metrics_helper import observed_cvm_dependency

import numpy as np
import pandas as pd
from scipy import stats
from itertools import product
from datetime import datetime
from functools import partial
import argparse
import h5py


def get_dist_string(dist_obj):
    name = dist_obj.dist.name

    # Combine positional args and keyword args into one list of strings
    args_str = [str(a) for a in dist_obj.args]
    kwds_str = [f"{k}={v}" for k, v in dist_obj.kwds.items()]

    # Join them together with commas
    params_join = ", ".join(args_str + kwds_str)

    return f"{name}({params_join})"
 
def load_hdf5(path):
    def read_obj(obj):

        # dataset
        if isinstance(obj, h5py.Dataset):
            return obj[()]

        # group
        if isinstance(obj, h5py.Group):
            out = {}

            # put attributes directly as keys
            out.update({k: v for k, v in obj.attrs.items()})

            for key, item in obj.items():
                out[key] = read_obj(item)

            return out

    with h5py.File(path, "r") as f:
        return {k: read_obj(v) for k, v in f.items()}

if __name__ == "__main__":
    nsim = 75
    n = [100, 200, 300]
    k = [3]
    rho = [0.2]
    alpha = [0.05]
    marginals = ['gaussian']
    edge_var = [1]

    method = [
        partial(RVPermutationTest, permutation_type="latent"),
        QAP,
        DiffusionCorrelation,
        partial(ObservedCVM, test_function=partial(observed_cvm_dependency, degree=2)),
        partial(ObservedCVM, test_function=partial(observed_cvm_dependency, degree=1)),
    ]

    npermutations = [200]
    metrics = [ComputeAll()]
    assortativity = [-0.8, -0.5, 0.5, 0.8]
    sparsity_bias = [0.3, 0.7]
    prob_switch = [0.2, 0.5]
    assignment_mode = ['random', 'correlated']
    block_probs_type = ['random', 'identical']
    
    setup = [ 
        (partial(GaussianNetwork, sbm=True), ASE),
        (partial(GaussianNetwork, sbm=True), ASE),
        (partial(BernoulliNetwork, sbm=True), ASE),
        (partial(BernoulliNetwork, sbm=True), ASE),
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
        "npermutations",
        "sparsity_bias",
        "prob_switch",
        "assignment_mode",
        "block_probs_type",
        "assortativity"
    ]

    param_values = product(
        setup, method, n, k, alpha, marginals, rho, edge_var, npermutations, sparsity_bias, prob_switch, assignment_mode, block_probs_type, assortativity
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

    # save first here so if the pandas columns crashes i still have results
    out.to_csv(file_name, index=False)
    
    
    out["n"] = out["args"].apply(lambda x: x["n"])
    out["k"] = out["args"].apply(lambda x: x["k"])
    out["edge_var"] = out["args"].apply(lambda x: x.get("edge_var", "NA"))
    out["dgp"] = out["args"].apply(lambda x: x["setup"][0].func.__name__)
    out["solver"] = out["args"].apply(lambda x: x["setup"][1].__name__)
    out['rho'] = out["args"].apply(lambda x: x.get("rho", "NA"))
    out["method"] = out["args"].apply(lambda x: x.get("method_name", "NA"))
    out["marginals"] = out["args"].apply(lambda x: x.get("marginals").name if hasattr(x.get("marginals"), "name") else "NA")
    out["copula"] = out["args"].apply(lambda x: x["setup"][0].keywords.get("copula_model", "NA"))
    out['sparsity_bias'] = out["args"].apply(lambda x: x.get("sparsity_bias", "NA"))
    out['assignment_mode'] = out["args"].apply(lambda x: x.get("assignment_mode", "NA"))
    out['block_probs_type'] = out["args"].apply(lambda x: x.get("block_probs_type", "NA"))

    out['column_covariance'] = out['args'].apply(lambda x: x.get("column_covariance", "NA"))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    out.to_csv(file_name, index=False)
