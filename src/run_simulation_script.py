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
    marginals = ['gaussian', 'uniform -1 1', 'cauchy', 't 5', 'chi 5']
    edge_var = [1]

    method = [
        partial(RVPermutationTest, permutation_type="latent"),
        QAP,
        DiffusionCorrelation,
        partial(ObservedCVM, test_function=partial(observed_cvm_dependency, degree=2)),
        partial(ObservedCVM, test_function=partial(observed_cvm_dependency, degree=1)),
    ]

    npermutations = [200]
    df = [3]
    metrics = [ComputeAll()]
    approximation = ["F-distr"]
    
    column_covariance = [
        np.array([[10, 7, 0.0], [7, 5, 0.0], [0.0, 0.0, 1]]),
        np.eye(3)
        ]
    
    setup = [
        (partial(GaussianNetwork, copula_model='gaussian'), ASE),
        # (partial(GaussianNetwork, copula_model='clayton'), ASE),
        # (partial(GaussianNetwork, copula_model='gumbel'), ASE),
        (partial(GaussianNetwork, copula_model='student_t', df=3), ASE),
        (partial(GaussianNetwork, copula_model='mixture_uniform', weights=[0.5, 0.5], correlations=[0.5, -0.5]), ASE),
        
        (partial(BernoulliNetwork, copula_model='gaussian'), pgd_fit_wrapper),
        # (partial(BernoulliNetwork, copula_model='clayton'), pgd_fit_wrapper),
        # (partial(BernoulliNetwork, copula_model='gumbel'), pgd_fit_wrapper),
        (partial(BernoulliNetwork, copula_model='student_t', df=3), pgd_fit_wrapper),
        (partial(BernoulliNetwork, copula_model='mixture_uniform', weights=[0.5, 0.5], correlations=[0.5, -0.5]), pgd_fit_wrapper),
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
        "npermutations",
        "df",
        "column_covariance"
    ]

    param_values = product(
        setup, method, n, k, alpha, marginals, rho, edge_var, approximation, npermutations, df, column_covariance
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
    
    setup2 = [
        (partial(GaussianNetwork, copula_model='gaussian'), ASE),
        (partial(BernoulliNetwork, copula_model='gaussian'), pgd_fit_wrapper),
    ]
    rho2 = [0.0]
    
    rng2 = np.random.default_rng(2)    

    param_values2 = product(
        setup2, method, n, k, alpha, marginals, rho2, edge_var, approximation, npermutations, df, column_covariance
    )

    factorial_design2 = [dict(zip(param_names, v)) for v in param_values2]

    out2 = run_simulation(
        nsim=nsim,
        metrics=metrics,
        factorial_design=factorial_design2,
        rng=rng2,
        parallel=True,
    )
    
    out = pd.concat([out, pd.DataFrame(out2)], ignore_index=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    file_name = f"results/simulation_results_{timestamp}.csv"

    # save first here so if the pandas columns crashes i still have results
    out.to_csv(file_name, index=False)

    out["n"] = out["args"].apply(lambda x: x["n"])
    out["k"] = out["args"].apply(lambda x: x["k"])
    out["edge_var"] = out["args"].apply(lambda x: x.get("edge_var", "NA"))
    out["approximation"] = out["args"].apply(lambda x: x.get("approximation", "NA"))
    out["dgp"] = out["args"].apply(lambda x: x.get("dgp_name", "NA"))
    out["solver"] = out["args"].apply(lambda x: x.get('solver', "NA"))
    out['rho'] = out["args"].apply(lambda x: x.get("rho", "NA"))

    out["method"] = out["args"].apply(lambda x: x.get("method_name", "NA"))

    out["marginals"] = out["args"].apply(lambda x: x.get("marginals").name if hasattr(x.get("marginals"), "name") else "NA")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    out.to_csv(file_name, index=False)
