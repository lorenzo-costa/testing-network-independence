from src.dgp import GaussianNetwork, BernoulliNetwork
from src.metrics import ReturnMetric
from src.metrics import ComputeAll
from src.methods import FitIndependent
from src.solvers.binary_network import MLE_logistic
from src.solvers.weighted_network import MLE_gaussian, ASE
from src.solvers.MaMa_uuuuu import pgd_fit, pgd_fit_wrapper
from src.helper_functions.simulation_functions import run_simulation
from src.helper_functions.analyse_functions import aggregate_results
from src.metrics import rv_coefficient_adjusted
from src.helper_functions._metrics_helper import cvm_stat_multivariate

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


if __name__ == "__main__":
    
    nsim = 50
    n = [100, 200, 300]
    k = [3]
    rho = [0.2]
    alpha = [0.05]
    marginals = ['gaussian', 'uniform -1 1', 't 5', 'chi 5', 'cauchy']
    edge_var = [1, 3]
    method = [FitIndependent]

    npermutations = [100]
    df = [3]
    metrics = [ReturnMetric()]
    
    approximation = ["F-distr"]

    setup = [
        (partial(GaussianNetwork, copula_model='gaussian'), ASE),
        (partial(GaussianNetwork, copula_model='clayton'), ASE),
        (partial(GaussianNetwork, copula_model='gumbel'), ASE),
        (partial(GaussianNetwork, copula_model='student_t', df=3), ASE),
        (partial(GaussianNetwork, copula_model='mixture_uniform', weights=[0.5, 0.5], correlations=[0.98, -0.98]), ASE),
        
        (partial(BernoulliNetwork, copula_model='gaussian'), pgd_fit_wrapper),
        (partial(BernoulliNetwork, copula_model='clayton'), pgd_fit_wrapper),
        (partial(BernoulliNetwork, copula_model='gumbel'), pgd_fit_wrapper),
        (partial(BernoulliNetwork, copula_model='student_t', df=3), pgd_fit_wrapper),
        (partial(BernoulliNetwork, copula_model='mixture_uniform', weights=[0.5, 0.5], correlations=[0.98, -0.98]), pgd_fit_wrapper),
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
        "df",
        "edge_var",
        "approximation",
        "npermutations"
    ]

    param_values = product(
        setup, method, n, k, alpha, marginals, rho, df, edge_var, approximation, npermutations
    )

    factorial_design = [dict(zip(param_names, v)) for v in param_values]

    out = run_simulation(
        nsim=nsim,
        metrics=metrics,
        factorial_design=factorial_design,
        rng=rng,
        parallel=True,
    )
    print(len(out))

    out = pd.DataFrame(out)

    rho = [0]
    setup = [
        (partial(GaussianNetwork, copula_model='gaussian'), ASE),
        (partial(BernoulliNetwork, copula_model='gaussian'), pgd_fit_wrapper),
    ]

    rng = np.random.default_rng(2)    
    param_values = product(
        setup, method, n, k, alpha, marginals, rho, df, edge_var, approximation, npermutations
    )

    factorial_design = [dict(zip(param_names, v)) for v in param_values]

    out2 = run_simulation(
        nsim=nsim,
        metrics=metrics,
        factorial_design=factorial_design,
        rng=rng,
        parallel=True,
    )

    out2 = pd.DataFrame(out2)
    print(len(out2))
    out = pd.concat([out, out2], ignore_index=True)

    filename = 'results/data.h5'

    metadata = pd.DataFrame()

    metadata["n"] = out["args"].apply(lambda x: x["n"])
    metadata["k"] = out["args"].apply(lambda x: x["k"])
    metadata["edge_var"] = out["args"].apply(lambda x: x.get("edge_var", "NA"))
    metadata["approximation"] = out["args"].apply(lambda x: x.get("approximation", "NA"))
    metadata["dgp"] = out["args"].apply(lambda x: x.get("dgp_name", "NA"))
    if metadata['dgp'][0] is not None:
        metadata['dgp_name'] =  [x.split('_')[0] for x in metadata['dgp']]
        metadata['copula_type'] =  [x.split('_')[1] for x in metadata['dgp']]

    metadata["solver"] = out["args"].apply(lambda x: x["setup"][1].__name__)
    metadata['rho'] = out["args"].apply(lambda x: x.get("rho", "NA"))
    metadata["marginals"] = out["args"].apply(lambda x: x.get("marginals", "NA"))
    metadata['df'] = out["args"].apply(lambda x: x.get("df", "NA"))


    with h5py.File(filename, 'w') as hf:
        for i in range(len(out)):
            grp = hf.create_group(f'iteration_{i}')
            estimated_latent = out['ReturnMetric'].iloc[i]['estimated']
            true_latent = out['ReturnMetric'].iloc[i]['truth']

            # Store the arrays separately
            grp.create_dataset('estimated_X', data=estimated_latent[0], compression="gzip")
            grp.create_dataset('estimated_Z', data=estimated_latent[1], compression="gzip")
            grp.create_dataset('true_X', data=true_latent[0], compression="gzip")
            grp.create_dataset('true_Z', data=true_latent[1], compression="gzip")

            temp = metadata.iloc[i]
            # 4. Store metadata as attributes
            for key, value in temp.items():
                grp.attrs[key] = value
