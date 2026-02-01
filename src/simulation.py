from src.dgp import GaussianNetwork, BernoulliNetwork
from src.metrics import Rejection, FalseRejection, TrueRejection, FalseAcceptance, TrueAcceptance, RelativeFrobeniusNorm
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

if __name__ == '__main__':
    print("Starting simulation logistic rdpg")
    
    nsim = 200
    n = [10, 25, 50, 100]
    k = [2, 5]
    sigma = [0, 0.01, 0.1, 0.5]
    alpha = [0.05]
    marginal_z = [stats.norm, stats.beta(a=2, b=5)]
    marginal_x = [stats.norm, stats.beta(a=2, b=5)]
    marginal_x_params = [{'a': 2, 'b': 5}]
    marginal_z_params = [{'a': 2, 'b': 5}]
    solver = [MLE_gaussian, MLE_logistic]
    edge_var = [1, 3, 5]
    dgp = [GaussianNetwork, BernoulliNetwork]
    methods = [RVPermutationTest, LLKRatioTest]
    metrics = [ComputeAll()]
    rv_coefficient_function = [rv_coefficient_adjusted]
    approximation = ['F-distr']
    npermutations = [1000]

    rng = np.random.default_rng(1)

    param_names = ["dgp", "method", "n", "k", "sigma", "alpha", "marginal_z", "marginal_x", 
                "edge_var", "solver", "approximation", "npermutations"]

    param_values = product(dgp, methods, n, k, sigma, alpha, marginal_z, marginal_x, 
                       edge_var, solver, approximation, npermutations)

    # 3. Zip keys with values to create dictionaries
    factorial_design = [dict(zip(param_names, v)) for v in param_values]

    out = run_simulation(nsim=nsim, 
                         metrics=metrics, 
                         factorial_design=factorial_design, 
                         rng=rng, 
                         parallel=True)

    out = pd.DataFrame(out)
    out['n'] = out['args'].apply(lambda x: x['n'])
    out['k'] = out['args'].apply(lambda x: x['k'])
    out['edge_var'] = out['args'].apply(lambda x: x.get('edge_var', 'NA'))
    out['approximation'] = out['args'].apply(lambda x: x.get('approximation', 'NA'))
    out['dgp'] = out['args'].apply(lambda x: x['dgp'].__name__)
    out['solver'] = out['args'].apply(lambda x: x.get('solver', 'NA').__name__)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    out.to_csv(f"results/simulation_results_{timestamp}.csv", index=False)