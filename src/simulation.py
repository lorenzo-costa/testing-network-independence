from src.dgp import GaussianNetwork
from src.metrics import Rejection, FalseRejection, TrueRejection
from src.methods import RVPermutationTest, LLKRatioTest
from src.simulation_functions import run_simulation
from src.analyse_functions import aggregate_results

import numpy as np 
import pandas as pd
from scipy import stats
from itertools import product
from datetime import datetime
from functools import partial

if __name__ == '__main__':
    nsim = 10000
    n = [10, 50, 100, 150, 200, 250]
    k = [2, 5]
    sigma = [0, 0.5, 0.99]
    alpha = [0.05]
    npermutations = [2000]
    marginal_z = [stats.norm]
    marginal_x = [stats.norm]
    edge_var = [1, 5, 10]
    dgp = [GaussianNetwork]
    methods = [LLKRatioTest, RVPermutationTest]
    metrics = [FalseRejection(), TrueRejection(), Rejection()]
    approximation = ['F-distr']

    rng = np.random.default_rng(1)

    param_names = ["dgp", "method", "n", "k", "sigma", "alpha", "npermutations", 
                   "marginal_z", "marginal_x", "edge_var", "approximation"]
    param_values = product(dgp, methods, n, k, sigma, alpha, npermutations, 
                           marginal_z, marginal_x, edge_var, approximation)

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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    out.to_csv(f"results/simulation_results_{timestamp}.csv", index=False)