from src.dgp import GaussianNetwork, BernoulliNetwork
from src.metrics import Rejection, FalseRejection, TrueRejection, FalseAcceptance, TrueAcceptance, RelativeFrobeniusNorm
from src.methods import RVPermutationTest, LLKRatioTest, MLE_gaussian, MLE_logistic, ASE, FitIndependent
from src.simulation_functions import run_simulation
from src.analyse_functions import aggregate_results

import numpy as np 
import pandas as pd
from scipy import stats
from itertools import product
from datetime import datetime
from functools import partial

if __name__ == '__main__':
    print("Starting simulation logistic rdpg")
    
    nsim = 500
    n = [50, 100, 150, 200, 250]
    k = [2, 3, 5]
    sigma = [0, 0.01, 0.1, 0.5]
    alpha = [0.05]
    marginal_z = [stats.norm]
    marginal_x = [stats.norm]
    marginal_x_params = [{'a': 2, 'b': 5}]
    marginal_z_params = [{'a': 2, 'b': 5}]
    solver = [MLE_logistic, MLE_gaussian, ASE]
    edge_var = [1, 3, 5]
    dgp = [GaussianNetwork, BernoulliNetwork]
    methods = [FitIndependent]
    metrics = [RelativeFrobeniusNorm(gram_matrix=True)]
    approximation = ['F-distr']

    rng = np.random.default_rng(1)

    param_names = ["dgp", "method", "n", "k", "sigma", "alpha", "marginal_z", "marginal_x", 
                "edge_var", "solver", "approximation"]

    param_values = product(dgp, methods, n, k, sigma, alpha, marginal_z, marginal_x, 
                       edge_var, solver, approximation)

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
    #out['solver'] = out['args'].apply(lambda x: x.get('solver', 'NA').__name__)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    out.to_csv(f"results/simulation_results_{timestamp}.csv", index=False)