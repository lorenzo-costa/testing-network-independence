
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import itertools
from multiprocessing import Pool, cpu_count
import os
from functools import partial 

def run_scenario(metrics, args):
    n, k, sigma, edge_var, dgp, step_size, lambda_reg, niters, solver = args
    A, B, Z, X = dgp(n=n, k=k, sigma=sigma, edge_var=edge_var)
    true_M = np.block([[Z @ Z.T, X @ Z.T], [Z @ X.T, X @ X.T]])
    M_path, sigma_path = solver(A, B, k, 
                                niters=niters, step_size=step_size, 
                                lambda_reg=lambda_reg) 
    estimated_M = M_path[-1]
    estimated_sigma = sigma_path[-1]
        
    err_M = []
    for metric in metrics['matrix']:
        err_M.append(metric(estimated_M, true_M))
    err_sigma = []
    for metric in metrics['scalar']:
        err_sigma.append(metric(estimated_sigma, sigma))

    out = {'err_M': err_M, 'err_sigma': err_sigma, 'n': n, 
           'k': k, 'sigma': sigma, 'edge_var': edge_var, 'dgp': dgp, 
           'step_size': step_size, 'lambda_reg': lambda_reg, 
           'niters': niters, 'solver': solver}
    return out

def run_simulation_parallel(nsim, n, k, sigma, edge_var, dgp, metrics, 
                   solver, step_size, lambda_reg=None, rng=None, 
                   n_jobs=None, niters=10):
    if rng is None:
        rng = np.random.default_rng()
    
    if not isinstance(n, list):
        n = [n]
    if not isinstance(k, list):
        k = [k]
    if not isinstance(sigma, list):
        sigma = [sigma]
    if not isinstance(edge_var, list):
        edge_var = [edge_var]
    if not isinstance(dgp, list):
        dgp = [dgp]
    # if not isinstance(metrics, list):
    #     metrics = [metrics]
    if not isinstance(solver, list):
        solver = [solver]
    if not isinstance(step_size, list):
        step_size = [step_size]
    if not isinstance(lambda_reg, list):
        lambda_reg = [lambda_reg]

    factorial_design = list(itertools.product(
        n, k, sigma, edge_var, dgp, step_size, lambda_reg, solver
    ))
    results = []
    for i in range(nsim):
        print(f"Simulation {i+1} of {nsim}")
        sim_args = [
            (n, k, sigma, edge_var, dgp_func, step_size_val, lambda_reg, niters, solver_func)
            for (n, k, sigma, edge_var, dgp_func, step_size_val, lambda_reg, solver_func) in factorial_design
        ]
        
        if n_jobs is None:
            n_jobs = cpu_count()
        chunk_size = max(1, len(sim_args) // (n_jobs * 10))
        
        with Pool(processes=n_jobs) as pool:
            with tqdm(total=len(sim_args), desc="Running scenarios") as pbar:
                worker_func = partial(run_scenario, metrics)
                
                for out_scenario in pool.imap(
                    worker_func, sim_args, chunksize=chunk_size
                    ):
                    results.append(out_scenario)
                    pbar.update(1)

    return results

def run_simulation(nsim, n, k, sigma, edge_var, dgp, metrics, 
                   solver, step_size, lambda_reg=None, parallel=False, 
                   rng=None, niters=10, n_jobs=None):
    if parallel:
        return run_simulation_parallel(
            nsim=nsim,
            n=n,
            k=k,
            sigma=sigma,
            edge_var=edge_var,
            dgp=dgp,
            metrics=metrics,
            solver=solver,
            step_size=step_size,
            lambda_reg=lambda_reg,
            rng=rng,
            n_jobs=n_jobs,
            niters=niters
        )
        
    if rng is None:
        rng = np.random.default_rng()
    
    if not isinstance(n, list):
        n = [n]
    if not isinstance(k, list):
        k = [k]
    if not isinstance(sigma, list):
        sigma = [sigma]
    if not isinstance(edge_var, list):
        edge_var = [edge_var]
    if not isinstance(dgp, list):
        dgp = [dgp]
    # if not isinstance(metrics, list):
    #     metrics = [metrics]
    if not isinstance(solver, list):
        solver = [solver]
    if not isinstance(step_size, list):
        step_size = [step_size]
    if not isinstance(lambda_reg, list):
        lambda_reg = [lambda_reg]

    factorial_design = list(itertools.product(
        n, k, sigma, edge_var, dgp, step_size, lambda_reg, solver
    ))
    out = []
    for i in range(nsim):
        print(f"Simulation {i+1} of {nsim}")
        sim_args = [
            (n, k, sigma, edge_var, dgp_func, step_size_val, lambda_reg, niters, solver_func)
            for (n, k, sigma, edge_var, dgp_func, step_size_val, lambda_reg, solver_func) in factorial_design
        ]
        results = []
        for args in tqdm(sim_args, desc="Running scenarios"):
            scenario_out = run_scenario(metrics, args)
            results.append(scenario_out)
        out.append(results)
        
    return out