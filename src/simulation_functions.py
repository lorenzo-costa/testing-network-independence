import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import itertools
from multiprocessing import Pool, cpu_count
import os
from functools import partial 

# def run_scenario(metrics, args):
#     """Run a single scenario of the simulation.

#     Parameters
#     ----------
#     metrics : list of BaseMetric
#         list of metrics to compute
#     args : tuple
#         arguments for the simulation scenario. 
#         Needs to contain dpg (BaseDPG) at index 0 and 
#         method (BaseMethod) at index 1.

#     Returns
#     -------
#     dict
#         Dictionary containing the computed metrics.
#     """
#     dgp = args['dgp'](**args)
#     method = args['method'](**args)
#     data = dgp.generate()
#     method.fit(data)
#     estimated = method.get_estimated()
#     truth = method.get_truth()
#     # print(estimated, truth)
#     out_metrics = {}
#     for metric in metrics:
#         out_metrics[metric.get_name()] = metric(truth=truth, estimated=estimated)

#     out_metrics['args'] = args

#     return out_metrics

# def run_simulation_parallel(nsim, factorial_design, metrics, 
#                             rng=None, n_jobs=None):
#     if rng is None:
#         rng = np.random.default_rng()

#     if not isinstance(factorial_design, list):
#         raise ValueError("factorial_design must be a list of tuples")

#     results = []
#     for i in range(nsim):
#         print(f"Simulation {i+1} of {nsim}")
#         sim_args = [
#             args for args in factorial_design
#         ]
        
#         if n_jobs is None:
#             n_jobs = cpu_count()
#         chunk_size = max(1, len(sim_args) // (n_jobs * 10))
        
#         with Pool(processes=n_jobs) as pool:
#             with tqdm(total=len(sim_args), desc="Running scenarios") as pbar:
#                 worker_func = partial(run_scenario, metrics)
                
#                 for out_scenario in pool.imap(
#                     worker_func, sim_args, chunksize=chunk_size
#                     ):
#                     results.append(out_scenario)
#                     pbar.update(1)

#     return results

# def run_simulation(nsim, factorial_design, metrics, parallel=False, 
#                    rng=None, n_jobs=None):
#     if parallel:
#         return run_simulation_parallel(
#             nsim=nsim,
#             factorial_design=factorial_design,
#             metrics=metrics,
#             rng=rng,
#             n_jobs=n_jobs
#         )
        
#     if rng is None:
#         rng = np.random.default_rng()
    
#     results = []
#     for i in range(nsim):
#         print(f"Simulation {i+1} of {nsim}")
#         sim_args = [
#             args for args in factorial_design
#         ]
        
#         for args in tqdm(sim_args, desc="Running scenarios"):
#             scenario_out = run_scenario(metrics, args)
#             results.append(scenario_out)
        
#     return results

def run_scenario(metrics, args):
    """Run a single scenario of the simulation.
    Parameters
    ----------
    metrics : list of BaseMetric
        list of metrics to compute
    args : tuple
        arguments for the simulation scenario. 
        Needs to contain dpg (BaseDPG) at index 0 and 
        method (BaseMethod) at index 1.
    Returns
    -------
    dict
        Dictionary containing the computed metrics.
    """
    dgp = args['dgp'](**args)
    method = args['method'](**args)
    data = dgp.generate()
    method.fit(data)
    estimated = method.get_estimated()
    truth = method.get_truth()
    
    out_metrics = {}
    for metric in metrics:
        out_metrics[metric.get_name()] = metric(truth=truth, estimated=estimated)
    out_metrics['args'] = args
    return out_metrics


def run_single_simulation(args_tuple):
    """Run all scenarios for a single simulation iteration.
    
    Parameters
    ----------
    args_tuple : tuple
        Tuple of (sim_idx, factorial_design, metrics)
        
    Returns
    -------
    list
        List of results for all scenarios in this simulation
    """
    sim_idx, factorial_design, metrics = args_tuple
    # print(f"Simulation {sim_idx + 1}")
    results = []
    for args in factorial_design:
        scenario_out = run_scenario(metrics, args)
        results.append(scenario_out)
    return results


def run_simulation_parallel(nsim, factorial_design, metrics, 
                           rng=None, n_jobs=None):
    if rng is None:
        rng = np.random.default_rng()
    
    if not isinstance(factorial_design, list):
        raise ValueError("factorial_design must be a list of tuples")
    
    if n_jobs is None:
        n_jobs = cpu_count()
    
    # Create arguments for each simulation
    sim_args = [(i, factorial_design, metrics) for i in range(nsim)]
    
    chunk_size = max(1, nsim // (n_jobs * 10))
    
    results = []
    with Pool(processes=n_jobs) as pool:
        with tqdm(total=nsim, desc="Running simulations") as pbar:
            for sim_results in pool.imap(
                run_single_simulation, sim_args, chunksize=chunk_size
            ):
                results.extend(sim_results)
                pbar.update(1)
    
    return results


def run_simulation(nsim, factorial_design, metrics, parallel=False, 
                   rng=None, n_jobs=None):
    if parallel:
        return run_simulation_parallel(
            nsim=nsim,
            factorial_design=factorial_design,
            metrics=metrics,
            rng=rng,
            n_jobs=n_jobs
        )
    
    if rng is None:
        rng = np.random.default_rng()
    
    results = []
    for i in range(nsim):
        print(f"Simulation {i+1} of {nsim}")
        for args in tqdm(factorial_design, desc="Running scenarios"):
            scenario_out = run_scenario(metrics, args)
            results.append(scenario_out)
    
    return results
