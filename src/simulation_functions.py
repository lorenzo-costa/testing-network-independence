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
    args : dict
        arguments for the simulation scenario
        
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
    
    out_metrics = {
        metric.get_name(): metric(truth=truth, estimated=estimated)
        for metric in metrics
    }
    out_metrics['args'] = args
    return out_metrics


def run_scenario_wrapper(args_and_metrics):
    """Wrapper to unpack args for pool.map"""
    args, metrics = args_and_metrics
    return run_scenario(metrics, args)


def run_simulation_parallel(nsim, factorial_design, metrics, 
                           rng=None, n_jobs=None):
    """Run simulations in parallel with improved batching.
    
    Key optimizations:
    1. Flatten all work upfront (nsim * len(factorial_design) scenarios)
    2. Use single Pool.map instead of nested loops
    3. Better chunk sizing for load balancing
    4. Reduced overhead from pool management
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if not isinstance(factorial_design, list):
        raise ValueError("factorial_design must be a list of tuples")
    
    if n_jobs is None:
        n_jobs = cpu_count()
    
    # Create all scenario arguments upfront (flattened structure)
    all_scenarios = [
        (args, metrics) 
        for _ in range(nsim) 
        for args in factorial_design
    ]
    
    total_scenarios = len(all_scenarios)
    
    # Better chunk size: balance between overhead and load distribution
    chunk_size = max(1, total_scenarios // (n_jobs * 4))
    
    results = []
    with Pool(processes=n_jobs) as pool:
        with tqdm(total=total_scenarios, desc="Running scenarios") as pbar:
            # Use imap_unordered for better performance (order doesn't matter)
            for result in pool.imap_unordered(
                run_scenario_wrapper, 
                all_scenarios, 
                chunksize=chunk_size
            ):
                results.append(result)
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
