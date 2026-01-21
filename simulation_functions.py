
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import itertools
from multiprocessing import Pool, cpu_count
import os
from functools import partial 

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
    dgp = args[0]
    method = args[1]

    data = dgp(**args)
    method = method(**args)
    method.fit(data)
    estimated = method.get_estimated()
    truth = method.get_truth()
    
    out_metrics = {}
    for metric in metrics:
        out_metrics[metric.__name__] = metric(estimated, truth)

    return out_metrics

def run_simulation_parallel(nsim, factorial_design, metrics, 
                            rng=None, n_jobs=None):
    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(factorial_design, list):
        raise ValueError("factorial_design must be a list of tuples")

    results = []
    for i in range(nsim):
        print(f"Simulation {i+1} of {nsim}")
        sim_args = [
            args for args in factorial_design
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
    
    out = []
    for i in range(nsim):
        print(f"Simulation {i+1} of {nsim}")
        sim_args = [
            args for args in factorial_design
        ]
        results = []
        for args in tqdm(sim_args, desc="Running scenarios"):
            scenario_out = run_scenario(metrics, args)
            results.append(scenario_out)
        out.append(results)
        
    return out