import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
import itertools
from multiprocessing import Pool, cpu_count
import os
from functools import partial
from src.solvers import MLE_gaussian, MLE_logistic


# TODO:
# - this could be sped up by having dpg run once and then feed data to each arg combination
# (it has a specific name i don't remember not)
# - add intermediate save

def run_scenario(metrics, args, method_params=None):
    """Run a single scenario of the simulation.

    Parameters
    ----------
    metrics : list of BaseMetric
        list of metrics to compute
    args : dict
        arguments for the simulation scenario. Should contain 'setup' key with
        (dgp, method) tuple.

    Returns
    -------
    dict
        Dictionary containing the computed metrics.
    """
    dgp, solver = args["setup"]
    method = args["method"]

    dgp = dgp(**args)
    method = method(solver=solver, **args)
    
    args['method_name'] = method.get_name()
    args['dgp_name'] = dgp.get_name() 

    data = dgp.generate()
    method.fit(data, **(method_params if method_params else {}))
    results = method.get_estimated()

    out_metrics = {metric.get_name(): metric(results) for metric in metrics}

    out_metrics["args"] = args
    return out_metrics


def run_scenario_wrapper(args):
    """Wrapper to unpack args for pool.map"""
    args, metrics, method_params = args
    return run_scenario(metrics, args, method_params=method_params)


def run_simulation_parallel(
    nsim, factorial_design, metrics, method_params=None, rng=None, n_jobs=None
):
    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(factorial_design, list):
        raise ValueError("factorial_design must be a list of tuples")

    if n_jobs is None:
        n_jobs = cpu_count()

    # Create all scenario arguments upfront (flattened structure)
    all_scenarios = [
        (args, metrics, method_params) for _ in range(nsim) for args in factorial_design
    ]

    total_scenarios = len(all_scenarios)

    # Better chunk size: balance between overhead and load distribution
    chunk_size = max(1, total_scenarios // (n_jobs * 64))

    results = []
    with Pool(processes=n_jobs) as pool:
        with tqdm(total=total_scenarios, desc="Running scenarios") as pbar:
            # Use imap_unordered for better performance (order doesn't matter)
            for result in pool.imap_unordered(
                run_scenario_wrapper, all_scenarios, chunksize=chunk_size
            ):
                results.append(result)
                pbar.update(1)

    return results


def run_simulation(
    nsim,
    factorial_design,
    metrics,
    method_params=None,
    parallel=False,
    rng=None,
    n_jobs=None,
):
    if parallel:
        return run_simulation_parallel(
            nsim=nsim,
            factorial_design=factorial_design,
            metrics=metrics,
            method_params=method_params,
            rng=rng,
            n_jobs=n_jobs,
        )

    if rng is None:
        rng = np.random.default_rng()

    results = []
    for i in range(nsim):
        print(f"Simulation {i + 1} of {nsim}")
        for args in tqdm(factorial_design, desc="Running scenarios"):
            scenario_out = run_scenario(metrics, args, method_params=method_params)
            results.append(scenario_out)

    return results
