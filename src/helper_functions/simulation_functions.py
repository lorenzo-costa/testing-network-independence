import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# TODO:
# - this could be sped up by having dpg run once and then feed data to each arg combination
# (it has a specific name i don't remember not)
# - add intermediate save


def run_scenario(metrics, args, seed, method_params=None):
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
    rng = np.random.default_rng(seed)
    args["rng"] = rng

    if args.get("data") is None:
        dgp, solver = args["setup"]
        args["solver"] = solver
        dgp = dgp(**args)
        data = dgp.generate()
        args["dgp_name"] = dgp.get_name()
    else:
        data = args["data"]
        solver = ...  # i don't which placeholder value to use

    method = args["method"]
    force_k = args.get("force_k", None)
    if force_k is not None:
        args["true_k"] = args["k"]
        args["k"] = force_k
        method = method(k=force_k, **args)
    else:
        method = method(**args)

    args["method_name"] = method.get_name()

    method.fit(data, **(method_params if method_params else {}))
    results = method.get_estimated()

    density_A = (data["A"] == 0).sum() / data["A"].size
    density_B = (data["B"] == 0).sum() / data["B"].size

    out_metrics = {metric.get_name(): metric(results) for metric in metrics}

    out_metrics["args"] = args
    out_metrics["density"] = (density_A, density_B)
    return out_metrics


def run_scenario_wrapper(args):
    """Wrapper to unpack args for pool.map"""
    args, metrics, method_params, seed = args
    return run_scenario(metrics, args, seed=seed, method_params=method_params)


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
        (args, metrics, method_params) for i in range(nsim) for args in factorial_design
    ]

    total_scenarios = len(all_scenarios)

    child_seeds = rng.spawn(total_scenarios)

    all_scenarios_seed = [
        (*scenario, seed) for scenario, seed in zip(all_scenarios, child_seeds)
    ]

    # Shuffle scenarios for better parallelisation
    rng.shuffle(all_scenarios_seed)

    # Better chunk size: balance between overhead and load distribution
    chunk_size = max(1, total_scenarios // (n_jobs * 32))

    results = []
    with Pool(processes=n_jobs) as pool:
        with tqdm(total=total_scenarios, desc="Running scenarios") as pbar:
            # Use imap_unordered for better performance (order doesn't matter)
            for result in pool.imap_unordered(
                run_scenario_wrapper, all_scenarios_seed, chunksize=chunk_size
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
    """Run a simulation study.

    Parameters
    ----------
    nsim : _type_
        _description_
    factorial_design : _type_
        _description_
    metrics : _type_
        _description_
    method_params : _type_, optional
        _description_, by default None
    parallel : bool, optional
        _description_, by default False
    rng : _type_, optional
        _description_, by default None
    n_jobs : _type_, optional
        _description_, by default None
    data : dict, optional
       Dictionary containing keys 'estimate_latent_x', 'estimate_latent_y',
       'true_latent_x', and 'true_latent_y'.

    Returns
    -------
    _type_
        _description_
    """
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

    # for i in range(nsim):
    #     print(f"Simulation {i + 1} of {nsim}")
    #     for args in tqdm(factorial_design, desc="Running scenarios"):
    #         scenario_out = run_scenario(metrics, args, method_params=method_params)
    #         results.append(scenario_out)

    for i in range(nsim):
        sim_seeds = rng.spawn(len(factorial_design))

        for args, seed in zip(tqdm(factorial_design), sim_seeds):
            scenario_out = run_scenario(
                metrics, args, method_params=method_params, seed=seed
            )
            results.append(scenario_out)

    return results
