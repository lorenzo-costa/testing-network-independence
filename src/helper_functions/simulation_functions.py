import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from copy import copy

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

    is_null = dgp.is_null if hasattr(dgp, "is_null") else None

    density_A = (data["A"] == 0).sum() / data["A"].size

    out_metrics = {
        metric.get_name(): metric(results, is_null=is_null) for metric in metrics
    }

    out_metrics["args"] = args
    out_metrics["density"] = density_A
    return out_metrics


def run_scenario_wrapper(args):
    """Wrapper to unpack args for pool.map"""
    args, metrics, method_params, seed = args
    return run_scenario(metrics, args, seed=seed, method_params=method_params)


def _validate_shard(shard_index, num_shards):
    if not isinstance(shard_index, int) or not isinstance(num_shards, int):
        raise TypeError("shard_index and num_shards must be integers")
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")


def _local_scenario_count(nsim, design_size, shard_index, num_shards):
    return len(range(shard_index, nsim * design_size, num_shards))


def _scenario_tasks(
    nsim,
    factorial_design,
    metrics,
    method_params,
    rng,
    shard_index,
    num_shards,
):
    """Yield this shard's deterministic subset of the global scenario grid."""
    total_scenarios = nsim * len(factorial_design)
    child_seeds = rng.spawn(total_scenarios)
    for global_index in range(shard_index, total_scenarios, num_shards):
        design_index = global_index % len(factorial_design)
        # ``run_scenario`` augments its arguments, so each task needs its own
        # top-level mapping even when it uses the same factorial-design row.
        yield (
            copy(factorial_design[design_index]),
            metrics,
            method_params,
            child_seeds[global_index],
        )


def run_simulation_parallel(
    nsim,
    factorial_design,
    metrics,
    method_params=None,
    rng=None,
    n_jobs=None,
    batch_size=32,
    shard_index=0,
    num_shards=1,
    result_callback=None,
):
    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(factorial_design, list):
        raise ValueError("factorial_design must be a list")
    if not factorial_design:
        return [] if result_callback is None else None
    _validate_shard(shard_index, num_shards)

    if n_jobs is None:
        n_jobs = cpu_count()
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    local_total = _local_scenario_count(
        nsim, len(factorial_design), shard_index, num_shards
    )
    chunk_size = max(1, local_total // (n_jobs * batch_size))
    tasks = _scenario_tasks(
        nsim,
        factorial_design,
        metrics,
        method_params,
        rng,
        shard_index,
        num_shards,
    )
    results = [] if result_callback is None else None
    with Pool(processes=n_jobs) as pool:
        with tqdm(total=local_total, desc="Running scenarios") as pbar:
            for result in pool.imap_unordered(
                run_scenario_wrapper, tasks, chunksize=chunk_size
            ):
                if result_callback is None:
                    results.append(result)
                else:
                    result_callback(result)
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
    batch_size=32,
    shard_index=0,
    num_shards=1,
    result_callback=None,
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
       Dictionary containing ``A``, optional ``Z``, and observed ``Y``.
    batch_size : int, optional
        Number of scenarios to process in each batch when parallelizing, by default 32

    Returns
    -------
    _type_
        _description_
    """
    if not isinstance(factorial_design, list):
        raise ValueError("factorial_design must be a list")
    if not factorial_design:
        return [] if result_callback is None else None
    _validate_shard(shard_index, num_shards)

    if parallel:
        return run_simulation_parallel(
            nsim=nsim,
            factorial_design=factorial_design,
            metrics=metrics,
            method_params=method_params,
            rng=rng,
            n_jobs=n_jobs,
            batch_size=batch_size,
            shard_index=shard_index,
            num_shards=num_shards,
            result_callback=result_callback,
        )

    if rng is None:
        rng = np.random.default_rng()

    local_total = _local_scenario_count(
        nsim, len(factorial_design), shard_index, num_shards
    )
    tasks = _scenario_tasks(
        nsim,
        factorial_design,
        metrics,
        method_params,
        rng,
        shard_index,
        num_shards,
    )
    results = [] if result_callback is None else None
    for task in tqdm(tasks, total=local_total, desc="Running scenarios"):
        scenario_out = run_scenario_wrapper(task)
        if result_callback is None:
            results.append(scenario_out)
        else:
            result_callback(scenario_out)

    return results
