import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from threadpoolctl import threadpool_limits

# TODO:
# - this could be sped up by having dpg run once and then feed data to each arg combination
# (it has a specific name i don't remember not)
# - add intermediate save


_worker_blas_limiter = None


def _initialize_parallel_worker(blas_threads):
    """Keep native BLAS calls within a simulation worker from oversubscribing."""
    global _worker_blas_limiter
    _worker_blas_limiter = threadpool_limits(limits=blas_threads, user_api="blas")


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
        latent_sampler = getattr(dgp, "latent_sampler", None)
        if latent_sampler is not None:
            args["x_network_correlation"] = getattr(
                latent_sampler,
                "x_network_correlation",
                None,
            )
            args["eps_distribution"] = getattr(
                latent_sampler,
                "eps_distribution",
                None,
            )
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
    if getattr(method, "approximation", None) == "asymptotic":
        args["asymptotic_null"] = method.asymptotic_null

    method.fit(data, **(method_params if method_params else {}))
    if getattr(method, "effective_gamma", None) is not None:
        args["cca_gamma"] = method.effective_gamma
    results = method.get_estimated()

    is_null = dgp.is_null if hasattr(dgp, "is_null") else None

    out_metrics = {
        metric.get_name(): metric(results, is_null=is_null) for metric in metrics
    }

    out_metrics["args"] = args
    return out_metrics


def run_scenario_wrapper(args):
    """Wrapper to unpack args for pool.map"""
    args, metrics, method_params, seed = args
    return run_scenario(metrics, args, seed=seed, method_params=method_params)


def _validate_shard(shard_index, num_shards):
    """Validate and return a zero-based shard index and positive shard count."""
    if isinstance(num_shards, bool) or not isinstance(num_shards, int):
        raise ValueError("num_shards must be a positive integer.")
    if num_shards < 1:
        raise ValueError("num_shards must be a positive integer.")
    if isinstance(shard_index, bool) or not isinstance(shard_index, int):
        raise ValueError("shard_index must be an integer.")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards.")
    return shard_index, num_shards


def _build_seeded_scenarios(
    nsim,
    factorial_design,
    metrics,
    method_params,
    rng,
    *,
    shard_index=0,
    num_shards=1,
    shuffle=False,
):
    """Build the global task list, then select one deterministic shard."""
    shard_index, num_shards = _validate_shard(shard_index, num_shards)
    all_scenarios = [
        (args, metrics, method_params) for _ in range(nsim) for args in factorial_design
    ]
    child_seeds = rng.spawn(len(all_scenarios))
    seeded_scenarios = [
        (*scenario, seed) for scenario, seed in zip(all_scenarios, child_seeds)
    ]
    if shuffle:
        rng.shuffle(seeded_scenarios)
    return seeded_scenarios[shard_index::num_shards], len(seeded_scenarios)


def run_simulation_parallel(
    nsim,
    factorial_design,
    metrics,
    method_params=None,
    rng=None,
    n_jobs=None,
    batch_size=32,
    blas_threads=1,
    shard_index=0,
    num_shards=1,
    result_callback=None,
):
    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(factorial_design, list):
        raise ValueError("factorial_design must be a list of tuples")

    if n_jobs is None:
        n_jobs = cpu_count()
    if isinstance(blas_threads, bool) or not isinstance(blas_threads, int):
        raise ValueError("blas_threads must be a positive integer.")
    if blas_threads < 1:
        raise ValueError("blas_threads must be a positive integer.")

    # Every shard constructs the same shuffled global task/seed list before
    # selecting a disjoint strided slice. This makes separately launched Slurm
    # jobs reproducible and prevents duplicated simulations.
    all_scenarios_seed, global_total = _build_seeded_scenarios(
        nsim,
        factorial_design,
        metrics,
        method_params,
        rng,
        shard_index=shard_index,
        num_shards=num_shards,
        shuffle=True,
    )
    total_scenarios = len(all_scenarios_seed)

    # Better chunk size: balance between overhead and load distribution
    chunk_size = max(1, total_scenarios // (n_jobs * batch_size))
    # chunk_size = max(1, total_scenarios // (n_jobs * 32))

    results = [] if result_callback is None else None
    with Pool(
        processes=n_jobs,
        initializer=_initialize_parallel_worker,
        initargs=(blas_threads,),
    ) as pool:
        description = (
            "Running scenarios"
            if num_shards == 1
            else f"Running shard {shard_index + 1}/{num_shards}"
        )
        with tqdm(total=total_scenarios, desc=description) as pbar:
            # Use imap_unordered for better performance (order doesn't matter)
            for result in pool.imap_unordered(
                run_scenario_wrapper, all_scenarios_seed, chunksize=chunk_size
            ):
                if result_callback is None:
                    results.append(result)
                else:
                    result_callback(result)
                pbar.update(1)

    if num_shards > 1:
        print(
            f"Shard {shard_index + 1}/{num_shards} completed "
            f"{total_scenarios} of {global_total} scenarios."
        )
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
    blas_threads=1,
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
    blas_threads : int, optional
        Maximum BLAS threads in each parallel simulation worker, by default 1.
        Ignored when ``parallel=False``.
    shard_index : int, optional
        Zero-based shard to execute, by default 0.
    num_shards : int, optional
        Number of disjoint shards in the global simulation task list, by default 1.
    result_callback : callable, optional
        Called once in the parent process for each completed result. When supplied,
        results are not accumulated in memory and this function returns ``None``.

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
            batch_size=batch_size,
            blas_threads=blas_threads,
            shard_index=shard_index,
            num_shards=num_shards,
            result_callback=result_callback,
        )

    if rng is None:
        rng = np.random.default_rng()

    scenarios, _ = _build_seeded_scenarios(
        nsim,
        factorial_design,
        metrics,
        method_params,
        rng,
        shard_index=shard_index,
        num_shards=num_shards,
        shuffle=num_shards > 1,
    )
    results = [] if result_callback is None else None
    for args, scenario_metrics, scenario_method_params, seed in tqdm(scenarios):
        result = run_scenario(
            scenario_metrics,
            args,
            method_params=scenario_method_params,
            seed=seed,
        )
        if result_callback is None:
            results.append(result)
        else:
            result_callback(result)

    return results
