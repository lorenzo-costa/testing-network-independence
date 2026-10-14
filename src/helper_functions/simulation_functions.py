"""Generate each data group once per replicate, then run its method rows."""

from collections import OrderedDict
from copy import deepcopy
from functools import partial
import hashlib
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from src.methods._base_class import BaseEstimationMethod


# Compatibility for hand-built designs. Designs from load_config carry explicit
# _generation_keys, and custom designs can provide those keys as well.
_LEGACY_METHOD_COLUMNS = frozenset({
    "method", "alpha", "npermutations", "df", "approximation",
    "use_true_latent", "force_k", "test_label",
})


def _canonical(value):
    """Return a stable, comparable description of a design value."""
    if isinstance(value, partial):
        return ("partial", _canonical(value.func), _canonical(value.args),
                _canonical(value.keywords or {}))
    if isinstance(value, dict):
        return ("dict", tuple(sorted((str(k), _canonical(v)) for k, v in value.items())))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_canonical(v) for v in value))
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.tobytes().hex())
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if value is None or isinstance(value, (str, int, float, bool, bytes)):
        return value
    if callable(value):
        if hasattr(value, "__qualname__"):
            return ("callable", value.__module__, value.__qualname__)
        if hasattr(value, "__dict__"):
            return ("callable-object", type(value).__module__,
                    type(value).__qualname__, _canonical(vars(value)))
    if hasattr(value, "__dict__"):
        return ("object", type(value).__module__, type(value).__qualname__,
                _canonical(vars(value)))
    raise TypeError(f"Cannot form a deterministic simulation key for {type(value)!r}")


def _generation_args(row):
    if "_generation_keys" in row:
        return {key: row[key] for key in row["_generation_keys"]}
    if "_generation_args" in row:
        return row["_generation_args"]
    return {key: value for key, value in row.items()
            if key not in _LEGACY_METHOD_COLUMNS and not key.startswith("_")}


def _data_key(row):
    generation = _generation_args(row)
    dgp, _ = row["setup"]
    return _canonical((dgp, {key: value for key, value in generation.items()
                             if key != "setup"}))


def _groups(factorial_design):
    """Group rows by their DGP callable and explicit generation parameters."""
    groups = OrderedDict()
    for row in factorial_design:
        if "setup" not in row:
            raise ValueError("Each factorial-design row must contain a setup pair")
        key = _data_key(row)
        groups.setdefault(key, []).append(row)
    return list(groups.items())


def _root_entropy(rng):
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    return tuple(int(word) for word in generator.integers(0, 2**32, size=4, dtype=np.uint32))


def _child_rng(entropy, replicate, data_key, purpose, detail=None):
    descriptor = repr((data_key, purpose, detail)).encode("utf-8")
    digest = hashlib.blake2b(descriptor, digest_size=16).digest()
    words = np.frombuffer(digest, dtype="<u4")
    seed = np.random.SeedSequence([*entropy, replicate, *(int(word) for word in words)])
    return np.random.default_rng(seed)


def _method_class(method):
    while isinstance(method, partial):
        method = method.func
    return method


def _method_option(row, name, default=None):
    if name in row:
        return row[name]
    method = row["method"]
    while isinstance(method, partial):
        if name in (method.keywords or {}):
            return method.keywords[name]
        method = method.func
    return default


def _embedding_key(row, data):
    _, solver = row["setup"]
    dimension = row.get("force_k", row.get("k"))
    if dimension is None:
        dimension = np.asarray(data["Z"]).shape[1]
    options = row.get("solver_kwargs", {})
    return _canonical((solver, dimension, options)), dimension, options


def _run_group(rows, metrics, method_params, entropy, replicate, data_key):
    """Generate one dataset and cache all embeddings used by its methods."""
    generation = _generation_args(rows[0])
    dgp_type, _ = rows[0]["setup"]
    dgp_kwargs = deepcopy({key: value for key, value in generation.items()
                           if key != "setup"})
    dgp = deepcopy(dgp_type)(**dgp_kwargs,
                             rng=_child_rng(entropy, replicate, data_key, "data"))
    data = dgp.generate()
    dgp_name = dgp.get_name()
    is_null = getattr(dgp, "is_null", None)
    density = (data["A"] == 0).sum() / data["A"].size
    embeddings = {}
    results = []
    method_occurrences = {}

    for row in rows:
        method_type = _method_class(row["method"])
        needs_embedding = (isinstance(method_type, type)
                           and issubclass(method_type, BaseEstimationMethod)
                           and not _method_option(row, "use_true_latent", False))
        method_data = deepcopy(data)  # No method can change another method's input.
        embedding_key = None
        if needs_embedding:
            embedding_key, dimension, options = _embedding_key(row, data)
            if embedding_key not in embeddings:
                _, solver = row["setup"]
                embeddings[embedding_key] = solver(
                    deepcopy(data["A"]), k=dimension,
                    rng=_child_rng(entropy, replicate, data_key, "embedding", embedding_key),
                    **options,
                )[0]
            method_data["estimated_Z"] = deepcopy(embeddings[embedding_key])

        # Include method and embedding settings in the child stream. Duplicate
        # identical rows receive distinct streams via their occurrence number.
        method_identity = _canonical({key: value for key, value in row.items()
                                      if key not in {"_generation_args", "_generation_keys"}})
        occurrence = method_occurrences.get(method_identity, 0)
        method_occurrences[method_identity] = occurrence + 1
        method_rng = _child_rng(entropy, replicate, data_key, "method",
                                (method_identity, occurrence))
        args = deepcopy({key: value for key, value in row.items()
                         if key not in {"_generation_args", "_generation_keys"}})
        _, solver = row["setup"]
        solver = deepcopy(solver)
        if row.get("solver_kwargs"):
            solver = partial(solver, **row["solver_kwargs"])
        args["solver"] = solver
        args["rng"] = method_rng
        args["dgp_name"] = dgp_name
        if row.get("force_k") is not None:
            args["true_k"] = args["k"]
            args["k"] = row["force_k"]
        method = args["method"](**args)
        args["method_name"] = method.get_name()
        method.fit(method_data, **(method_params or {}))
        estimated = method.get_estimated()
        out = {metric.get_name(): metric(estimated, is_null=is_null)
               for metric in metrics}
        out["args"] = args
        out["density"] = density
        results.append(out)
    return results


def run_scenario(metrics, args, seed, method_params=None):
    """Run one design row; use ``run_simulation`` to share data across rows."""
    if "data" in args:
        raise ValueError("Direct data input is unsupported; pass a DGP setup")
    entropy = _root_entropy(seed)
    return _run_group([args], metrics, method_params, entropy, 0, _data_key(args))[0]


def _group_wrapper(task):
    rows, metrics, method_params, entropy, replicate, data_key = task
    return _run_group(rows, metrics, method_params, entropy, replicate, data_key)


def _validate_shard(shard_index, num_shards):
    if not isinstance(shard_index, int) or not isinstance(num_shards, int):
        raise TypeError("shard_index and num_shards must be integers")
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")


def shard_row_count(nsim, factorial_design, shard_index=0, num_shards=1):
    """Count result rows assigned to a shard of whole data groups."""
    _validate_shard(shard_index, num_shards)
    groups = _groups(factorial_design)
    return sum(len(groups[index % len(groups)][1])
               for index in range(shard_index, nsim * len(groups), num_shards)) if groups else 0


def _group_tasks(nsim, groups, metrics, method_params, entropy, shard_index, num_shards):
    total = nsim * len(groups)
    for index in range(shard_index, total, num_shards):
        replicate, group_index = divmod(index, len(groups))
        data_key, rows = groups[group_index]
        yield (rows, metrics, method_params, entropy, replicate, data_key)


def run_simulation_parallel(
    nsim, factorial_design, metrics, method_params=None, rng=None, n_jobs=None,
    batch_size=32, shard_index=0, num_shards=1, result_callback=None,
):
    """Process independent replicate/data groups in worker processes."""
    return run_simulation(
        nsim, factorial_design, metrics, method_params=method_params,
        parallel=True, rng=rng, n_jobs=n_jobs, batch_size=batch_size,
        shard_index=shard_index, num_shards=num_shards,
        result_callback=result_callback,
    )


def run_simulation(
    nsim, factorial_design, metrics, method_params=None, parallel=False,
    rng=None, n_jobs=None, batch_size=32, shard_index=0, num_shards=1,
    result_callback=None,
):
    """Run all method rows for each independent replicate/data group.

    Shards partition whole groups, so all method rows for a generated dataset
    stay on one shard. A result callback receives individual rows as they finish.
    """
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

    groups = _groups(factorial_design)
    entropy = _root_entropy(rng)
    local_total = len(range(shard_index, nsim * len(groups), num_shards))
    tasks = _group_tasks(nsim, groups, metrics, method_params, entropy,
                         shard_index, num_shards)
    results = [] if result_callback is None else None

    def consume(group_results):
        for result in group_results:
            if result_callback is None:
                results.append(result)
            else:
                result_callback(result)

    if parallel:
        chunk_size = max(1, local_total // (n_jobs * batch_size))
        with Pool(processes=n_jobs) as pool:
            with tqdm(total=local_total, desc="Running data groups") as progress:
                for group_results in pool.imap_unordered(_group_wrapper, tasks,
                                                          chunksize=chunk_size):
                    consume(group_results)
                    progress.update(1)
    else:
        for task in tqdm(tasks, total=local_total, desc="Running data groups"):
            consume(_group_wrapper(task))
    return results
