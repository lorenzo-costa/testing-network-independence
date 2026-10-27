"""Generate each data group once per replicate, then run its method rows."""

from collections import OrderedDict
from copy import deepcopy
from functools import partial
import hashlib
from multiprocessing import Pool, SimpleQueue, cpu_count
from queue import Queue as ThreadQueue
from threading import Thread

import numpy as np
from tqdm import tqdm

from src.methods._base_class import BaseEstimationMethod


def _canonical(value):
    """Return a stable, comparable description of a design value."""
    if isinstance(value, partial):
        return ("partial", _canonical(value.func), _canonical(value.args),
                _canonical(value.keywords or {}))
    if isinstance(value, dict):
        entries = ((_canonical(k), _canonical(v)) for k, v in value.items())
        return ("dict", tuple(sorted(entries, key=repr)))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_canonical(v) for v in value))
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.tobytes().hex())
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, float):
        return ("float", value.hex())
    if value is None or isinstance(value, (str, int, bool, bytes)):
        return (type(value).__name__, value)
    if callable(value):
        if hasattr(value, "__qualname__"):
            name = (value.__module__, value.__qualname__)
            # Separate independently created local functions/classes. Their
            # names alone do not identify captured state or mutable behavior.
            if "<locals>" in value.__qualname__ or value.__name__ == "<lambda>":
                return ("local-callable", *name, id(value))
            return ("callable", *name)
        if hasattr(value, "__dict__"):
            return ("callable-object", type(value).__module__,
                    type(value).__qualname__, id(value))
    if hasattr(value, "__dict__"):
        return ("object", type(value).__module__, type(value).__qualname__,
                id(value))
    raise TypeError(f"Cannot form a deterministic simulation key for {type(value)!r}")


def _generation_args(row):
    if "_generation_keys" in row:
        return {key: row[key] for key in row["_generation_keys"]}
    if "_generation_args" in row:
        return row["_generation_args"]
    # A lone legacy scenario needs no grouping decision. Preserve all of its
    # original arguments instead of guessing which custom fields affect a DGP.
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _data_key(row):
    generation = _generation_args(row)
    dgp, _ = row["setup"]
    return _canonical((dgp, {key: value for key, value in generation.items()
                             if key != "setup"}))


def _groups(factorial_design):
    """Group rows by their DGP callable and explicit generation parameters."""
    if len(factorial_design) > 1 and any(
        "_generation_keys" not in row and "_generation_args" not in row
        for row in factorial_design
    ):
        raise ValueError(
            "Multirow designs require _generation_keys or _generation_args "
            "on every row; DGP inputs cannot be inferred from method options"
        )
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


def _embedding_key(method, data):
    solver = method.solver
    dimension = method.k
    if dimension is None:
        if data.get("Z") is None:
            raise ValueError("Number of dimensions (k) must be specified.")
        dimension = np.asarray(data["Z"]).shape[1]
    return _canonical((solver, dimension)), dimension, solver


def _run_group(rows, metrics, method_params, entropy, replicate, data_key,
               result_callback=None):
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
    embeddings = {}
    results = []
    method_occurrences = {}

    for row in rows:
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
        method_callable = args["method"]
        if (
            isinstance(method_callable, partial)
            and (method_callable.keywords or {}).get("conditional_generator") == "from_dgp"
        ):
            if method_callable.keywords.get("test_type") != "model_X":
                raise ValueError("conditional_generator: from_dgp requires test_type: model_X")
            if not callable(getattr(getattr(dgp, "latent_sampler", None), "sample", None)):
                raise ValueError("The DGP must expose a latent_sampler with sample(X) for Model-X")
            args["method"] = partial(
                method_callable.func,
                *method_callable.args,
                **{
                    **method_callable.keywords,
                    "conditional_generator": deepcopy(dgp.latent_sampler),
                },
            )
        if row.get("force_k") is not None:
            args["true_k"] = args["k"]
            args["k"] = row["force_k"]
        method = args["method"](**args)
        args["method_name"] = method.get_name()
        method_data = deepcopy(data)  # No method can change another method's input.
        if (isinstance(method, BaseEstimationMethod)
                and not method.use_true_latent
                and "estimated_Z" not in method_data):
            embedding_key, dimension, effective_solver = _embedding_key(method, data)
            if embedding_key not in embeddings:
                embeddings[embedding_key] = effective_solver(
                    deepcopy(data["A"]), k=dimension,
                    rng=_child_rng(entropy, replicate, data_key, "embedding", embedding_key),
                )[0]
            method_data["estimated_Z"] = deepcopy(embeddings[embedding_key])
        method.fit(method_data, **(method_params or {}))
        estimated = method.get_estimated()
        density = (method_data["A"] == 0).sum() / method_data["A"].size
        out = {metric.get_name(): metric(estimated, is_null=is_null)
               for metric in metrics}
        out["args"] = args
        out["density"] = density
        if result_callback is None:
            results.append(out)
        else:
            result_callback(out)
    return results


def run_scenario(metrics, args, seed, method_params=None):
    """Run one design row; use ``run_simulation`` to share data across rows."""
    if "data" in args:
        raise ValueError("Direct data input is unsupported; pass a DGP setup")
    entropy = _root_entropy(seed)
    return _run_group([args], metrics, method_params, entropy, 0, _data_key(args))[0]


_RESULT_QUEUE = None


def _initialize_result_queue(queue):
    global _RESULT_QUEUE
    _RESULT_QUEUE = queue


def _group_wrapper(task):
    rows, metrics, method_params, entropy, replicate, data_key = task
    return _run_group(rows, metrics, method_params, entropy, replicate, data_key)


def _stream_group_wrapper(task):
    rows, metrics, method_params, entropy, replicate, data_key = task
    _run_group(rows, metrics, method_params, entropy, replicate, data_key,
               result_callback=lambda result: _RESULT_QUEUE.put(("result", result)))
    _RESULT_QUEUE.put(("done", None))


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
        if result_callback is None:
            with Pool(processes=n_jobs) as pool:
                with tqdm(total=local_total, desc="Running data groups") as progress:
                    for group_results in pool.imap_unordered(_group_wrapper, tasks,
                                                              chunksize=chunk_size):
                        consume(group_results)
                        progress.update(1)
        else:
            # SimpleQueue sends each row synchronously. The forwarding thread
            # lets the parent receive those rows while the completion thread
            # advances the pool iterator and reports worker failures.
            queue = SimpleQueue()
            inbox = ThreadQueue()

            def forward_results():
                while True:
                    message = queue.get()
                    inbox.put(message)
                    if message[0] == "finished":
                        return

            forwarder = Thread(target=forward_results, daemon=True)
            forwarder.start()
            try:
                with Pool(processes=n_jobs, initializer=_initialize_result_queue,
                          initargs=(queue,)) as pool:
                    worker_error = []

                    def drain_completions():
                        try:
                            for _ in pool.imap_unordered(_stream_group_wrapper, tasks,
                                                         chunksize=chunk_size):
                                pass
                        except BaseException as exc:
                            worker_error.append(exc)
                        finally:
                            queue.put(("finished", None))

                    coordinator = Thread(target=drain_completions, daemon=True)
                    coordinator.start()
                    completed = 0
                    with tqdm(total=local_total, desc="Running data groups") as progress:
                        while True:
                            kind, payload = inbox.get()
                            if kind == "result":
                                result_callback(payload)
                            elif kind == "done":
                                completed += 1
                                progress.update(1)
                            else:
                                break
                    coordinator.join()
                    forwarder.join()
                    if worker_error:
                        raise worker_error[0]
                    if completed != local_total:
                        raise RuntimeError("Simulation workers stopped before all groups completed")
            finally:
                queue.close()
    else:
        for task in tqdm(tasks, total=local_total, desc="Running data groups"):
            if result_callback is None:
                consume(_group_wrapper(task))
            else:
                _run_group(*task, result_callback=result_callback)
    return results
