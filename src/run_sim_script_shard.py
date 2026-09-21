"""Run one deterministic shard of a YAML-configured simulation study."""

import argparse
from datetime import datetime
import os
from pathlib import Path

import pandas as pd

from src.helper_functions.simulation_functions import run_simulation
from src.load_config import (
    build_factorial_design_multi,
    flatten_args_columns,
    load_config,
)


CSV_BATCH_SIZE = 1000


class _CsvResultWriter:
    """Write simulation results in bounded batches while preserving CSV shape."""

    def __init__(self, output_path, batch_size=CSV_BATCH_SIZE):
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self.buffer = []
        self.rows_written = 0
        self._wrote_header = False

    def add(self, result):
        self.buffer.append(result)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        frame = pd.DataFrame(self.buffer)
        flatten_args_columns(frame)
        frame.to_csv(
            self.output_path,
            mode="a" if self._wrote_header else "w",
            header=not self._wrote_header,
            index=False,
        )
        self.rows_written += len(frame)
        self._wrote_header = True
        self.buffer.clear()

    def close(self):
        self.flush()


def _environment_int(name, default=None):
    """Return an integer Slurm environment variable when it is defined."""
    value = os.environ.get(name)
    return default if value is None else int(value)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run one deterministic shard of a configured simulation study."
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        default=["config.yaml"],
        help="One or more YAML config files.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=_environment_int("SLURM_ARRAY_TASK_ID", 0),
        help="Zero-based shard index; defaults to SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=_environment_int("SLURM_ARRAY_TASK_COUNT", 1),
        help="Total shard count; defaults to SLURM_ARRAY_TASK_COUNT.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=_environment_int("SLURM_CPUS_PER_TASK"),
        help="Local worker processes; defaults to SLURM_CPUS_PER_TASK.",
    )
    return parser


def _validate_cli_args(parser, args):
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must satisfy 0 <= index < num-shards")
    if args.n_jobs is not None and args.n_jobs < 1:
        parser.error("--n-jobs must be positive")


def _output_path(config, shard_index, num_shards):
    output = config["output"]
    results_dir = Path(output["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    prefix = output.get("file_prefix", "simulation_results")
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    run_id = job_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    return results_dir / (
        f"{prefix}_{run_id}_shard-{shard_index:03d}-of-{num_shards:03d}.csv"
    )


def main(argv=None):
    """Run this process's shard and return its CSV path."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_cli_args(parser, args)

    configs = [load_config(path) for path in args.config]
    factorial = build_factorial_design_multi(configs)
    simulation = configs[0]["simulation"]
    global_total = simulation["nsim"] * len(factorial)
    local_total = len(range(args.shard_index, global_total, args.num_shards))
    configured_n_jobs = simulation.get("n_jobs")
    n_jobs = args.n_jobs
    if n_jobs is None:
        n_jobs = None if configured_n_jobs == -1 else configured_n_jobs

    print(
        f"Shard {args.shard_index + 1}/{args.num_shards}: "
        f"{local_total} of {global_total} scenarios, {n_jobs or 'all'} local workers."
    )
    started = datetime.now()
    output_path = _output_path(configs[0], args.shard_index, args.num_shards)
    writer = _CsvResultWriter(output_path)
    run_simulation(
        nsim=simulation["nsim"],
        metrics=configs[0]["metrics"],
        factorial_design=factorial,
        rng=configs[0]["rng"],
        parallel=simulation.get("parallel", True),
        n_jobs=n_jobs,
        batch_size=simulation.get("batch_size", 32),
        blas_threads=simulation.get("blas_threads", 1),
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        result_callback=writer.add,
    )
    writer.close()
    print(
        f"Saved {writer.rows_written} rows to {output_path} "
        f"in {datetime.now() - started}."
    )
    return str(output_path)


if __name__ == "__main__":
    main()
