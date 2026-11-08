"""Resolve optional execution controls from a loaded simulation config."""


def execution_options(config):
    """Return ``run_simulation`` options for a loaded YAML configuration.

    An optional top-level ``execution`` mapping may override ``parallel``,
    ``n_jobs``, and ``batch_size``. Shard-specific worker counts are supplied
    by the shard runner's command-line interface.
    """
    execution = config.get("execution") or {}
    return {
        "nsim": config["simulation"]["nsim"],
        "metrics": config["metrics"],
        "rng": config["rng"],
        "parallel": execution.get("parallel", True),
        "n_jobs": execution.get("n_jobs"),
        "batch_size": execution.get("batch_size", 32),
    }
