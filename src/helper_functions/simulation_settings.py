"""Resolve run-level options from the first configuration in a study."""


def execution_options(config):
    simulation = config["simulation"]
    return {
        "nsim": simulation["nsim"],
        "metrics": config["metrics"],
        "rng": config["rng"],
        "parallel": simulation.get("parallel", True),
        "n_jobs": None if simulation.get("n_jobs") == -1 else simulation.get("n_jobs"),
        "batch_size": simulation.get("batch_size", 32),
        "blas_threads": simulation.get("blas_threads", 1),
    }
