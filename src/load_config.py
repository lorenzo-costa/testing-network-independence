"""
load_config.py
--------------
Loads config.yaml and resolves string identifiers back into Python callables.

Usage in run_simulation_script.py:
    from load_config import load_config
    cfg = load_config("config.yaml")
"""

import yaml
from functools import partial

# ── DGP classes ──────────────────────────────────────────────────────────────
from src.dgp import GaussianNetwork, BernoulliNetwork

# ── Solver functions ──────────────────────────────────────────────────────────
from src.solvers.weighted_network import ASE
from src.solvers.MaMa_uuuuu import pgd_fit_wrapper

# ── Test methods ──────────────────────────────────────────────────────────────
from src.methods import RVPermutationTest, QAP, DiffusionCorrelation, ObservedCVM
from src.helper_functions._metrics_helper import observed_cvm_dependency

# ── Metrics ───────────────────────────────────────────────────────────────────
from src.metrics import ComputeAll


# ---------------------------------------------------------------------------
# Registry: maps YAML string identifiers → Python objects / factories
# Add new DGPs, solvers, or methods here as your project grows.
# ---------------------------------------------------------------------------

DGP_REGISTRY = {
    "GaussianNetwork": GaussianNetwork,
    "BernoulliNetwork": BernoulliNetwork,
}

SOLVER_REGISTRY = {
    "ASE": ASE,
    "pgd_fit_wrapper": pgd_fit_wrapper,
}

METHOD_REGISTRY = {
    "RVPermutationTest": RVPermutationTest,
    "QAP": QAP,
    "DiffusionCorrelation": DiffusionCorrelation,
    "ObservedCVM": ObservedCVM,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_dgp(entry: dict):
    """
    Turn a setup entry from YAML into a (partial(DGP, ...), Solver) tuple,
    matching the original `setup` list format.
    """
    dgp_cls = DGP_REGISTRY[entry["dgp"]]
    solver = SOLVER_REGISTRY[entry["solver"]]

    # Build kwargs for the DGP constructor (everything except 'dgp' and 'solver')
    dgp_kwargs = {
        k: v for k, v in entry.items()
        if k not in ("dgp", "solver")
    }
    return (partial(dgp_cls, **dgp_kwargs), solver)


def _resolve_method(entry: dict):
    """
    Turn a method entry from YAML into a callable (or partial).
    Handles the special case of ObservedCVM which wraps observed_cvm_dependency.
    """
    name = entry["name"]
    kwargs = entry.get("kwargs", {}) or {}
    cls = METHOD_REGISTRY[name]

    if name == "ObservedCVM":
        degree = kwargs.get("degree", 2)
        return partial(cls, test_function=partial(observed_cvm_dependency, degree=degree))

    if kwargs:
        return partial(cls, **kwargs)

    return cls


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    """
    Load config.yaml and return a fully resolved config dict ready for use
    in run_simulation_script.py.

    Resolved keys
    -------------
    cfg["simulation"]   – scalar/list simulation params (nsim, n, k, …)
    cfg["methods"]      – resolved list of callables + permutation/df/approx params
    cfg["setups"]       – list of (partial(DGP), Solver) tuples  (H1)
    cfg["null_setups"]  – same format, for H0 runs
    cfg["metrics"]      – list of metric objects
    cfg["output"]       – output directory string
    cfg["rng"]          – np.random.Generator seeded from config
    """
    import numpy as np

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # ── Simulation scalars / lists ────────────────────────────────────────
    sim = raw["simulation"]

    # ── Methods ───────────────────────────────────────────────────────────
    methods_cfg = raw["methods"]
    resolved_methods = [_resolve_method(m) for m in methods_cfg["list"]]

    # ── Setups (H1) ───────────────────────────────────────────────────────
    resolved_setups = [_resolve_dgp(s) for s in raw["setups"]]

    # ── Null setups (H0) ──────────────────────────────────────────────────
    null_cfg = raw["null_setups"]
    resolved_null_setups = [_resolve_dgp(s) for s in null_cfg["setups"]]

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics_cfg = raw["metrics"]
    metrics = [ComputeAll()] if metrics_cfg.get("compute_all") else []

    # ── RNG ───────────────────────────────────────────────────────────────
    rng = np.random.default_rng(sim["seed"])

    return {
        "simulation": sim,
        "rng": rng,
        "methods": {
            "list": resolved_methods,
            "npermutations": methods_cfg["npermutations"],
            "df": methods_cfg["df"],
            "approximation": methods_cfg["approximation"],
            "use_true_latent": methods_cfg["use_true_latent"],
        },
        "setups": resolved_setups,
        "null_setups": {
            "rho": null_cfg["rho"],
            "setups": resolved_null_setups,
        },
        "metrics": metrics,
        "output": raw["output"],
    }