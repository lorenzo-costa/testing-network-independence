"""Load and expand YAML configurations for the multiple-network linear model.

Public flow: ``load_config(path)`` → ``build_factorial_design(config)`` →
scenario rows. The registries below are the single place to add supported
networks, solvers, and methods.
"""

from functools import partial

import numpy as np
import yaml

from .configuration.linear_model import build_linear_model_rows
from .configuration.validation import _as_sweep, _resolve_linear_model_simulation
from .dgp import BernoulliNetwork, GaussianNetwork
from .helper_functions.multiple_network_factories import make_network
from .latent_samplers import CopulaSampler, MultipleNetworksSampler
from .methods import (
    CanonicalCorrelationTest,
    DistanceCorrelationTest,
    GraphCorrelationTest,
    MRQAP,
    RVTest,
)
from .metrics import ComputeAll
from .solvers.MaMa_uuuuu import pgd_fit_wrapper
from .solvers.weighted_network import ASE


DGP_REGISTRY = {
    "GaussianNetwork": GaussianNetwork,
    "BernoulliNetwork": BernoulliNetwork,
}

SOLVER_REGISTRY = {
    "ASE": ASE,
    "pgd_fit_wrapper": pgd_fit_wrapper,
}

LATENT_SAMPLER_REGISTRY = {
    "MultipleNetworksSampler": MultipleNetworksSampler,
    "CopulaSampler": CopulaSampler,
}

METHOD_REGISTRY = {
    "RVtest": RVTest,
    "RVTest": RVTest,
    "DiffusionCorrelation": DistanceCorrelationTest,
    "DistanceCorrelationTest": DistanceCorrelationTest,
    "GraphCorrelationTest": GraphCorrelationTest,
    "CanonicalCorrelation": CanonicalCorrelationTest,
    "CanonicalCorrelationTest": CanonicalCorrelationTest,
    "MRQAP": MRQAP,
}


def _resolve_method(entry: dict):
    """Resolve one configured method and its fixed keyword arguments."""
    if not isinstance(entry, dict):
        raise TypeError("Each methods.list entry must be a mapping")
    try:
        method = METHOD_REGISTRY[entry["name"]]
    except KeyError as error:
        raise ValueError(f"Unknown method: {entry.get('name')!r}") from error
    kwargs = entry.get("kwargs") or {}
    if not isinstance(kwargs, dict):
        raise TypeError("Method kwargs must be a mapping")
    return partial(method, **kwargs) if kwargs else method


def _resolve_methods_block(methods_cfg: dict) -> dict:
    """Normalize the method settings used by the linear-model grid."""
    if not isinstance(methods_cfg, dict) or "list" not in methods_cfg:
        raise ValueError("methods must contain a non-empty list")
    methods = [_resolve_method(entry) for entry in methods_cfg["list"]]
    if not methods:
        raise ValueError("methods.list must not be empty")
    return {
        "list": methods,
        "npermutations": _as_sweep(
            methods_cfg.get("npermutations", 200), "methods.npermutations"
        ),
        "use_true_latent": (
            None
            if methods_cfg.get("use_true_latent") is None
            else _as_sweep(methods_cfg["use_true_latent"], "methods.use_true_latent")
        ),
    }


def _resolve_linear_model_setup(entry: dict) -> tuple:
    """Resolve a multiple-network DGP and its embedding solver."""
    if not isinstance(entry, dict):
        raise TypeError("Each setup must be a mapping")
    try:
        dgp_cls = DGP_REGISTRY[entry["dgp"]]
        solver = SOLVER_REGISTRY[entry["solver"]]
    except KeyError as error:
        raise ValueError(
            "Each setup must name a registered 'dgp' and 'solver'"
        ) from error

    sampler_name = entry.get("latent_sampler", "MultipleNetworksSampler")
    try:
        latent_sampler = LATENT_SAMPLER_REGISTRY[sampler_name]
    except KeyError as error:
        available = ", ".join(sorted(LATENT_SAMPLER_REGISTRY))
        raise ValueError(
            f"Unknown latent_sampler: {sampler_name!r}. Available names: {available}."
        ) from error

    dgp_kwargs = entry.get("dgp_kwargs") or {}
    solver_kwargs = entry.get("solver_kwargs") or {}
    if not isinstance(dgp_kwargs, dict):
        raise TypeError("setup.dgp_kwargs must be a mapping")
    if not isinstance(solver_kwargs, dict):
        raise TypeError("setup.solver_kwargs must be a mapping")

    null_target = entry.get("null_target")
    if null_target not in {None, "independence", "zero_covariance"}:
        raise ValueError(
            "setup.null_target must be 'independence', 'zero_covariance', or null"
        )
    if null_target == "zero_covariance":
        if latent_sampler is not CopulaSampler:
            raise ValueError(
                "setup.null_target='zero_covariance' currently requires "
                "latent_sampler: CopulaSampler"
            )
        if not CopulaSampler.has_zero_covariance_configuration(
            dgp_kwargs.get("copula_model"),
            dgp_kwargs.get("rho", 0),
            dgp_kwargs.get("copula_params", {}),
            dgp_kwargs.get("marginals"),
            dgp_kwargs.get("cross_covariance"),
        ):
            raise ValueError(
                "setup.null_target='zero_covariance' requires a supported "
                "zero-covariance copula configuration"
            )

    dgp_factory = partial(
        make_network,
        dgp_cls,
        latent_sampler_class=latent_sampler,
        null_target=null_target,
        network_kwargs=dgp_kwargs,
    )
    return dgp_factory, partial(solver, **solver_kwargs) if solver_kwargs else solver


def load_config(path: str = "experiments/linear_model_config.yaml") -> dict:
    """Load one supported ``linear_model`` YAML configuration."""
    with open(path, encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise TypeError("Configuration must be a YAML mapping")
    if raw.get("experiment_type") != "linear_model":
        raise ValueError("Only experiment_type: linear_model is supported.")
    if "setups" not in raw or not isinstance(raw["setups"], list):
        raise ValueError("linear_model configurations require a setups list")

    setups = [_resolve_linear_model_setup(entry) for entry in raw["setups"]]
    requires_signal = any(
        setup[0].keywords["latent_sampler_class"] is MultipleNetworksSampler
        for setup in setups
    )
    simulation = _resolve_linear_model_simulation(
        raw.get("simulation"), require_signal_parameter=requires_signal
    )
    return {
        "experiment_type": "linear_model",
        "simulation": simulation,
        "rng": np.random.default_rng(simulation["seed"]),
        "methods": _resolve_methods_block(raw.get("methods")),
        "setups": setups,
        "metrics": [ComputeAll()] if raw.get("metrics", {}).get("compute_all") else [],
        "output": raw.get("output"),
    }


def build_factorial_design(config: dict) -> list[dict]:
    """Build ordered scenario rows for one loaded linear-model configuration."""
    if config.get("experiment_type") != "linear_model":
        raise ValueError("Only linear_model configurations can build a design.")
    return build_linear_model_rows(config)


def build_factorial_design_multi(configs: list[dict]) -> list[dict]:
    """Concatenate independently loaded linear-model designs in input order."""
    return [row for config in configs for row in build_factorial_design(config)]


def flatten_args_columns(df, extra_cols: dict | None = None):
    """Expand common scenario metadata from ``args`` into columns in place."""
    fields = (
        ("n", "n"),
        ("p", "p"),
        ("d_x", "d_x"),
        ("d_y", "d_y"),
        ("snr", "snr"),
        ("requested_snr", "requested_snr"),
        ("b_active_network_fraction", "b_active_network_fraction"),
        ("x_network_correlation", "x_network_correlation"),
        ("eps_distribution", "eps_distribution"),
        ("latent_sampler", "latent_sampler"),
        ("null_target", "null_target"),
        ("hypothesis", "hypothesis"),
        ("edge_var", "edge_var"),
        ("approximation", "approximation"),
        ("asymptotic_null", "asymptotic_null"),
        ("dgp", "dgp_name"),
        ("solver", "solver"),
        ("method", "method_name"),
    )
    for column, argument in fields:
        df[column] = df["args"].apply(lambda args, key=argument: args.get(key, "NA"))
    if extra_cols:
        for column, extractor in extra_cols.items():
            df[column] = df["args"].apply(extractor)
    return df
