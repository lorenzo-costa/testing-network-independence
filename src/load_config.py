"""
load_config.py
--------------
Universal config loader for all simulation experiments.

Supports experiment types auto-detected from YAML structure:
  - "standard"       -> main study + observed CVM sweep (same structure, different values)
  - "diff_marginals" -> asymmetric per-network marginal distributions

Public API
----------
    cfg          = load_config("config.yaml")      # auto-detects type
    design       = build_factorial_design(cfg)
    df           = flatten_args_columns(df)        # common post-processing
"""

import yaml
import numpy as np
from functools import partial
from itertools import product as iproduct

# -- DGP classes --------------------------------------------------------------
from src.dgp import GaussianNetwork, BernoulliNetwork

# -- Solvers ------------------------------------------------------------------
from src.solvers.weighted_network import ASE
from src.solvers.MaMa_uuuuu import pgd_fit_wrapper

# -- Test methods -------------------------------------------------------------
from src.methods import *

# -- Metrics ------------------------------------------------------------------
from src.metrics import ComputeAll


# =============================================================================
# Registries  --  extend here when adding new DGPs / solvers / methods
# =============================================================================

DGP_REGISTRY = {
    "GaussianNetwork": GaussianNetwork,
    "BernoulliNetwork": BernoulliNetwork,
}

SOLVER_REGISTRY = {
    "ASE": ASE,
    "pgd_fit_wrapper": pgd_fit_wrapper,
}

METHOD_REGISTRY = {
    "RVtest": RVTest,
    "MultivariateACTest": MultivariateACTest,
    "DistanceCorrelationTest": DistanceCorrelationTest,
    "DiffusionCorrelation": DistanceCorrelationTest,
    "MRQAP": MRQAP,
}

# =============================================================================
# Internal resolvers
# =============================================================================


def _resolve_method(entry: dict):
    """
    Convert a YAML method entry into a callable (or partial).
    """
    name = entry["name"]
    kwargs = entry.get("kwargs") or {}
    if "approximation" in kwargs:
        raise ValueError("'approximation' is no longer supported; use 'test_type'.")
    cls = METHOD_REGISTRY[name]

    return partial(cls, **kwargs) if kwargs else cls


def _resolve_copula_setup(entry: dict):
    """
    Resolve one copula-based setup entry into a (partial(DGP, ...), Solver) tuple.

    Keeps copula_model as a standalone DGP argument.
    Packs every other setup-specific field into copula_params.
    """
    dgp_cls = DGP_REGISTRY[entry["dgp"]]
    solver = SOLVER_REGISTRY[entry["solver"]]

    reserved = {
        "dgp",
        "solver",
        "conditional_copula",
        "post_nonlinear_noise",
        "copula_model",
        "rdgp",
        "rdpg",
        "rdpg_distr",
    }
    copula_keys = {
        "center_latent",
        "column_covariance",
        "column_covariance_y",
        "cross_covariance",
    }
    conditional_keys = {
        "C",
        "rho",
        "marginals",
        "marginals_by_class",
        "means",
        "covariances",
        "class_probabilities",
        "p",
        "center_latent",
        "column_covariance",
        "column_covariance_y",
        "cross_covariance",
    }
    post_nonlinear_keys = {
        "C",
        "rho",
        "class_probabilities",
        "p",
        "center_latent",
        "function_type",
        "error_covariance",
        "stratum_covariance",
        "column_covariance_z",
        "column_covariance_y",
        "column_covariance",
        "cross_correlation_template",
    }
    if entry.get("conditional_copula") is not None:
        reserved.update(conditional_keys)
    if entry.get("post_nonlinear_noise") is not None:
        reserved.update(post_nonlinear_keys)
    if entry.get("copula_model") is not None:
        reserved.update(copula_keys)

    copula_params = {k: v for k, v in entry.items() if k not in reserved}

    dgp_kwargs = {
        "conditional_copula": entry.get("conditional_copula"),
        "post_nonlinear_noise": entry.get("post_nonlinear_noise"),
        "copula_model": entry.get("copula_model"),
        "rdpg": entry.get("rdpg", False),
        "rdpg_distr": entry.get("rdpg_distr", None),
        "copula_params": copula_params,
    }
    if entry.get("post_nonlinear_noise") is not None:
        dgp_kwargs.pop("copula_params")
    if entry.get("conditional_copula") is not None:
        dgp_kwargs.update(
            {key: entry[key] for key in conditional_keys if key in entry}
        )
    if entry.get("copula_model") is not None:
        dgp_kwargs.update({key: entry[key] for key in copula_keys if key in entry})
    if entry.get("post_nonlinear_noise") is not None:
        dgp_kwargs.update(
            {key: entry[key] for key in post_nonlinear_keys if key in entry}
        )

    return (partial(dgp_cls, **dgp_kwargs), solver)


def _as_sweep(value, name: str) -> list:
    """Normalise a scalar or list-valued config field into a factorial sweep."""
    if value is None:
        return [None]
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name} must not be an empty list")
        return value
    return [value]


def _resolve_methods_block(methods_cfg: dict) -> dict:
    """Parse the YAML methods block into a normalised dict for product sweeps."""
    if "approximation" in methods_cfg:
        raise ValueError("'approximation' is no longer supported; use 'test_type'.")
    return {
        "list": [_resolve_method(m) for m in methods_cfg["list"]],
        "npermutations": methods_cfg.get("npermutations", [200]),
        "df": methods_cfg.get("df", [3]),
        "test_type": methods_cfg.get("test_type"),
        "use_true_latent": methods_cfg.get("use_true_latent"),  # None when absent
    }


def _resolve_noise_options(simulation_cfg: dict) -> dict:
    """Validate optional functional-noise controls from the simulation block.

    ``noise_scale`` and ``snr`` are numeric and ``noise_type`` is a string.
    Each may also be supplied as a list to sweep several values; scalars are
    normalised to one-element lists for the factorial design.  ``snr`` is the
    signal fraction and maps to the functional sampler's ``rho`` parameter.
    """
    def _as_list(value):
        return value if isinstance(value, list) else [value]

    resolved = {}
    if "snr" in simulation_cfg:
        if "rho" in simulation_cfg:
            raise ValueError("Specify only one of simulation.snr and simulation.rho")
        snrs = _as_list(simulation_cfg["snr"])
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in snrs):
            raise TypeError("simulation.snr must be a number or a list of numbers")
        if not snrs or any(not 0 < value <= 1 for value in snrs):
            raise ValueError("simulation.snr must contain values satisfying 0 < snr <= 1")
        resolved["rho"] = snrs

    if "noise_scale" in simulation_cfg:
        scales = _as_list(simulation_cfg["noise_scale"])
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in scales):
            raise TypeError("simulation.noise_scale must be a number or a list of numbers")
        resolved["noise_scale"] = scales

    if "noise_type" in simulation_cfg:
        types = _as_list(simulation_cfg["noise_type"])
        if any(not isinstance(value, str) for value in types):
            raise TypeError("simulation.noise_type must be a string or a list of strings")
        resolved["noise_type"] = types

    return resolved


def _resolve_functionals_block(functionals_cfg):
    """
    Normalise a ``simulation.functionals`` sweep.

    Each entry is retained as a plain, serialisable descriptor so downstream
    simulation code can dispatch the functional by name while receiving its
    keyword arguments unchanged.  A bare string is also accepted as shorthand
    for ``{"name": <string>, "kwargs": {}}``. Optional
    ``kwargs_by_predictor_distribution`` overrides individual kwargs for
    matching predictor distributions when the design rows are built.
    """
    if functionals_cfg is None:
        return None
    if not isinstance(functionals_cfg, list):
        raise TypeError("simulation.functionals must be a list of strings or mappings")

    resolved = []
    for entry in functionals_cfg:
        if isinstance(entry, str):
            entry = {"name": entry}
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError(
                "Each simulation.functionals entry must contain a non-empty 'name'"
            )
        kwargs = entry.get("kwargs") or {}
        if not isinstance(kwargs, dict):
            raise TypeError("functional kwargs must be a mapping")
        descriptor = {"functional_form": entry["name"], "function_params": kwargs}
        if "kwargs_by_predictor_distribution" in entry:
            overrides = entry["kwargs_by_predictor_distribution"]
            if not isinstance(overrides, dict) or any(
                not isinstance(distribution, str)
                or not isinstance(params, dict)
                for distribution, params in overrides.items()
            ):
                raise TypeError(
                    "functional kwargs_by_predictor_distribution must map "
                    "predictor names to parameter mappings"
                )
            descriptor["kwargs_by_predictor_distribution"] = overrides
        resolved.append(descriptor)
    return resolved


def _resolve_predictor_noise_sweeps(predictor_noise_cfg):
    """Validate paired functional predictor-distribution and noise sweeps."""
    if not isinstance(predictor_noise_cfg, list) or not predictor_noise_cfg:
        raise TypeError("simulation.predictor_noise_sweeps must be a non-empty list")

    resolved = []
    for entry in predictor_noise_cfg:
        if not isinstance(entry, dict):
            raise TypeError("Each predictor_noise_sweeps entry must be a mapping")
        distribution = entry.get("predictor_distribution")
        noise_scale = entry.get("noise_scale")
        if not isinstance(distribution, str) or not distribution:
            raise TypeError(
                "predictor_distribution in predictor_noise_sweeps must be a string"
            )
        if isinstance(noise_scale, bool) or not isinstance(noise_scale, (int, float)):
            raise TypeError(
                "noise_scale in predictor_noise_sweeps must be a number"
            )
        resolved.append(
            {
                "predictor_distribution": distribution,
                "noise_scale": noise_scale,
            }
        )
    return resolved


# =============================================================================
# Experiment-type detection
# =============================================================================


def _detect_experiment_type(raw: dict) -> str:
    """
    Infer experiment type from YAML structure (no explicit tag required).

    Detection priority (most specific first):
      1. "conditioning_mixed"   -- setups select both conditioning samplers
      2. "post_nonlinear_noise" -- a setup selects that sampler
      3. "conditional_copula"   -- a setup selects that sampler
      4. Remaining experiment types
    """
    setups_raw = raw.get("setups", {})

    if isinstance(setups_raw, list):
        has_post_nonlinear = any(
            entry.get("post_nonlinear_noise") for entry in setups_raw
        )
        has_conditional_copula = any(
            entry.get("conditional_copula") is not None for entry in setups_raw
        )
        if has_post_nonlinear and has_conditional_copula:
            return "conditioning_mixed"
        if has_post_nonlinear:
            return "post_nonlinear_noise"
        if has_conditional_copula:
            return "conditional_copula"

    marginals = raw.get("simulation", {}).get("marginals", [])
    if marginals and isinstance(marginals[0], dict):
        return "diff_marginals"

    if "functionals" in raw.get("simulation", {}):
        return "functionals"

    if "column_covariance" in raw.get("simulation", {}):
        return "asymptotic"

    return "standard"


# =============================================================================
# Public: load_config
# =============================================================================


def load_config(path: str = "config.yaml") -> dict:
    """
    Load a YAML config file and return a fully resolved config dict.

    Returned keys
    -------------
    experiment_type : str
        One of "standard", "diff_marginals", "functionals", or "asymptotic".
    simulation : dict
        Raw simulation block: nsim, n, k, rho, alpha, edge_var, marginals, seed.
    rng : np.random.Generator
        Seeded RNG ready for use.
    methods : dict
        list            -- resolved callables
        npermutations   -- list of ints
        df              -- list of ints
        test_type       -- list of strings or None when absent
        use_true_latent -- list of bools or None when not applicable
    setups : list
        (partial(DGP, ...), Solver) tuples for the H1 run. Functional
        setups may instead map predictor distributions to these tuples;
        build_factorial_design selects the matching tuple for each row.
    null_setups : dict | None
        {"rho": [...], "setups": [...]} for H0 runs, or None.
    extra_params : dict
        others    -> {}
    metrics : list
    output : dict
        results_dir, file_prefix
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    exp_type = _detect_experiment_type(raw)
    sim_raw = raw["simulation"]
    # Keep the raw YAML shape, but normalise optional functional sweeps so
    # callers receive a consistent descriptor format.
    if "functionals" in sim_raw:
        sim_raw = dict(sim_raw)
        sim_raw["functionals"] = _resolve_functionals_block(sim_raw["functionals"])
        sim_raw.update(_resolve_noise_options(sim_raw))
        if "predictor_noise_sweeps" in sim_raw:
            if "noise_scale" in sim_raw or "predictor_distribution" in sim_raw:
                raise ValueError(
                    "predictor_noise_sweeps cannot be combined with noise_scale or "
                    "predictor_distribution."
                )
            sim_raw["predictor_noise_sweeps"] = _resolve_predictor_noise_sweeps(
                sim_raw["predictor_noise_sweeps"]
            )

    methods = _resolve_methods_block(raw["methods"])
    metrics = [ComputeAll()] if raw.get("metrics", {}).get("compute_all") else []
    rng = np.random.default_rng(sim_raw["seed"])

    # -- Resolve setups -------------------------------------------------------
    if exp_type == "functionals":
        setups = []
        for entry in raw["setups"]:
            if isinstance(entry["solver"], dict):
                if not entry["solver"]:
                    raise ValueError("Functional solver mapping must not be empty")
                resolved = {
                    distribution: _resolve_copula_setup(dict(entry, solver=name))
                    for distribution, name in entry["solver"].items()
                }
                setups.append(resolved)
            else:
                setups.append(_resolve_copula_setup(entry))
    else:
        # standard, diff_marginals, asymptotic
        # all use copula-style setup entries
        setups = [_resolve_copula_setup(e) for e in raw["setups"]]

    # -- Experiment-specific extra params -------------------------------------
    extra_params = {}
    if exp_type == "asymptotic":
        extra_params["asymptotic"] = {
            "column_covariance": sim_raw.get("column_covariance"),
        }

    out = {
        "experiment_type": exp_type,
        "simulation": sim_raw,
        "rng": rng,
        "methods": methods,
        "setups": setups,
        "extra_params": extra_params,
        "metrics": metrics,
        "output": raw.get("output"),
    }

    return out


# =============================================================================
# Public: build_factorial_design
# =============================================================================


def _build_single_design(exp: str, cfg: dict) -> list[dict]:
    """
    Build method rows and label their data-generation arguments.
    All types draw from the same cfg; extra_params branches are simply
    ignored when the corresponding type is not active.
    """
    sim = cfg["simulation"]
    mth = cfg["methods"]
    sets = cfg["setups"]

    def _label_generation_args(rows, method_names):
        # The grid builder knows which columns came from method controls. Keep
        # that distinction explicit instead of guessing from DGP signatures,
        # many of which accept arbitrary **kwargs.
        for row in rows:
            row["_generation_keys"] = tuple(
                name for name in row if name not in method_names
            )
        return rows

    def _standard_rows(setups_list, rho_list, include_marginals=None):
        if include_marginals is None:
            include_marginals = exp != "post_nonlinear_noise"
        names = [
            "setup",
            "method",
            "n",
            "k",
            "ky",
            "alpha",
            "rho",
            "edge_var",
            "npermutations",
            "df",
        ]
        vals = [
            setups_list,
            mth["list"],
            sim["n"],
            sim["k"],
            sim.get("ky", [1]),
            sim["alpha"],
            rho_list,
            sim["edge_var"],
            mth["npermutations"],
            mth["df"],
        ]
        if include_marginals:
            marginals = sim.get("marginals", None)
            marginals_y = sim.get("marginals_y", None)
            marginals_z = sim.get("marginals_z", None)

            if marginals is None:
                # Functional-dependence configs do not require a marginal sweep.
                # Preserve a stable row schema by carrying an explicit None.
                if (
                    exp == "functionals"
                    and marginals_y is None
                    and marginals_z is None
                ):
                    marginals = [None]
                elif marginals_y is None:
                    if marginals_z is None:
                        raise ValueError(
                            "At least one of 'marginals', 'marginals_y', or "
                            "'marginals_z' must be specified."
                        )
                    marginals_y = marginals_z
                if marginals is None:
                    if marginals_z is None:
                        marginals_z = marginals_y

                    marginals = [
                        {"y": marginal_y, "z": marginal_z}
                        for marginal_y, marginal_z in iproduct(
                            marginals_y, marginals_z
                        )
                    ]

            names.append("marginals")
            vals.append(marginals)

        # Functional sweeps are stored as paired descriptors so each form stays
        # attached to its own parameters (rather than forming a cross-product).
        if sim.get("functionals") is not None:
            names.append("_functional")
            vals.append(sim["functionals"])
            for option in ("predictor_df", "center_latent"):
                if option in sim:
                    names.append(option)
                    vals.append(_as_sweep(sim[option], f"simulation.{option}"))
        if "predictor_noise_sweeps" in sim:
            names.append("_predictor_noise")
            vals.append(sim["predictor_noise_sweeps"])
        elif "noise_scale" in sim:
            names.append("noise_scale")
            vals.append(sim["noise_scale"])
        if "noise_type" in sim:
            names.append("noise_type")
            vals.append(sim["noise_type"])
        if "column_covariance" in sim:
            names.append("column_covariance")
            vals.append(sim["column_covariance"])
        if (
            "predictor_noise_sweeps" not in sim
            and "predictor_distribution" in sim
        ):
            names.append("predictor_distribution")
            vals.append(
                _as_sweep(
                    sim["predictor_distribution"],
                    "simulation.predictor_distribution",
                )
            )

        if mth["test_type"] is not None:
            names.append("test_type")
            vals.append(mth["test_type"])
        if mth["use_true_latent"] is not None:
            names.append("use_true_latent")
            vals.append(mth["use_true_latent"])
        if "function_type" in sim:
            names.append("function_type")
            vals.append(sim["function_type"])
        rows = [dict(zip(names, v)) for v in iproduct(*vals)]
        for row in rows:
            predictor_noise = row.pop("_predictor_noise", None)
            if predictor_noise is not None:
                row.update(predictor_noise)
            if exp == "functionals" and isinstance(row["setup"], dict):
                distribution = row.get("predictor_distribution", "gaussian")
                if distribution not in row["setup"]:
                    raise ValueError(
                        f"No functional solver configured for predictor_distribution "
                        f"{distribution!r}"
                    )
                row["setup"] = row["setup"][distribution]
            functional = row.pop("_functional", None)
            if functional is not None:
                row["functional_form"] = functional["functional_form"]
                row["function_params"] = dict(functional["function_params"])
                overrides = functional.get("kwargs_by_predictor_distribution", {})
                row["function_params"].update(
                    overrides.get(row.get("predictor_distribution", "gaussian"), {})
                )
        method_names = {"method", "alpha", "npermutations", "df"}
        if mth["test_type"] is not None:
            method_names.add("test_type")
        if mth["use_true_latent"] is not None:
            method_names.add("use_true_latent")
        return _label_generation_args(rows, method_names)

    # -- Mixed conditional-null study -----------------------------------------
    if exp == "conditioning_mixed":
        conditional_setups = [
            setup
            for setup in sets
            if setup[0].keywords.get("conditional_copula") is not None
        ]
        post_nonlinear_setups = [
            setup
            for setup in sets
            if setup[0].keywords.get("post_nonlinear_noise") is not None
        ]
        return _standard_rows(
            conditional_setups,
            sim["rho"],
            include_marginals=True,
        ) + _standard_rows(
            post_nonlinear_setups,
            sim["rho"],
            include_marginals=False,
        )

    # -- Standard / observed / diff_marginals ---------------------------------
    h1 = _standard_rows(sets, sim["rho"])

    return h1


def build_factorial_design(cfg: dict) -> list[dict]:
    """
    Build method rows for one or more experiment types.

    cfg["experiment_type"] may now be a string (single type, backward-compatible)
    or a list of strings (multiple types whose rows are concatenated).

    Each row includes internal ``_generation_keys`` metadata so the runner
    groups by DGP inputs while retaining every original method row.
    """
    exp_types = cfg["experiment_type"]
    if isinstance(exp_types, str):
        exp_types = [exp_types]

    all_h1 = []

    for exp in exp_types:
        h1 = _build_single_design(exp, cfg)
        all_h1.extend(h1)

    return all_h1


def build_factorial_design_multi(
    cfgs: list[dict],
) -> list[dict]:
    """
    Build a combined factorial design from a list of independently loaded configs.
    Each cfg is resolved for its own single experiment type.
    Rows from all types are concatenated with their generation-key metadata.
    """
    all_h1 = []
    for cfg in cfgs:
        h1 = build_factorial_design(cfg)  # existing single-type fn
        all_h1.extend(h1)
    return all_h1


# =============================================================================
# Public: flatten_args_columns
# =============================================================================


def flatten_args_columns(df, extra_cols: dict = None):
    """
    Extract the nested 'args' dict into flat DataFrame columns in-place.

    Parameters
    ----------
    df : pd.DataFrame
        Output from run_simulation; must have an 'args' column.
    extra_cols : dict, optional
        {column_name: extractor_fn} for experiment-specific columns.
        Each extractor receives one args dict and returns a scalar.

    Returns
    -------
    df : pd.DataFrame  (modified in-place; also returned for chaining)
    """
    df["n"] = df["args"].apply(lambda x: x["n"])
    df["k"] = df["args"].apply(lambda x: x["k"])
    df["edge_var"] = df["args"].apply(lambda x: x.get("edge_var", "NA"))
    df["test_type"] = df["args"].apply(lambda x: x.get("test_type", "NA"))
    df["dgp"] = df["args"].apply(lambda x: x.get("dgp_name", "NA"))
    df["solver"] = df["args"].apply(lambda x: x.get("solver", "NA"))
    df["rho"] = df["args"].apply(lambda x: x.get("rho", "NA"))
    df["method"] = df["args"].apply(lambda x: x.get("method_name", "NA"))
    df["marginals"] = df["args"].apply(
        lambda x: x.get("marginals").name
        if hasattr(x.get("marginals"), "name")
        else "NA"
    )

    if extra_cols:
        for col, fn in extra_cols.items():
            df[col] = df["args"].apply(fn)

    return df
