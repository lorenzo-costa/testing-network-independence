"""
load_config.py
--------------
Universal config loader for all simulation experiments.

Supports six experiment types, auto-detected from YAML structure:
  - "standard"       -> main study + observed CVM sweep (same structure, different values)
  - "lee2019"        -> latent functional-relationship study (Lee et al. 2019)
  - "diff_marginals" -> asymmetric per-network marginal distributions
  - "sbm"            -> covariate-aware stochastic block model study
  - "multiness"      -> multi-network study with common/individual latent dimensions

Public API
----------
    cfg          = load_config("config.yaml")      # auto-detects type
    h1, h0       = build_factorial_design(cfg)     # h0 is None for lee2019
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
    "DiffusionCorrelation": DistanceCorrelationTest,
    "CanonicalCorrelation": CanonicalCorrelationTest,
    "QAP": QAP,
    "MRQAP": MRQAP,
}

# Latent-sim shapes that do NOT accept sim_kwargs={'noise': True}
_NO_NOISE_SIMS = {"multimodal_independence"}


# =============================================================================
# Internal resolvers
# =============================================================================


def _resolve_method(entry: dict):
    """
    Convert a YAML method entry into a callable (or partial).
    """
    name = entry["name"]
    kwargs = entry.get("kwargs") or {}
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
    conditional_keys = {
        "C",
        "rho",
        "marginals",
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
    if entry.get("post_nonlinear_noise") is not None:
        dgp_kwargs.update(
            {key: entry[key] for key in post_nonlinear_keys if key in entry}
        )

    return (partial(dgp_cls, **dgp_kwargs), solver)


def _resolve_lee2019_setups(setups_cfg: dict) -> list:
    """
    Expand gaussian_latent_sims / bernoulli_latent_sims name-lists into
    (partial(DGP, latent_sim=...), partial(ASE, k=...)) tuples.
    multimodal_independence is special-cased: it receives no sim_kwargs.
    """
    ase_k = setups_cfg.get("ase_k", None)
    if ase_k is None:
        solver = ASE
    else:
        solver = partial(ASE, k=ase_k)
    rdpg = setups_cfg.get("bernoulli_rdpg", "minmax")
    result = []

    for sim_name in setups_cfg["gaussian_latent_sims"]:
        if sim_name in _NO_NOISE_SIMS:
            dgp = partial(GaussianNetwork, latent_sim=sim_name)
        else:
            dgp = partial(
                GaussianNetwork, latent_sim=sim_name, sim_kwargs={"noise": True}
            )
        result.append((dgp, solver))

    for sim_name in setups_cfg["bernoulli_latent_sims"]:
        if sim_name in _NO_NOISE_SIMS:
            dgp = partial(BernoulliNetwork, rdpg=rdpg, latent_sim=sim_name)
        else:
            dgp = partial(
                BernoulliNetwork,
                rdpg=rdpg,
                latent_sim=sim_name,
                sim_kwargs={"noise": True},
            )
        result.append((dgp, solver))

    return result


def _as_sweep(value, name: str) -> list:
    """Normalise a scalar or list-valued config field into a factorial sweep."""
    if value is None:
        return [None]
    if isinstance(value, list):
        if not value:
            raise ValueError(f"{name} must not be an empty list")
        return value
    return [value]


def _resolve_sbm_setups(setups_cfg: list) -> list:
    """Resolve SBM DGP/solver pairs.

    An SBM config may omit ``sbm: true`` in each setup because the top-level
    ``sbm`` block already identifies the experiment.  The flag is injected
    unconditionally here, while any other setup-specific DGP keyword arguments
    are preserved.
    """
    if not isinstance(setups_cfg, list) or not setups_cfg:
        raise TypeError("SBM setups must be a non-empty list of mappings")

    resolved = []
    for entry in setups_cfg:
        if not isinstance(entry, dict):
            raise TypeError("Each SBM setup must be a mapping")
        try:
            dgp_cls = DGP_REGISTRY[entry["dgp"]]
            solver = SOLVER_REGISTRY[entry["solver"]]
        except KeyError as exc:
            raise ValueError(
                "Each SBM setup must name a registered 'dgp' and 'solver'"
            ) from exc

        dgp_kwargs = {
            key: value for key, value in entry.items() if key not in {"dgp", "solver"}
        }
        dgp_kwargs["sbm"] = True
        resolved.append((partial(dgp_cls, **dgp_kwargs), solver))

    return resolved


def _resolve_sbm_block(sbm_cfg: dict) -> dict:
    """Normalise optional SBM-specific sweeps.

    ``assortativity`` defaults to ``0.5`` when it is not specified.  The
    covariate-aware fields (``sbm_covariate_sampling`` and ``y_distribution``) are optional
    as a pair so this loader accepts both the supplied covariate-aware setup and
    earlier block/assignment-style SBM configurations.
    """
    if not isinstance(sbm_cfg, dict):
        raise TypeError("sbm must be a mapping")

    has_sampling = "sbm_covariate_sampling" in sbm_cfg
    has_y_distribution = "y_distribution" in sbm_cfg
    if has_sampling != has_y_distribution:
        raise ValueError(
            "sbm.sbm_covariate_sampling and sbm.y_distribution must be supplied together"
        )

    resolved = {
        "assortativity": _as_sweep(
            sbm_cfg.get("assortativity", 0.5), "sbm.assortativity"
        )
    }

    for name in (
        "sbm_covariate_sampling",
        "y_distribution",
        "sparsity_bias",
        "prob_switch",
        "assignment_mode",
        "block_probs_type",
        "block_probs",
        "y_upper_bound",
        "y_probabilities",
        "softmax_intercept",
        "softmax_slope",
        "directed",
        "self_loops",
    ):
        if name in sbm_cfg:
            resolved[name] = _as_sweep(sbm_cfg[name], f"sbm.{name}")

    return resolved


def _resolve_methods_block(methods_cfg: dict) -> dict:
    """Parse the YAML methods block into a normalised dict for product sweeps."""
    return {
        "list": [_resolve_method(m) for m in methods_cfg["list"]],
        "npermutations": methods_cfg.get("npermutations", [200]),
        "df": methods_cfg.get("df", [3]),
        "approximation": methods_cfg.get(
            "approximation"
        ),  # None when absent (e.g. multiness)
        "use_true_latent": methods_cfg.get("use_true_latent"),  # None when absent
    }


def _resolve_noise_options(simulation_cfg: dict) -> dict:
    """Validate optional functional-noise controls from the simulation block.

    ``noise_scale`` is an integer and ``noise_type`` is a string.  Both may
    also be supplied as lists to sweep several values; scalar values are
    normalised to one-element lists for the factorial design.
    """
    def _as_list(value):
        return value if isinstance(value, list) else [value]

    resolved = {}
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
    for ``{"name": <string>, "kwargs": {}}``.
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
        resolved.append(
            {"functional_form": entry["name"], "function_params": kwargs}
        )
    return resolved


# =============================================================================
# Experiment-type detection
# =============================================================================


def _detect_experiment_type(raw: dict) -> str:
    """
    Infer experiment type from YAML structure (no explicit tag required).

    Detection priority (most specific first):
      1. "sbm"                  -- top-level ``sbm`` block is present
      2. "lee2019"              -- setups contain ``gaussian_latent_sims``
      3. "conditioning_mixed"   -- setups select both conditioning samplers
      4. "post_nonlinear_noise" -- a setup selects that sampler
      5. "conditional_copula"   -- a setup selects that sampler
      6. Remaining legacy experiment types
    """
    if "sbm" in raw:
        return "sbm"

    setups_raw = raw.get("setups", {})
    if isinstance(setups_raw, dict) and "gaussian_latent_sims" in setups_raw:
        return "lee2019"

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

    if "dim_common" in raw.get("simulation", {}):
        return "multiness"

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
        One of "standard", "lee2019", "diff_marginals", "sbm", "multiness",
        "functionals", or "asymptotic".
    simulation : dict
        Raw simulation block: nsim, n, k, rho, alpha, edge_var, marginals, seed.
        multiness also carries: dim_common, dim_individual, shared_latent_type.
    rng : np.random.Generator
        Seeded RNG ready for use.
    methods : dict
        list            -- resolved callables
        npermutations   -- list of ints
        df              -- list of ints (None for  multiness)
        approximation   -- list of strings or None when absent (multiness )
        use_true_latent -- list of bools or None when not applicable
    setups : list
        (partial(DGP, ...), Solver) tuples for the H1 run.
    null_setups : dict | None
        {"rho": [...], "setups": [...]} for H0 runs, or None.
    extra_params : dict
        lee2019   -> {"sparsity": {"make_sparse": [...], "sparsity_bias": [...]}}
        multiness -> {}   (extra dims live directly in cfg["simulation"])
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

    methods = _resolve_methods_block(raw["methods"])
    metrics = [ComputeAll()] if raw.get("metrics", {}).get("compute_all") else []
    rng = np.random.default_rng(sim_raw["seed"])

    # -- Resolve setups -------------------------------------------------------
    if exp_type == "lee2019":
        setups = _resolve_lee2019_setups(raw["setups"])
    elif exp_type == "sbm":
        setups = _resolve_sbm_setups(raw["setups"])
    else:
        # standard, diff_marginals, multiness, functionals, asymptotic
        # all use copula-style setup entries
        setups = [_resolve_copula_setup(e) for e in raw["setups"]]

    # -- Experiment-specific extra params -------------------------------------
    extra_params = {}
    if exp_type == "lee2019":
        sp = raw.get("sparsity", {"make_sparse": [False], "sparsity_bias": [1]})
        extra_params["sparsity"] = {
            "make_sparse": sp["make_sparse"],
            "sparsity_bias": sp["sparsity_bias"],
        }
    elif exp_type == "sbm":
        extra_params["sbm"] = _resolve_sbm_block(raw["sbm"])
    elif exp_type == "asymptotic":
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


def _build_single_design(exp: str, cfg: dict) -> tuple[list[dict], list[dict] | None]:
    """
    Build the H1/H0 parameter rows for a single experiment type.
    All types draw from the same cfg; extra_params branches are simply
    ignored when the corresponding type is not active.
    """
    sim = cfg["simulation"]
    mth = cfg["methods"]
    sets = cfg["setups"]

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
        if "noise_scale" in sim:
            names.append("noise_scale")
            vals.append(sim["noise_scale"])
        if "noise_type" in sim:
            names.append("noise_type")
            vals.append(sim["noise_type"])
        if "column_covariance" in sim:
            names.append("column_covariance")
            vals.append(sim["column_covariance"])

        if mth["approximation"] is not None:
            names.append("approximation")
            vals.append(mth["approximation"])
        if mth["use_true_latent"] is not None:
            names.append("use_true_latent")
            vals.append(mth["use_true_latent"])
        if "function_type" in sim:
            names.append("function_type")
            vals.append(sim["function_type"])
        rows = [dict(zip(names, v)) for v in iproduct(*vals)]
        for row in rows:
            functional = row.pop("_functional", None)
            if functional is not None:
                row["functional_form"] = functional["functional_form"]
                row["function_params"] = functional["function_params"]
        return rows

    # -- Multiness ------------------------------------------------------------
    if exp == "multiness":
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
            "dim_common",
            "dim_individual",
            "shared_latent_type",
        ]
        vals = [
            sets,
            mth["list"],
            sim["n"],
            sim["k"],
            sim.get("ky", [1]),
            sim["alpha"],
            sim["rho"],
            sim["edge_var"],
            mth["npermutations"],
            mth["df"],
            sim["dim_common"],
            sim["dim_individual"],
            sim["shared_latent_type"],
        ]
        if mth["use_true_latent"] is not None:
            names.append("use_true_latent")
            vals.append(mth["use_true_latent"])

        h1 = [dict(zip(names, v)) for v in iproduct(*vals)]
        return h1

    # -- Lee 2019 -------------------------------------------------------------
    if exp == "lee2019":
        sp = cfg["extra_params"]["sparsity"]
        names = [
            "setup",
            "method",
            "n",
            "k",
            "ky",
            "alpha",
            "edge_var",
            "npermutations",
            "df",
            "make_sparse",
            "sparsity_bias",
        ]
        vals = [
            sets,
            mth["list"],
            sim["n"],
            sim["k"],
            sim.get("ky", [1]),
            sim["alpha"],
            sim["edge_var"],
            mth["npermutations"],
            mth["df"],
            sp["make_sparse"],
            sp["sparsity_bias"],
        ]
        if mth["use_true_latent"] is not None:
            names.append("use_true_latent")
            vals.append(mth["use_true_latent"])

        return [dict(zip(names, v)) for v in iproduct(*vals)]

    # -- Covariate-aware SBM ---------------------------------------------------
    if exp == "sbm":
        sbm = cfg["extra_params"]["sbm"]
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
            "assortativity",
        ]
        vals = [
            sets,
            mth["list"],
            sim["n"],
            sim["k"],
            sim.get("ky", [1]),
            sim["alpha"],
            sim.get("rho", [None]),
            sim.get("edge_var", [None]),
            mth["npermutations"],
            mth["df"],
            sbm["assortativity"],
        ]

        # Covariate-aware and legacy SBM controls are included only when the
        # YAML supplies them, so neither family needs dummy parameters.
        for name in (
            "sbm_covariate_sampling",
            "y_distribution",
            "sparsity_bias",
            "prob_switch",
            "assignment_mode",
            "block_probs_type",
            "block_probs",
            "y_upper_bound",
            "y_probabilities",
            "softmax_intercept",
            "softmax_slope",
            "directed",
            "self_loops",
        ):
            if name in sbm:
                names.append(name)
                vals.append(sbm[name])

        # A marginal sweep is not required for SBM configs, but preserve it
        # when a hybrid config explicitly supplies one.
        if "marginals" in sim:
            names.append("marginals")
            vals.append(sim["marginals"])

        if mth["approximation"] is not None:
            names.append("approximation")
            vals.append(mth["approximation"])
        if mth["use_true_latent"] is not None:
            names.append("use_true_latent")
            vals.append(mth["use_true_latent"])

        return [dict(zip(names, v)) for v in iproduct(*vals)]

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


def build_factorial_design(cfg: dict) -> tuple[list[dict], list[dict] | None]:
    """
    Build the combined parameter grid for one or more experiment types.

    cfg["experiment_type"] may now be a string (single type, backward-compatible)
    or a list of strings (multiple types whose rows are concatenated).

    Returns
    -------
    (factorial_h1, factorial_h0)
        factorial_h0 is None when no type in the list produces H0 rows.
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
) -> tuple[list[dict], list[dict] | None]:
    """
    Build a combined factorial design from a list of independently loaded configs.
    Each cfg is resolved for its own single experiment type.
    Rows from all types are concatenated; H0 rows are concatenated where present.
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
    df["approximation"] = df["args"].apply(lambda x: x.get("approximation", "NA"))
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
