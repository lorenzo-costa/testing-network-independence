"""
load_config.py
--------------
Universal config loader for all simulation experiments.

Supports five experiment types, auto-detected from YAML structure:
  - "standard"       -> main study + observed CVM sweep (same structure, different values)
  - "lee2019"        -> latent functional-relationship study (Lee et al. 2019)
  - "diff_marginals" -> asymmetric per-network marginal distributions
  - "sbm"            -> stochastic block model misspecification study
  - "multiness"      -> multi-network study with common/individual latent dimensions

Public API
----------
    cfg          = load_config("config.yaml")      # auto-detects type
    h1, h0       = build_factorial_design(cfg)     # h0 is None for lee2019 / sbm
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
from src.methods import RVtest, QAP, DiffusionCorrelation, ObservedCVM
from src.helper_functions._metrics_helper import observed_cvm_dependency

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
    "RVtest": RVtest,
    "QAP": QAP,
    "DiffusionCorrelation": DiffusionCorrelation,
    "ObservedCVM": ObservedCVM,
}

# Latent-sim shapes that do NOT accept sim_kwargs={'noise': True}
_NO_NOISE_SIMS = {"multimodal_independence"}


# =============================================================================
# Internal resolvers
# =============================================================================


def _resolve_method(entry: dict):
    """
    Convert a YAML method entry into a callable (or partial).
    ObservedCVM is special-cased because it wraps observed_cvm_dependency.
    """
    name = entry["name"]
    kwargs = entry.get("kwargs") or {}
    cls = METHOD_REGISTRY[name]

    if name == "ObservedCVM":
        degree = kwargs.get("degree", 2)
        return partial(
            cls, test_function=partial(observed_cvm_dependency, degree=degree)
        )

    return partial(cls, **kwargs) if kwargs else cls


def _resolve_standard_setup(entry: dict):
    """
    Resolve one copula-based setup entry into a (partial(DGP, ...), Solver) tuple.
    Used by: standard, observed, diff_marginals experiments.
    """
    dgp_cls = DGP_REGISTRY[entry["dgp"]]
    solver = SOLVER_REGISTRY[entry["solver"]]
    dgp_kwargs = {k: v for k, v in entry.items() if k not in ("dgp", "solver")}
    return (partial(dgp_cls, **dgp_kwargs), solver)


def _resolve_lee2019_setups(setups_cfg: dict) -> list:
    """
    Expand gaussian_latent_sims / bernoulli_latent_sims name-lists into
    (partial(DGP, latent_sim=...), partial(ASE, k=...)) tuples.
    multimodal_independence is special-cased: it receives no sim_kwargs.
    """
    ase_k = setups_cfg.get("ase_k", 2)
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


def _resolve_sbm_setups(setups_list: list) -> list:
    """
    Resolve SBM setups: sbm=True is injected automatically; no copula params.
    """
    return [
        (partial(DGP_REGISTRY[e["dgp"]], sbm=True), SOLVER_REGISTRY[e["solver"]])
        for e in setups_list
    ]


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

    
# =============================================================================
# Experiment-type detection
# =============================================================================


def _detect_experiment_type(raw: dict) -> str:
    """
    Infer experiment type from YAML structure (no explicit tag required).

    Detection priority (most specific first):
      1. "sbm"           -- top-level `sbm:` key is present
      2. "lee2019"       -- `setups` is a dict with a `gaussian_latent_sims` key
      3. "multiness"     -- simulation block contains `dim_common` key
      4. "diff_marginals"-- first marginals entry is a dict (has 'x'/'y' keys)
      5. "standard"      -- everything else (main study + observed CVM sweep)
    """
    if "sbm" in raw:
        return "sbm"

    setups_raw = raw.get("setups", {})
    if isinstance(setups_raw, dict) and "gaussian_latent_sims" in setups_raw:
        return "lee2019"

    if "dim_common" in raw.get("simulation", {}):
        return "multiness"

    marginals = raw.get("simulation", {}).get("marginals", [])
    if marginals and isinstance(marginals[0], dict):
        return "diff_marginals"

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
        One of "standard", "lee2019", "diff_marginals", "sbm", "multiness".
    simulation : dict
        Raw simulation block: nsim, n, k, rho, alpha, edge_var, marginals, seed.
        multiness also carries: dim_common, dim_individual, shared_latent_type.
    rng : np.random.Generator
        Seeded RNG ready for use.
    methods : dict
        list            -- resolved callables
        npermutations   -- list of ints
        df              -- list of ints (None for sbm / multiness)
        approximation   -- list of strings or None when absent (multiness / sbm)
        use_true_latent -- list of bools or None when not applicable
    setups : list
        (partial(DGP, ...), Solver) tuples for the H1 run.
    null_setups : dict | None
        {"rho": [...], "setups": [...]} for H0 runs, or None.
    extra_params : dict
        lee2019   -> {"sparsity": {"make_sparse": [...], "sparsity_bias": [...]}}
        sbm       -> {"sbm": {"assortativity": [...], ...}}
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
    methods = _resolve_methods_block(raw["methods"])
    metrics = [ComputeAll()] if raw.get("metrics", {}).get("compute_all") else []
    rng = np.random.default_rng(sim_raw["seed"])

    # -- Resolve setups -------------------------------------------------------
    if exp_type == "lee2019":
        setups = _resolve_lee2019_setups(raw["setups"])
    elif exp_type == "sbm":
        setups = _resolve_sbm_setups(raw["setups"])
    else:
        # standard, diff_marginals, multiness — all use copula-style setup entries
        setups = [_resolve_standard_setup(e) for e in raw["setups"]]

    # -- Resolve null setups (standard-family experiments only) ---------------
    null_setups = None
    if "null_setups" in raw:
        nc = raw["null_setups"]
        null_setups = {
            "rho": nc["rho"],
            "setups": [_resolve_standard_setup(e) for e in nc["setups"]],
        }

    # -- Experiment-specific extra params -------------------------------------
    extra_params = {}
    if exp_type == "lee2019":
        sp = raw["sparsity"]
        extra_params["sparsity"] = {
            "make_sparse": sp["make_sparse"],
            "sparsity_bias": sp["sparsity_bias"],
        }
    elif exp_type == "sbm":
        sbm = raw["sbm"]
        extra_params["sbm"] = {
            "assortativity": sbm["assortativity"],
            "sparsity_bias": sbm["sparsity_bias"],
            "prob_switch": sbm["prob_switch"],
            "assignment_mode": sbm["assignment_mode"],
            "block_probs_type": sbm["block_probs_type"],
        }
    elif exp_type == "asymptotic":
        extra_params["asymptotic"] = {
            "column_covariance": sim_raw["column_covariance"],
        }

    return {
        "experiment_type": exp_type,
        "simulation": sim_raw,
        "rng": rng,
        "methods": methods,
        "setups": setups,
        "null_setups": null_setups,
        "extra_params": extra_params,
        "metrics": metrics,
        "output": raw["output"],
    }


# =============================================================================
# Public: build_factorial_design
# =============================================================================


def build_factorial_design(cfg: dict) -> tuple:
    """
    Build the parameter grid for a loaded config.

    Returns
    -------
    (factorial_h1, factorial_h0)
        Both are lists of dicts for run_simulation(factorial_design=...).
        factorial_h0 is None for lee2019 and sbm (no null/H0 run).
    """
    exp = cfg["experiment_type"]
    sim = cfg["simulation"]
    mth = cfg["methods"]
    sets = cfg["setups"]

    # Helper: build standard-family factorial for a given setup list and rho
    def _standard_rows(setups_list, rho_list):
        names = [
            "setup",
            "method",
            "n",
            "k",
            "alpha",
            "marginals",
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
            sim["alpha"],
            sim["marginals"],
            rho_list,
            sim["edge_var"],
            mth["npermutations"],
            mth["df"],
        ]
        # approximation is optional — absent in multiness configs
        if mth["approximation"] is not None:
            names.append("approximation")
            vals.append(mth["approximation"])
        if mth["use_true_latent"] is not None:
            names.append("use_true_latent")
            vals.append(mth["use_true_latent"])
        return [dict(zip(names, v)) for v in iproduct(*vals)]

    # -- Multiness ------------------------------------------------------------
    if exp == "multiness":
        names = [
            "setup",
            "method",
            "n",
            "k",
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

        h0 = None
        if cfg["null_setups"]:
            null = cfg["null_setups"]
            names_h0 = [n if n != "rho" else "rho" for n in names]  # same schema
            vals_h0 = [
                null["setups"] if n == "setup" else null["rho"] if n == "rho" else v
                for n, v in zip(names, vals)
            ]
            h0 = [dict(zip(names_h0, v)) for v in iproduct(*vals_h0)]

        return h1, h0

    # -- Lee 2019 -------------------------------------------------------------
    if exp == "lee2019":
        sp = cfg["extra_params"]["sparsity"]
        names = [
            "setup",
            "method",
            "n",
            "k",
            "alpha",
            "marginals",
            "rho",
            "edge_var",
            "approximation",
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
            sim["alpha"],
            sim["marginals"],
            sim["rho"],
            sim["edge_var"],
            mth["approximation"],
            mth["npermutations"],
            mth["df"],
            sp["make_sparse"],
            sp["sparsity_bias"],
        ]
        return [dict(zip(names, v)) for v in iproduct(*vals)], None

    # -- SBM ------------------------------------------------------------------
    if exp == "sbm":
        sbm = cfg["extra_params"]["sbm"]
        names = [
            "setup",
            "method",
            "n",
            "k",
            "alpha",
            "marginals",
            "rho",
            "edge_var",
            "npermutations",
            "sparsity_bias",
            "prob_switch",
            "assignment_mode",
            "block_probs_type",
            "assortativity",
        ]
        vals = [
            sets,
            mth["list"],
            sim["n"],
            sim["k"],
            sim["alpha"],
            sim["marginals"],
            sim["rho"],
            sim["edge_var"],
            mth["npermutations"],
            sbm["sparsity_bias"],
            sbm["prob_switch"],
            sbm["assignment_mode"],
            sbm["block_probs_type"],
            sbm["assortativity"],
        ]
        return [dict(zip(names, v)) for v in iproduct(*vals)], None

    # -- Standard / observed / diff_marginals ---------------------------------
    h1 = _standard_rows(sets, sim["rho"])

    h0 = None
    if cfg["null_setups"]:
        null = cfg["null_setups"]
        h0 = _standard_rows(null["setups"], null["rho"])

    return h1, h0


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
