#!/usr/bin/env python3
"""Load, preprocess, and plot simulation results.

Populate ``RESULT_FILES`` after all shards finish, or pass the filenames with
``--files``. Optional matching MRQAP and asymptotic RV shards can be supplied
with ``--mrqap-files`` and ``--asymptotic-files``. Networks, true/estimated
latent modes, error distributions, and X-network correlations are plotted
separately. Adjacency-level MRQAP results are shown in both latent-mode panels.

The reusable pipeline is ``merge_result_shards`` followed by
``preprocess_results`` and ``plot_metric_grid``. The study-specific
``generate_linear_model_figures`` wrapper preserves the command-line output.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.analysis.processing import (  # noqa: E402
    PLOT_CHUNKSIZE,
    DEFAULT_COLUMN_ALIASES as DEFAULT_COLUMN_ALIASES,
    DEFAULT_NUMERIC_COLUMNS,
    merge_result_shards,
    merge_result_shard_sets,
    _is_missing_value as _is_missing_value,
    _expand_mapping_column as _expand_mapping_column,
    _overlay_expanded_columns as _overlay_expanded_columns,
    preprocess_results,
    filter_results,
    aggregate_metric,
)
from src.analysis.plotting import (  # noqa: E402
    _column_list as _column_list,
    _ordered_values as _ordered_values,
    plot_metric_grid,
)
from src.analysis.styles import METHOD_LABELS, COLORS, MARKERS, LINESTYLES  # noqa: E402
from src.analysis.io import iter_shard_outputs  # noqa: E402

# Fill these after the three cluster shards have completed. Paths are resolved
# relative to --results-dir.
RESULT_FILES: tuple[str, ...] = (
    "linear_model_results_61599204_shard-000-of-003.csv",
    "linear_model_results_61599204_shard-001-of-003.csv",
    "linear_model_results_61599204_shard-002-of-003.csv",
)
MRQAP_RESULT_FILES: tuple[str, ...] = (
    "linear_model_mrqap_results_61679617_shard-000-of-003.csv",
    "linear_model_mrqap_results_61679617_shard-001-of-003.csv",
    "linear_model_mrqap_results_61679617_shard-002-of-003.csv",
)

ASYMPTOTIC_RESULT_FILES: tuple[str, ...] = (
    "linear_model_asymptotic_results_61636044_shard-000-of-003.csv",
    "linear_model_asymptotic_results_61636044_shard-001-of-003.csv",
    "linear_model_asymptotic_results_61636044_shard-002-of-003.csv",
)

P_ONE_RESULT_FILE = "linear_model_results_20260924_1130.csv"
REDUCED_GRID_P_VALUES = (10, 100)
REDUCED_GRID_SNR_VALUES = (1,)

NETWORK_LABELS = {
    "GaussianNetwork": "Gaussian weighted network",
    "BernoulliNetwork": "Bernoulli binary network",
}
LATENT_MODE_LABELS = {
    False: "Estimated latent positions",
    True: "True latent positions",
}
ERROR_DISTRIBUTION_LABELS = {
    "multivariate_gaussian": "Gaussian errors",
    "student_t_3": r"Student-$t_3$ errors",
}
METHOD_ORDER = (
    "RVTest_permutation",
    "RVTest_asymptotic_independence",
    "RVTest_asymptotic_zero_covariance",
    "CCA",
    "DC",
    "MRQAP",
)


PNG_DPI = 600

ASYMPTOTIC_NULL_OPTIONS = ("split", "independence", "zero_covariance")
ASYMPTOTIC_METHODS = {
    "independence": "RVTest_asymptotic_independence",
    "zero_covariance": "RVTest_asymptotic_zero_covariance",
}


LINEAR_MODEL_CONFIG_FIELDS = (
    "n",
    "p",
    "d_x",
    "d_y",
    "snr",
    "b_active_network_fraction",
    "hypothesis",
    "alpha",
    "x_network_correlation",
    "eps_distribution",
    "use_true_latent",
    "method",
    "dgp_name",
    "approximation",
    "asymptotic_null",
)
LINEAR_MODEL_METRIC_FIELDS = (
    "Rejection",
    "RelativeFrobeniusNorm_Y",
)
LINEAR_MODEL_GROUP_COLUMNS = (
    "dgp_name",
    "p",
    "snr",
    "n",
    "method",
    "alpha",
    "use_true_latent",
    "x_network_correlation",
    "eps_distribution",
)

__all__ = [
    "aggregate_frobenius_errors",
    "aggregate_metric",
    "aggregate_rejection_rates",
    "configure_plot_style",
    "filter_results",
    "generate_linear_model_figures",
    "linear_model_method_label",
    "merge_result_shard_sets",
    "merge_result_shards",
    "plot_metric_grid",
    "prepare_active_fraction_results",
    "prepare_p_one_testing_results",
    "prepare_linear_model_results",
    "preprocess_linear_model_results",
    "preprocess_results",
    "select_asymptotic_null_results",
]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the sharded multiple-network linear-model experiment."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the shard CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "linear_model_figures",
        help="Directory in which to save PNG figures.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Primary shard filenames; overrides RESULT_FILES.",
    )
    parser.add_argument(
        "--asymptotic-files",
        nargs="+",
        default=None,
        help=(
            "Asymptotic shard filenames; overrides ASYMPTOTIC_RESULT_FILES. "
            "Only RV rows are retained."
        ),
    )
    parser.add_argument(
        "--mrqap-files",
        nargs="+",
        default=None,
        help=(
            "Global MRQAP shard filenames; overrides MRQAP_RESULT_FILES. "
            "Their adjacency-level rejection curves are added to both latent-mode "
            "testing plots."
        ),
    )
    parser.add_argument(
        "--asymptotic-null",
        choices=ASYMPTOTIC_NULL_OPTIONS,
        required=True,
        help=(
            "How to handle asymptotic RV results: 'split' draws independence "
            "and zero-covariance lines; either null name retains only that line."
        ),
    )
    parser.add_argument(
        "--testing-only",
        action="store_true",
        help="Save only power and type-I-error figures.",
    )
    parser.add_argument(
        "--reduced-grid",
        action="store_true",
        help="Exclude p=10, p=100, and SNR=1 and save separate reduced-grid files.",
    )
    parser.add_argument(
        "--p-one-file",
        default=P_ONE_RESULT_FILE,
        help="Unsharded result file providing the p=1 null and alternative rows.",
    )
    return parser.parse_args(argv)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": PNG_DPI,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "figure.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "lines.linewidth": 1.3,
            "lines.markersize": 4.5,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.facecolor": "white",
        }
    )


def _validate_linear_model_results(results: pd.DataFrame) -> None:
    required = (
        "n",
        "p",
        "snr",
        "x_network_correlation",
        "eps_distribution",
        "method",
        "dgp_name",
        "Rejection",
    )
    missing = [column for column in required if column not in results]
    if missing:
        raise ValueError(f"Processed results are missing columns: {missing}")
    invalid = results[list(required)].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} rows have missing plotting fields.")

    adjacency_rows = results["method"] == "MRQAP"
    invalid_latent_mode = ~adjacency_rows & results["use_true_latent"].isna()
    if invalid_latent_mode.any():
        raise ValueError(
            f"{int(invalid_latent_mode.sum())} non-MRQAP rows have no "
            "use_true_latent value."
        )
    missing_frobenius = ~adjacency_rows & results["RelativeFrobeniusNorm_Y"].isna()
    if missing_frobenius.any():
        raise ValueError(
            f"{int(missing_frobenius.sum())} non-MRQAP rows have no Y Frobenius "
            "metric."
        )


def preprocess_linear_model_results(
    raw_results: pd.DataFrame,
    *,
    include_methods: Sequence[str] | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Apply study-specific normalization after generic preprocessing."""
    results = preprocess_results(
        raw_results,
        config_fields=LINEAR_MODEL_CONFIG_FIELDS,
        metric_fields=LINEAR_MODEL_METRIC_FIELDS,
        numeric_columns=(
            *DEFAULT_NUMERIC_COLUMNS,
            "b_active_network_fraction",
            "x_network_correlation",
            "RelativeFrobeniusNorm_Y",
        ),
        method_labeler=linear_model_method_label,
    )
    results["x_network_correlation"] = results["x_network_correlation"].map(
        _network_correlation
    )
    results["eps_distribution"] = results["eps_distribution"].map(_error_distribution)
    results["hypothesis"] = results["hypothesis"].map(_clean_text)
    results["use_true_latent"] = results["use_true_latent"].map(_parse_latent_mode)
    results["dgp_name"] = results["dgp_name"].map(
        lambda value: _clean_text(value).split("_")[0]
    )
    if include_methods is not None:
        results = results[results["method"].isin(include_methods)].copy()
    results = results.reset_index(drop=True)
    if validate and not results.empty:
        _validate_linear_model_results(results)
    return results


def prepare_linear_model_results(
    results_dir: Path,
    filenames: Sequence[str],
    include_methods: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Stream, preprocess, and concatenate linear-model shard rows.

    For staged processing, call :func:`merge_result_shards` followed
    by :func:`preprocess_linear_model_results` instead.
    """
    frames = []
    for chunk in iter_shard_outputs(
        results_dir,
        filenames,
        chunksize=PLOT_CHUNKSIZE,
        usecols=("args", "ComputeAll"),
    ):
        processed = preprocess_linear_model_results(
            chunk,
            include_methods=include_methods,
            validate=False,
        )
        if not processed.empty:
            frames.append(processed)
    if not frames:
        detail = " matching the method filter" if include_methods is not None else ""
        raise ValueError(f"The shard files contain no result rows{detail}.")
    results = pd.concat(frames, ignore_index=True)
    _validate_linear_model_results(results)
    return results


def prepare_p_one_testing_results(
    results_dir: Path,
    filename: str | Path,
) -> pd.DataFrame:
    """Load an unsharded p=1 file containing null and alternative rows."""
    frames = []
    root = Path(results_dir).expanduser().resolve()
    path = Path(filename)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"Missing p=1 result file: {path}")

    for chunk in pd.read_csv(
        path,
        chunksize=PLOT_CHUNKSIZE,
        usecols=("args", "ComputeAll"),
    ):
        chunk["source_file"] = path.name
        processed = preprocess_linear_model_results(chunk, validate=False)
        selected = processed[processed["p"] == 1].copy()
        if not selected.empty:
            frames.append(selected)

    if not frames:
        raise ValueError(f"The p=1 result file has no p=1 rows: {path}")

    results = pd.concat(frames, ignore_index=True)
    _validate_linear_model_results(results)
    if not (results["snr"] == 0).any() or not (results["snr"] > 0).any():
        raise ValueError(
            f"The p=1 result file must contain null and alternative rows: {path}"
        )
    return results


def prepare_active_fraction_results(
    results_dir: Path,
    filenames: Sequence[str],
) -> pd.DataFrame:
    """Load active-fraction shards for the shared testing-plot pipeline.

    The plotting internals use ``snr`` as their effect-size coordinate. Active
    fractions are copied into that column after preprocessing; the rendered
    figure labels still identify the coordinate as active fraction.
    """
    frames = []
    for chunk in iter_shard_outputs(
        results_dir,
        filenames,
        chunksize=PLOT_CHUNKSIZE,
        usecols=("args", "ComputeAll"),
    ):
        processed = preprocess_linear_model_results(chunk, validate=False)
        fractions = pd.to_numeric(
            processed["b_active_network_fraction"],
            errors="coerce",
        )
        if fractions.isna().any():
            raise ValueError(
                "Active-fraction result rows have missing "
                "b_active_network_fraction values."
            )
        processed["snr"] = fractions
        frames.append(processed)

    if not frames:
        raise ValueError("The active-fraction shard files contain no result rows.")
    results = pd.concat(frames, ignore_index=True)
    _validate_linear_model_results(results)
    return results


def _clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip().strip("'\"")


def _parse_latent_mode(value) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = _clean_text(value).lower()
    if normalized in {"", "na", "nan", "none"}:
        return None
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid use_true_latent value: {value!r}")


def _error_distribution(value) -> str:
    """Use the DGP default for result files that predate recorded metadata."""
    normalized = _clean_text(value)
    if normalized.lower() in {"", "na", "nan", "none"}:
        return "multivariate_gaussian"
    return normalized


def _network_correlation(value) -> float:
    """Use the DGP default for result files that predate recorded metadata."""
    normalized = _clean_text(value).lower()
    if normalized in {"", "na", "nan", "none"}:
        return 0.0
    try:
        return float(normalized)
    except ValueError as error:
        raise ValueError(f"Invalid x_network_correlation value: {value!r}") from error


def expand_adjacency_results_across_latent_modes(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Copy adjacency-only MRQAP rows into both latent comparison panels."""
    adjacency = (results["method"] == "MRQAP") & results["use_true_latent"].isna()
    if not adjacency.any():
        return results.copy()

    ordinary = results.loc[~adjacency]
    mrqap = results.loc[adjacency]
    return pd.concat(
        [
            ordinary,
            mrqap.assign(use_true_latent=False),
            mrqap.assign(use_true_latent=True),
        ],
        ignore_index=True,
    )


def select_asymptotic_null_results(
    results: pd.DataFrame,
    asymptotic_null: str,
) -> pd.DataFrame:
    """Split asymptotic RV rows by null model or retain one requested model.

    Parameters
    ----------
    results
        Preprocessed simulation rows.
    asymptotic_null
        One of ``"split"``, ``"independence"``, or ``"zero_covariance"``.
        Split mode relabels both variants as distinct methods. A specific null
        filters out the other variant. There is intentionally no pooled mode.
    """
    if asymptotic_null not in ASYMPTOTIC_NULL_OPTIONS:
        choices = ", ".join(repr(value) for value in ASYMPTOTIC_NULL_OPTIONS)
        raise ValueError(f"asymptotic_null must be one of: {choices}.")
    if "method" not in results:
        raise ValueError("Processed results are missing column: method")

    selected = results.copy()
    asymptotic_methods = {"RVTest_asymptotic", *ASYMPTOTIC_METHODS.values()}
    asymptotic_rows = selected["method"].isin(asymptotic_methods)
    if not asymptotic_rows.any():
        return selected

    if "asymptotic_null" not in selected:
        selected["asymptotic_null"] = None
    null_models = selected["asymptotic_null"].map(_clean_text)
    for null_model, method in ASYMPTOTIC_METHODS.items():
        inferred = selected["method"] == method
        null_models.loc[inferred] = null_model

    invalid = asymptotic_rows & ~null_models.isin(ASYMPTOTIC_METHODS)
    if invalid.any():
        invalid_values = sorted(set(null_models.loc[invalid]))
        raise ValueError(
            "Asymptotic RV rows have missing or unsupported asymptotic_null "
            f"values: {invalid_values}."
        )

    selected.loc[asymptotic_rows, "asymptotic_null"] = null_models.loc[asymptotic_rows]
    if asymptotic_null != "split":
        keep = ~asymptotic_rows | (null_models == asymptotic_null)
        selected = selected.loc[keep].copy()
        asymptotic_rows = selected["method"].isin(asymptotic_methods)
        if not asymptotic_rows.any():
            raise ValueError(
                "No asymptotic RV rows found for "
                f"asymptotic_null={asymptotic_null!r}."
            )

    selected.loc[asymptotic_rows, "method"] = selected.loc[
        asymptotic_rows, "asymptotic_null"
    ].map(ASYMPTOTIC_METHODS)
    return selected.reset_index(drop=True)


def linear_model_method_label(config: Mapping[str, Any]) -> str:
    """Return the compact method label used by the linear-model figures."""
    method = _clean_text(config.get("method"))
    if method == "RVTest":
        approximation = _clean_text(config.get("approximation"))
        return f"RVTest_{approximation}"
    if method == "CanonicalCorrelationTest":
        return "CCA"
    if method == "DistanceCorrelationTest":
        return "DC"
    return method


def _aggregate_metric(
    results: pd.DataFrame,
    value_column: str,
    mean_column: str,
    sem_column: str,
) -> pd.DataFrame:
    return aggregate_metric(
        results,
        value_column,
        LINEAR_MODEL_GROUP_COLUMNS,
        mean_column=mean_column,
        error="sem",
        error_column=sem_column,
    )


def aggregate_rejection_rates(
    results: pd.DataFrame,
    groupby: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute rejection-rate means, optionally retaining extra variants."""
    return aggregate_metric(
        results,
        "Rejection",
        LINEAR_MODEL_GROUP_COLUMNS if groupby is None else groupby,
        mean_column="rejection_rate",
        error="sem",
        error_column="rejection_sem",
    )


def aggregate_frobenius_errors(
    results: pd.DataFrame,
    groupby: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute mean Y relative-Frobenius errors and their SEMs."""
    return aggregate_metric(
        results,
        "RelativeFrobeniusNorm_Y",
        LINEAR_MODEL_GROUP_COLUMNS if groupby is None else groupby,
        mean_column="frobenius_error",
        error="sem",
        error_column="frobenius_sem",
    )


def _method_handles(methods: Sequence[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            markerfacecolor="none",
            label=METHOD_LABELS[method],
        )
        for method in methods
    ]


def _available_methods(data: pd.DataFrame) -> tuple[str, ...]:
    available = set(data["method"])
    unknown = sorted(available - set(METHOD_ORDER))
    if unknown:
        raise ValueError(f"Add plotting styles for methods: {unknown}")
    return tuple(method for method in METHOD_ORDER if method in available)


def _plot_curves(
    ax: Axes,
    data: pd.DataFrame,
    methods: Sequence[str],
    *,
    mean_column: str,
    sem_column: str,
    upper_clip: float | None = None,
) -> None:
    duplicates = data.duplicated(["n", "method"], keep=False)
    if duplicates.any():
        raise ValueError("A plot panel contains duplicate method/n combinations.")

    for method in methods:
        subset = data[data["method"] == method].sort_values("n")
        if subset.empty:
            continue
        x = subset["n"].to_numpy(dtype=float)
        mean = subset[mean_column].to_numpy(dtype=float)
        sem = subset[sem_column].to_numpy(dtype=float)
        ax.plot(
            x,
            mean,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            markerfacecolor="none",
            markeredgewidth=0.9,
            zorder=3,
        )
        lower = np.clip(mean - sem, 0, upper_clip)
        upper = mean + sem
        if upper_clip is not None:
            upper = np.clip(upper, 0, upper_clip)
        ax.fill_between(
            x,
            lower,
            upper,
            color=COLORS[method],
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )

    if not data.empty:
        ax.set_xticks(sorted(data["n"].unique()))
    ax.grid(axis="y", color="#E2E2E2", linewidth=0.45)


def _latent_mode_slug(use_true_latent: bool) -> str:
    return "true_latent" if use_true_latent else "estimated_latent"


def _latent_mode_label(use_true_latent: bool) -> str:
    return LATENT_MODE_LABELS[use_true_latent]


def _setting_label(eps_distribution: str, x_network_correlation: float) -> str:
    error_label = ERROR_DISTRIBUTION_LABELS.get(
        eps_distribution,
        eps_distribution.replace("_", " "),
    )
    return f"{error_label} — X-network correlation = {x_network_correlation:g}"


def _setting_slug(eps_distribution: str, x_network_correlation: float) -> str:
    distribution_slug = "_".join(eps_distribution.lower().split())
    correlation_slug = f"{x_network_correlation:g}".replace("-", "m").replace(".", "p")
    return f"{distribution_slug}_xcorr_{correlation_slug}"


def _select_setting(
    aggregated: pd.DataFrame,
    network: str,
    use_true_latent: bool,
    eps_distribution: str,
    x_network_correlation: float,
) -> pd.DataFrame:
    return aggregated[
        (aggregated["dgp_name"] == network)
        & (aggregated["use_true_latent"] == use_true_latent)
        & (aggregated["eps_distribution"] == eps_distribution)
        & np.isclose(
            aggregated["x_network_correlation"],
            x_network_correlation,
        )
    ].copy()


def _frobenius_upper_limit(data: pd.DataFrame) -> float:
    upper_values = data["frobenius_error"] + data["frobenius_sem"]
    return max(0.05, float(upper_values.max()) * 1.05)


def _save_figure(fig: Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename}.png"
    print(f"Saving {path}")
    fig.savefig(
        path,
        dpi=PNG_DPI,
        metadata={"Creator": "visualise_linear_model.py"},
    )
    plt.close(fig)
    return path


def plot_power_grid(
    aggregated: pd.DataFrame,
    output_dir: Path,
    network: str,
    use_true_latent: bool,
    eps_distribution: str,
    x_network_correlation: float,
    show_setting: bool = True,
    filename_suffix: str | None = None,
    effect_label: str = "SNR",
) -> Path:
    """Plot rows of positive effect sizes and columns of p facets."""
    data = _select_setting(
        aggregated,
        network,
        use_true_latent,
        eps_distribution,
        x_network_correlation,
    )
    data = data[data["snr"] > 0].copy()
    if data.empty:
        raise ValueError(
            f"No alternative rows found for {network}, "
            f"use_true_latent={use_true_latent}, "
            f"eps_distribution={eps_distribution}, "
            f"x_network_correlation={x_network_correlation:g}."
        )
    p_values = tuple(sorted(data["p"].unique()))
    snr_values = tuple(sorted(data["snr"].unique()))
    methods = _available_methods(data)

    fig, axes = plt.subplots(
        len(snr_values),
        len(p_values),
        figsize=(max(12.9, 2.15 * len(p_values)), 1.8 * len(snr_values) + 0.8),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for row, snr in enumerate(snr_values):
        for column, p_value in enumerate(p_values):
            ax = axes[row, column]
            panel = data[(data["p"] == p_value) & (data["snr"] == snr)]
            _plot_curves(
                ax,
                panel,
                methods,
                mean_column="rejection_rate",
                sem_column="rejection_sem",
                upper_clip=1.0,
            )
            ax.set_ylim(-0.02, 1.02)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            if row == 0:
                ax.set_title(f"p = {int(p_value)}")
            if column == len(p_values) - 1:
                ax.annotate(
                    f"{effect_label} = {snr:g}",
                    xy=(1.04, 0.5),
                    xycoords="axes fraction",
                    rotation=270,
                    va="center",
                    annotation_clip=False,
                )

    label = NETWORK_LABELS.get(network, network)
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.82))
    mode_label = _latent_mode_label(use_true_latent)
    title = f"Power — {label} — {mode_label}"
    if show_setting:
        title += f"\n{_setting_label(eps_distribution, x_network_correlation)}"
    fig.suptitle(title, y=0.995)
    fig.supxlabel(r"Network size, $n$")
    fig.supylabel("Power")
    fig.legend(
        handles=_method_handles(methods),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncols=min(3, len(methods)),
        frameon=False,
    )
    slug = network.removesuffix("Network").lower()
    mode_slug = _latent_mode_slug(use_true_latent)
    filename = f"power_{slug}_{mode_slug}"
    if show_setting:
        filename += f"_{_setting_slug(eps_distribution, x_network_correlation)}"
    if filename_suffix:
        filename += f"_{filename_suffix}"
    return _save_figure(
        fig,
        output_dir,
        filename,
    )


def plot_type_i_error_by_p(
    aggregated: pd.DataFrame,
    output_dir: Path,
    network: str,
    use_true_latent: bool,
    eps_distribution: str,
    x_network_correlation: float,
    show_setting: bool = True,
    filename_suffix: str | None = None,
) -> Path:
    """Create one row of SNR-zero type-I-error facets over p."""
    data = _select_setting(
        aggregated,
        network,
        use_true_latent,
        eps_distribution,
        x_network_correlation,
    )
    data = data[data["snr"] == 0].copy()
    if data.empty:
        raise ValueError(
            f"No null rows found for {network}, "
            f"use_true_latent={use_true_latent}, "
            f"eps_distribution={eps_distribution}, "
            f"x_network_correlation={x_network_correlation:g}."
        )
    methods = _available_methods(data)
    alpha_values = pd.to_numeric(data.get("alpha"), errors="coerce")
    alpha = 0.05 if alpha_values.isna().all() else float(alpha_values.dropna().iloc[0])
    label = NETWORK_LABELS.get(network, network)
    slug = network.removesuffix("Network").lower()
    p_values = tuple(sorted(data["p"].unique()))
    fig, axes = plt.subplots(
        1,
        len(p_values),
        figsize=(max(15.6, 2.6 * len(p_values)), 4.2),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    upper_limit = max(0.15, float(data["rejection_rate"].max()) + 0.04)
    for column, p_value in enumerate(p_values):
        panel = data[data["p"] == p_value]
        ax = axes[0, column]
        _plot_curves(
            ax,
            panel,
            methods,
            mean_column="rejection_rate",
            sem_column="rejection_sem",
            upper_clip=1.0,
        )
        ax.axhline(
            alpha,
            color="#555555",
            linestyle="--",
            linewidth=1,
            label=rf"Nominal $\alpha={alpha:g}$",
        )
        ax.set_ylim(0, upper_limit)
        ax.set_title(f"p = {int(p_value)}")

    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.72))
    mode_label = _latent_mode_label(use_true_latent)
    title = f"Type I error — {label} — {mode_label}"
    if show_setting:
        title += f"\n{_setting_label(eps_distribution, x_network_correlation)}"
    fig.suptitle(title, y=0.995)
    fig.supxlabel(r"Network size, $n$")
    fig.supylabel("Type I error rate")
    handles = _method_handles(methods)
    handles.append(
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle="--",
            label=rf"Nominal $\alpha={alpha:g}$",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        frameon=False,
        ncols=min(4, len(handles)),
    )
    mode_slug = _latent_mode_slug(use_true_latent)
    filename = f"type_i_error_{slug}_{mode_slug}"
    if show_setting:
        filename += f"_{_setting_slug(eps_distribution, x_network_correlation)}"
    if filename_suffix:
        filename += f"_{filename_suffix}"
    return _save_figure(
        fig,
        output_dir,
        filename,
    )


def plot_type_i_error_two_row(
    aggregated: pd.DataFrame,
    output_dir: Path,
    network: str,
    use_true_latent: bool,
    eps_distribution: str,
    x_network_correlation: float,
    show_setting: bool = True,
    filename_suffix: str | None = None,
) -> Path:
    """Create a two-row type-I-error facet layout for four to six p values."""
    data = _select_setting(
        aggregated,
        network,
        use_true_latent,
        eps_distribution,
        x_network_correlation,
    )
    data = data[data["snr"] == 0].copy()
    if data.empty:
        raise ValueError(
            f"No null rows found for {network}, "
            f"use_true_latent={use_true_latent}, "
            f"eps_distribution={eps_distribution}, "
            f"x_network_correlation={x_network_correlation:g}."
        )

    p_values = tuple(sorted(data["p"].unique()))
    if len(p_values) not in {4, 5, 6}:
        raise ValueError(
            "The two-row type-I-error layout requires four to six p values; "
            f"found {len(p_values)}."
        )

    methods = _available_methods(data)
    alpha_values = pd.to_numeric(data.get("alpha"), errors="coerce")
    alpha = 0.05 if alpha_values.isna().all() else float(alpha_values.dropna().iloc[0])
    upper_limit = max(0.15, float(data["rejection_rate"].max()) + 0.04)

    fig = plt.figure(figsize=(11.5, 6.5), layout="constrained")
    grid = fig.add_gridspec(2, 6)
    if len(p_values) == 4:
        first_row = (
            (0, slice(1, 3)),
            (0, slice(3, 5)),
        )
        second_row = (
            (1, slice(1, 3)),
            (1, slice(3, 5)),
        )
    else:
        first_row = (
            (0, slice(0, 2)),
            (0, slice(2, 4)),
            (0, slice(4, 6)),
        )
        second_row = (
            (1, slice(1, 3)),
            (1, slice(3, 5)),
        )
        if len(p_values) == 6:
            second_row = (
                (1, slice(0, 2)),
                (1, slice(2, 4)),
                (1, slice(4, 6)),
            )
    spans = (*first_row, *second_row)
    axes = []
    for index, (row, columns) in enumerate(spans):
        shared = axes[0] if axes else None
        ax = fig.add_subplot(grid[row, columns], sharex=shared, sharey=shared)
        axes.append(ax)

        p_value = p_values[index]
        panel = data[data["p"] == p_value]
        _plot_curves(
            ax,
            panel,
            methods,
            mean_column="rejection_rate",
            sem_column="rejection_sem",
            upper_clip=1.0,
        )
        ax.axhline(
            alpha,
            color="#555555",
            linestyle="--",
            linewidth=1,
        )
        ax.set_ylim(0, upper_limit)
        ax.set_title(f"p = {int(p_value)}")
        first_in_row = index == 0 or spans[index - 1][0] != row
        ax.tick_params(labelleft=first_in_row)
        if first_in_row:
            ax.set_ylabel("Type I error rate")

    label = NETWORK_LABELS.get(network, network)
    mode_label = _latent_mode_label(use_true_latent)
    title = f"Type I error — {label} — {mode_label}"
    if show_setting:
        title += f"\n{_setting_label(eps_distribution, x_network_correlation)}"
    fig.suptitle(title, y=0.995)
    fig.supxlabel(r"Network size, $n$")

    handles = _method_handles(methods)
    handles.append(
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle="--",
            label=rf"Nominal $\alpha={alpha:g}$",
        )
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        frameon=False,
        ncols=min(4, len(handles)),
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        legend_top = 0.76 if len(handles) > 4 else 0.82
        layout_engine.set(rect=(0.0, 0.0, 1.0, legend_top))

    slug = network.removesuffix("Network").lower()
    mode_slug = _latent_mode_slug(use_true_latent)
    filename = f"type_i_error_two_row_{slug}_{mode_slug}"
    if show_setting:
        filename += f"_{_setting_slug(eps_distribution, x_network_correlation)}"
    if filename_suffix:
        filename += f"_{filename_suffix}"
    return _save_figure(fig, output_dir, filename)


def plot_frobenius_grid(
    aggregated: pd.DataFrame,
    output_dir: Path,
    network: str,
    use_true_latent: bool,
    eps_distribution: str,
    x_network_correlation: float,
    show_setting: bool = True,
    filename_suffix: str | None = None,
) -> Path:
    """Plot Y errors with SNR rows and p columns over n."""
    data = _select_setting(
        aggregated,
        network,
        use_true_latent,
        eps_distribution,
        x_network_correlation,
    )
    data = data[data["snr"] > 0].copy()
    if data.empty:
        raise ValueError(
            f"No alternative Frobenius rows found for {network}, "
            f"use_true_latent={use_true_latent}, "
            f"eps_distribution={eps_distribution}, "
            f"x_network_correlation={x_network_correlation:g}."
        )
    p_values = tuple(sorted(data["p"].unique()))
    snr_values = tuple(sorted(data["snr"].unique()))
    methods = _available_methods(data)
    upper_limit = _frobenius_upper_limit(data)

    fig, axes = plt.subplots(
        len(snr_values),
        len(p_values),
        figsize=(2.15 * len(p_values), 1.8 * len(snr_values) + 0.8),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for row, snr in enumerate(snr_values):
        for column, p_value in enumerate(p_values):
            ax = axes[row, column]
            panel = data[(data["p"] == p_value) & (data["snr"] == snr)]
            _plot_curves(
                ax,
                panel,
                methods,
                mean_column="frobenius_error",
                sem_column="frobenius_sem",
            )
            ax.set_ylim(0, upper_limit)
            if row == 0:
                ax.set_title(f"p = {int(p_value)}")
            if column == len(p_values) - 1:
                ax.annotate(
                    f"SNR = {snr:g}",
                    xy=(1.04, 0.5),
                    xycoords="axes fraction",
                    rotation=270,
                    va="center",
                    annotation_clip=False,
                )

    label = NETWORK_LABELS.get(network, network)
    mode_label = _latent_mode_label(use_true_latent)
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.88))
    title = f"Y relative Frobenius error — {label} — {mode_label}"
    if show_setting:
        title += f"\n{_setting_label(eps_distribution, x_network_correlation)}"
    fig.suptitle(title, y=0.995)
    fig.supxlabel(r"Network size, $n$")
    fig.supylabel(r"Relative Frobenius error, $\|\hat Y-Y\|_F/\|Y\|_F$")
    fig.legend(
        handles=_method_handles(methods),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncols=len(methods),
        frameon=False,
    )
    slug = network.removesuffix("Network").lower()
    mode_slug = _latent_mode_slug(use_true_latent)
    filename = f"frobenius_y_{slug}_{mode_slug}"
    if show_setting:
        filename += f"_{_setting_slug(eps_distribution, x_network_correlation)}"
    if filename_suffix:
        filename += f"_{filename_suffix}"
    return _save_figure(
        fig,
        output_dir,
        filename,
    )


def plot_null_frobenius_by_p(
    aggregated: pd.DataFrame,
    output_dir: Path,
    network: str,
    use_true_latent: bool,
    eps_distribution: str,
    x_network_correlation: float,
    show_setting: bool = True,
    filename_suffix: str | None = None,
) -> Path:
    """Plot SNR-zero Y relative-Frobenius error in one row of p facets."""
    data = _select_setting(
        aggregated,
        network,
        use_true_latent,
        eps_distribution,
        x_network_correlation,
    )
    data = data[data["snr"] == 0].copy()
    if data.empty:
        raise ValueError(
            f"No null Frobenius rows found for {network}, "
            f"use_true_latent={use_true_latent}, "
            f"eps_distribution={eps_distribution}, "
            f"x_network_correlation={x_network_correlation:g}."
        )
    methods = _available_methods(data)
    p_values = tuple(sorted(data["p"].unique()))
    upper_limit = _frobenius_upper_limit(data)
    fig, axes = plt.subplots(
        1,
        len(p_values),
        figsize=(2.6 * len(p_values), 3.3),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for column, p_value in enumerate(p_values):
        panel = data[data["p"] == p_value]
        ax = axes[0, column]
        _plot_curves(
            ax,
            panel,
            methods,
            mean_column="frobenius_error",
            sem_column="frobenius_sem",
        )
        ax.set_ylim(0, upper_limit)
        ax.set_title(f"p = {int(p_value)}")

    label = NETWORK_LABELS.get(network, network)
    mode_label = _latent_mode_label(use_true_latent)
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.82))
    title = f"Y relative Frobenius error under H0 — {label} — {mode_label}"
    if show_setting:
        title += f"\n{_setting_label(eps_distribution, x_network_correlation)}"
    fig.suptitle(title, y=0.995)
    fig.supxlabel(r"Network size, $n$")
    fig.supylabel(r"Relative Frobenius error, $\|\hat Y-Y\|_F/\|Y\|_F$")
    fig.legend(
        handles=_method_handles(methods),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        frameon=False,
        ncols=len(methods),
    )
    slug = network.removesuffix("Network").lower()
    mode_slug = _latent_mode_slug(use_true_latent)
    filename = f"frobenius_y_null_{slug}_{mode_slug}"
    if show_setting:
        filename += f"_{_setting_slug(eps_distribution, x_network_correlation)}"
    if filename_suffix:
        filename += f"_{filename_suffix}"
    return _save_figure(
        fig,
        output_dir,
        filename,
    )


def _select_reduced_grid(results: pd.DataFrame) -> pd.DataFrame:
    """Return the optional plot subset with selected p and SNR values removed."""
    return results[
        ~results["p"].isin(REDUCED_GRID_P_VALUES)
        & ~results["snr"].isin(REDUCED_GRID_SNR_VALUES)
    ].copy()


def generate_linear_model_figures(
    results: pd.DataFrame,
    output_dir: str | Path,
    *,
    asymptotic_null: str,
    additional_testing_results: pd.DataFrame | None = None,
    effect_label: str = "SNR",
    reduced_grid: bool = False,
    testing_only: bool = False,
    apply_style: bool = True,
) -> list[Path]:
    """Generate figures with split or selected asymptotic RV null models."""
    results = select_asymptotic_null_results(results, asymptotic_null)
    if reduced_grid:
        results = _select_reduced_grid(results)
    _validate_linear_model_results(results)
    if apply_style:
        configure_plot_style()
    output_dir = Path(output_dir)
    frobenius_results = expand_adjacency_results_across_latent_modes(results)
    testing_results = results
    if additional_testing_results is not None:
        additional_testing_results = select_asymptotic_null_results(
            additional_testing_results,
            asymptotic_null,
        )
        if reduced_grid:
            additional_testing_results = _select_reduced_grid(
                additional_testing_results
            )
        _validate_linear_model_results(additional_testing_results)
        testing_results = pd.concat(
            [testing_results, additional_testing_results],
            ignore_index=True,
        )
    testing_results = expand_adjacency_results_across_latent_modes(testing_results)
    rejection_rates = aggregate_rejection_rates(testing_results)
    frobenius_errors = (
        None
        if testing_only
        else aggregate_frobenius_errors(
            frobenius_results[frobenius_results["RelativeFrobeniusNorm_Y"].notna()]
        )
    )
    filename_suffix = f"asymptotic_{asymptotic_null}"
    if reduced_grid:
        filename_suffix += "_reduced_grid"
    outputs = []
    available_networks = testing_results["dgp_name"].drop_duplicates().tolist()
    networks = [name for name in NETWORK_LABELS if name in available_networks]
    networks.extend(name for name in available_networks if name not in networks)
    for network in networks:
        network_results = testing_results[testing_results["dgp_name"] == network]
        settings = (
            network_results[["eps_distribution", "x_network_correlation"]]
            .drop_duplicates()
            .sort_values(["eps_distribution", "x_network_correlation"])
        )
        show_setting = len(settings) > 1
        latent_modes = [
            mode
            for mode in LATENT_MODE_LABELS
            if mode in set(network_results["use_true_latent"].dropna())
        ]
        for setting in settings.itertuples(index=False):
            for use_true_latent in latent_modes:
                outputs.append(
                    plot_power_grid(
                        rejection_rates,
                        output_dir,
                        network,
                        use_true_latent,
                        setting.eps_distribution,
                        setting.x_network_correlation,
                        show_setting,
                        filename_suffix=filename_suffix,
                        effect_label=effect_label,
                    )
                )
                outputs.append(
                    plot_type_i_error_by_p(
                        rejection_rates,
                        output_dir,
                        network,
                        use_true_latent,
                        setting.eps_distribution,
                        setting.x_network_correlation,
                        show_setting,
                        filename_suffix=filename_suffix,
                    )
                )
                outputs.append(
                    plot_type_i_error_two_row(
                        rejection_rates,
                        output_dir,
                        network,
                        use_true_latent,
                        setting.eps_distribution,
                        setting.x_network_correlation,
                        show_setting,
                        filename_suffix=filename_suffix,
                    )
                )
                if frobenius_errors is not None:
                    outputs.append(
                        plot_frobenius_grid(
                            frobenius_errors,
                            output_dir,
                            network,
                            use_true_latent,
                            setting.eps_distribution,
                            setting.x_network_correlation,
                            show_setting,
                            filename_suffix=filename_suffix,
                        )
                    )
                    outputs.append(
                        plot_null_frobenius_by_p(
                            frobenius_errors,
                            output_dir,
                            network,
                            use_true_latent,
                            setting.eps_distribution,
                            setting.x_network_correlation,
                            show_setting,
                            filename_suffix=filename_suffix,
                        )
                    )
    return outputs


def main(argv=None) -> list[Path]:
    args = parse_args(argv)
    filenames = tuple(args.files) if args.files is not None else RESULT_FILES
    asymptotic_filenames = (
        tuple(args.asymptotic_files)
        if args.asymptotic_files is not None
        else ASYMPTOTIC_RESULT_FILES
    )
    mrqap_filenames = (
        tuple(args.mrqap_files) if args.mrqap_files is not None else MRQAP_RESULT_FILES
    )
    if not filenames:
        raise ValueError(
            "No shard files configured. Populate RESULT_FILES or pass --files."
        )

    primary_results = prepare_linear_model_results(args.results_dir, filenames)
    result_frames = [primary_results]
    if mrqap_filenames:
        result_frames.append(
            prepare_linear_model_results(
                args.results_dir,
                mrqap_filenames,
                include_methods=("MRQAP",),
            )
        )
    if asymptotic_filenames:
        result_frames.append(
            prepare_linear_model_results(
                args.results_dir,
                asymptotic_filenames,
                include_methods=("RVTest_asymptotic",),
            )
        )
    results = pd.concat(result_frames, ignore_index=True)
    p_one_testing_results = prepare_p_one_testing_results(
        args.results_dir,
        args.p_one_file,
    )
    outputs = generate_linear_model_figures(
        results,
        args.output_dir,
        asymptotic_null=args.asymptotic_null,
        additional_testing_results=p_one_testing_results,
        reduced_grid=args.reduced_grid,
        testing_only=args.testing_only,
    )
    print(f"Saved {len(outputs)} figures to {args.output_dir}")
    return outputs


if __name__ == "__main__":
    main()
