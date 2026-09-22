#!/usr/bin/env python3
"""Plot testing performance and Y recovery for the linear-model study.

Populate ``RESULT_FILES`` after all shards finish, or pass the filenames with
``--files``. Optional matching MRQAP and asymptotic RV shards can be supplied
with ``--mrqap-files`` and ``--asymptotic-files``. Networks, true/estimated
latent modes, error distributions, and X-network correlations are plotted
separately. Adjacency-level MRQAP results are shown in both latent-mode panels.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from results.results_processing import (  # noqa: E402
    iter_shard_outputs,
    parse_config_string,
    parse_result_string,
)


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

ASYMPTOTIC_RESULT_FILES: tuple[str, ...] = ()

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
METHOD_ORDER = ("RVTest_permutation", "RVTest_asymptotic", "CCA", "DC", "MRQAP")

METHOD_LABELS = {
    "RVTest_permutation": "RV (permutation)",
    "RVTest_asymptotic": "RV (asymptotic)",
    "CCA": "CCA",
    "DC": "MGC",
    "MRQAP": "MRQAP (adjacency)",
}
COLORS = {
    "RVTest_permutation": "#E69F00",
    "RVTest_asymptotic": "#E69F00",
    "CCA": "#0072B2",
    "DC": "#009E73",
    "MRQAP": "#CC79A7",
}
MARKERS = {
    "RVTest_permutation": "o",
    "RVTest_asymptotic": "D",
    "CCA": "s",
    "DC": "^",
    "MRQAP": "v",
}
LINESTYLES = {
    "RVTest_permutation": "-",
    "RVTest_asymptotic": "--",
    "CCA": "-",
    "DC": "-",
    "MRQAP": "-",
}
PNG_DPI = 600
PLOT_CHUNKSIZE = 5_000


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
        "--testing-only",
        action="store_true",
        help="Save only power and type-I-error figures.",
    )
    return parser.parse_args(argv)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": PNG_DPI,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "figure.titlesize": 10,
            "legend.fontsize": 8,
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


def prepare_linear_model_results(
    results_dir: Path,
    filenames: Sequence[str],
    include_methods: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Stream shards, optionally filtering methods before parsing metrics."""
    included = None if include_methods is None else set(include_methods)
    frames = []
    for chunk in iter_shard_outputs(
        results_dir,
        filenames,
        chunksize=PLOT_CHUNKSIZE,
        usecols=("args", "ComputeAll"),
    ):
        configs = chunk["args"].map(parse_config_string)
        methods = configs.map(_method_label)
        if included is not None:
            keep = methods.isin(included)
            chunk = chunk.loc[keep]
            configs = configs.loc[keep]
            methods = methods.loc[keep]
        if chunk.empty:
            continue
        metrics = chunk["ComputeAll"].map(parse_result_string)
        frames.append(
            pd.DataFrame(
                {
                    "n": configs.map(lambda value: value.get("n")),
                    "p": configs.map(lambda value: value.get("p")),
                    "d_x": configs.map(lambda value: value.get("d_x")),
                    "d_y": configs.map(lambda value: value.get("d_y")),
                    "snr": configs.map(lambda value: value.get("snr")),
                    "alpha": configs.map(lambda value: value.get("alpha")),
                    "x_network_correlation": configs.map(
                        lambda value: _network_correlation(
                            value.get("x_network_correlation")
                        )
                    ),
                    "eps_distribution": configs.map(
                        lambda value: _error_distribution(value.get("eps_distribution"))
                    ),
                    "use_true_latent": configs.map(
                        lambda value: _parse_latent_mode(value.get("use_true_latent"))
                    ),
                    "method": methods,
                    "dgp_name": configs.map(
                        lambda value: _clean_text(value.get("dgp_name")).split("_")[0]
                    ),
                    "Rejection": metrics.map(lambda value: value.get("Rejection")),
                    "RelativeFrobeniusNorm_Y": metrics.map(
                        lambda value: value.get("RelativeFrobeniusNorm_Y")
                    ),
                }
            )
        )

    if not frames:
        detail = " matching the method filter" if included is not None else ""
        raise ValueError(f"The shard files contain no result rows{detail}.")
    results = pd.concat(frames, ignore_index=True)
    numeric_columns = (
        "n",
        "p",
        "d_x",
        "d_y",
        "snr",
        "alpha",
        "x_network_correlation",
        "Rejection",
        "RelativeFrobeniusNorm_Y",
    )
    for column in numeric_columns:
        results[column] = pd.to_numeric(results[column], errors="coerce")

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


def _method_label(config: dict) -> str:
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
    grouping = [
        "dgp_name",
        "p",
        "snr",
        "n",
        "method",
        "alpha",
        "use_true_latent",
        "x_network_correlation",
        "eps_distribution",
    ]
    aggregated = (
        results.groupby(grouping, dropna=False)[value_column]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": mean_column,
                "sem": sem_column,
                "count": "replicates",
            }
        )
    )
    aggregated[sem_column] = aggregated[sem_column].fillna(0.0)
    return aggregated


def aggregate_rejection_rates(results: pd.DataFrame) -> pd.DataFrame:
    """Compute rejection-rate means, SEMs, and replicate counts."""
    return _aggregate_metric(
        results,
        "Rejection",
        "rejection_rate",
        "rejection_sem",
    )


def aggregate_frobenius_errors(results: pd.DataFrame) -> pd.DataFrame:
    """Compute mean Y relative-Frobenius errors and their SEMs."""
    return _aggregate_metric(
        results,
        "RelativeFrobeniusNorm_Y",
        "frobenius_error",
        "frobenius_sem",
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
) -> Path:
    """Plot rows of positive-SNR facets and columns of p facets."""
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
                    f"SNR = {snr:g}",
                    xy=(1.04, 0.5),
                    xycoords="axes fraction",
                    rotation=270,
                    va="center",
                    annotation_clip=False,
                )

    label = NETWORK_LABELS.get(network, network)
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.88))
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
        ncols=len(methods),
        frameon=False,
    )
    slug = network.removesuffix("Network").lower()
    mode_slug = _latent_mode_slug(use_true_latent)
    filename = f"power_{slug}_{mode_slug}"
    if show_setting:
        filename += f"_{_setting_slug(eps_distribution, x_network_correlation)}"
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
        figsize=(2.6 * len(p_values), 3.3),
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
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.82))
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
        ncols=len(handles),
    )
    mode_slug = _latent_mode_slug(use_true_latent)
    filename = f"type_i_error_{slug}_{mode_slug}"
    if show_setting:
        filename += f"_{_setting_slug(eps_distribution, x_network_correlation)}"
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
) -> Path:
    """Create a centered three-over-two type-I-error facet layout."""
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
    if len(p_values) != 5:
        raise ValueError(
            "The two-row type-I-error layout requires exactly five p values; "
            f"found {len(p_values)}."
        )

    methods = _available_methods(data)
    alpha_values = pd.to_numeric(data.get("alpha"), errors="coerce")
    alpha = 0.05 if alpha_values.isna().all() else float(alpha_values.dropna().iloc[0])
    upper_limit = max(0.15, float(data["rejection_rate"].max()) + 0.04)

    fig = plt.figure(figsize=(9.2, 5.8), layout="constrained")
    grid = fig.add_gridspec(2, 6)
    spans = (
        (0, slice(0, 2)),
        (0, slice(2, 4)),
        (0, slice(4, 6)),
        (1, slice(1, 3)),
        (1, slice(3, 5)),
    )
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
        ax.tick_params(labelleft=index in {0, 3})
        if index in {0, 3}:
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
        ncols=len(handles),
    )
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.82))

    slug = network.removesuffix("Network").lower()
    mode_slug = _latent_mode_slug(use_true_latent)
    filename = f"type_i_error_two_row_{slug}_{mode_slug}"
    if show_setting:
        filename += f"_{_setting_slug(eps_distribution, x_network_correlation)}"
    return _save_figure(fig, output_dir, filename)


def plot_frobenius_grid(
    aggregated: pd.DataFrame,
    output_dir: Path,
    network: str,
    use_true_latent: bool,
    eps_distribution: str,
    x_network_correlation: float,
    show_setting: bool = True,
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
    return _save_figure(
        fig,
        output_dir,
        filename,
    )


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
    configure_plot_style()
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
    results = expand_adjacency_results_across_latent_modes(
        pd.concat(result_frames, ignore_index=True)
    )
    rejection_rates = aggregate_rejection_rates(results)
    frobenius_errors = (
        None
        if args.testing_only
        else aggregate_frobenius_errors(
            results[results["RelativeFrobeniusNorm_Y"].notna()]
        )
    )
    outputs = []
    for network in NETWORK_LABELS:
        network_results = results[results["dgp_name"] == network]
        settings = (
            network_results[["eps_distribution", "x_network_correlation"]]
            .drop_duplicates()
            .sort_values(["eps_distribution", "x_network_correlation"])
        )
        show_setting = len(settings) > 1
        for setting in settings.itertuples(index=False):
            for use_true_latent in LATENT_MODE_LABELS:
                outputs.append(
                    plot_power_grid(
                        rejection_rates,
                        args.output_dir,
                        network,
                        use_true_latent,
                        setting.eps_distribution,
                        setting.x_network_correlation,
                        show_setting,
                    )
                )
                outputs.append(
                    plot_type_i_error_by_p(
                        rejection_rates,
                        args.output_dir,
                        network,
                        use_true_latent,
                        setting.eps_distribution,
                        setting.x_network_correlation,
                        show_setting,
                    )
                )
                outputs.append(
                    plot_type_i_error_two_row(
                        rejection_rates,
                        args.output_dir,
                        network,
                        use_true_latent,
                        setting.eps_distribution,
                        setting.x_network_correlation,
                        show_setting,
                    )
                )
                if frobenius_errors is not None:
                    outputs.append(
                        plot_frobenius_grid(
                            frobenius_errors,
                            args.output_dir,
                            network,
                            use_true_latent,
                            setting.eps_distribution,
                            setting.x_network_correlation,
                            show_setting,
                        )
                    )
                    outputs.append(
                        plot_null_frobenius_by_p(
                            frobenius_errors,
                            args.output_dir,
                            network,
                            use_true_latent,
                            setting.eps_distribution,
                            setting.x_network_correlation,
                            show_setting,
                        )
                    )
    print(f"Saved {len(outputs)} figures to {args.output_dir}")
    return outputs


if __name__ == "__main__":
    main()
