#!/usr/bin/env python3
"""Plot testing performance and Y recovery for the linear-model study.

Populate ``RESULT_FILES`` and ``ASYMPTOTIC_RESULT_FILES`` after all shards
finish, or pass the filenames with ``--files`` and ``--asymptotic-files``.
Only asymptotic RV rows are retained from the asymptotic run. Networks and
true/estimated latent modes are plotted separately.
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
ASYMPTOTIC_RESULT_FILES: tuple[str, ...] = (
    "linear_model_asymptotic_results_61628751_shard-000-of-003.csv",
    "linear_model_asymptotic_results_61628751_shard-001-of-003.csv",
    "linear_model_asymptotic_results_61628751_shard-002-of-003.csv",
)

NETWORK_LABELS = {
    "GaussianNetwork": "Gaussian weighted network",
    "BernoulliNetwork": "Bernoulli binary network",
}
LATENT_MODE_LABELS = {
    False: "Estimated latent positions",
    True: "True latent positions",
}
METHOD_ORDER = ("RVTest_permutation", "RVTest_asymptotic", "CCA", "DC")
METHOD_LABELS = {
    "RVTest_permutation": "RV (permutation)",
    "RVTest_asymptotic": "RV (asymptotic)",
    "CCA": "CCA",
    "DC": "MGC",
}
COLORS = {
    "RVTest_permutation": "#E69F00",
    "RVTest_asymptotic": "#E69F00",
    "CCA": "#0072B2",
    "DC": "#009E73",
}
MARKERS = {
    "RVTest_permutation": "o",
    "RVTest_asymptotic": "D",
    "CCA": "s",
    "DC": "^",
}
LINESTYLES = {
    "RVTest_permutation": "-",
    "RVTest_asymptotic": "--",
    "CCA": "-",
    "DC": "-",
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
                    "use_true_latent": configs.map(
                        lambda value: _parse_latent_mode(
                            value.get("use_true_latent")
                        )
                    ),
                    "method": methods,
                    "dgp_name": configs.map(
                        lambda value: _clean_text(value.get("dgp_name")).split(
                            "_"
                        )[0]
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
        "Rejection",
        "RelativeFrobeniusNorm_Y",
    )
    for column in numeric_columns:
        results[column] = pd.to_numeric(results[column], errors="coerce")

    required = (
        "n",
        "p",
        "snr",
        "method",
        "dgp_name",
        "use_true_latent",
        "Rejection",
        "RelativeFrobeniusNorm_Y",
    )
    missing = [column for column in required if column not in results]
    if missing:
        raise ValueError(f"Processed results are missing columns: {missing}")
    invalid = results[list(required)].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} rows have missing plotting fields.")
    return results


def _clean_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip().strip("'\"")


def _parse_latent_mode(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = _clean_text(value).lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid use_true_latent value: {value!r}")


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
) -> Path:
    """Plot rows of positive-SNR facets and columns of p facets."""
    data = aggregated[
        (aggregated["dgp_name"] == network)
        & (aggregated["snr"] > 0)
        & (aggregated["use_true_latent"] == use_true_latent)
    ].copy()
    if data.empty:
        raise ValueError(
            f"No alternative rows found for {network}, "
            f"use_true_latent={use_true_latent}."
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
    fig.suptitle(f"Power — {label} — {mode_label}", y=0.995)
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
    return _save_figure(fig, output_dir, f"power_{slug}_{mode_slug}")


def plot_type_i_error_by_p(
    aggregated: pd.DataFrame,
    output_dir: Path,
    network: str,
    use_true_latent: bool,
) -> Path:
    """Create one row of SNR-zero type-I-error facets over p."""
    data = aggregated[
        (aggregated["dgp_name"] == network)
        & (aggregated["snr"] == 0)
        & (aggregated["use_true_latent"] == use_true_latent)
    ].copy()
    if data.empty:
        raise ValueError(
            f"No null rows found for {network}, "
            f"use_true_latent={use_true_latent}."
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
    fig.suptitle(f"Type I error — {label} — {mode_label}", y=0.995)
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
    return _save_figure(fig, output_dir, f"type_i_error_{slug}_{mode_slug}")


def plot_frobenius_grid(
    aggregated: pd.DataFrame,
    output_dir: Path,
    network: str,
    use_true_latent: bool,
) -> Path:
    """Plot Y errors with SNR rows and p columns over n."""
    data = aggregated[
        (aggregated["dgp_name"] == network)
        & (aggregated["snr"] > 0)
        & (aggregated["use_true_latent"] == use_true_latent)
    ].copy()
    if data.empty:
        raise ValueError(
            f"No alternative Frobenius rows found for {network}, "
            f"use_true_latent={use_true_latent}."
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
    fig.suptitle(f"Y relative Frobenius error — {label} — {mode_label}", y=0.995)
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
    return _save_figure(fig, output_dir, f"frobenius_y_{slug}_{mode_slug}")


def plot_null_frobenius_by_p(
    aggregated: pd.DataFrame,
    output_dir: Path,
    network: str,
    use_true_latent: bool,
) -> Path:
    """Plot SNR-zero Y relative-Frobenius error in one row of p facets."""
    data = aggregated[
        (aggregated["dgp_name"] == network)
        & (aggregated["snr"] == 0)
        & (aggregated["use_true_latent"] == use_true_latent)
    ].copy()
    if data.empty:
        raise ValueError(
            f"No null Frobenius rows found for {network}, "
            f"use_true_latent={use_true_latent}."
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
    fig.suptitle(
        f"Y relative Frobenius error under H0 — {label} — {mode_label}",
        y=0.995,
    )
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
    return _save_figure(
        fig,
        output_dir,
        f"frobenius_y_null_{slug}_{mode_slug}",
    )


def main(argv=None) -> list[Path]:
    args = parse_args(argv)
    filenames = tuple(args.files) if args.files is not None else RESULT_FILES
    asymptotic_filenames = (
        tuple(args.asymptotic_files)
        if args.asymptotic_files is not None
        else ASYMPTOTIC_RESULT_FILES
    )
    if not filenames:
        raise ValueError(
            "No shard files configured. Populate RESULT_FILES or pass --files."
        )
    if not asymptotic_filenames:
        raise ValueError(
            "No asymptotic shard files configured. Populate "
            "ASYMPTOTIC_RESULT_FILES or pass --asymptotic-files."
        )

    configure_plot_style()
    primary_results = prepare_linear_model_results(args.results_dir, filenames)
    asymptotic_rv_results = prepare_linear_model_results(
        args.results_dir,
        asymptotic_filenames,
        include_methods=("RVTest_asymptotic",),
    )
    results = pd.concat(
        [primary_results, asymptotic_rv_results],
        ignore_index=True,
    )
    rejection_rates = aggregate_rejection_rates(results)
    frobenius_errors = aggregate_frobenius_errors(results)
    outputs = []
    for network in NETWORK_LABELS:
        for use_true_latent in LATENT_MODE_LABELS:
            outputs.append(
                plot_power_grid(
                    rejection_rates,
                    args.output_dir,
                    network,
                    use_true_latent,
                )
            )
            outputs.append(
                plot_type_i_error_by_p(
                    rejection_rates,
                    args.output_dir,
                    network,
                    use_true_latent,
                )
            )
            outputs.append(
                plot_frobenius_grid(
                    frobenius_errors,
                    args.output_dir,
                    network,
                    use_true_latent,
                )
            )
            outputs.append(
                plot_null_frobenius_by_p(
                    frobenius_errors,
                    args.output_dir,
                    network,
                    use_true_latent,
                )
            )
    print(f"Saved {len(outputs)} figures to {args.output_dir}")
    return outputs


if __name__ == "__main__":
    main()
