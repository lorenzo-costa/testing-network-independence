#!/usr/bin/env python3
"""Plot power and type-I error for the multiple-network linear-model study.

Populate ``RESULT_FILES`` after all shards finish, or pass the three filenames
with ``--files``. Gaussian and Bernoulli networks are plotted separately.
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

from results.results_processing import process_shard_results  # noqa: E402


# Fill these after the three cluster shards have completed. Paths are resolved
# relative to --results-dir.
RESULT_FILES: tuple[str, ...] = (
    "linear_model_results_61598026_shard-000-of-003.csv",
    "linear_model_results_61598026_shard-001-of-003.csv",
    "linear_model_results_61598026_shard-002-of-003.csv",
)

NETWORK_LABELS = {
    "GaussianNetwork": "Gaussian weighted network",
    "BernoulliNetwork": "Bernoulli binary network",
}
METHOD_ORDER = ("RVTest_permutation", "CCA", "DC")
METHOD_LABELS = {
    "RVTest_permutation": "RV",
    "CCA": "CCA",
    "DC": "MGC",
}
COLORS = {
    "RVTest_permutation": "#E69F00",
    "CCA": "#0072B2",
    "DC": "#009E73",
}
MARKERS = {
    "RVTest_permutation": "o",
    "CCA": "s",
    "DC": "^",
}
PNG_DPI = 600


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
        help="Shard filenames; overrides RESULT_FILES.",
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
    results_dir: Path, filenames: Sequence[str]
) -> pd.DataFrame:
    """Load all shards and coerce the fields used by the figures."""
    results = process_shard_results(results_dir, filenames)
    numeric_columns = ("n", "p", "d_x", "d_y", "snr", "alpha", "Rejection")
    for column in numeric_columns:
        results[column] = pd.to_numeric(results[column], errors="coerce")

    required = ("n", "p", "snr", "method", "dgp_name", "Rejection")
    missing = [column for column in required if column not in results]
    if missing:
        raise ValueError(f"Processed results are missing columns: {missing}")
    invalid = results[list(required)].isna().any(axis=1)
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} rows have missing plotting fields.")
    return results


def aggregate_rejection_rates(results: pd.DataFrame) -> pd.DataFrame:
    """Compute rejection-rate means, SEMs, and replicate counts."""
    grouping = ["dgp_name", "p", "snr", "n", "method", "alpha"]
    aggregated = (
        results.groupby(grouping, dropna=False)["Rejection"]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "rejection_rate",
                "sem": "rejection_sem",
                "count": "replicates",
            }
        )
    )
    aggregated["rejection_sem"] = aggregated["rejection_sem"].fillna(0.0)
    return aggregated


def _method_handles(methods: Sequence[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            marker=MARKERS[method],
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


def _plot_curves(ax: Axes, data: pd.DataFrame, methods: Sequence[str]) -> None:
    duplicates = data.duplicated(["n", "method"], keep=False)
    if duplicates.any():
        raise ValueError("A plot panel contains duplicate method/n combinations.")

    for method in methods:
        subset = data[data["method"] == method].sort_values("n")
        if subset.empty:
            continue
        x = subset["n"].to_numpy(dtype=float)
        mean = subset["rejection_rate"].to_numpy(dtype=float)
        sem = subset["rejection_sem"].to_numpy(dtype=float)
        ax.plot(
            x,
            mean,
            color=COLORS[method],
            marker=MARKERS[method],
            markerfacecolor="none",
            markeredgewidth=0.9,
            zorder=3,
        )
        ax.fill_between(
            x,
            np.clip(mean - sem, 0, 1),
            np.clip(mean + sem, 0, 1),
            color=COLORS[method],
            alpha=0.12,
            linewidth=0,
            zorder=2,
        )

    if not data.empty:
        ax.set_xticks(sorted(data["n"].unique()))
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(axis="y", color="#E2E2E2", linewidth=0.45)


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
) -> Path:
    """Plot rows of p facets and columns of positive-SNR facets."""
    data = aggregated[
        (aggregated["dgp_name"] == network) & (aggregated["snr"] > 0)
    ].copy()
    if data.empty:
        raise ValueError(f"No alternative rows found for {network}.")
    p_values = tuple(sorted(data["p"].unique()))
    snr_values = tuple(sorted(data["snr"].unique()))
    methods = _available_methods(data)

    fig, axes = plt.subplots(
        len(p_values),
        len(snr_values),
        figsize=(2.15 * len(snr_values), 1.8 * len(p_values) + 0.8),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for row, p_value in enumerate(p_values):
        for column, snr in enumerate(snr_values):
            ax = axes[row, column]
            panel = data[(data["p"] == p_value) & (data["snr"] == snr)]
            _plot_curves(ax, panel, methods)
            if row == 0:
                ax.set_title(f"SNR = {snr:g}")
            if column == len(snr_values) - 1:
                ax.annotate(
                    f"p = {int(p_value)}",
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
    fig.suptitle(f"Power — {label}", y=0.995)
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
    return _save_figure(fig, output_dir, f"power_{slug}")


def plot_type_i_error_by_p(
    aggregated: pd.DataFrame,
    output_dir: Path,
    network: str,
) -> Path:
    """Create one row of SNR-zero type-I-error facets over p."""
    data = aggregated[
        (aggregated["dgp_name"] == network) & (aggregated["snr"] == 0)
    ].copy()
    if data.empty:
        raise ValueError(f"No null rows found for {network}.")
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
        _plot_curves(ax, panel, methods)
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
    fig.suptitle(f"Type I error — {label}", y=0.995)
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
    return _save_figure(fig, output_dir, f"type_i_error_{slug}")


def main(argv=None) -> list[Path]:
    args = parse_args(argv)
    filenames = tuple(args.files) if args.files is not None else RESULT_FILES
    if not filenames:
        raise ValueError(
            "No shard files configured. Populate RESULT_FILES or pass --files."
        )

    configure_plot_style()
    results = prepare_linear_model_results(args.results_dir, filenames)
    aggregated = aggregate_rejection_rates(results)
    outputs = []
    for network in NETWORK_LABELS:
        outputs.append(plot_power_grid(aggregated, args.output_dir, network))
        outputs.append(plot_type_i_error_by_p(aggregated, args.output_dir, network))
    print(f"Saved {len(outputs)} figures to {args.output_dir}")
    return outputs


if __name__ == "__main__":
    main()
