#!/usr/bin/env python3
"""Plot selected Bernoulli copula simulations for ``n=200`` and ``k_y=3``.

The single output figure keeps the Gaussian, Clayton, and mixture copulas. It
compares DC, AC with ``k=1``, AC with ``k=sqrt(n)``, and the permutation and
asymptotic RV tests across all simulated Y and Z marginal distributions.

Run from anywhere with:

    python results/visualise_bernoulli_copulas_n200_ky3.py

By default, the PNG is saved alongside this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from results.results_processing import process_results
from results.visualise_results_chatterjee import (
    COLORS,
    MARKERS,
    METHOD_LABELS,
    PNG_DPI,
    configure_plot_style,
    display_marginal_label,
    read_result_batch,
)
from src.helper_functions.analyse_functions import aggregate_results


RESULT_FILES = ("simulation_results_20260730_2244.csv",)
NETWORK_SIZE = 200
KY = 3

COPULAS = ("gaussian", "clayton", "mixture")
MARGINALS_Y = ("gaussian", "chi df=5", "cauchy")
MARGINALS_Z = ("gaussian", "chi df=5", "unif(-1, 1)")
METHODS = (
    "DC",
    "AC_1",
    "AC_sqrt",
    "RVTest_permutation",
    "RVTest_asymptotic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the simulation CSV (default: script directory).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "bernoulli_copulas_n200_ky3_selected_methods.png"
        ),
        help="Path for the generated PNG.",
    )
    return parser.parse_args()


def prepare_results(results_dir: Path) -> pd.DataFrame:
    """Load, filter, and aggregate the selected simulation results."""
    results = process_results(read_result_batch(results_dir, RESULT_FILES))
    selected = results[
        (results["dgp_name"] == "BernoulliNetwork")
        & (results["n"] == NETWORK_SIZE)
        & (results["k"] == KY)
        & results["copula"].isin(COPULAS)
        & results["marginal_y"].isin(MARGINALS_Y)
        & results["marginal_z"].isin(MARGINALS_Z)
        & results["method"].isin(METHODS)
    ].copy()

    if selected.empty:
        raise ValueError("No simulation rows matched the requested configuration.")

    aggregated = aggregate_results(
        selected,
        y_axis="Rejection",
        x_axis="marginal_y",
        factors=["method", "copula", "marginal_z"],
    )

    keys = ["method", "copula", "marginal_z", "marginal_y"]
    if aggregated.duplicated(keys).any():
        raise ValueError("The aggregated results contain duplicate plotted points.")

    expected = pd.MultiIndex.from_product(
        [METHODS, COPULAS, MARGINALS_Z, MARGINALS_Y],
        names=keys,
    )
    observed = pd.MultiIndex.from_frame(aggregated[keys])
    missing = expected.difference(observed)
    if len(missing):
        raise ValueError(f"Missing {len(missing)} expected method/facet combinations.")

    return aggregated


def method_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            linestyle="none",
            marker=MARKERS[method],
            markersize=4.5,
            markerfacecolor="none",
            markeredgewidth=0.9,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]


def plot_results(data: pd.DataFrame, output_file: Path) -> None:
    fig, axes = plt.subplots(
        len(MARGINALS_Z),
        len(COPULAS),
        figsize=(7.0, 5.5),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )

    positions = np.arange(len(MARGINALS_Y), dtype=float)
    offsets = dict(zip(METHODS, np.linspace(-0.22, 0.22, len(METHODS))))

    for row, marginal_z in enumerate(MARGINALS_Z):
        for column, copula in enumerate(COPULAS):
            ax = axes[row, column]
            panel = data[
                (data["marginal_z"] == marginal_z)
                & (data["copula"] == copula)
            ]

            for method in METHODS:
                subset = (
                    panel[panel["method"] == method]
                    .set_index("marginal_y")
                    .reindex(MARGINALS_Y)
                )
                power = subset["Rejection_mean"].to_numpy(dtype=float)
                valid = np.isfinite(power)
                ax.plot(
                    positions[valid] + offsets[method],
                    power[valid],
                    color=COLORS[method],
                    linestyle="none",
                    marker=MARKERS[method],
                    markerfacecolor="none",
                    markeredgewidth=0.9,
                    markersize=4.5,
                    zorder=3,
                )

            ax.set_xlim(-0.5, len(MARGINALS_Y) - 0.5)
            ax.set_ylim(-0.05, 1.08)
            ax.set_xticks(positions)
            ax.set_xticklabels(
                [display_marginal_label(value) for value in MARGINALS_Y]
            )
            ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
            ax.grid(axis="y", color="#E2E2E2", linewidth=0.45)

            if row == 0:
                ax.set_title(copula.replace("_", " ").title())
            if column == len(COPULAS) - 1:
                ax.annotate(
                    f"Z marginal: {display_marginal_label(marginal_z)}",
                    xy=(1.04, 0.5),
                    xycoords="axes fraction",
                    ha="left",
                    va="center",
                    rotation=270,
                    fontsize=8,
                    annotation_clip=False,
                )

    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.86))
    fig.suptitle(
        rf"Power across Y marginals — Bernoulli binary network, "
        rf"$n={NETWORK_SIZE}$, $d^Y={KY}$"
    )
    fig.supxlabel("Y marginal distribution")
    fig.supylabel("Power")
    fig.legend(
        handles=method_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncols=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=1.3,
        handletextpad=0.5,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {output_file}")
    fig.savefig(
        output_file,
        dpi=PNG_DPI,
        metadata={"Creator": Path(__file__).name},
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    configure_plot_style()
    data = prepare_results(results_dir)
    plot_results(data, output_file)


if __name__ == "__main__":
    main()
