#!/usr/bin/env python3
"""Plot the selected RDPG null-study Y marginals in one horizontal row."""

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
from results.visualise_results_chatterjee import configure_plot_style, style_probability_axis
from src.helper_functions.analyse_functions import aggregate_results


MARGINALS = (
    ("cauchy", "Cauchy"),
    ("gaussian", "Gaussian"),
    ("pareto 1.5", "Pareto (shape 1.5)"),
)
METHODS = (
    ("DC", "#0072B2", "x", "DC"),
    ("AC_1", "#D55E00", "D", r"AC asymptotic ($k=1$)"),
    ("AC_sqrt", "#009E73", "^", r"AC asymptotic ($k=\sqrt{n}$)"),
    ("AC_adaptive", "#CC79A7", "s", "AC adaptive (permutation)"),
    ("CanonicalCorrelationTest", "#222222", "o", "CCA"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=REPOSITORY_ROOT / "results/simulation_results_65130774.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results/chatterjee_figures/65130774/31_null_rdpg_ky3.png"
        ),
    )
    args = parser.parse_args()

    configure_plot_style()
    data = process_results(pd.read_csv(args.source))
    marginal_names = tuple(name for name, _ in MARGINALS)
    method_names = tuple(name for name, *_ in METHODS)
    data = data[
        (data["dgp_name"] == "BernoulliNetwork")
        & (data["rho"] == 0)
        & (data["marginal_z"] == "uniform 0 0.5773502692")
        & data["marginal_y"].isin(marginal_names)
        & data["method"].isin(method_names)
    ]
    if data.empty:
        raise ValueError(f"No matching RDPG null results in {args.source}")
    summary = aggregate_results(
        data,
        y_axis="FalseRejection",
        x_axis="n",
        factors=["method", "marginal_y"],
    )
    expected = len(MARGINALS) * len(METHODS) * data["n"].nunique()
    if len(summary) != expected:
        raise ValueError(f"Expected {expected} null-study points, found {len(summary)}")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.67, wspace=0.15)
    handles = []
    for method, color, marker, label in METHODS:
        handles.append(
            Line2D(
                [0], [0], color=color, marker=marker, markerfacecolor="none",
                label=label,
            )
        )
    for ax, (marginal, label) in zip(axes, MARGINALS):
        panel = summary[summary["marginal_y"] == marginal]
        for method, color, marker, _ in METHODS:
            curve = panel[panel["method"] == method].sort_values("n")
            x = curve["n"].to_numpy(dtype=float)
            mean = curve["FalseRejection_mean"].to_numpy(dtype=float)
            sem = curve["FalseRejection_sem"].to_numpy(dtype=float)
            ax.plot(x, mean, color=color, marker=marker, markerfacecolor="none")
            ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.08, linewidth=0)
        ax.set_title(f"Y marginal: {label}")
        ax.set_xticks([100, 200, 500, 1000])
        style_probability_axis(ax, null=True)

    fig.suptitle("Type I error", y=0.98)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=3, frameon=False)
    fig.supxlabel("Number of nodes", y=0.045)
    fig.supylabel("Type I error rate", x=0.015)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=600, metadata={"Creator": "plot_rdpg_null.py"})
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
