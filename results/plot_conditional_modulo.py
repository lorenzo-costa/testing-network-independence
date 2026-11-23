#!/usr/bin/env python3
"""Plot conditional-modulo power or null rejection by network size and noise."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from results.results_processing import extract_argument, process_results
from results.visualise_results_chatterjee import (
    add_shared_legend,
    configure_plot_style,
    plot_line_panel,
    save_figure,
    style_probability_axis,
)
from src.helper_functions.analyse_functions import aggregate_results

CONDITIONAL_MODULO_METHODS = (
    "DC", "AC_1", "AC_sqrt", "AC_adaptive", "CanonicalCorrelationTest",
)


def plot_power(
    input_paths: list[Path], output_dir: Path, output_name: str | None = None,
    *, null_results: bool = False,
) -> Path:
    raw = pd.concat((pd.read_csv(path) for path in input_paths), ignore_index=True)
    null = raw["args"].map(lambda value: extract_argument(value, "independent_response"))
    noise = raw["args"].map(lambda value: extract_argument(value, "noise_scale"))
    if not null.map(lambda value: isinstance(value, bool)).all():
        raise ValueError("Every row must identify whether it is a null simulation.")
    if not pd.to_numeric(noise, errors="coerce").notna().all():
        raise ValueError("Every row must identify a numeric noise scale.")

    selected = null if null_results else ~null
    data = process_results(raw.loc[selected].copy())
    data["noise_scale"] = noise.loc[selected].to_numpy(dtype=float)
    if set(data["method"]) != set(CONDITIONAL_MODULO_METHODS):
        raise ValueError("The result files have unexpected or missing methods.")
    counts = data.groupby(["noise_scale", "n", "method"]).size()
    if counts.empty or counts.nunique() != 1 or counts.iloc[0] < 2:
        raise ValueError("Each noise, size, and method combination needs equal replication.")

    metric = "FalseRejection" if null_results else "Rejection"
    summary = aggregate_results(
        data, y_axis=metric, x_axis="n", factors=["noise_scale", "method"]
    )
    scales = sorted(summary["noise_scale"].unique())
    expected = len(scales) * data["n"].nunique() * len(CONDITIONAL_MODULO_METHODS)
    if len(summary) != expected:
        raise ValueError("Some noise, size, and method combinations are missing.")

    configure_plot_style()
    fig, axes = plt.subplots(
        1, len(scales), figsize=(4.0 * len(scales) + 0.4, 4.4),
        sharex=True, sharey=True, squeeze=False, layout="constrained",
    )
    for ax, scale in zip(axes.flat, scales):
        panel = summary[summary["noise_scale"] == scale]
        plot_line_panel(
            ax, panel, y_mean=f"{metric}_mean", y_sem=f"{metric}_sem",
            methods=CONDITIONAL_MODULO_METHODS,
        )
        style_probability_axis(ax, null=null_results)
        if null_results:
            ax.axhline(0.05, color="#777777", linestyle="--", linewidth=0.8, zorder=1)
        ax.set_title(rf"Noise scale $\sigma={scale:g}$")

    fig.supxlabel(r"Network size, $n$")
    fig.supylabel("Null rejection rate" if null_results else "Power")
    add_shared_legend(fig, lines=True, methods=CONDITIONAL_MODULO_METHODS)
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_name is None:
        suffix = "null_rejection_by_n" if null_results else "power_by_n"
        output_name = "_".join(path.stem for path in input_paths) + f"_{suffix}"
    save_figure(fig, output_dir, output_name)
    return output_dir / f"{output_name}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Conditional-modulo result CSV files")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "chatterjee_figures",
    )
    parser.add_argument("--output-name", help="Output PNG filename without the extension")
    parser.add_argument("--null", action="store_true", help="Plot null rejection rates")
    args = parser.parse_args()
    plot_power(args.inputs, args.output_dir, args.output_name, null_results=args.null)


if __name__ == "__main__":
    main()
