#!/usr/bin/env python3
"""Plot conditional-modulo power by network size, faceted by noise scale."""

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
    RDPG_BY_N_METHODS,
    add_shared_legend,
    configure_plot_style,
    plot_line_panel,
    save_figure,
    style_probability_axis,
)
from src.helper_functions.analyse_functions import aggregate_results


def plot_power(input_path: Path, output_dir: Path) -> Path:
    raw = pd.read_csv(input_path)
    null = raw["args"].map(lambda value: extract_argument(value, "independent_response"))
    noise = raw["args"].map(lambda value: extract_argument(value, "noise_scale"))
    if not null.map(lambda value: isinstance(value, bool)).all():
        raise ValueError("Every row must identify whether it is a null simulation.")
    if not pd.to_numeric(noise, errors="coerce").notna().all():
        raise ValueError("Every row must identify a numeric noise scale.")

    data = process_results(raw.loc[~null].copy())
    data["noise_scale"] = noise.loc[~null].to_numpy(dtype=float)
    if set(data["method"]) != set(RDPG_BY_N_METHODS):
        raise ValueError("The result file has unexpected or missing methods.")
    counts = data.groupby(["noise_scale", "n", "method"]).size()
    if counts.empty or counts.nunique() != 1 or counts.iloc[0] < 2:
        raise ValueError("Each noise, size, and method combination needs equal replication.")

    power = aggregate_results(
        data, y_axis="Rejection", x_axis="n", factors=["noise_scale", "method"]
    )
    scales = sorted(power["noise_scale"].unique())
    expected = len(scales) * data["n"].nunique() * len(RDPG_BY_N_METHODS)
    if len(power) != expected:
        raise ValueError("Some noise, size, and method combinations are missing.")

    configure_plot_style()
    fig, axes = plt.subplots(
        1, len(scales), figsize=(4.0 * len(scales) + 0.4, 4.4),
        sharex=True, sharey=True, squeeze=False, layout="constrained",
    )
    for ax, scale in zip(axes.flat, scales):
        panel = power[power["noise_scale"] == scale]
        plot_line_panel(
            ax, panel, y_mean="Rejection_mean", y_sem="Rejection_sem",
            methods=RDPG_BY_N_METHODS,
        )
        style_probability_axis(ax)
        ax.set_title(rf"Noise scale $\sigma={scale:g}$")

    fig.suptitle("Conditional modulo power by network size")
    fig.supxlabel(r"Network size, $n$")
    fig.supylabel("Power")
    add_shared_legend(fig, lines=True, methods=RDPG_BY_N_METHODS)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_dir, f"{input_path.stem}_power_by_n")
    return output_dir / f"{input_path.stem}_power_by_n.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Conditional-modulo result CSV")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "chatterjee_figures",
    )
    args = parser.parse_args()
    plot_power(args.input, args.output_dir)


if __name__ == "__main__":
    main()
