#!/usr/bin/env python3
"""Compare known and estimated conditional laws at one noise scale."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from results.plot_conditional_modulo import CONDITIONAL_MODULO_METHODS
from results.results_processing import extract_argument, process_results
from results.visualise_results_chatterjee import (
    add_shared_legend,
    configure_plot_style,
    plot_line_panel,
    save_figure,
    style_probability_axis,
)
from src.helper_functions.analyse_functions import aggregate_results


def summarize(path: Path, noise_scale: float, *, null: bool) -> pd.DataFrame:
    raw = pd.read_csv(path)
    null_flags = raw["args"].map(
        lambda value: extract_argument(value, "independent_response")
    )
    noise_scales = pd.to_numeric(
        raw["args"].map(lambda value: extract_argument(value, "noise_scale")),
        errors="coerce",
    )
    if not null_flags.map(lambda value: isinstance(value, bool)).all():
        raise ValueError(f"Missing null status in {path}")
    if noise_scales.isna().any():
        raise ValueError(f"Missing noise scale in {path}")

    selected = null_flags.eq(null) & np.isclose(noise_scales, noise_scale)
    data = process_results(raw.loc[selected].copy())
    if data.empty:
        raise ValueError(f"No matching rows in {path} at noise scale {noise_scale}")
    if set(data["method"]) != set(CONDITIONAL_MODULO_METHODS):
        raise ValueError(f"Unexpected or missing methods in {path}")
    counts = data.groupby(["n", "method"]).size()
    expected = data["n"].nunique() * len(CONDITIONAL_MODULO_METHODS)
    if len(counts) != expected or counts.nunique() != 1 or counts.iloc[0] < 2:
        raise ValueError(f"Incomplete or uneven simulation groups in {path}")

    metric = "FalseRejection" if null else "Rejection"
    return aggregate_results(data, y_axis=metric, x_axis="n", factors=["method"])


def plot_comparison(
    known_path: Path,
    estimated_path: Path,
    noise_scale: float,
    output_dir: Path,
) -> tuple[Path, Path]:
    configure_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for null, metric, ylabel, suffix in (
        (False, "Rejection", "Power", "power_by_n"),
        (True, "FalseRejection", "Type I error rate", "type1_by_n"),
    ):
        known = summarize(known_path, noise_scale, null=null)
        estimated = summarize(estimated_path, noise_scale, null=null)
        if set(known["n"]) != set(estimated["n"]):
            raise ValueError("Known and estimated results have different network sizes")

        fig, axes = plt.subplots(
            1, 2, figsize=(8.4, 4.4), sharex=True, sharey=True,
            layout="constrained",
        )
        for ax, summary, title in zip(
            axes,
            (known, estimated),
            ("Known conditional law", "Estimated conditional law"),
        ):
            plot_line_panel(
                ax, summary, y_mean=f"{metric}_mean", y_sem=f"{metric}_sem",
                methods=CONDITIONAL_MODULO_METHODS,
            )
            style_probability_axis(ax, null=null)
            if null:
                ax.axhline(
                    0.05, color="#777777", linestyle="--", linewidth=0.8,
                    zorder=1,
                )
            ax.set_title(title)

        fig.supxlabel(r"Network size, $n$")
        fig.supylabel(ylabel)
        add_shared_legend(fig, lines=True, methods=CONDITIONAL_MODULO_METHODS)
        name = f"conditional_modulo_true_vs_estimated_noise_{noise_scale:g}".replace(
            ".", "p"
        ) + f"_{suffix}"
        save_figure(fig, output_dir, name)
        outputs.append(output_dir / f"{name}.png")

    return tuple(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--known", type=Path,
        default=Path(__file__).resolve().parent / "conditional_modulo_66671385.csv",
    )
    parser.add_argument(
        "--estimated", type=Path,
        default=Path(__file__).resolve().parent / "conditional_modulo_estimated.csv",
    )
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "chatterjee_figures",
    )
    args = parser.parse_args()
    plot_comparison(args.known, args.estimated, args.noise_scale, args.output_dir)


if __name__ == "__main__":
    main()
