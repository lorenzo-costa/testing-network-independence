"""Plot the merged 65807617 power and 65807660 null RDPG stratum batches.

Run from the repository root with::

    python -m results.plot_rdpg_stratum_batches
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results.results_processing import extract_argument, process_results
from results.visualise_results_chatterjee import (
    PNG_DPI,
    RDPG_STRATUM_Y_PAIRS,
    add_shared_legend,
    assert_unique,
    configure_plot_style,
    plot_line_panel,
    style_probability_axis,
)
from src.helper_functions.analyse_functions import aggregate_results


POWER_FILE = "conditional_copula_rdpg_stratum_marginals_65807617.csv"
NULL_FILE = "conditional_copula_rdpg_stratum_marginals_65807660.csv"
AC_METHODS = ("AC_1", "AC_sqrt", "AC_adaptive")
ALL_METHODS = (*AC_METHODS, "DC")
METHOD_LABELS = {
    "AC_1": r"AC ($M=1$)",
    "AC_sqrt": r"AC ($M=\sqrt{n}$)",
    "AC_adaptive": "AC (adaptive)",
    "DC": "DC (unrestricted MGC)",
}


def load_batch(path: Path, *, rho: float) -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_csv(path)
    data = process_results(raw)
    strata = raw["args"].map(
        lambda value: extract_argument(value, "marginals_by_class")
    )
    if not strata.map(lambda value: isinstance(value, list) and len(value) == 2).all():
        raise ValueError(f"Every row in {path} must specify two strata")
    data["stratum_y_pair"] = strata.map(
        lambda value: (value[0]["y"], value[1]["y"])
    )

    if set(data["stratum_y_pair"]) != set(RDPG_STRATUM_Y_PAIRS):
        raise ValueError(f"Unexpected Y marginal assignment in {path}")
    if set(data["method"]) != set(ALL_METHODS):
        raise ValueError(f"Unexpected method in {path}")
    if set(data["n"]) != {100, 200, 500, 1000}:
        raise ValueError(f"Expected four network sizes in {path}")
    if not np.allclose(data["rho"].to_numpy(dtype=float), rho):
        raise ValueError(f"Expected rho={rho:g} throughout {path}")

    counts = data.groupby(["n", "stratum_y_pair", "method"]).size()
    if len(counts) != 4 * len(RDPG_STRATUM_Y_PAIRS) * len(ALL_METHODS):
        raise ValueError(f"One or more simulation scenarios are absent from {path}")
    if not counts.loc[[100, 200, 500]].eq(200).all():
        raise ValueError(f"Expected 200 runs for each n=100, 200, and 500 scenario in {path}")
    if not counts.loc[1000].between(1, 200).all():
        raise ValueError(f"Invalid n=1000 run count in {path}")
    return data, counts


def plot_batch(
    data: pd.DataFrame,
    counts: pd.Series,
    output_dir: Path,
    *,
    metric: str,
    filename: str,
    methods: tuple[str, ...],
) -> Path:
    aggregated = aggregate_results(
        data,
        y_axis=metric,
        x_axis="n",
        factors=["method", "stratum_y_pair"],
    )
    assert_unique(
        aggregated, ["n", "method", "stratum_y_pair"],
        f"{metric} RDPG stratum curves",
    )

    fig, axes = plt.subplots(
        2, 2, figsize=(9.2, 5.8), sharex=True, sharey=True,
        squeeze=False, layout="constrained",
    )
    is_null = metric == "FalseRejection"
    for ax, (pair, (y0, y1)) in zip(axes.flat, RDPG_STRATUM_Y_PAIRS.items()):
        panel = aggregated[aggregated["stratum_y_pair"] == pair]
        plot_line_panel(
            ax, panel, y_mean=f"{metric}_mean", y_sem=f"{metric}_sem",
            methods=methods,
        )
        style_probability_axis(ax, null=is_null)
        if is_null:
            ax.axhline(0.05, color="#777777", linestyle="--", linewidth=0.8, zorder=1)
        ax.set_title(rf"$Y\mid X=0$: {y0}     $Y\mid X=1$: {y1}")

    fig.suptitle("Type I error by network size" if is_null else "Power by network size")
    incomplete = counts.loc[1000][counts.loc[1000] < 200]
    x_label = "Number of nodes"
    if not incomplete.empty:
        x_label += (
            f"  (n=1000: {int(counts.loc[1000].min())}–"
            f"{int(counts.loc[1000].max())} runs per point)"
        )
    fig.supxlabel(x_label)
    fig.supylabel("Type I error rate" if is_null else "Power")
    add_shared_legend(fig, lines=True, methods=methods)
    for text, method in zip(fig.legends[-1].get_texts(), methods):
        text.set_text(METHOD_LABELS[method])

    path = output_dir / f"{filename}.png"
    fig.savefig(path, dpi=PNG_DPI, metadata={"Creator": "plot_rdpg_stratum_batches.py"})
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "rdpg_stratum_figures" / "65807617_65807660",
    )
    args = parser.parse_args()
    configure_plot_style()
    power, power_counts = load_batch(args.results_dir / POWER_FILE, rho=0.5)
    null, null_counts = load_batch(args.results_dir / NULL_FILE, rho=0.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for data, counts, metric, prefix in (
        (power, power_counts, "Rejection", "27_rdpg_stratum_power_by_n_ky3_rho_0.5"),
        (null, null_counts, "FalseRejection", "31_rdpg_stratum_type_i_error"),
    ):
        for methods, suffix in ((ALL_METHODS, "all_methods"), (AC_METHODS, "ac")):
            path = plot_batch(
                data, counts, args.output_dir, metric=metric,
                filename=f"{prefix}_{suffix}", methods=methods,
            )
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
