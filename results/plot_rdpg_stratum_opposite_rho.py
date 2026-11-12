"""Plot the 65851842 opposite-stratum-correlation RDPG power batch.

Run from the repository root with::

    python -m results.plot_rdpg_stratum_opposite_rho
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
    add_shared_legend,
    assert_unique,
    configure_plot_style,
    plot_line_panel,
    style_probability_axis,
)
from src.helper_functions.analyse_functions import aggregate_results


RESULT_FILE = "conditional_copula_rdpg_stratum_opposite_rho_65851842.csv"
METHODS = ("AC_1", "AC_sqrt", "AC_adaptive", "DC")
METHOD_LABELS = {
    "AC_1": r"AC ($M=1$)",
    "AC_sqrt": r"AC ($M=\sqrt{n}$)",
    "AC_adaptive": "AC (adaptive)",
    "DC": "DC (Model-X MGC)",
}
EXPECTED_RHO = np.asarray([0.3, -0.3])


def load_batch(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    data = process_results(raw)

    correlations = raw["args"].map(lambda value: extract_argument(value, "rho"))
    valid_correlations = correlations.map(
        lambda value: isinstance(value, list)
        and len(value) == 2
        and np.allclose(np.asarray(value, dtype=float), EXPECTED_RHO)
    )
    if not valid_correlations.all():
        raise ValueError(f"Expected rho=[0.3, -0.3] throughout {path}")

    if set(data["method"]) != set(METHODS):
        raise ValueError(f"Unexpected method in {path}")
    if set(data["n"]) != {100, 200, 500, 1000}:
        raise ValueError(f"Expected four network sizes in {path}")

    counts = data.groupby(["n", "method"]).size()
    if len(counts) != 4 * len(METHODS) or not counts.eq(200).all():
        raise ValueError(f"Expected 200 simulations for every size and method in {path}")
    return data


def plot_power(data: pd.DataFrame, output_dir: Path) -> Path:
    aggregated = aggregate_results(
        data,
        y_axis="Rejection",
        x_axis="n",
        factors=["method"],
    )
    assert_unique(aggregated, ["n", "method"], "opposite-rho power curves")

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(7.0, 4.0),
        layout="constrained",
    )
    plot_line_panel(
        ax,
        aggregated,
        y_mean="Rejection_mean",
        y_sem="Rejection_sem",
        methods=METHODS,
    )
    style_probability_axis(ax)
    ax.set_title(
        r"Gaussian copula: $\rho_{X=0}=0.3$, $\rho_{X=1}=-0.3$",
        fontsize=9,
    )

    fig.suptitle("Power by network size")
    fig.supxlabel("Number of nodes")
    fig.supylabel("Power")
    add_shared_legend(fig, lines=True, methods=METHODS)
    for text, method in zip(fig.legends[-1].get_texts(), METHODS):
        text.set_text(METHOD_LABELS[method])

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "27_rdpg_stratum_opposite_rho_power_by_n_ky3.png"
    fig.savefig(
        path,
        dpi=PNG_DPI,
        metadata={"Creator": "plot_rdpg_stratum_opposite_rho.py"},
    )
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "rdpg_stratum_figures"
            / "65851842"
        ),
    )
    args = parser.parse_args()

    configure_plot_style()
    data = load_batch(args.results_dir / RESULT_FILE)
    path = plot_power(data, args.output_dir)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
