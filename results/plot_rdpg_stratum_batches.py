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
    method_handles,
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
    pairs: dict[tuple[str, str], tuple[str, str]] | None = None,
    figsize: tuple[float, float] | None = None,
    show_title: bool = True,
    font_size: float | None = None,
    legend_columns: int | None = None,
    y_max: float | None = None,
    show_run_counts: bool = True,
    show_panel_titles: bool = True,
    axes_margins: tuple[float, float, float, float] | None = None,
    legend_bbox: tuple[float, float] = (0.5, 0.95),
    legend_font_size: float | None = None,
    xlabel_y: float | None = None,
    ylabel_x: float | None = None,
) -> Path:
    selected_pairs = RDPG_STRATUM_Y_PAIRS if pairs is None else pairs
    aggregated = aggregate_results(
        data[data["stratum_y_pair"].isin(selected_pairs)],
        y_axis=metric,
        x_axis="n",
        factors=["method", "stratum_y_pair"],
    )
    assert_unique(
        aggregated, ["n", "method", "stratum_y_pair"],
        f"{metric} RDPG stratum curves",
    )

    single_panel = len(selected_pairs) == 1
    fig, axes = plt.subplots(
        1 if single_panel else 2,
        1 if single_panel else 2,
        figsize=figsize or ((7.0, 4.0) if single_panel else (9.2, 5.8)),
        sharex=True, sharey=True,
        squeeze=False, layout=None if axes_margins is not None else "constrained",
    )
    if axes_margins is not None:
        left, right, bottom, top = axes_margins
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    is_null = metric == "FalseRejection"
    for ax, (pair, (y0, y1)) in zip(axes.flat, selected_pairs.items()):
        panel = aggregated[aggregated["stratum_y_pair"] == pair]
        plot_line_panel(
            ax, panel, y_mean=f"{metric}_mean", y_sem=f"{metric}_sem",
            methods=methods,
        )
        style_probability_axis(ax, null=is_null)
        if y_max is not None:
            ax.set_ylim(0, y_max)
            ax.set_yticks(np.arange(0, y_max + 0.001, 0.05))
        if is_null:
            ax.axhline(0.05, color="#777777", linestyle="--", linewidth=0.8, zorder=1)
        if show_panel_titles:
            ax.set_title(
                rf"$Y\mid X=0$: {y0}     $Y\mid X=1$: {y1}",
                fontsize=None if font_size is None else font_size + 1,
            )
        if font_size is not None:
            ax.tick_params(labelsize=font_size - 0.5)

    if show_title:
        fig.suptitle("Type I error by network size" if is_null else "Power by network size")
    selected_counts = counts[
        counts.index.get_level_values("stratum_y_pair").isin(selected_pairs)
    ]
    incomplete = selected_counts.loc[1000][selected_counts.loc[1000] < 200]
    x_label = "Number of nodes"
    if show_run_counts and not incomplete.empty:
        x_label += (
            f"  (n=1000: {int(counts.loc[1000].min())}–"
            f"{int(counts.loc[1000].max())} runs per point)"
        )
    label_options = {} if font_size is None else {"fontsize": font_size + 1}
    if xlabel_y is not None:
        label_options["y"] = xlabel_y
    fig.supxlabel(x_label, **label_options)
    y_label_options = {} if font_size is None else {"fontsize": font_size + 1}
    if ylabel_x is not None:
        y_label_options["x"] = ylabel_x
    fig.supylabel("Type I error rate" if is_null else "Power", **y_label_options)
    if axes_margins is None:
        add_shared_legend(fig, lines=True, methods=methods)
    else:
        fig.legend(
            handles=method_handles(lines=True, methods=methods),
            loc="upper center",
            bbox_to_anchor=legend_bbox,
            ncols=legend_columns or 3,
            frameon=False,
            handlelength=2.2,
            columnspacing=0.8,
            handletextpad=0.4,
        )
    legend = fig.legends[-1]
    if axes_margins is None and legend_columns is not None:
        legend.set_ncols(legend_columns)
        legend.set_bbox_to_anchor(legend_bbox)
    effective_legend_font_size = (
        legend_font_size
        if legend_font_size is not None
        else (None if font_size is None else font_size - 0.5)
    )
    if effective_legend_font_size is not None:
        for text in legend.get_texts():
            text.set_fontsize(effective_legend_font_size)
    for text, method in zip(fig.legends[-1].get_texts(), methods):
        text.set_text(METHOD_LABELS[method])
    if not show_title and single_panel:
        layout_engine = fig.get_layout_engine()
        if layout_engine is not None:
            layout_engine.set(rect=(0.0, 0.0, 1.0, 0.76))

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

    gaussian_pair = ("gaussian -2", "gaussian 2")
    path = plot_batch(
        null,
        null_counts,
        args.output_dir,
        metric="FalseRejection",
        filename="conditional_type_I",
        methods=ALL_METHODS,
        pairs={gaussian_pair: RDPG_STRATUM_Y_PAIRS[gaussian_pair]},
        figsize=(5.0, 5.0),
        show_title=False,
        font_size=15,
        legend_columns=2,
        y_max=0.15,
        show_run_counts=False,
        show_panel_titles=False,
        axes_margins=(0.18, 0.97, 0.16, 0.60),
        legend_bbox=(0.5, 0.94),
        legend_font_size=13,
        xlabel_y=0.045,
        ylabel_x=0.015,
    )
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
