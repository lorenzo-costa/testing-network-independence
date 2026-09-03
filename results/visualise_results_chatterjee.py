#!/usr/bin/env python3
"""Create paper-quality figures from ``visualise_results_chatterjee.ipynb``.

The script applies the notebook's row-matched replacement of
``RVTest_asymptotic`` results to the original ky=1 simulation batches. It also
reads the ky=3 copula, null, and rho-sweep batches directly, without replacing
their asymptotic RV results. Every figure is written as a 600-DPI PNG.

Run from anywhere with:

    python results/visualise_results_chatterjee.py

By default, figures are saved under ``results/chatterjee_figures``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
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

from results.results_processing import process_results
from src.helper_functions.analyse_functions import aggregate_results


OLD_RESULT_FILES = (
    "simulation_results_20260723_1332.csv",  # Gaussian network
    "simulation_results_20260723_1407.csv",  # Bernoulli network
    "simulation_results_20260723_1155.csv",  # functional alternatives
    "simulation_results_20260723_1322.csv",  # latent simulations
    "simulation_results_20260723_1554.csv",  # null simulations
)

NEW_RESULT_FILES = (
    "simulation_results_20260726_0347.csv",  # Gaussian network
    "simulation_results_20260726_0353.csv",  # Bernoulli network
    "simulation_results_20260726_0339.csv",  # functional alternatives
    "simulation_results_20260726_0342.csv",  # latent simulations
    "simulation_results_20260726_0400.csv",  # null simulations
)

KY3_GAUSSIAN_COPULA_RESULT_FILES = (
    "simulation_results_20260730_2105.csv",
)

KY3_BERNOULLI_COPULA_RESULT_FILES = (
    "simulation_results_20260730_2244.csv",
)

KY3_NULL_RESULT_FILES = (
    "simulation_results_20260818_0854.csv",  # Bernoulli network
    "simulation_results_20260818_0806.csv",  # Gaussian network
)

KY3_RHO_GAUSSIAN_RESULT_FILES = (
    "simulation_results_20260730_0858.csv",
)

KY3_RHO_BERNOULLI_RESULT_FILES = (
    "simulation_results_20260730_0928.csv",
)

COLUMNS_TO_REPLACE = (
    "RelativeFrobeniusNorm_x",
    "RelativeFrobeniusNorm_z",
    "ProcrustesDistance_x",
    "ProcrustesDistance_z",
    "FalseRejection",
    "Power",
    "Rejection",
    "avg_rel_frob_z",
    "avg_proc_dist_z",
)

COLUMNS_TO_MATCH = (
    "density",
    "n",
    "k",
    "edge_var",
    "approximation",
    "solver",
    "rho",
    "method",
    "marginals",
    "density_A",
    "density_B",
    "make_sparse",
    "use_true_x",
    "use_true_z",
    "latent_sim",
    "copula",
    "marginal_y",
    "marginal_z",
    "degree",
    "NN_number",
    "functional_form",
    "sbm_covariate_sampling",
    "assortativity",
    "x_distribution",
    "adaptive_m",
    "column_covariance",
    "dgp_name",
    "column_covariance_type",
    "column_covariance_condition_number",
)

COLORS = {
    "DC": "#0072B2",
    "AC_1": "#D55E00",
    "AC_sqrt": "#009E73",
    "AC_adaptive": "#CC79A7",
    "RVTest_permutation": "#E69F00",
    "RVTest_asymptotic": "#222222",
}

MARKERS = {
    "DC": "x",
    "AC_1": "D",
    "AC_sqrt": "^",
    "AC_adaptive": "s",
    "RVTest_permutation": "v",
    "RVTest_asymptotic": "o",
}

LINESTYLES = {
    "DC": "-",
    "AC_1": "-",
    "AC_sqrt": "-",
    "AC_adaptive": "-",
    "RVTest_permutation": "-",
    "RVTest_asymptotic": "-",
}

METHOD_LABELS = {
    "DC": "DC",
    "AC_1": r"AC ($k=1$)",
    "AC_sqrt": r"AC ($k=\sqrt{n}$)",
    "AC_adaptive": "AC (adaptive)",
    "RVTest_permutation": "RV (permutation)",
    "RVTest_asymptotic": "RV (asymptotic)",
}

METHODS = tuple(COLORS)

COPULAS = ("gaussian", "student_t", "clayton", "mixture")
MARGINALS_Z = ("gaussian", "chi df=5")
MARGINALS_Z_KY3 = (*MARGINALS_Z, "unif(-1, 1)")
MARGINALS_Y = ("gaussian", "chi df=5", "cauchy")
RHO_SWEEP_MARGINALS_Y = ("gaussian", "chi df=5")

LATENT_SIMULATIONS = (
    "linear",
    "exponential",
    "cubic",
    "step",
    "circle",
    "ellipse",
    "spiral",
    "quadratic",
    "fourth_root",
    "log",
    "w_shaped",
    "two_parabolas",
    "bernoulli",
    "square",
    "diamond",
    "sin_sixteen_pi",
)

FUNCTIONAL_FORMS = (
    "linear",
    "interaction",
    "pareto",
    "max",
    "abs_max",
    "radial",
    "sine",
    "tanh_product",
)

CATEGORY_LABELS = {
    "student_t": "Student t",
    "fourth_root": "Fourth\nroot",
    "w_shaped": "W-shaped",
    "two_parabolas": "Two\nparabolas",
    "sin_sixteen_pi": r"$\sin(16\pi)$",
    "abs_max": "Absolute\nmaximum",
    "tanh_product": "Tanh\nproduct",
}

MARGINAL_LABELS = {
    "gaussian": r"$\mathcal{N}(0, 1)$",
    "chi df=5": r"$\chi^2_5$",
    "unif(-1, 1)": r"$U(-1, 1)$",
}

PNG_DPI = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create paper-quality versions of all active figures in "
            "visualise_results_chatterjee.ipynb."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the simulation CSV files (default: script directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "chatterjee_figures",
        help="Directory in which to save PNG figures.",
    )
    return parser.parse_args()


def configure_plot_style() -> None:
    """Set physical sizes, fonts, strokes, and output defaults."""
    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 4.8),
            "figure.dpi": 120,
            "savefig.dpi": PNG_DPI,
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "figure.titlesize": 10,
            "legend.fontsize": 7.5,
            "legend.title_fontsize": 7.5,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.5,
            "lines.markeredgewidth": 0.9,
            "patch.linewidth": 0.6,
            "axes.linewidth": 0.6,
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.transparent": False,
        }
    )


def read_result_batch(results_dir: Path, filenames: tuple[str, ...]) -> pd.DataFrame:
    missing = [filename for filename in filenames if not (results_dir / filename).is_file()]
    if missing:
        formatted = "\n".join(f"  - {results_dir / filename}" for filename in missing)
        raise FileNotFoundError(f"Missing simulation result files:\n{formatted}")

    frames = []
    for filename in filenames:
        path = results_dir / filename
        print(f"Reading {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def make_comparable(value: object) -> object:
    """Convert nested dataframe values into hashable row-key components."""
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, np.ndarray):
        return ("array", make_comparable(value.tolist()))

    if isinstance(value, Mapping):
        items = (
            (make_comparable(key), make_comparable(item))
            for key, item in value.items()
        )
        return ("dict", tuple(sorted(items, key=repr)))

    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(make_comparable(item) for item in value))

    try:
        if pd.isna(value):
            return ("missing",)
    except (TypeError, ValueError):
        pass

    return value


def make_row_keys(frame: pd.DataFrame) -> pd.Series:
    return frame[list(COLUMNS_TO_MATCH)].apply(
        lambda row: tuple(make_comparable(value) for value in row),
        axis=1,
    )


def replace_asymptotic_results(
    old_results: pd.DataFrame,
    new_results: pd.DataFrame,
) -> pd.DataFrame:
    """Apply cell 7's one-to-one replacement of matching asymptotic-test rows."""
    method = "RVTest_asymptotic"
    target_positions = np.flatnonzero(old_results["method"].eq(method).to_numpy())
    source_positions = np.flatnonzero(new_results["method"].eq(method).to_numpy())

    target = old_results.iloc[target_positions].copy()
    source = new_results.iloc[source_positions].copy()
    target["_position"] = target_positions
    source["_position"] = source_positions
    target["_key"] = make_row_keys(target)
    source["_key"] = make_row_keys(source)
    target["_occurrence"] = target.groupby("_key", sort=False).cumcount()
    source["_occurrence"] = source.groupby("_key", sort=False).cumcount()

    matches = target[["_position", "_key", "_occurrence"]].merge(
        source[["_position", "_key", "_occurrence"]],
        on=["_key", "_occurrence"],
        how="inner",
        suffixes=("_target", "_source"),
        validate="one_to_one",
    )

    if len(matches) != 47_400:
        raise RuntimeError(
            f"Expected 47,400 matching asymptotic rows, found {len(matches)}."
        )

    result = old_results.copy()
    target_columns = result.columns.get_indexer(COLUMNS_TO_REPLACE)
    source_columns = new_results.columns.get_indexer(COLUMNS_TO_REPLACE)
    if (target_columns < 0).any() or (source_columns < 0).any():
        raise KeyError("One or more replacement columns are missing.")

    result.iloc[
        matches["_position_target"].to_numpy(),
        target_columns,
    ] = new_results.iloc[
        matches["_position_source"].to_numpy(),
        source_columns,
    ].to_numpy()
    return result


def prepare_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    old_results = process_results(read_result_batch(results_dir, OLD_RESULT_FILES))
    new_results = process_results(read_result_batch(results_dir, NEW_RESULT_FILES))
    all_results = replace_asymptotic_results(old_results, new_results)

    gaussian = all_results[all_results["dgp_name"] == "GaussianNetwork"].copy()
    bernoulli = all_results[all_results["dgp_name"] == "BernoulliNetwork"].copy()
    return gaussian, bernoulli


def prepare_ky3_copula_results(
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the ky=3 copula batches without replacing asymptotic RV results."""
    gaussian = process_results(
        read_result_batch(results_dir, KY3_GAUSSIAN_COPULA_RESULT_FILES)
    )
    bernoulli = process_results(
        read_result_batch(results_dir, KY3_BERNOULLI_COPULA_RESULT_FILES)
    )
    return gaussian, bernoulli


def prepare_ky3_null_results(
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the ky=3 null batch without replacing asymptotic RV results."""
    results = process_results(read_result_batch(results_dir, KY3_NULL_RESULT_FILES))
    gaussian = results[results["dgp_name"] == "GaussianNetwork"].copy()
    bernoulli = results[results["dgp_name"] == "BernoulliNetwork"].copy()
    return gaussian, bernoulli


def prepare_ky3_rho_results(
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the ky=3 Gaussian-copula rho sweeps for both network models."""
    gaussian = process_results(
        read_result_batch(results_dir, KY3_RHO_GAUSSIAN_RESULT_FILES)
    )
    bernoulli = process_results(
        read_result_batch(results_dir, KY3_RHO_BERNOULLI_RESULT_FILES)
    )
    return gaussian, bernoulli


def aggregate_null_results(data: pd.DataFrame) -> pd.DataFrame:
    null_factors = [
        "method",
        "copula",
        "marginal_z",
        "marginal_y",
        "column_covariance_condition_number",
    ]
    return aggregate_results(
        data[data["rho"] == 0.0],
        y_axis="FalseRejection",
        x_axis="n",
        factors=null_factors,
    )


def aggregate_alternative_results(data: pd.DataFrame) -> pd.DataFrame:
    alternative_factors = [
        "marginal_y",
        "marginal_z",
        "method",
        "latent_sim",
        "make_sparse",
        "functional_form",
        "copula",
        "sbm_covariate_sampling",
        "assortativity",
        "x_distribution",
    ]
    return aggregate_results(
        data[data["rho"] != 0.0],
        y_axis="Rejection",
        x_axis="n",
        factors=alternative_factors,
    )


def aggregate_rho_results(data: pd.DataFrame, *, n: int) -> pd.DataFrame:
    """Aggregate power over rho at one network size for every marginal pair."""
    data = data[
        (data["n"] == n)
        & (data["rho"] > 0.0)
        & (data["rho"] <= 0.4)
        & (data["copula"] == "gaussian")
        & data["marginal_z"].isin(MARGINALS_Z_KY3)
        & data["marginal_y"].isin(RHO_SWEEP_MARGINALS_Y)
        & data["method"].isin(METHODS)
    ].copy()
    return aggregate_results(
        data,
        y_axis="Rejection",
        x_axis="rho",
        factors=["method", "marginal_z", "marginal_y"],
    )


def aggregate_inputs(
    gaussian: pd.DataFrame,
    bernoulli: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        aggregate_null_results(gaussian),
        aggregate_null_results(bernoulli),
        aggregate_alternative_results(gaussian),
        aggregate_alternative_results(bernoulli),
    )


def display_label(value: object) -> str:
    text = str(value)
    return CATEGORY_LABELS.get(text, text.replace("_", " ").title())


def display_marginal_label(value: object) -> str:
    text = str(value)
    return MARGINAL_LABELS.get(text, display_label(text))


def assert_unique(data: pd.DataFrame, keys: list[str], context: str) -> None:
    """Prevent future data additions from silently drawing duplicate points."""
    duplicates = data.duplicated(keys, keep=False)
    if duplicates.any():
        raise ValueError(
            f"{context} contains {int(duplicates.sum())} rows duplicated on {keys}. "
            "Add the omitted configuration as a facet or filter before plotting."
        )


def method_handles(*, lines: bool) -> list[Line2D]:
    handles = []
    for method in METHODS:
        handles.append(
            Line2D(
                [0],
                [0],
                color=COLORS[method],
                linestyle=LINESTYLES[method] if lines else "none",
                linewidth=1.25,
                marker=MARKERS[method],
                markersize=4.5,
                markerfacecolor="none",
                markeredgewidth=0.9,
                label=METHOD_LABELS[method],
            )
        )
    return handles


def add_shared_legend(fig: Figure, *, lines: bool) -> None:
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        # Reserve a stable header band: title at the top, legend immediately below.
        layout_engine.set(rect=(0.0, 0.0, 1.0, 0.86))
    fig.legend(
        handles=method_handles(lines=lines),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncols=3,
        frameon=False,
        handlelength=2.5 if lines else 1.2,
        columnspacing=1.3,
        handletextpad=0.5,
    )


def style_probability_axis(ax: Axes, *, null: bool = False) -> None:
    if null:
        # Leave room above the largest observed rates and their SEM ribbons.
        ax.set_ylim(0, 0.20)
        ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
    else:
        # Markers and symmetric error bars at exactly 0 or 1 need visual headroom.
        ax.set_ylim(-0.05, 1.08)
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])

    ax.grid(axis="y", color="#E2E2E2", linewidth=0.45)
    ax.margins(x=0.04)


def plot_line_panel(
    ax: Axes,
    data: pd.DataFrame,
    *,
    y_mean: str,
    y_sem: str,
    x_column: str = "n",
    band_alpha: float = 0.13,
) -> None:
    assert_unique(data, [x_column, "method"], "Line panel")
    for method in METHODS:
        subset = data[data["method"] == method].sort_values(x_column)
        if subset.empty:
            continue

        x = subset[x_column].to_numpy(dtype=float)
        mean = subset[y_mean].to_numpy(dtype=float)
        sem = subset[y_sem].to_numpy(dtype=float)
        ax.plot(
            x,
            mean,
            color=COLORS[method],
            linestyle=LINESTYLES[method],
            marker=MARKERS[method],
            markerfacecolor="none",
            markeredgewidth=0.9,
            zorder=3,
        )
        ax.fill_between(
            x,
            mean - sem,
            mean + sem,
            color=COLORS[method],
            alpha=band_alpha,
            linewidth=0,
            zorder=2,
        )

    if not data.empty:
        ax.set_xticks(sorted(data[x_column].unique()))


def plot_scatter_panel(
    ax: Axes,
    data: pd.DataFrame,
    *,
    category: str,
    categories: tuple[str, ...],
    y_mean: str = "Rejection_mean",
    category_labeler: Callable[[object], str] = display_label,
) -> None:
    assert_unique(data, [category, "method"], "Scatter panel")
    base_positions = np.arange(len(categories), dtype=float)
    offsets = dict(zip(METHODS, np.linspace(-0.25, 0.25, len(METHODS))))

    for method in METHODS:
        subset = data[data["method"] == method].set_index(category)
        subset = subset.reindex(categories)
        mean = pd.to_numeric(subset[y_mean], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(mean)
        if not valid.any():
            continue

        ax.plot(
            base_positions[valid] + offsets[method],
            mean[valid],
            color=COLORS[method],
            linestyle="none",
            marker=MARKERS[method],
            markerfacecolor="none",
            markeredgewidth=0.9,
            markersize=4.5,
            zorder=3,
        )

    ax.set_xlim(-0.55, len(categories) - 0.45)
    ax.set_xticks(base_positions)
    ax.set_xticklabels([category_labeler(value) for value in categories])


def add_facet_labels(
    axes: np.ndarray,
    *,
    column_values: tuple[str, ...],
    column_prefix: str,
    row_values: tuple[str, ...] = (),
    row_prefix: str = "",
    column_labeler: Callable[[object], str] = display_label,
    row_labeler: Callable[[object], str] = display_label,
) -> None:
    for column, value in enumerate(column_values):
        axes[0, column].set_title(f"{column_prefix}{column_labeler(value)}")

    for row, value in enumerate(row_values):
        axes[row, -1].annotate(
            f"{row_prefix}{row_labeler(value)}",
            xy=(1.04, 0.5),
            xycoords="axes fraction",
            ha="left",
            va="center",
            rotation=270,
            fontsize=8,
            annotation_clip=False,
        )


def save_figure(fig: Figure, output_dir: Path, filename: str) -> None:
    png_path = output_dir / f"{filename}.png"
    metadata = {"Creator": "visualise_results_chatterjee.py"}
    print(f"Saving {png_path}")
    fig.savefig(png_path, dpi=PNG_DPI, metadata=metadata)
    plt.close(fig)


def plot_null_network(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    title: str,
    filename: str,
    ky: int,
) -> None:
    condition_number = pd.to_numeric(
        data["column_covariance_condition_number"],
        errors="coerce",
    )
    data = data[np.isclose(condition_number, 1.0, equal_nan=False)].copy()
    marginal_order = ("gaussian", "chi df=5", "cauchy")
    columns = tuple(value for value in marginal_order if value in set(data["marginal_z"]))
    rows = tuple(value for value in marginal_order if value in set(data["marginal_y"]))

    assert_unique(
        data,
        ["n", "method", "marginal_z", "marginal_y"],
        title,
    )
    fig, axes = plt.subplots(
        len(rows),
        len(columns),
        figsize=(7.0, 5.5),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for row, marginal_y in enumerate(rows):
        for column, marginal_z in enumerate(columns):
            ax = axes[row, column]
            panel = data[
                (data["marginal_y"] == marginal_y)
                & (data["marginal_z"] == marginal_z)
            ]
            plot_line_panel(
                ax,
                panel,
                y_mean="FalseRejection_mean",
                y_sem="FalseRejection_sem",
                band_alpha=0.08,
            )
            style_probability_axis(ax, null=True)

    add_facet_labels(
        axes,
        column_values=columns,
        column_prefix="Z marginal: ",
        row_values=rows,
        row_prefix="Y marginal: ",
        column_labeler=display_marginal_label,
        row_labeler=display_marginal_label,
    )
    fig.suptitle(f"{title}, $k_y={ky}$")
    fig.supxlabel(r"Network size, $n$")
    fig.supylabel("Type I error rate")
    add_shared_legend(fig, lines=True)
    save_figure(fig, output_dir, filename)


def plot_null_figures(
    null_gaussian: pd.DataFrame,
    null_bernoulli: pd.DataFrame,
    output_dir: Path,
    *,
    ky: int,
    figure_numbers: tuple[int, int],
) -> None:
    plot_null_network(
        null_gaussian,
        output_dir,
        title="Type I error under independence — Gaussian weighted network",
        filename=f"{figure_numbers[0]:02d}_null_gaussian_ky{ky}",
        ky=ky,
    )
    plot_null_network(
        null_bernoulli,
        output_dir,
        title="Type I error under independence — Bernoulli binary network",
        filename=f"{figure_numbers[1]:02d}_null_bernoulli_ky{ky}",
        ky=ky,
    )


def filtered_copula_results(
    data: pd.DataFrame,
    *,
    marginals_z: tuple[str, ...] = MARGINALS_Z,
) -> pd.DataFrame:
    return data[
        data["copula"].isin(COPULAS)
        & data["marginal_z"].isin(marginals_z)
        & data["marginal_y"].isin(MARGINALS_Y)
        & data["method"].isin(METHODS)
    ].copy()


def plot_copula_network(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    network_name: str,
    filename_prefix: str,
    first_number: int,
    chi_squared_number: int,
    uniform_number: int | None,
    point_n: int,
    ky: int,
    marginals_z: tuple[str, ...],
) -> None:
    scatter_data = data[data["n"] == point_n].copy()
    assert_unique(
        scatter_data,
        ["method", "copula", "marginal_z", "marginal_y"],
        f"{network_name} copula point ranges",
    )
    fig, axes = plt.subplots(
        len(marginals_z),
        len(COPULAS),
        figsize=(7.0, 5.5 if len(marginals_z) == 3 else 4.2),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for row, marginal_z in enumerate(marginals_z):
        for column, copula in enumerate(COPULAS):
            ax = axes[row, column]
            panel = scatter_data[
                (scatter_data["marginal_z"] == marginal_z)
                & (scatter_data["copula"] == copula)
            ]
            plot_scatter_panel(
                ax,
                panel,
                category="marginal_y",
                categories=MARGINALS_Y,
                category_labeler=display_marginal_label,
            )
            style_probability_axis(ax)

    add_facet_labels(
        axes,
        column_values=COPULAS,
        column_prefix="",
        row_values=marginals_z,
        row_prefix="Z marginal: ",
        row_labeler=display_marginal_label,
    )
    fig.suptitle(
        f"Power across Y marginals — {network_name}, "
        f"$n={point_n}$, $k_y={ky}$"
    )
    fig.supxlabel("Y marginal distribution")
    fig.supylabel("Power")
    add_shared_legend(fig, lines=False)
    save_figure(
        fig,
        output_dir,
        f"{first_number:02d}_{filename_prefix}_copulas_n{point_n}_ky{ky}",
    )

    by_n_specs = [
        (
            "gaussian",
            f"{first_number + 1:02d}_{filename_prefix}_copulas_by_n_ky{ky}",
        ),
        (
            "chi df=5",
            (
                f"{chi_squared_number:02d}_{filename_prefix}_copulas_by_n_"
                f"z_chi_squared_ky{ky}"
            ),
        ),
    ]
    if uniform_number is not None:
        by_n_specs.append(
            (
                "unif(-1, 1)",
                (
                    f"{uniform_number:02d}_{filename_prefix}_copulas_by_n_"
                    f"z_uniform_ky{ky}"
                ),
            )
        )
    for marginal_z, filename in by_n_specs:
        line_data = data[data["marginal_z"] == marginal_z].copy()
        assert_unique(
            line_data,
            ["n", "method", "copula", "marginal_y"],
            f"{network_name} copula curves with Z marginal {marginal_z}",
        )
        fig, axes = plt.subplots(
            len(MARGINALS_Y),
            len(COPULAS),
            figsize=(7.0, 5.2),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        for row, marginal_y in enumerate(MARGINALS_Y):
            for column, copula in enumerate(COPULAS):
                ax = axes[row, column]
                panel = line_data[
                    (line_data["marginal_y"] == marginal_y)
                    & (line_data["copula"] == copula)
                ]
                plot_line_panel(
                    ax,
                    panel,
                    y_mean="Rejection_mean",
                    y_sem="Rejection_sem",
                )
                style_probability_axis(ax)

        add_facet_labels(
            axes,
            column_values=COPULAS,
            column_prefix="",
            row_values=MARGINALS_Y,
            row_prefix="Y marginal: ",
            row_labeler=display_marginal_label,
        )
        fig.suptitle(
            f"Power by network size — {network_name} "
            f"(Z marginal: {display_marginal_label(marginal_z)}), "
            f"$k_y={ky}$"
        )
        fig.supxlabel(r"Network size, $n$")
        fig.supylabel("Power")
        add_shared_legend(fig, lines=True)
        save_figure(fig, output_dir, filename)


def plot_copula_figures(
    alternative_gaussian: pd.DataFrame,
    alternative_bernoulli: pd.DataFrame,
    output_dir: Path,
    *,
    ky: int,
    point_n: int,
    first_numbers: tuple[int, int],
    chi_squared_numbers: tuple[int, int],
    uniform_numbers: tuple[int, int] | None = None,
) -> None:
    marginals_z = MARGINALS_Z_KY3 if uniform_numbers is not None else MARGINALS_Z
    binary_uniform_number = uniform_numbers[0] if uniform_numbers is not None else None
    gaussian_uniform_number = (
        uniform_numbers[1] if uniform_numbers is not None else None
    )
    plot_copula_network(
        filtered_copula_results(alternative_bernoulli, marginals_z=marginals_z),
        output_dir,
        network_name="Bernoulli binary network",
        filename_prefix="binary",
        first_number=first_numbers[0],
        chi_squared_number=chi_squared_numbers[0],
        uniform_number=binary_uniform_number,
        point_n=point_n,
        ky=ky,
        marginals_z=marginals_z,
    )
    plot_copula_network(
        filtered_copula_results(alternative_gaussian, marginals_z=marginals_z),
        output_dir,
        network_name="Gaussian weighted network",
        filename_prefix="gaussian",
        first_number=first_numbers[1],
        chi_squared_number=chi_squared_numbers[1],
        uniform_number=gaussian_uniform_number,
        point_n=point_n,
        ky=ky,
        marginals_z=marginals_z,
    )


def plot_rho_sweep_network(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    network_name: str,
    filename: str,
    n: int,
    ky: int,
) -> None:
    """Plot power over rho for every Y- and Z-marginal combination."""
    assert_unique(
        data,
        ["rho", "method", "marginal_z", "marginal_y"],
        f"{network_name} Gaussian-copula rho sweep",
    )
    fig, axes = plt.subplots(
        len(RHO_SWEEP_MARGINALS_Y),
        len(MARGINALS_Z_KY3),
        figsize=(7.0, 4.2),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for row, marginal_y in enumerate(RHO_SWEEP_MARGINALS_Y):
        for column, marginal_z in enumerate(MARGINALS_Z_KY3):
            ax = axes[row, column]
            panel = data[
                (data["marginal_y"] == marginal_y)
                & (data["marginal_z"] == marginal_z)
            ]
            plot_line_panel(
                ax,
                panel,
                y_mean="Rejection_mean",
                y_sem="Rejection_sem",
                x_column="rho",
            )
            style_probability_axis(ax)

    add_facet_labels(
        axes,
        column_values=MARGINALS_Z_KY3,
        column_prefix="Z marginal: ",
        row_values=RHO_SWEEP_MARGINALS_Y,
        row_prefix="Y marginal: ",
        column_labeler=display_marginal_label,
        row_labeler=display_marginal_label,
    )
    fig.suptitle(
        f"Power by dependence strength — {network_name}, Gaussian copula, "
        f"$n={n}$, $k_y={ky}$"
    )
    fig.supxlabel(r"Dependence parameter, $\rho$")
    fig.supylabel("Power")
    add_shared_legend(fig, lines=True)
    save_figure(fig, output_dir, filename)


def plot_rho_sweep_figures(
    gaussian: pd.DataFrame,
    bernoulli: pd.DataFrame,
    output_dir: Path,
    *,
    n: int,
    ky: int,
    figure_numbers: tuple[int, int],
) -> None:
    """Create the Gaussian- and Bernoulli-network rho-sweep figures."""
    plot_rho_sweep_network(
        aggregate_rho_results(gaussian, n=n),
        output_dir,
        network_name="Gaussian weighted network",
        filename=f"{figure_numbers[0]:02d}_gaussian_copula_rho_n{n}_ky{ky}",
        n=n,
        ky=ky,
    )
    plot_rho_sweep_network(
        aggregate_rho_results(bernoulli, n=n),
        output_dir,
        network_name="Bernoulli binary network",
        filename=f"{figure_numbers[1]:02d}_binary_copula_rho_n{n}_ky{ky}",
        n=n,
        ky=ky,
    )


def plot_latent_network(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    network_name: str,
    filename: str,
    ky: int,
) -> None:
    data = data[
        data["latent_sim"].isin(LATENT_SIMULATIONS)
        & data["method"].isin(METHODS)
        & (data["n"] == 300)
    ].copy()
    assert_unique(
        data,
        ["method", "latent_sim"],
        f"{network_name} latent simulations",
    )

    category_rows = (
        LATENT_SIMULATIONS[:8],
        LATENT_SIMULATIONS[8:],
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 4.5),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for row, categories in enumerate(category_rows):
        ax = axes[row, 0]
        panel = data[data["latent_sim"].isin(categories)]
        plot_scatter_panel(
            ax,
            panel,
            category="latent_sim",
            categories=categories,
        )
        style_probability_axis(ax)
        ax.set_title(f"({chr(ord('a') + row)})", loc="left")

    fig.suptitle(
        f"Power across latent dependence structures — {network_name}, "
        f"$n=300$, $k_y={ky}$"
    )
    fig.supxlabel("Latent dependence structure")
    fig.supylabel("Power")
    add_shared_legend(fig, lines=False)
    save_figure(fig, output_dir, filename)


def plot_latent_simulation_figures(
    alternative_gaussian: pd.DataFrame,
    alternative_bernoulli: pd.DataFrame,
    output_dir: Path,
    *,
    ky: int,
) -> None:
    plot_latent_network(
        alternative_bernoulli,
        output_dir,
        network_name="Bernoulli binary network",
        filename=f"07_binary_latent_simulations_n300_ky{ky}",
        ky=ky,
    )
    plot_latent_network(
        alternative_gaussian,
        output_dir,
        network_name="Gaussian weighted network",
        filename=f"08_gaussian_latent_simulations_n300_ky{ky}",
        ky=ky,
    )


def plot_functional_curves(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    network_name: str,
    filename: str,
    ky: int,
) -> None:
    categories = FUNCTIONAL_FORMS
    panel_data = data[data["functional_form"].isin(categories)].copy()
    assert_unique(
        panel_data,
        ["n", "method", "functional_form"],
        f"{network_name} functional curves",
    )
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(7.0, 5.0),
        sharex=True,
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    for index, category in enumerate(categories):
        row, column = divmod(index, 4)
        ax = axes[row, column]
        plot_line_panel(
            ax,
            panel_data[panel_data["functional_form"] == category],
            y_mean="Rejection_mean",
            y_sem="Rejection_sem",
        )
        style_probability_axis(ax)
        ax.set_title(display_label(category))

    fig.suptitle(f"Power by network size — {network_name}, $k_y={ky}$")
    fig.supxlabel(r"Network size, $n$")
    fig.supylabel("Power")
    add_shared_legend(fig, lines=True)
    save_figure(fig, output_dir, filename)


def plot_functional_figures(
    data: pd.DataFrame,
    output_dir: Path,
    *,
    network_name: str,
    scatter_n: int,
    filename_prefix: str,
    figure_numbers: tuple[int, int],
    ky: int,
) -> None:
    data = data[
        data["functional_form"].isin(FUNCTIONAL_FORMS)
        & data["method"].isin(METHODS)
    ].copy()

    point_data = data[data["n"] == scatter_n].copy()
    assert_unique(
        point_data,
        ["method", "functional_form"],
        f"{network_name} functional point ranges",
    )
    fig, axes = plt.subplots(
        1,
        1,
        figsize=(7.0, 2.9),
        squeeze=False,
        layout="constrained",
    )
    ax = axes[0, 0]
    plot_scatter_panel(
        ax,
        point_data,
        category="functional_form",
        categories=FUNCTIONAL_FORMS,
    )
    style_probability_axis(ax)
    fig.suptitle(
        f"Power across functional alternatives — {network_name}, "
        f"$n={scatter_n}$, $k_y={ky}$"
    )
    fig.supxlabel("Functional alternative")
    fig.supylabel("Power")
    add_shared_legend(fig, lines=False)
    save_figure(
        fig,
        output_dir,
        (
            f"{figure_numbers[0]:02d}_{filename_prefix}_functional_forms_"
            f"n{scatter_n}_ky{ky}"
        ),
    )

    plot_functional_curves(
        data,
        output_dir,
        network_name=network_name,
        filename=(
            f"{figure_numbers[1]:02d}_{filename_prefix}_functional_forms_"
            f"by_n_ky{ky}"
        ),
        ky=ky,
    )


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_plot_style()
    gaussian, bernoulli = prepare_results(results_dir)
    gaussian_ky3, bernoulli_ky3 = prepare_ky3_copula_results(results_dir)
    null_gaussian_ky3_raw, null_bernoulli_ky3_raw = prepare_ky3_null_results(
        results_dir
    )
    rho_gaussian_ky3, rho_bernoulli_ky3 = prepare_ky3_rho_results(results_dir)
    (
        null_gaussian,
        null_bernoulli,
        alternative_gaussian,
        alternative_bernoulli,
    ) = aggregate_inputs(gaussian, bernoulli)
    alternative_gaussian_ky3 = aggregate_alternative_results(gaussian_ky3)
    alternative_bernoulli_ky3 = aggregate_alternative_results(bernoulli_ky3)
    null_gaussian_ky3 = aggregate_null_results(null_gaussian_ky3_raw)
    null_bernoulli_ky3 = aggregate_null_results(null_bernoulli_ky3_raw)

    plot_null_figures(
        null_gaussian,
        null_bernoulli,
        output_dir,
        ky=1,
        figure_numbers=(1, 2),
    )
    plot_copula_figures(
        alternative_gaussian,
        alternative_bernoulli,
        output_dir,
        ky=1,
        point_n=300,
        first_numbers=(3, 5),
        chi_squared_numbers=(17, 18),
    )
    plot_latent_simulation_figures(
        alternative_gaussian,
        alternative_bernoulli,
        output_dir,
        ky=1,
    )
    plot_functional_figures(
        alternative_bernoulli,
        output_dir,
        network_name="Bernoulli binary network",
        scatter_n=300,
        filename_prefix="binary",
        figure_numbers=(9, 10),
        ky=1,
    )
    plot_functional_figures(
        alternative_gaussian,
        output_dir,
        network_name="Gaussian weighted network",
        scatter_n=500,
        filename_prefix="gaussian",
        figure_numbers=(11, 12),
        ky=1,
    )
    plot_copula_figures(
        alternative_gaussian_ky3,
        alternative_bernoulli_ky3,
        output_dir,
        ky=3,
        point_n=200,
        first_numbers=(13, 15),
        chi_squared_numbers=(19, 20),
        uniform_numbers=(23, 24),
    )
    plot_null_figures(
        null_gaussian_ky3,
        null_bernoulli_ky3,
        output_dir,
        ky=3,
        figure_numbers=(21, 22),
    )
    plot_rho_sweep_figures(
        rho_gaussian_ky3,
        rho_bernoulli_ky3,
        output_dir,
        n=200,
        ky=3,
        figure_numbers=(25, 26),
    )

    print(f"Saved 26 figures as PNG to {output_dir}")


if __name__ == "__main__":
    main()
