"""Faceted metric plots independent of study-specific figure selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import cycle
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from .processing import filter_results, aggregate_metric, _is_missing_value
from .styles import METHOD_LABELS, COLORS, MARKERS, LINESTYLES


def _column_list(columns: str | Sequence[str] | None) -> list[str]:
    if columns is None:
        return []
    return [columns] if isinstance(columns, str) else list(columns)


def _ordered_values(values: pd.Series, requested: Sequence[Any] | None) -> list[Any]:
    if requested is not None:
        return list(requested)
    unique = values.drop_duplicates().tolist()
    try:
        return sorted(unique)
    except TypeError:
        return unique


def plot_metric_grid(
    results: pd.DataFrame,
    *,
    value: str = "Rejection",
    x: str = "n",
    series: str | Sequence[str] = "method",
    row: str | None = None,
    col: str | None = "p",
    filters: Mapping[str, Any] | None = None,
    row_order: Sequence[Any] | None = None,
    col_order: Sequence[Any] | None = None,
    series_order: Sequence[Any] | None = None,
    error: str | None = "sem",
    labels: Mapping[Any, str] | None = None,
    colors: Mapping[Any, str] | None = None,
    markers: Mapping[Any, str] | None = None,
    linestyles: Mapping[Any, str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    reference_y: float | None = None,
    reference_label: str | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] | None = None,
    sharex: bool = True,
    sharey: bool = True,
    output_path: str | Path | None = None,
) -> tuple[Figure, np.ndarray, pd.DataFrame]:
    """Aggregate and plot a reusable faceted line grid.

    ``series`` may contain several columns, so method variants can be drawn as
    distinct lines without modifying the input. The returned tuple contains
    the figure, the two-dimensional axes array, and the aggregated data used
    by the plot.
    """
    data = filter_results(results, filters)
    series_columns = _column_list(series)
    facet_columns = [column for column in (row, col) if column is not None]
    group_columns = list(dict.fromkeys([*facet_columns, *series_columns, x]))
    summary = aggregate_metric(
        data,
        value,
        group_columns,
        error=error,
    )
    mean_column = f"{value}_mean"
    error_column = f"{value}_{error}" if error is not None else None

    row_values = [None] if row is None else _ordered_values(summary[row], row_order)
    col_values = [None] if col is None else _ordered_values(summary[col], col_order)
    if not row_values or not col_values:
        raise ValueError("No rows remain after applying the plot filters.")

    series_frame = summary[series_columns].drop_duplicates()
    keys = [tuple(record) for record in series_frame.itertuples(index=False, name=None)]
    if series_order is not None:
        normalized_order = [
            item if isinstance(item, tuple) else (item,) for item in series_order
        ]
        available = set(keys)
        ordered = [key for key in normalized_order if key in available]
        keys = ordered + [key for key in keys if key not in ordered]

    labels = {} if labels is None else dict(labels)
    colors = {} if colors is None else dict(colors)
    markers = {} if markers is None else dict(markers)
    linestyles = {} if linestyles is None else dict(linestyles)
    palette = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    marker_cycle = cycle(("o", "s", "^", "v", "D", "P", "X", "*"))
    style_by_key = {}
    for key in keys:
        lookup = key[0] if len(key) == 1 else key
        default_color = COLORS.get(lookup, next(palette))
        default_marker = MARKERS.get(lookup, next(marker_cycle))
        default_linestyle = LINESTYLES.get(lookup, "-")
        if lookup in labels:
            display = labels[lookup]
        elif len(key) == 1:
            display = METHOD_LABELS.get(lookup, str(lookup))
        else:
            display = " | ".join(
                f"{column}={'NA' if _is_missing_value(value_) else value_}"
                for column, value_ in zip(series_columns, key)
            )
        style_by_key[key] = {
            "label": display,
            "color": colors.get(lookup, default_color),
            "marker": markers.get(lookup, default_marker),
            "linestyle": linestyles.get(lookup, default_linestyle),
        }

    if figsize is None:
        figsize = (3.0 * len(col_values), 2.5 * len(row_values) + 0.5)
    fig, axes = plt.subplots(
        len(row_values),
        len(col_values),
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=False,
        layout="constrained",
    )
    for row_index, row_value in enumerate(row_values):
        for col_index, col_value in enumerate(col_values):
            ax = axes[row_index, col_index]
            panel = summary
            if row is not None:
                if _is_missing_value(row_value):
                    panel = panel[panel[row].map(_is_missing_value)]
                else:
                    panel = panel[panel[row] == row_value]
            if col is not None:
                if _is_missing_value(col_value):
                    panel = panel[panel[col].map(_is_missing_value)]
                else:
                    panel = panel[panel[col] == col_value]
            for key in keys:
                line = panel
                for column, series_value in zip(series_columns, key):
                    if _is_missing_value(series_value):
                        line = line[line[column].map(_is_missing_value)]
                    else:
                        line = line[line[column] == series_value]
                line = line.sort_values(x)
                if line.empty:
                    continue
                style = style_by_key[key]
                x_values = pd.to_numeric(line[x], errors="coerce").to_numpy()
                means = line[mean_column].to_numpy(dtype=float)
                ax.plot(
                    x_values,
                    means,
                    label=style["label"],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    markerfacecolor="none",
                )
                if error_column is not None:
                    errors = line[error_column].to_numpy(dtype=float)
                    ax.fill_between(
                        x_values,
                        means - errors,
                        means + errors,
                        color=style["color"],
                        alpha=0.12,
                        linewidth=0,
                    )
            if reference_y is not None:
                ax.axhline(
                    reference_y,
                    color="#555555",
                    linestyle="--",
                    linewidth=1,
                    label=reference_label,
                )
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_xticks(
                sorted(pd.to_numeric(panel[x], errors="coerce").dropna().unique())
            )
            ax.grid(axis="y", color="#E2E2E2", linewidth=0.45)
            title_parts = []
            if row is not None:
                title_parts.append(f"{row} = {row_value}")
            if col is not None:
                title_parts.append(f"{col} = {col_value}")
            if title_parts:
                ax.set_title(" — ".join(title_parts))

    fig.supxlabel(xlabel or x)
    fig.supylabel(ylabel or value)
    if title:
        fig.suptitle(title)
    unique_handles = {}
    for ax in axes.flat:
        handles, legend_labels = ax.get_legend_handles_labels()
        unique_handles.update(zip(legend_labels, handles))
    if unique_handles:
        fig.legend(
            unique_handles.values(),
            unique_handles.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99 if title is None else 0.95),
            ncols=max(1, len(unique_handles)),
            frameon=False,
        )
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    return fig, axes, summary
