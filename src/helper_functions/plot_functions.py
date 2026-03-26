import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
import logging
import re


# Suppress weird matplotlib category warning for boxplots
logging.getLogger("matplotlib.category").setLevel(logging.ERROR)

def visualise_latent(X_list, 
                    Z_list, 
                    titles='Latent Positions Scatterplot', 
                    figsize=(18, 6),
                    sharex=False, 
                    sharey=False,
                    kdplot=True,
                    shape=None,
                    save_path=None,
                    k=0):
    """Visualise correlaton structure of latent positions

    Parameters
    ----------
    X_list : _type_
        _description_
    Z_list : _type_
        _description_
    title : str, optional
        _description_, by default 'Latent Positions Scatterplot'
    figsize : tuple, optional
        _description_, by default (18, 6)
    sharex : bool, optional
        _description_, by default True
    sharey : bool, optional
        _description_, by default True
    kdplot : bool, optional
        _description_, by default True
    k : int, optional
        _description_, by default 1
    """

    if shape is None:
        fig, axes = plt.subplots(1, len(X_list), figsize=figsize, sharex=sharex, sharey=sharey)
    else:
        fig, axes = plt.subplots(shape[0], shape[1], figsize=figsize, sharex=sharex, sharey=sharey)
        axes = axes.flatten()

    for ax, i in zip(axes, range(len(X_list))):
        z, x = X_list[i][:, k], Z_list[i][:, k]

        ax.scatter(z, x, alpha=0.4, s=10, color='royalblue', label='Samples')
        
        # countour density may make more clear
        if kdplot:
            sns.kdeplot(x=z, y=x, ax=ax, levels=5, color='black', linewidths=1.5)
        
        if isinstance(titles, list):
            title = titles[i]
        else:
            title = titles
        ax.set_title(title, fontsize=14, weight='bold')
        # ax.set_xlim(-4, 4)
        # ax.set_ylim(-4, 4)
        ax.set_xlabel("Latent Z", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # diagonal perfect correlation line
        #ax.plot([-4, 4], [-4, 4], 'r--', alpha=0.5, label='Perfect Correlation')
        
        ax.set_ylabel("Latent X", fontsize=12)

    #axes[0].set_ylabel("Latent X", fontsize=12)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close() 


def create_dashed_boxed_message(message):
    """Wraps the given message string in a box frame made of dashes and pipes.

    Args:
        message: The string to be placed inside the box.

    Returns:
        A formatted string with the message centered in a dashed box.
    """

    padding = 4
    content_width = len(message)
    line_width = content_width + padding
    horizontal_line = "-" * line_width
    message_line = f"|  {message}  |"

    return f"\n\n{horizontal_line}\n{message_line}\n{horizontal_line}\n\n"


def plot_with_bands(x_axis, y_axis, **kwargs):
    """Plot lines with confidence/error bands for each method.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data to plot.
    x_axis : str
        The name of the column to be used for the x-axis.
    y_axis : str
        The name of the column to be used for the y-axis.
    factors : list, optional
        A list of column names to be used as additional factors for grouping,
        by default None
    plot_bands : str, optional
        Name of the column containing the standard error for the y-axis values,
        if None no bands are drawn, by default None
    colors : dict, optional
        A dictionary mapping factor values to colors, by default None
    linestyles : dict, optional
        A dictionary mapping factor values to linestyles, by default None
    """
    data = kwargs.pop("data")
    factors = kwargs.pop("factors", None)
    se_bands = kwargs.pop("se_bands", None)
    colors = kwargs.pop("colors", None)
    linestyles = kwargs.pop("linestyles", None)
    hline = kwargs.pop("hline", None)
    hline_name = kwargs.pop("hline_name", "Reference Line")

    ax = plt.gca()
    hue_variable = factors[0] if factors is not None and len(factors) >= 1 else None

    if hue_variable is not None:
        for hue_var in data[hue_variable].unique():
            subset = data[data[hue_variable] == hue_var].sort_values(x_axis)
            line = ax.plot(
                subset[x_axis],
                subset[y_axis],
                marker="o",
                linestyle=linestyles[hue_var] if linestyles is not None else "-",
                color=colors[hue_var] if colors is not None else None,
                label=hue_var,
            )
            color = line[0].get_color()

            if se_bands is not None:
                ax.fill_between(
                    subset[x_axis],
                    subset[y_axis] - subset[se_bands],
                    subset[y_axis] + subset[se_bands],
                    alpha=0.2,
                    color=color,
                )
    else:
        # assume single line
        subset = data.sort_values(x_axis)
        line = ax.plot(
            subset[x_axis], subset[y_axis], marker="o", linestyle="-", label=None
        )
        color = line[0].get_color()

        if se_bands is not None:
            ax.fill_between(
                subset[x_axis],
                subset[y_axis] - subset[se_bands],
                subset[y_axis] + subset[se_bands],
                alpha=0.2,
                color=color,
            )
    
    if hline is not None:
        ax.axhline(hline, color='black', linestyle='-', label=hline_name)
        
        
def plot_scatter_markers(x_axis, y_axis, **kwargs):
    """Plot scatter points with per-method marker styles and hollow/solid rendering.
 
    Designed to be passed as ``plotting_function`` to :func:`plot_grid`, but can
    also be called standalone via ``plt.gca()``.
 
    Each method is drawn with its own marker shape and color.  Markers that are
    *not* ``'x'`` (or ``'+'``) are rendered hollow (face transparent, colored
    edge), while cross-style markers are drawn solid so their line-only geometry
    remains visible.
 
    Parameters
    ----------
    x_axis : str
        Column name for the x-axis.
    y_axis : str
        Column name for the y-axis (typically a ``*_mean`` column produced by
        :func:`analyse_functions.aggregate_results`).
    data : pd.DataFrame
        DataFrame slice passed in by :class:`~seaborn.FacetGrid` (or supplied
        directly).
    factors : list
        Column names used as grouping factors.  ``factors[0]`` is treated as
        the *hue* variable whose unique values are looked up in ``colors`` and
        ``markers``.
    colors : dict, optional
        Mapping of hue-variable value → matplotlib color string.  Falls back to
        the default color cycle when a value is missing.
    markers : dict, optional
        Mapping of hue-variable value → matplotlib marker string.  Falls back to
        ``'o'`` when a value is missing.
    marker_size : int, optional
        Scatter point size (``s`` argument), by default 80.
    linewidths : float, optional
        Edge line width for hollow markers and stroke width for ``'x'``/``'+'``
        markers, by default 2.5.
    ylim : tuple, optional
        ``(ymin, ymax)`` limits for the y-axis.  When *not* supplied and
        ``y_axis`` looks like a rejection/power column the axis is clamped to
        ``(-0.05, 1.05)`` automatically.
    yticks : list, optional
        Explicit y-tick positions.
    yticklabels : list, optional
        Labels matching ``yticks``.
    y_axis_title : str, optional
        Y-axis label, by default ``'Power'``.
    spine_color : str, optional
        Edge color applied to all four spines, by default ``'#555555'``.
    tick_direction : str, optional
        Direction for both axes ticks (``'in'``, ``'out'``, ``'inout'``),
        by default ``'in'``.
    hline : float, optional
        Draw a horizontal reference line at this y value.
    hline_name : str, optional
        Label for the horizontal reference line, by default ``'Reference Line'``.
    """
    
    data = kwargs.pop("data")
    factors = kwargs.pop("factors", None)
    colors = kwargs.pop("colors", None)
    markers = kwargs.pop("markers", None)
    marker_size = kwargs.pop("marker_size", 80)
    linewidths = kwargs.pop("linewidths", 2.5)
    ylim = kwargs.pop("ylim", None)
    yticks = kwargs.pop("yticks", None)
    yticklabels = kwargs.pop("yticklabels", None)
    y_axis_title = kwargs.pop("y_axis_title", "Power")
    spine_color = kwargs.pop("spine_color", "#555555")
    tick_direction = kwargs.pop("tick_direction", "in")
    hline = kwargs.pop("hline", None)
    hline_name = kwargs.pop("hline_name", "Reference Line")
 
    ax = plt.gca()
 
    # Resolve hue variable from factors (mirrors plot_with_bands convention)
    hue_variable = factors[0] if factors is not None and len(factors) >= 1 else None
 
    if hue_variable is not None:
        for method in data[hue_variable].unique():
            subset = data[data[hue_variable] == method].sort_values(x_axis)
            color = colors[method] if (colors is not None and method in colors) else None
            marker = markers[method] if (markers is not None and method in markers) else "o"
 
            # Cross/plus markers are line-only — draw solid
            if marker in ("x", "+"):
                ax.scatter(
                    subset[x_axis],
                    subset[y_axis],
                    marker=marker,
                    color=color,
                    s=marker_size,
                    linewidths=linewidths,
                    label=method,
                    zorder=3,
                )
            else:
                # All other markers: hollow (transparent face, colored edge)
                ax.scatter(
                    subset[x_axis],
                    subset[y_axis],
                    marker=marker,
                    edgecolors=color,
                    facecolors="none",
                    s=marker_size,
                    linewidths=linewidths,
                    label=method,
                    zorder=3,
                )
    else:
        # No hue — single series, plain scatter
        subset = data.sort_values(x_axis)
        ax.scatter(subset[x_axis], subset[y_axis], s=marker_size, zorder=3)
 
    # --- y-axis formatting ---
    power_pattern = re.compile(
        r"(?:power|rejection|reject|type.?i)", re.IGNORECASE
    )
    auto_power = power_pattern.search(y_axis) is not None
 
    if ylim is not None:
        ax.set_ylim(*ylim)
    elif auto_power:
        ax.set_ylim(-0.05, 1.05)
 
    if yticks is not None:
        ax.set_yticks(yticks)
        if yticklabels is not None:
            ax.set_yticklabels(yticklabels, rotation=90, va="center")
    elif auto_power and ylim is None:
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"], rotation=90, va="center")
 
    if y_axis_title:
        ax.set_ylabel(y_axis_title, fontsize=16)
 
    # --- tick styling ---
    ax.tick_params(axis="y", direction=tick_direction, length=6, pad=10)
    ax.tick_params(axis="x", direction=tick_direction, length=8, pad=15)
 
    # --- spine styling ---
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)
 
    # --- optional horizontal reference line ---
    if hline is not None:
        ax.axhline(hline, color="black", linestyle="-", label=hline_name)



def plot_boxplot(x_axis, y_axis, **kwargs):
    """Create a boxplot for the given x and y axes.

    Parameters
    ----------
    x_axis : str
        The name of the column to be used for the x-axis.
    y_axis : str
        The name of the column to be used for the y-axis.
    data : pd.DataFrame
        DataFrame containing the data to plot.
    factors : list, optional
        A list of column names to be used as additional factors for grouping,
        by default None
    log_y_axis : bool, optional
        Whether to use a logarithmic scale for the y-axis, by default False
    log_x_axis : bool, optional
        Whether to use a logarithmic scale for the x-axis, by default False
    n_boxplots : int, optional
        Number of boxplots to display along the x-axis, by default 5
    colors : dict, optional
        A dictionary mapping factor values to colors, by default None
    """
    data = kwargs.pop("data")
    factors = kwargs.pop("factors", None)
    log_y_axis = kwargs.pop("log_y_axis", False)
    log_x_axis = kwargs.pop("log_x_axis", False)
    n_boxplots = kwargs.pop("n_boxplots", 5)
    colors = kwargs.pop("colors", None)

    ax = plt.gca()

    temp = data.copy()

    if log_x_axis is True:
        temp[x_axis] = np.log10(temp[x_axis])
    if log_y_axis is True:
        temp[y_axis] = np.log10(temp[y_axis])

    if n_boxplots < len(temp[x_axis].unique()):
        # Select n_boxplots evenly spaced along x
        df_values = sorted(temp[x_axis].unique())
        selected_dfs = np.linspace(0, len(df_values) - 1, n_boxplots, dtype=int)
        selected_dfs = [df_values[i] for i in selected_dfs]
        temp = temp[temp[x_axis].isin(selected_dfs)]

    hue_variable = None
    if factors is not None and len(factors) >= 1:
        hue_variable = factors[0]

    if hue_variable is not None:
        # Create palette from colors dict if provided
        palette = None
        if colors is not None:
            # Get unique hue values in the data
            hue_order = sorted(temp[hue_variable].unique())
            palette = [colors.get(hue_val, None) for hue_val in hue_order]

        sns.boxplot(
            data=temp, x=x_axis, y=y_axis, hue=hue_variable, palette=palette, ax=ax
        )
    else:
        sns.boxplot(data=temp, x=x_axis, y=y_axis, ax=ax)


def plot_heatmap(x_axis, y_axis, **kwargs):
    """Plot a heatmap inside a single FacetGrid facet.
 
    Within each facet the data is already filtered to one (col, row) combination
    by :class:`~seaborn.FacetGrid`.  This function pivots the remaining data so
    that ``x_axis`` runs along the columns of the heatmap and ``factors[0]``
    (the *hue* / method variable) runs along the rows, with ``y_axis`` as the
    cell colour intensity.
 
    Designed to be passed as ``plotting_function`` to :func:`plot_grid`, but
    can also be called standalone against ``plt.gca()``.
 
    Parameters
    ----------
    x_axis : str
        Column used for heatmap columns (e.g. ``'latent_sim'``).
    y_axis : str
        Column whose values fill the cells (e.g. ``'TrueRejection_mean'``).
    data : pd.DataFrame
        DataFrame slice passed in by :class:`~seaborn.FacetGrid` (or provided
        directly when calling standalone).
    factors : list
        ``factors[0]`` is used as the row-index of the heatmap (typically the
        method / hue variable).
    cmap : str, optional
        Matplotlib / seaborn colormap name, by default ``'RdYlGn'``.
    vmin : float, optional
        Minimum value of the colour scale.  Defaults to ``0`` when the column
        name suggests a power/rejection metric, otherwise the data minimum.
    vmax : float, optional
        Maximum value of the colour scale.  Defaults to ``1`` when the column
        name suggests a power/rejection metric, otherwise the data maximum.
    fmt : str, optional
        ``annot`` format string, by default ``'.2f'``.
    annot : bool, optional
        Whether to annotate cells with their values, by default ``True``.
    annot_fontsize : int, optional
        Font size for cell annotations, by default ``8``.
    linewidths : float, optional
        Width of lines separating cells, by default ``0.5``.
    linecolor : str, optional
        Colour of the cell separator lines, by default ``'white'``.
    row_order : list, optional
        Explicit ordering of hue-variable values (rows).  Defaults to sorted
        order of the unique values found in the facet slice.
    col_order : list, optional
        Explicit ordering of x-axis values (columns).  Defaults to sorted
        ascending order.
    cbar : bool, optional
        Whether to draw the colour-bar, by default ``False`` (keeps facets
        compact; a shared colour-bar can be added afterwards if needed).
    x_tick_rotation : float, optional
        Rotation of x-axis tick labels in degrees, by default ``0``.
    y_tick_rotation : float, optional
        Rotation of y-axis tick labels in degrees, by default ``0``.
    """
    data        = kwargs.pop("data")
    factors     = kwargs.pop("factors", None)
    cmap        = kwargs.pop("cmap", "RdYlGn")
    fmt         = kwargs.pop("fmt", ".2f")
    annot       = kwargs.pop("annot", True)
    annot_fontsize  = kwargs.pop("annot_fontsize", 8)
    linewidths  = kwargs.pop("linewidths", 0.5)
    linecolor   = kwargs.pop("linecolor", "white")
    row_order   = kwargs.pop("row_order", None)
    col_order   = kwargs.pop("col_order", None)
    cbar        = kwargs.pop("cbar", False)
    x_tick_rotation = kwargs.pop("x_tick_rotation", 0)
    y_tick_rotation = kwargs.pop("y_tick_rotation", 0)
 
    # Auto-detect sensible colour-scale bounds for power/rejection metrics
    power_pattern = re.compile(r"(?:power|rejection|reject|type.?i)", re.IGNORECASE)
    auto_power = power_pattern.search(y_axis) is not None
    vmin = kwargs.pop("vmin", 0.0 if auto_power else data[y_axis].min())
    vmax = kwargs.pop("vmax", 1.0 if auto_power else data[y_axis].max())
 
    hue_variable = factors[0] if factors is not None and len(factors) >= 1 else None
 
    if hue_variable is None:
        raise ValueError(
            "plot_heatmap requires at least one entry in `factors` to use as "
            "the heatmap row variable."
        )
 
    # Determine row/column ordering
    if row_order is None:
        row_order = sorted(data[hue_variable].unique())
    if col_order is None:
        col_order = sorted(data[x_axis].unique())
 
    # Pivot to (hue_variable) × (x_axis) matrix
    pivot = (
        data.pivot_table(index=hue_variable, columns=x_axis, values=y_axis, aggfunc="mean")
        .reindex(index=row_order, columns=col_order)
    )
 
    ax = plt.gca()
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        fmt=fmt,
        annot_kws={"size": annot_fontsize},
        linewidths=linewidths,
        linecolor=linecolor,
        cbar=cbar,
    )
 
    # Tick label rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation=x_tick_rotation, ha="right" if x_tick_rotation else "center")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=y_tick_rotation, va="center")
 
    # Clear axis labels — plot_grid sets them centrally
    ax.set_xlabel("")
    ax.set_ylabel("")


def plot_grid(grouped_stats, x_axis, y_axis, factors, plotting_function=None, **kwargs):
    """Plot a grid of plots using the specified plotting function.
 
    Parameters
    ----------
    results : pd.DataFrame
        DataFrame containing the data to plot.
    x_axis : str
        The name of the column to be used for the x-axis.
    y_axis : str
        The name of the column to be used for the y-axis.
    factors : list
        A list of column names to be used as additional factors for grouping.
    plotting_function : callable, optional
        A function to use for plotting, by default None
    height : float, optional
        Height of each facet in inches, by default 1.3
    aspect : float, optional
        Aspect ratio of each facet, by default 1.3
    group_variables : bool, optional
        Whether to aggregate results by computing mean and standard error
        for each combination of factors, by default False
    se_bands : str, optional
        The name of the column to use for plotting standard error bands, by default None
    log_y_axis : bool, optional
        Whether to use a logarithmic scale for the y-axis, by default False
    log_x_axis : bool, optional
        Whether to use a logarithmic scale for the x-axis, by default False
    name_conversion : dict, optional
        A dictionary mapping variable names to more descriptive names for
        axis labels and titles, by default {}
    add_legend : bool, optional
        Whether to add a legend to the plot, by default True
    no_facet_y_axis : bool, optional
        Whether to hide y-axis tick labels and ticks on every facet, by default False.
        Useful when the shared y-axis label is sufficient and per-cell tick text
        would be redundant or cluttered.
    save_path : str, optional
        Path to save the plot, by default None
 
 
    Returns
    -------
    sns.FacetGrid
        The FacetGrid object containing the plots.
    """
    if plotting_function is None:
        raise ValueError("plotting_function must be provided.")
 
    height = kwargs.get("height", 1.3)
    save_path = kwargs.get("save_path", None)
    se_bands = kwargs.get("se_bands", None)
    log_y_axis = kwargs.get("log_y_axis", False)
    log_x_axis = kwargs.get("log_x_axis", False)
    aspect = kwargs.get("aspect", 1.3)
    name_conversion = kwargs.get("name_conversion", {})
    add_legend = kwargs.get("add_legend", True)
    title = kwargs.get("title", None)
    x_axis_title = kwargs.get("x_axis_title", None)
    y_axis_title = kwargs.get("y_axis_title", None)
    share_x = kwargs.get("share_x", True)
    share_y = kwargs.get("share_y", True)
    flip_x_axis = kwargs.get("flip_x_axis", False)
    no_facet_y_axis = kwargs.get("no_facet_y_axis", False)
    hide_row_titles = kwargs.get("hide_row_titles", False)
    show_row_names = kwargs.get("show_row_names", True)
 
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
 
    hue_variable = factors[0] if len(factors) >= 2 else None
    aggregate_x = factors[1] if len(factors) >= 2 else factors[0]
    aggregate_y = factors[2] if len(factors) >= 3 else None
 
    g = sns.FacetGrid(
        grouped_stats,
        row=aggregate_y,
        col=aggregate_x,
        sharey=share_y,
        sharex=share_x,
        height=height,
        aspect=aspect,
        margin_titles=not hide_row_titles,
    )
 
    g.map_dataframe(
        plotting_function,
        x_axis=x_axis,
        y_axis=y_axis,
        factors=factors,
        **kwargs,
    )
    # remove default x/y axis labels and tick labels from all subplots
    for ax in g.axes.flat:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        if no_facet_y_axis:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
 
    # Set x and y axis labels only in central places
    if x_axis_title is None:
        x_axis_title = (
            "Log " + name_conversion.get(x_axis, x_axis).replace("_", " ").title()
            if log_x_axis
            else name_conversion.get(x_axis, x_axis).replace("_", " ").title()
        )
    g.axes[-1, g.axes.shape[1] // 2].set_xlabel(x_axis_title)
 
    if y_axis_title is None:
        y_axis_title = (
            "Log " + name_conversion.get(y_axis, y_axis).replace("_", " ").title()
            if log_y_axis
            else name_conversion.get(y_axis, y_axis).replace("_", " ").title()
        )
 
    g.axes[g.axes.shape[0] // 2, 0].set_ylabel(y_axis_title)
 
    if len(factors) >= 2:
        # column facet titles
        for ax in range(g.axes.shape[1]):
            # put percentage sign for fraction variables
 
            if re.search(
                r"(?<![a-z])(?:percentage|fraction|prop)(?![a-z])",
                aggregate_x,
                re.IGNORECASE,
            ):
                facet_title = f"{int(g.col_names[ax] * 100)}% {name_conversion.get(aggregate_x, aggregate_x).replace('_', ' ').title()}"
            elif aggregate_x[0] == "_":
                facet_title = ""
            else:
                facet_title = f"{name_conversion.get(aggregate_x, aggregate_x).replace('_', ' ').title()}: {g.col_names[ax]}"
            g.axes[0, ax].set_title(facet_title)
 
        # custom row facet labels
        if aggregate_y is not None:
            if hide_row_titles:
                # seaborn stores margin title Text artists in _margin_titles_texts
                for t in getattr(g, "_margin_titles_texts", []):
                    t.set_visible(False)
                # also clear any texts attached directly to the rightmost axes
                for ax in range(g.axes.shape[0]):
                    for t in g.axes[ax, -1].texts:
                        t.set_visible(False)
            else:
                for ax in range(g.axes.shape[0]):
                    # put percentage sign for fraction variables
                    if re.search(
                        r"(?<![a-z])(?:percentage|fraction|prop)(?![a-z])",
                        aggregate_y,
                        re.IGNORECASE,
                    ):
                        text = f"{int(g.row_names[ax] * 100)}\\% {name_conversion.get(aggregate_y, aggregate_y).replace('_', ' ').title()}"
                    else:
                        if show_row_names:
                            text = f"{name_conversion.get(aggregate_y, aggregate_y).replace('_', ' ').title()}: {g.row_names[ax]}"
                        else:
                            text = f"{g.row_names[ax]}"
                    g.axes[ax, -1].texts[0].set_text(text)
 
    if add_legend is True:
        g.add_legend()
 
    # code to make the title centered above the grid not the legend
    plot_center_x = (
        g.axes[0, 0].get_position().x0 + g.axes[0, -1].get_position().x1
    ) / 2
    if title is not None:
        g.figure.suptitle(
            title,
            y=1.02,
            x=plot_center_x,
        )
    elif title=="":
        pass
    else:
        if log_y_axis is True:
            g.figure.suptitle(
                "Log "
                + name_conversion.get(x_axis, x_axis)
                + " vs Log "
                + name_conversion.get(y_axis, y_axis)
                if log_x_axis
                else name_conversion.get(x_axis, x_axis)
                + " vs Log "
                + name_conversion.get(y_axis, y_axis),
                y=1.02,
                x=plot_center_x,
            )
        else:
            g.figure.suptitle(
                "Log "
                + name_conversion.get(x_axis, x_axis)
                + " vs "
                + name_conversion.get(y_axis, y_axis)
                if log_x_axis
                else name_conversion.get(x_axis, x_axis)
                + " vs "
                + name_conversion.get(y_axis, y_axis),
                y=1.02,
                x=plot_center_x,
            )
    
    # flip x axis
    if flip_x_axis is True:
        print('flipping')
        for ax in g.axes.flat:
            left, right = ax.get_xlim()
            # Only flip if the axis is in standard ascending order (not flipped yet)
            if left < right:
                # ax.invert_xaxis() is slightly cleaner than ax.set_xlim(right, left)
                ax.invert_xaxis()
 
    if save_path is not None:
        plt.savefig(save_path + ".png", dpi=300, bbox_inches="tight")
        plt.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    return g