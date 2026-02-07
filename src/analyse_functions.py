# function to put together the results of simulations.
# quick and dirty approach, these were simply copied from BHreplication code

def _ratio_helper(df, factors, ratio_variable, y_axis, num, den):
    df_ratio = df.pivot_table(
        index=factors, columns=ratio_variable, values=y_axis + "_mean"
    ).reset_index()
    df_ratio[y_axis + "_ratio"] = df_ratio[num] / df_ratio[den]

    return df_ratio

def aggregate_results(
    results,
    y_axis,
    x_axis,
    factors=None,
    log_x_axis=False,
    log_y_axis=False,
    transform=None,
):
    """Compute dataset with mean and standard error for each group.

    Parameters
    ----------
    results : pd.DataFrame
        Input DataFrame to group and aggregate.
    y_axis : str
        The name of the column to be used for the y-axis.
    x_axis : str
        The name of the column to be used for the x-axis.
    factors : list, optional
        A list of column names to be used as additional factors for grouping,
        by default None
    log_x_axis : bool, optional
        Whether to use a logarithmic scale for the x-axis, by default True
    log_y_axis : bool, optional
        Whether to use a logarithmic scale for the y-axis, by default False
    transform : callable, optional
        A function to apply to the df after aggregation, by default None

    Returns
    -------
    pd.DataFrame
        DataFrame containing the aggregated results with mean and standard error for each group
    """
    if factors is None:
        factors = []

    grouping = [x_axis] + factors

    grouped_stats = (
        results.groupby(grouping).agg({y_axis: ["mean", "sem"]}).reset_index()
    )
    grouped_stats.columns = grouping + [
        f"{y_axis}_mean",
        f"{y_axis}_sem",
    ]

    if log_y_axis is True:
        grouped_stats[f"{y_axis}_mean"] = np.log10(grouped_stats[f"{y_axis}_mean"])
        grouped_stats[f"{y_axis}_sem"] = (
            grouped_stats[f"{y_axis}_sem"] / grouped_stats[f"{y_axis}_mean"]
        )

    if log_x_axis is True:
        grouped_stats[x_axis] = np.log10(grouped_stats[x_axis])

    if transform is not None:
        grouped_stats = transform(grouped_stats)

    return grouped_stats


def analyse_function(results, x_axis, y_axis, factors, **kwargs):
    group_variables = kwargs.get("group_variables", False)
    log_y_axis = kwargs.get("log_y_axis", False)
    log_x_axis = kwargs.get("log_x_axis", False)
    ratio_variable = kwargs.get("ratio_variable", None)

    results = results.copy()

    if group_variables is True:
        grouped_stats = aggregate_results(
            results,
            x_axis=x_axis,
            y_axis=y_axis,
            factors=factors + ([ratio_variable] if ratio_variable is not None else []),
            log_x_axis=log_x_axis,
            log_y_axis=log_y_axis,
        )
        if ratio_variable is not None:
            den, num = sorted(results[ratio_variable].unique())
            grouped_stats = _ratio_helper(
                grouped_stats,
                factors=factors + [x_axis],
                ratio_variable=ratio_variable,
                y_axis=y_axis,
                num=num,
                den=den,
            )
    else:
        # for consistency, for boxplot we don't aggregate
        grouped_stats = results.copy()

    if len(factors) < 2:
        # for consistency this forces FaceGrid to plot a single cell
        grouped_stats = grouped_stats.copy()
        grouped_stats["_single_facet"] = " "

    else:
        hue_variable = factors[0] if len(factors) >= 2 else None
        aggregate_x = factors[1] if len(factors) >= 2 else factors[0]
        aggregate_y = factors[2] if len(factors) >= 3 else None

        if aggregate_y:
            # for consistency, if only one aggregating variable plot a row
            grouped_stats[aggregate_y] = pd.Categorical(
                grouped_stats[aggregate_y],
                categories=sorted(grouped_stats[aggregate_y].unique(), reverse=True),
                ordered=True,
            )
    return grouped_stats
