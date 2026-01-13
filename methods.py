import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm
import pandas as pd

def solve_independent(A, k=2):
    evals, evectors = eigsh(A, k=k, which='LM')
    evals = np.maximum(evals-0.5, 0)
    xhat = evectors @ np.diag(np.sqrt(evals))
    return xhat, evals

def rv_coefficient(A, B):
    num = np.trace((A.T @ B) @ (B.T @ A))
    den = norm(A.T @ A, 'fro') * norm(B.T @ B, 'fro')
    return num / den if den != 0 else 0

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



def llk_gradient(Z, X, A, B, sigma):
    n = Z.shape[0]
    grad = np.zeros((2*n, 2*n))
    grad[:n, :n] = Z@Z.T - A
    grad[n:, n:] = X@X.T - B
    theta_sigma = np.block([[np.eye(n), -sigma * np.eye(n)], [- sigma * np.eye(n), np.eye(n)]])
    theta_sigma *= 1/(1-sigma**2)
    grad = grad - theta_sigma
    return grad

def solve_dependent(A, B, k, niters, step_size=None):
    n = A.shape[0]
    Z_init, _ = solve_independent(B, k=k)
    X_init, _ = solve_independent(A, k=k)
        
    ZZ = Z_init @ Z_init.T
    XX = X_init @ X_init.T
    XZ = Z_init @ X_init.T
    ZX = X_init @ Z_init.T

    sigma = np.random.randn()
    
    M = np.block([[ZZ, XZ], [ZX, XX]])
    
    Ms = [M.copy()]
    sigma_list = [sigma]
    
    # don't remember where this comes from
    if step_size is None:
        eta = 4 * np.sqrt(2) / np.sqrt(3) * np.sqrt(n)
    else:
        eta = step_size
    
    for _ in range(niters):
        # gradient part
        grad = llk_gradient(Z_init, X_init, A, B, sigma=sigma)
        Y = M - grad
        
        # proximal operator part
        svd = np.linalg.svd(Y, full_matrices=False)
        U, S, Vt = svd
        Smax = np.maximum(S - eta, 0)
        M = U @ np.diag(Smax) @ Vt
        Ms.append(M.copy())
        
        # find sigma solving gradient equal to zero
        S_diag = np.trace(M[:n, :n]) + np.trace(M[n:, n:])
        S_cross = np.trace(M[:n, n:])

        coeff_a = n * k
        coeff_b = -S_cross
        coeff_c = S_diag - (n * k)
        coeff_d = -S_cross
        roots = np.roots([coeff_a, coeff_b, coeff_c, coeff_d])

        real_roots = roots.real[np.abs(roots.imag) < 1e-5]

        # 2. Keep only roots strictly in range (-1, 1)
        valid_roots = real_roots[(real_roots > -1.0) & (real_roots < 1.0)]
        
        if len(valid_roots) == 0:
            sigma = 0.0
        elif len(valid_roots) == 1:
            sigma = valid_roots[0]
        else:
            #don't really like this idea is to minimise the profile likelihood
            profile_llk = lambda s: (n * k * np.log(1 - s**2)) + (S_diag - 2 * s * S_cross) / (1 - s**2)

            sigma = valid_roots[np.argmin([profile_llk(r) for r in valid_roots])]
        sigma_list.append(sigma)

    return Ms, sigma_list