import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm
import pandas as pd


class BaseMethod:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses should implement this!")
    def name(self):
        raise NotImplementedError("Subclasses should implement this!")
    def get_estimated(self):
        raise NotImplementedError("Subclasses should implement this!")
    def get_truth(self):
        raise NotImplementedError("Subclasses should implement this!")
    

def solve_independent(A, k=2, rng=None, **kwargs):
    if rng is None:
        rng = np.random.default_rng()
    v0 = rng.normal(size=A.shape[0])
    evals, evectors = eigsh(A, k=k, which='LM', v0=v0)
    evals = np.maximum(evals-0.5, 0)
    xhat = evectors @ np.diag(np.sqrt(evals))
    return [xhat], [evals]

def rv_coefficient(A, B):
    num = np.trace((A.T @ B) @ (B.T @ A))
    den = norm(A.T @ A, 'fro') * norm(B.T @ B, 'fro')
    return num / den if den != 0 else 0

def llk_gradient(M, A, B, sigma, n):
    """
    Computes the gradient of the objective function w.r.t M.
    Objective: ||P_diag(M-Y)||^2 + 1/(1-sigma^2)*tr(M * Theta)
    """
    scale = 1.0 / (1 - sigma**2)
    
    grad = np.zeros_like(M)
    
    idx = np.arange(n)
    grad[idx, idx] += scale
    grad[n + idx, n + idx] += scale
    
    
    grad[:n, :n] += 2* (M[:n, :n] - A)
    grad[n:, n:] += 2* (M[n:, n:] - B)
    
    theta_sigma = np.block([[np.zeros((n, n)), -sigma * np.eye(n)],
                             [-sigma * np.eye(n), np.zeros((n, n))]])

    grad += scale * theta_sigma
    
    return 2 * grad

def solve_dependent(A, B, k, niters, lambda_reg=None, delta_reg=1.0, 
                    fit_sigma=True, sigma=None, rng=None, step_size=1.0):
    if rng is None:
        rng = np.random.default_rng()
        
    if fit_sigma is False:
        if not isinstance(sigma, (int, float)):
            raise Exception('If fit_sigma is False, need to specify a int or float sigma')
    n = A.shape[0]
    
    Z_init, _ = solve_independent(A, k=k, rng=rng) 
    X_init, _ = solve_independent(B, k=k, rng=rng)
    Z_init = Z_init[0]
    X_init = X_init[0]
    ZZ = Z_init @ Z_init.T
    XX = X_init @ X_init.T
    XZ = X_init @ Z_init.T
    ZX = Z_init @ X_init.T

    # Initial Random Sigma
    if fit_sigma is True:
        #sigma = rng.uniform(-0.5, 0.5)
        sigma = rv_coefficient(Z_init, X_init)

    # Construct M
    M = np.block([[ZZ, XZ], [ZX, XX]])
    
    Ms = [M.copy()]
    sigma_list = [sigma]
    # Default Regularization parameter (lambda)
    if lambda_reg is None or lambda_reg == -1:
        # from multiness paper, example 1 constant variance sigma=1
        if delta_reg is None:
            delta_reg = 0.309
        
        lambda_reg = (2 + delta_reg) * np.sqrt(2 * n)
    
    # TODO check this
    # Lipschitz constant of gradient possibly scaled by step_size
    L = 2 / step_size
    
    for i in range(niters):
        
        if fit_sigma is True:
            S_diag = np.trace(M[:n, :n]) + np.trace(M[n:, n:])
            S_cross = np.trace(M[:n, n:])

            # Polynomial coefficients for cubic equation
            coeff_a = n * k
            coeff_b = -S_cross
            coeff_c = S_diag - (n * k)
            coeff_d = -S_cross
            
            roots = np.roots([coeff_a, coeff_b, coeff_c, coeff_d])
            
            # Filter for real roots in (-1, 1)
            real_roots = roots.real[np.abs(roots.imag) < 1e-5]
            valid_roots = real_roots[(real_roots > -0.99) & (real_roots < 0.99)]
            if len(valid_roots) == 0:
                sigma = 0.0 
                print('Warning: no valid roots setting sigma to zero')
            elif len(valid_roots) == 1:
                sigma = valid_roots[0]
            else:
                print('multiple root')
                # Pick root minimizing profile likelihood
                # i do not really like this, maybe there is a better way
                def profile_cost(s):
                    term1 = n * k * np.log(1 - s**2)
                    term2 = (S_diag - 2 * s * S_cross) / (1 - s**2)
                    return term1 + term2
                
                sigma = valid_roots[np.argmin([profile_cost(r) for r in valid_roots])]
        
        # Step 1: Gradient Descent on M ---
        grad = llk_gradient(M, A, B, sigma, n)

        Y = M - 1 / L * grad
        Y = (Y + Y.T) / 2

        # Step 2: Proximal Operator to enforce SPD
        evals, evecs = np.linalg.eigh(Y)
        
        evals_prox = np.maximum(evals, 0)

        M = evecs @ np.diag(evals_prox) @ evecs.T
        # Enforce symmetry explicitly (do we really need this?)
        M = (M + M.T) / 2
        
        Ms.append(M.copy())
        sigma_list.append(sigma)

    return Ms, sigma_list

def gradient_descent_solver(A, B, k, niters, lambda_reg=None, delta_reg=1.0, 
                    fit_sigma=True, sigma=None, rng=None, step_size=1.0):
    if rng is None:
        rng = np.random.default_rng()
        
    if fit_sigma is False:
        if not isinstance(sigma, (int, float)):
            raise Exception('If fit_sigma is False, need to specify a int or float sigma')
    n = A.shape[0]
    
    Z_init, _ = solve_independent(A, k=k, rng=rng) 
    X_init, _ = solve_independent(B, k=k, rng=rng)
    Z = Z_init[0]
    X = X_init[0]

    # Initial Random Sigma
    if fit_sigma is True:
        #sigma = rng.uniform(-0.5, 0.5)
        sigma = rv_coefficient(Z_init, X_init)
    
    Zs = [Z_init.copy()]
    Xs = [X_init.copy()]
    sigma_list = [sigma]
    
    # Default Regularization parameter (lambda)
    if lambda_reg is None or lambda_reg == -1:
        # from multiness paper, example 1 constant variance sigma=1
        if delta_reg is None:
            delta_reg = 0.309
        
        lambda_reg = (2 + delta_reg) * np.sqrt(2 * n)
    
    # TODO check this
    # Lipschitz constant of gradient possibly scaled by step_size
    L = 2 / step_size
    
    for i in range(niters):
        
        # compute gradient
        grad_x = 2 * A @ X - 2 * Z @ Z.T @ Z - 1/(1-sigma**2) * (Z - sigma * X)
        grad_z = 2 * B @ Z - 2 * X @ X.T @ X - 1/(1-sigma**2) * (X - sigma * Z)

        X = X - step_size * grad_x
        Z = Z - step_size * grad_z
        
        XX = X @ X.T
        ZZ = Z @ Z.T
        ZX = Z @ X.T
        XZ = X @ Z.T

        if fit_sigma is True:
            S_diag = np.trace(M[:n, :n]) + np.trace(M[n:, n:])
            S_cross = np.trace(M[:n, n:])

            # Polynomial coefficients for cubic equation
            coeff_a = n * k
            coeff_b = -S_cross
            coeff_c = S_diag - (n * k)
            coeff_d = -S_cross
            
            roots = np.roots([coeff_a, coeff_b, coeff_c, coeff_d])
            
            # Filter for real roots in (-1, 1)
            real_roots = roots.real[np.abs(roots.imag) < 1e-5]
            valid_roots = real_roots[(real_roots > -0.99) & (real_roots < 0.99)]
            if len(valid_roots) == 0:
                sigma = 0.0 
                print('Warning: no valid roots setting sigma to zero')
            elif len(valid_roots) == 1:
                sigma = valid_roots[0]
            else:
                print('multiple root')
                # Pick root minimizing profile likelihood
                # i do not really like this, maybe there is a better way
                def profile_cost(s):
                    term1 = n * k * np.log(1 - s**2)
                    term2 = (S_diag - 2 * s * S_cross) / (1 - s**2)
                    return term1 + term2
                
                sigma = valid_roots[np.argmin([profile_cost(r) for r in valid_roots])]
        
        
        Ms.append(M.copy())
        sigma_list.append(sigma)

    return Ms, sigma_list

def objective_function(n, k, M, A, B, sigma, lambda_reg):
    M_zz = M[:n, :n]
    M_xx = M[n:,n:]
    M_xz = M[:n, n:]
    M_zx = M[n:, :n]
    
    first = n*k*np.log(1-sigma**2)

    second = norm(M_zz-A, ord='fro') + norm(M_xx-B, ord='fro')
    
    factor = 1/(1-sigma**2)
    third = np.trace(M_xx) + np.trace(M_zz) - sigma*np.trace(M_xz) - sigma*np.trace(M_zx)

    regularization = lambda_reg * norm(M, ord='nuc')
    
    out = first + second + factor * third + regularization

    return out

def solver_grid(A, B, k, niters, lambda_reg=None, grid=0.1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
        
    if isinstance(grid, (int, float)):
        grid = np.arange(0, 1, grid)
    
    llk_values = {}
    n = A.shape[0]
    out_M = {}
    out_sigma = grid
    for sigma in grid:
        Ms, sigma_list = solve_dependent(A, B, k, niters, lambda_reg, 
                                         fit_sigma=False, sigma=sigma, rng=rng)
        estimated_M = Ms[-1]
        out_M[sigma] = Ms

        obj_value = objective_function(n, k, estimated_M, A, B, sigma, lambda_reg=4 * np.sqrt(2) / np.sqrt(3) * np.sqrt(n))
        llk_values[sigma] = obj_value

    return llk_values, out_M, out_sigma
 

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
