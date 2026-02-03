# Function to fit the MLE in the dependent network model case
# note None of this work well, just leaving them here to not lose them

import numpy as np
from methods import solve_independent
from metrics import rv_coefficient

def solve_dependent(A, B, k, niters, 
                    lambda_reg=None, delta_reg=1.0, 
                    fit_sigma=True, sigma=None, 
                    rng=None, step_size=1.0):
    
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
        
        threshold = lambda_reg / L 

        # Soft-thresholding operator: relu(x - threshold)
        evals_prox = np.maximum(evals - threshold, 0)

        M = evecs @ np.diag(evals_prox) @ evecs.T
        # Enforce symmetry explicitly (do we really need this?)
        M = (M + M.T) / 2
        
        Ms.append(M.copy())
        sigma_list.append(sigma)

    return Ms, sigma_list

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