import numpy as np
import numba as nb
from scipy.optimize import minimize
from scipy.special import expit
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm

# algorithms from Ma & Ma (2020)

#################################
# projected gradient descent method

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_theta(Z, alpha=None, beta=None, X=None):
    """
    Θ = Z Z^T + α 1^T + 1 α^T + β X
    """
    if alpha is None:
        alpha = np.zeros(Z.shape[0])
    if beta is None:
        beta = 0.0
    if X is None:
        X = np.zeros((Z.shape[0], Z.shape[0]))

    return Z @ Z.T + np.add.outer(alpha, alpha) + beta * X

def project_Z(Z, M=None):
    """
    Project Z onto C_Z:
      1. Center columns  (JZ = Z)
      2. Clip each row to have L2-norm at most M^(1/3)
    """
    # Step 1: center
    Z = Z - Z.mean(axis=0)

    if M is not None:
        # Step 2: row-wise projection onto ball of radius M^(1/3)
        radius = M ** (1.0 / 3.0)
        row_norms = np.linalg.norm(Z, axis=1, keepdims=True)   # (n, 1)
        # scale down only rows that exceed the radius
        scale = np.where(row_norms > radius, radius / row_norms, 1.0)
        Z = Z * scale

    return Z

def project_alpha(alpha):
    return alpha  # no projection

def project_beta(beta):
    return beta   # no projection

def pgd_fit(A, k, 
            X=None, 
            eta_Z=1e-3, 
            eta_alpha=1e-3, 
            eta_beta=1e-3, 
            num_iters=100,
            Z0=None, 
            alpha0=None, 
            beta0=None, 
            rng=None, 
            init='svt',
            tau_init=1e-2,
            M_init=1e-2,
            return_history=False):
    """
    Projected Gradient Descent for latent space network model.

    Parameters
    ----------
    A         : (n, n) adjacency matrix
    X         : (n, n) covariate matrix
    k         : latent space dimension
    eta_Z     : step size for Z
    eta_alpha : step size for alpha
    eta_beta  : step size for beta
    T         : number of iterations
    Z0        : initial Z  (n x k), random if None
    alpha0    : initial alpha (n,), zeros if None
    beta0     : initial beta (scalar), zero if None

    Returns
    -------
    Z_hat, alpha_hat, beta_hat
    """
    if rng is None:
        rng = np.random.default_rng()
        
    n = A.shape[0]
    
    if X is None:
        X = np.zeros((n, n))

    # Initialise
    if init == 'svt':
        alpha, Z, beta = svt_init(A, k, tau=tau_init, M1=M_init, X=X)
    else:
        Z     = rng.standard_normal((n, k)) if Z0 is None else Z0.copy()
        alpha = np.zeros(n)                 if alpha0 is None else alpha0.copy()
        beta  = 0.0                         if beta0 is None else float(beta0)

    if return_history:
        history = [(Z.copy(), alpha.copy(), beta)]
    
    for t in range(num_iters):
        Theta   = compute_theta(Z, alpha, beta, X)
        residual = A - sigmoid(Theta)          # (A - σ(Θ))

        # --- gradient steps (ascending on log-likelihood) ---
        Z_tilde     = Z     + 2 * eta_Z    * (residual @ Z)
        alpha_tilde = alpha + 2 * eta_alpha * (residual @ np.ones(n))
        if X is not None:
            beta_tilde  = beta  +     eta_beta  * np.sum(residual * X)

        # --- projection ---
        Z     = project_Z(Z_tilde)
        alpha = project_alpha(alpha_tilde)
        if X is not None:
            beta  = project_beta(beta_tilde)

        if return_history:
            history.append((Z.copy(), alpha.copy(), beta))
    
    if return_history:
        return Z, alpha, beta, history

    return Z, alpha, beta

def pgd_fit_wrapper(A, k, 
            X=None, 
            eta_Z=1e-3, 
            eta_alpha=1e-3, 
            eta_beta=1e-3, 
            num_iters=100,
            Z0=None, 
            alpha0=None, 
            beta0=None, 
            rng=None, 
            init='svt',
            tau_init=1e-2,
            M_init=1e-2,
            return_history=False):
    """Wrapper for pgd_fit returning Z+alpha"""

    
    if return_history:
        Z, alpha, beta, history = pgd_fit(A, k, X=X, eta_Z=eta_Z, eta_alpha=eta_alpha, eta_beta=eta_beta,
                             num_iters=num_iters, Z0=Z0, alpha0=alpha0, beta0=beta0,
                             rng=rng, init=init, tau_init=tau_init,
                             M_init=M_init, return_history=return_history)
        
        return Z+alpha[:, None], beta, history
    
    Z, alpha, beta = pgd_fit(A, k, X=X, eta_Z=eta_Z, eta_alpha=eta_alpha, eta_beta=eta_beta,
                             num_iters=num_iters, Z0=Z0, alpha0=alpha0, beta0=beta0,
                             rng=rng, init=init, tau_init=tau_init,
                             M_init=M_init, return_history=return_history)
    
    return Z+alpha[:, None], beta

######################################
# initialisation strategies

def svt_init(A, k, tau, M1, X=None, fit_intercept=True):
    """
    Algorithm 3: SVT-based initialisation for Algorithm 1.

    Parameters
    ----------
    A   : (n, n) adjacency matrix
    X   : (n, n) covariate matrix
    k   : latent space dimension
    tau : singular value threshold
    M1  : controls the clipping interval [0.5*exp(-M1), 0.5]

    Returns
    -------
    alpha0 : (n,)   initial node effects
    Z0     : (n, k) initial latent positions
    beta0  : scalar initial covariate coefficient
    """
    n = A.shape[0]
    if X is None:
        X = np.zeros((n, n))
    
    ones = np.ones(n)
    J = np.eye(n) - np.outer(ones, ones) / n   # centering matrix

    # ------------------------------------------------------------------
    # Step 2: SVD thresholding → P̂ → Θ̂
    # ------------------------------------------------------------------
    U, s, Vt = np.linalg.svd(A)

    # Keep only components with singular value >= tau
    mask = s >= tau
    P_tilde = (U[:, mask] * s[mask]) @ Vt[mask, :]

    # Elementwise clip to [0.5 * exp(-M1), 0.5]
    lo = 0.5 * np.exp(-M1)
    # hi = 0.5
    # this prevents initialising all alpha >0
    hi = 1.0 - 0.5 * np.exp(-M1)
    P_hat = np.clip(P_tilde, lo, hi)

    # Symmetrize and apply logit to get Θ̂
    P_sym = (P_hat + P_hat.T) / 2.0
    Theta_hat = np.log(P_sym / (1.0 - P_sym))   # logit

    # ------------------------------------------------------------------
    # Step 3: Least-squares fit for alpha^0, beta^0
    #   min_{alpha, beta} || Θ̂ - (alpha 1^T + 1 alpha^T + beta X) ||_F^2
    #
    # Vectorise: each (i,j) gives one equation
    #   Θ̂_{ij} = alpha_i + alpha_j + beta * X_{ij}
    # ------------------------------------------------------------------
    # Build design matrix  (n^2) x (n+1)
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
    ii_flat = ii.ravel()
    jj_flat = jj.ravel()

    # Columns 0..n-1 for alpha, column n for beta
    row_idx = np.arange(n * n)
    data_i  = np.ones(n * n)
    data_j  = np.ones(n * n)

    # Sparse-style construction using dense array (fine for moderate n)
    B = np.zeros((n * n, n + 1))
    B[row_idx, ii_flat] += 1.0   # alpha_i
    B[row_idx, jj_flat] += 1.0   # alpha_j  (diagonal: alpha_i counted twice → handled by lstsq)
    B[:, n] = X.ravel()          # beta
    
    y = Theta_hat.ravel()
    
    if fit_intercept:
        constraint_row = np.zeros(n + 1)
        constraint_row[:n] = 1.0   # sum of alphas = 0
        B = np.vstack([B, constraint_row])
        y = np.append(y, 0.0)

    params, _, _, _ = np.linalg.lstsq(B, y, rcond=None)
    alpha0 = params[:n]
    beta0  = params[n]

    # ------------------------------------------------------------------
    # Step 4: Project residual onto PSD cone
    #   R = J (Θ̂ - alpha^0 1^T - 1 (alpha^0)^T - beta^0 X) J
    #   Ĝ = P_{S_+^n}(R)
    # ------------------------------------------------------------------
    Theta_resid = Theta_hat - np.add.outer(alpha0, alpha0) - beta0 * X
    R = J @ Theta_resid @ J

    # Eigendecomposition (R is symmetric by construction)
    eigvals, eigvecs = np.linalg.eigh(R)

    # Project onto PSD cone: threshold negative eigenvalues to 0
    eigvals_plus = np.maximum(eigvals, 0.0)
    # G_hat = (eigvecs * eigvals_plus) @ eigvecs.T

    # ------------------------------------------------------------------
    # Step 5: Top-k eigen-components → Z^0
    #   G_hat = U_k D_k U_k^T  =>  Z^0 = U_k D_k^{1/2}
    # ------------------------------------------------------------------
    # eigh returns eigenvalues in ascending order → take last k
    idx    = np.argsort(eigvals_plus)[::-1][:k]
    Uk     = eigvecs[:, idx]                   # (n, k)
    Dk     = np.diag(eigvals_plus[idx])        # (k, k)
    Z0     = Uk @ np.sqrt(Dk)                  # (n, k)

    return alpha0, Z0, beta0


