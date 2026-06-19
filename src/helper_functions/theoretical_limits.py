import numpy as np


def theoretical_rv_limit(
        lambdas,
        gammas,
        kappa= 0.0,
        A= None,
        U =None,
        V = None,
        n_mc=200_000,
        method="general",
        rng=None,
        tol=1e-12):
    """Sample from theoretical limit for RV distribution:
    
    \frac{\sum_{r=1}^{pq}\lambda_{r}\chi^{2}_{1, ij}(\eta_{r})}
    
    {\sqrt{ tr(\Sigma_{XX}^{2})tr(\Sigma_{ZZ}^{2}) }}
    
    

    Parameters
    ----------
    lambdas : array-like
        Eigenvalues of the covariance of the latent positions of the first network.
        _description_
    gammas : array-like
        Eigenvalues of the covariance of the latent positions of the second network.
    kappa : float, optional
        Kurtosis for elliptical distribution limit, by default 0.0
    A : array-like, optional
        Mean matrix for non-centrality, by default None
    U : array-like, optional
        Left singular vectors for non-centrality, by default None
    V : array-like, optional
        Right singular vectors for non-centrality, by default None
    n_mc : int, optional
        Number of Monte Carlo samples to draw, by default 200_000
    method : str, optional
        Method to compute limit, one of {"general", "independent_dirichlet", "central"}, by default "general"
    rng : np.random.Generator, optional
        Random number generator to use for sampling, by default None
    tol : float, optional
        Tolerance for numerical stability, by default 1e-12

    Returns
    -------
    array-like
        Samples from the asymptotic distribution of the RV under the null.

    """
    if rng is None:
        rng = np.random.default_rng()

    lambdas = np.asarray(lambdas, dtype=float)
    gammas = np.asarray(gammas, dtype=float)

    p, q = len(lambdas), len(gammas)

    denom = np.sqrt(np.sum(lambdas**2) * np.sum(gammas**2))

    if denom <= tol:
        raise ValueError("Degenerate denominator: eigenvalues have zero squared trace.")

    if method == "general":
        C = (1.0 + kappa) / denom
        eta_scale = 1.0 + kappa

        if eta_scale <= 0:
            raise ValueError("For method='general', require 1 + kappa > 0.")

    elif method == "independent_dirichlet":
        C = 1.0 / denom
        eta_scale = 1.0

    elif method == "central":
        C = 1.0 / denom
        eta_scale = 1.0
        A = None

    else:
        raise ValueError(
            "method must be one of {'general', 'independent_dirichlet', 'central'}."
        )

    etas = np.zeros((p, q), dtype=float)

    if A is not None:
        if U is None or V is None:
            raise ValueError("If A is provided, U and V must also be provided.")

        A = np.asarray(A, dtype=float)
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)

        if A.shape != (p, q):
            raise ValueError(f"A must have shape {(p, q)}, got {A.shape}.")

        if U.shape != (p, p):
            raise ValueError(f"U must have shape {(p, p)}, got {U.shape}.")

        if V.shape != (q, q):
            raise ValueError(f"V must have shape {(q, q)}, got {V.shape}.")

        for i in range(p):
            for j in range(q):
                weight = lambdas[i] * gammas[j]

                if weight > tol:
                    proj = float(U[:, i].T @ A @ V[:, j])
                    etas[i, j] = proj**2 / (eta_scale * weight)

                else:
                    # If weight is zero, this component is non-random.
                    # If proj is nonzero in a zero-variance direction, it contributes
                    # a deterministic shift proj^2 to ||G||^2.
                    etas[i, j] = 0.0

    draws = np.zeros(n_mc, dtype=float)
    deterministic_shift = 0.0

    for i in range(p):
        for j in range(q):
            weight = lambdas[i] * gammas[j]

            if weight > tol:
                draws += weight * rng.noncentral_chisquare(
                    df=1,
                    nonc=etas[i, j],
                    size=n_mc,
                )

            elif A is not None:
                # Zero-variance direction. The Gaussian part contributes nothing,
                # but a mean component can contribute deterministically.
                proj = float(U[:, i].T @ A @ V[:, j])
                deterministic_shift += proj**2

    draws += deterministic_shift

    return C * draws



def dirichlet_cov(alpha: np.ndarray) -> np.ndarray:
    """
    Covariance matrix of Dirichlet(alpha).
    """
    alpha = np.asarray(alpha, dtype=float)

    if alpha.ndim != 1:
        raise ValueError("alpha must be a 1D array.")

    if np.any(alpha <= 0):
        raise ValueError("Dirichlet parameters must be positive.")

    alpha0 = np.sum(alpha)

    Sigma = (
        alpha0 * np.diag(alpha) - np.outer(alpha, alpha)
    ) / (alpha0**2 * (alpha0 + 1.0))

    return Sigma


def dirichlet_rv_limit_params(
        alpha_x: np.ndarray,
        alpha_z: np.ndarray,
        drop_zero: bool = True,
        tol: float = 1e-12,
):
    """
    Given Dirichlet parameters for independent blocks

        X ~ Dirichlet(alpha_x),
        Z ~ Dirichlet(alpha_z),

    return the parameters needed for theoretical_rv_limit.

    Returns
    -------
    params : dict
        Contains:
            Sigma_X : covariance matrix of X
            Sigma_Z : covariance matrix of Z
            lambdas : nonzero eigenvalues of Sigma_X
            gammas  : nonzero eigenvalues of Sigma_Z
            U       : corresponding eigenvectors of Sigma_X
            V       : corresponding eigenvectors of Sigma_Z
            tr_X2   : tr(Sigma_X^2)
            tr_Z2   : tr(Sigma_Z^2)
    """
    Sigma_X = dirichlet_cov(alpha_x)
    Sigma_Z = dirichlet_cov(alpha_z)

    # eigh is for symmetric matrices.
    lambdas_all, U_all = np.linalg.eigh(Sigma_X)
    gammas_all, V_all = np.linalg.eigh(Sigma_Z)

    # Sort descending.
    idx_x = np.argsort(lambdas_all)[::-1]
    idx_z = np.argsort(gammas_all)[::-1]

    lambdas_all = lambdas_all[idx_x]
    gammas_all = gammas_all[idx_z]

    U_all = U_all[:, idx_x]
    V_all = V_all[:, idx_z]

    # Clean tiny negative numerical eigenvalues.
    lambdas_all = np.where(np.abs(lambdas_all) < tol, 0.0, lambdas_all)
    gammas_all = np.where(np.abs(gammas_all) < tol, 0.0, gammas_all)

    if drop_zero:
        keep_x = lambdas_all > tol
        keep_z = gammas_all > tol

        lambdas = lambdas_all[keep_x]
        gammas = gammas_all[keep_z]

        U = U_all[:, keep_x]
        V = V_all[:, keep_z]

    else:
        lambdas = lambdas_all
        gammas = gammas_all
        U = U_all
        V = V_all

    params = {
        "Sigma_X": Sigma_X,
        "Sigma_Z": Sigma_Z,
        "lambdas": lambdas,
        "gammas": gammas,
        "U": U,
        "V": V,
        "tr_X2": np.sum(lambdas_all**2),
        "tr_Z2": np.sum(gammas_all**2),
    }

    return params