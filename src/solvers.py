import numpy as np
import numba as nb
from scipy.optimize import minimize
from scipy.special import expit
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm


def logistic_grad(params, X, y, mu=None):
    """Utility function for scipy optimizer returning loss and gradient for logistic regression"""
    if mu is None:
        coef = params[:-1]
        mu = params[-1]
    else:
        coef = params

    logits = X @ coef + mu
    loss = np.sum(np.logaddexp(0, logits) - y * logits)
    p = expit(logits)
    error = p - y
    grad_w = X.T @ error
    return loss, grad_w


@nb.njit(cache=True)
def logistic_grad_fixed_mu(coef, X, y, mu):
    """Fast version of logistic gradient with fixed intercept compiled using numba"""
    n, d = X.shape
    grad_w = np.zeros(d)
    loss = 0.0

    for i in range(n):
        # Calculate logit for this row
        logit = np.dot(X[i], coef) + mu

        # Numerically stable log-loss: log(1 + exp(z))
        if logit > 0:
            loss += logit + np.log(1.0 + np.exp(-logit)) - y[i] * logit
        else:
            loss += np.log(1.0 + np.exp(logit)) - y[i] * logit

        # Probability and Error
        p = 1.0 / (1.0 + np.exp(-logit))
        err = p - y[i]

        # Accumulate gradient
        for j in range(d):
            grad_w[j] += X[i, j] * err

    return loss, grad_w


def solve_logistic_scipy(X, y, mu=None):
    """Solve logistic regression with (possibly) fixed intercept and positive coefficients

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    mu : float, optional
        Intercept term, by default None. If None estimate it

    Returns
    -------
    np.ndarray
        Coefficients of the logistic regression
    float
        Intercept term
    """

    n_samples, n_features = X.shape

    if mu is None:
        initial_params = np.zeros(n_features + 1)
        bounds = [(0, None)] * n_features + [(None, None)]
        # jac=True tells scipy the objective function returns (loss, gradient)
        res = minimize(
            logistic_grad,
            initial_params,
            args=(X, y),
            method="L-BFGS-B",
            bounds=bounds,
            jac=True,
        )
    else:
        initial_params = np.zeros(n_features)
        bounds = [(0, None)] * n_features
        # jac=True tells scipy the objective function returns (loss, gradient)
        res = minimize(
            logistic_grad_fixed_mu,
            initial_params,
            args=(X, y, mu),
            method="L-BFGS-B",
            bounds=bounds,
            jac=True,
        )

    return res.x[:n_features], res.x[-1] if mu is None else mu


def MLE_logistic(A, k=2, rng=None, **kwargs):
    """Maximum Likelihood Estimation for Logistic link adjacency matrix

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix
    k : int, optional
        Number of latent dimensions, by default 2
    rng : np.random.Generator, optional
        Random number generator, by default None

    Returns
    -------
    np.ndarray, np.ndarray
        Estimated latent positions, estimated eigenvalues
    """
    if rng is None:
        rng = np.random.default_rng()

    # useful quantities
    n = A.shape[0]

    # in the paper it seems to use the frob norm NOT squared, i get better results
    # squaring it
    a_norm_scaled = 1 / (n * (n - 1)) * norm(A, "fro") ** 2
    eps = 1e-10
    a_norm_scaled = np.clip(a_norm_scaled, eps, 1.0 - eps)

    # mean centered matrix
    A_centered = A - a_norm_scaled

    # from paper, mle of \mu
    mu_hat = -np.log(a_norm_scaled / (1 - a_norm_scaled))

    # use this to fix randomness in eigsh
    v0 = rng.standard_normal(size=A_centered.shape[0])

    try:
        evals, evectors = eigsh(A_centered, k=k, which="LM", v0=v0)
    except:
        A_dense = A_centered.toarray() if hasattr(A_centered, "toarray") else A_centered
        evals, evectors = np.linalg.eigh(A_dense)

    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evectors = evectors[:, idx]

    # build the matrix of features
    X_big = np.zeros((n * (n - 1) // 2, k))

    for i in range(k):
        t = np.outer(evectors[:, i], evectors[:, i])
        X_big[:, i] = t[np.triu_indices(n, k=1)]

    # define as target the upper diagonal part of A (equal to lower since
    # A symmetric)
    target = A[np.triu_indices(n, k=1)]

    # solve logistic regression with fixed mu and positive constrained coefs
    coefs, mu = solve_logistic_scipy(X_big, target, mu=mu_hat)

    xhat = evectors * np.sqrt(coefs)

    return xhat, evals

def ASE(A, k=2, rng=None, **kwargs):
    """Adjacency Spectral Embedding

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix
    k : int, optional
        Number of dimensions for the latent space, by default 2
    rng : np.random.Generator, optional
        Random number generator, by default None

    Returns
    -------
    np.ndarray, np.ndarray
        Estimated latent positions, estimated eigenvalues
    """
    if rng is None:
        rng = np.random.default_rng()

    # v0 for fixing randomness
    v0 = rng.standard_normal(size=A.shape[0])

    # don't remember why LA instead of LM
    try:
        evals, evectors = eigsh(A, k=k, which="LM", v0=v0)
    except:
        A_dense = A.toarray() if hasattr(A, "toarray") else A
        evals, evectors = np.linalg.eigh(A_dense)

    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evectors = evectors[:, idx]

    evals = np.maximum(evals, 0)

    xhat = evectors * np.sqrt(evals)

    return xhat, evals


def MLE_gaussian(A, k=2, rng=None, shrink=0.5, **kwargs):
    """Maximum Likelihood Estimation for Gaussian adjacency matrix

    Parameters
    ----------
    A : np.ndarray
        Adjacency matrix
    k : int, optional
        Number of latent dimensions, by default 2
    rng : np.random.Generator, optional
        Random number generator, by default None
    shrink : float, optional
        Shrinkage parameter, by default 0.5 (coming from MLE computation)

    Returns
    -------
    np.ndarray, np.ndarray
        Estimated latent positions, estimated eigenvalues
    """
    if rng is None:
        rng = np.random.default_rng()

    v0 = rng.standard_normal(size=A.shape[0])
    try:
        evals, evectors = eigsh(A, k=k, which="LM", v0=v0)
    except:
        A_dense = A.toarray() if hasattr(A, "toarray") else A
        evals, evectors = np.linalg.eigh(A_dense)

    # manual sorting just to be sure
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evectors = evectors[:, idx]

    evals = np.clip(np.maximum(evals-0.5, 0), 0, 1e10) # clip for numerical stability
    
    xhat = evectors * np.sqrt(evals)

    return xhat, evals
