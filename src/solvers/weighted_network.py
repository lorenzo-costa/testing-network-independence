import numpy as np
import numba as nb
from scipy.optimize import minimize
from scipy.special import expit
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm



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

    try:
        # "LM" (Largest Magnitude) is correct here to mimic SVD
        evals, evectors = eigsh(A, k=k, which="LM", v0=v0)
    except:
        A_dense = A.toarray() if hasattr(A, "toarray") else A
        evals, evectors = np.linalg.eigh(A_dense)

    # FIX: Sort by absolute magnitude to match "LM" behavior, 
    # and explicitly slice to keep only the top k dimensions.
    idx = np.argsort(np.abs(evals))[::-1][:k]
    evals = evals[idx]
    evectors = evectors[:, idx]

    # Convert to absolute values (mimicking singular values)
    evals = np.abs(evals)

    xhat = evectors @ np.diag(np.sqrt(evals))

    return xhat, evals


def MLE_gaussian(A, k=2, rng=None, shrink=0.5, **kwargs):
    """Maximum Likelihood Estimation for Gaussian adjacency matrix.

    Note: this assumes latent positions have mean zero.

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

    evals = np.clip(np.maximum(evals - 0.5, 0), 0, 1e10)  # clip for numerical stability

    xhat = evectors * np.sqrt(evals)

    return xhat, evals