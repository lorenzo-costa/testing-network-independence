import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm


def normalised_laplacian(A):
    A = (A + A.T) / 2

    D = np.array(A.sum(axis=1)).flatten()

    d_inv_sqrt = np.zeros_like(D)

    # where logic with result written to out matrix,
    # pretty cool numpy feature
    np.power(D, -0.5, where=D != 0, out=d_inv_sqrt)

    D_inv_sqrt_mat = diags(d_inv_sqrt)
    L_norm = D_inv_sqrt_mat @ A @ D_inv_sqrt_mat

    return L_norm


def diffusion_maps(L, t=1, k=None):
    if k is None:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # TODO: implement some sort of k selection technique
    else:
        eigenvalues, eigenvectors = eigsh(L, k=k, which="LM")
    idx = eigenvalues.argsort()[::-1]
    lambdas = eigenvalues[idx]
    phis = eigenvectors[:, idx]

    U = []
    ts = np.arange(t) + 1
    for i in ts:
        U_t = phis * (lambdas**i)
        U.append(U_t)

    return np.array(U)


def solve_independent(A, k=2):
    evals, evectors = eigsh(A, k=k, which="LM")
    evals = np.maximum(evals - 0.5, 0)
    xhat = evectors @ np.diag(np.sqrt(evals))
    return xhat, evals


def rv_coefficient(A, B):
    num = np.trace((A.T @ B) @ (B.T @ A))
    den = norm(A.T @ A, "fro") * norm(B.T @ B, "fro")
    return num / den if den != 0 else 0


def rv_coef_test(A, B, n_perm=100, rng=None, k=2):
    if rng is None:
        rng = np.random.default_rng()

    Zhat, evals_A = solve_independent(A, k=k)
    Xhat, evals_B = solve_independent(B, k=k)

    rv_est = rv_coefficient(Zhat, Xhat)

    rv_distr = []

    for i in range(n_perm):
        perm = rng.permutation(n)
        X_perm = X[perm, :]
        rv_perm = rv_coefficient(X_perm, Z)
        rv_distr.append(rv_perm)

    p_val = np.mean(rv_distr >= rv_est)

    return p_val
