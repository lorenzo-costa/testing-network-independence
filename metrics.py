import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm

def rv_coefficient(A, B):
    num = np.trace((A.T @ B) @ (B.T @ A))
    den = norm(A.T @ A, 'fro') * norm(B.T @ B, 'fro')
    return num / den if den != 0 else 0

def mse(X, Xhat):
    return ((X-Xhat)**2).mean()

def relative_frobenius_norm(X, Xhat):
    return norm(Xhat-X, 'fro') / norm(X, 'fro') if norm(X, 'fro') != 0 else 0