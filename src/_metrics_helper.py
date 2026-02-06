import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm
from scipy.linalg import blas

def rv_coefficient(A, B):
    AtB = A.T @ B
    # Flattening to 1D and using dot(x, x) is often faster than sum(x*x)
    temp_num = AtB.ravel()
    num = temp_num.dot(temp_num)
    
    AtA = A.T @ A
    BtB = B.T @ B
    
    a_flat = AtA.ravel()
    b_flat = BtB.ravel()
    den = np.sqrt(a_flat.dot(a_flat) * b_flat.dot(b_flat))
    
    return num / den if den != 0 else 0

def rv_coefficient_adjusted(A, B):
    """Adjusted version of RV coef from Mordant Gilles; Segers Johan (2022).
    
    Given Sigma_XX pxp matrix, Sigma_ZZ qxq matrix (here Sigma_XX = AA^T) define:
    - Lambda_x, Lambda_y the diagonal matrices of eigenvalues of Sigma_XX, Sigma_ZZ
    - Pi = [I_q, O_px(q-p)]
    The adjusted RV coefficient is defined as:
        RV(Sigma_XX, Sigma_ZZ) = Tr(Sigma_XX Sigma_ZZ)/Tr(Lambda_X Pi Lambda_Z)
        
    """
    AtB = A.T @ B
    # Flattening to 1D and using dot(x, x) is often faster than sum(x*x)
    temp_num = AtB.ravel()
    num = temp_num.dot(temp_num)

    # note evals of A^TA are square of singular values 
    # so use svd to avoid matrix computation
    sx = np.linalg.svd(A, compute_uv=False)
    sy = np.linalg.svd(B, compute_uv=False)
    m = min(len(sx), len(sy))
    den = np.sum((sx[:m]**2) * (sy[:m]**2))

    return num / den if den != 0 else 0
    

def mse(X, Xhat):
    return ((X-Xhat)**2).mean()

def relative_frobenius_norm(X, Xhat, inplace=True):
    if inplace is False:
        den = norm(X, 'fro')
        if den == 0:
            return 0
        num = norm(Xhat - X, 'fro')
        return num / den
        
    else:
        X_flat = X.ravel()
        Xhat_flat = Xhat.ravel()
        
        # 1. Compute norm of X directly via BLAS Level 1 (dnrm2)
        # This is the fastest way to get the Frobenius norm
        den = blas.dnrm2(X_flat)
        
        if den == 0:
            return 0
        
        # 2. Compute the norm of the difference
        # To avoid 'Xhat - X' creating a huge new matrix, we use 'axpy'
        # This computes: y = a*x + y -> Xhat = -1*X + Xhat
        # WARNING: This modifies Xhat in place for speed.
        # If you can't modify Xhat, use Xhat.copy() first (but that's slower).
        
        # Copy Xhat to avoid destroying original data
        diff = np.copy(Xhat_flat)
        blas.daxpy(X_flat, diff, a=-1.0)
        num = blas.dnrm2(diff)
        
        return num / den