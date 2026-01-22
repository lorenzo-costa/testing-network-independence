import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm
from scipy.linalg import blas

# def rv_coefficient(A, B):
#     AtB = A.T @ B
#     num = np.sum(AtB * AtB)  # trace((A.T @ B) @ (B.T @ A))
    
#     AtA = A.T @ A
#     BtB = B.T @ B
#     den = np.sqrt(np.sum(AtA * AtA) * np.sum(BtB * BtB))
    
#     return num / den if den != 0 else 0

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

class BaseMetric:
    def __init__(self):
        pass

    def __call__(self, estimated, truth):
        raise NotImplementedError("Subclasses should implement this!")

    def get_name(self):
        raise NotImplementedError("Subclasses should implement this!")

class RVCoefficient(BaseMetric):
    def __call__(self, estimated, truth):
        return rv_coefficient(estimated, truth)

    def get_name(self):
        return "RV Coefficient"

class MSE(BaseMetric):
    def __call__(self, estimated, truth):
        return ((truth-estimated)**2).mean()

    def get_name(self):
        return "Mean Squared Error"

class RelativeFrobeniusNorm(BaseMetric):
    """Relative Frobenius Norm, computed as ||Xhat - X||_F / ||X||_F"""
    def __call__(self, estimated, truth):
        num = norm(estimated-truth, 'fro')
        den = norm(truth, 'fro')
        return num / den if den != 0 else 0

    def get_name(self):
        return "Relative Frobenius Norm"

class Rejection(BaseMetric):
    def __call__(self, truth, estimated):
        if estimated == 1:
            return True
        return False

    def get_name(self):
        return "Rejection"
    
class FalseRejection(BaseMetric):
    """False Rejection (i.e. Type I error)"""
    def __call__(self, estimated, truth):
        # Type I Error: Null is True (1) AND we Reject (1)
        if truth == 1 and estimated == 1:
            return True
        # Correct Decision: Null is True (1) but we do NOT Reject (0)
        elif truth == 1 and estimated == 0:
            return False
        # If truth is 0 (Null is false), Type I error is impossible
        return False

    def get_name(self):
        return "FalseRejection"

class TrueAcceptance(BaseMetric):
    """True Acceptance (i.e. Type II error)"""
    def __call__(self, estimated, truth):
        if truth == 0 and estimated == 1:
            return True
        return False

    def get_name(self):
        return "TrueAcceptance"

class TrueRejection(BaseMetric):
    """True Rejection"""
    def __call__(self, estimated, truth):
        if truth == 1 and estimated == 1:
            return True
        return False

    def get_name(self):
        return "TrueRejection"

class TrueNegative(BaseMetric):
    """True Negative"""
    def __call__(self, estimated, truth):
        if truth == 0 and estimated == 0:
            return True
        return False

    def get_name(self):
        return "TrueNegative"