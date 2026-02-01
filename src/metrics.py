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

class AdjustedRVCoefficient(BaseMetric):
    def __call__(self, estimated, truth):
        return rv_coefficient_adjusted(estimated, truth)

    def get_name(self):
        return "Adjusted RV Coefficient"

class MSE(BaseMetric):
    def __call__(self, estimated, truth):
        return ((truth-estimated)**2).mean()

    def get_name(self):
        return "Mean Squared Error"

class RelativeFrobeniusNorm(BaseMetric):
    """Relative Frobenius Norm, computed as ||Xhat - X||_F / ||X||_F"""
    def __init__(self, gram_matrix=False):
        super().__init__()
        # when feeding the estimate latent positions we compute the gram matrix to 
        # get rid of orthogonal invariance
        self.gram_matrix = gram_matrix

    def __call__(self, estimated=None, truth=None, fit_out=None):
        # another very messy implementation, change this TODO
        if fit_out is not None:
            estimated = fit_out[:2]
            truth = fit_out[2:]
            
        if isinstance(estimated, tuple):
            out = []
            for i in range(len(estimated)):
                if self.gram_matrix:
                    # Compute the Gram matrix for both estimated and truth
                    est = estimated[i] @ estimated[i].T
                    true = truth[i] @ truth[i].T
                else:
                    est = estimated[i]
                    true = truth[i]
                    
                num = norm(est-true, 'fro')
                den = norm(true, 'fro')
                out.append(num / den if den != 0 else 0)
            return out
        
        if self.gram_matrix:
            # Compute the Gram matrix for both estimated and truth
            estimated = estimated @ estimated.T
            truth = truth @ truth.T
            
        num = norm(estimated-truth, 'fro')
        den = norm(truth, 'fro')
        return num / den if den != 0 else 0
        

    def get_name(self):
        return "RelativeFrobeniusNorm"

class Rejection(BaseMetric):
    def __call__(self, truth, estimated, fit_out=None):
        if estimated == 1:
            return True
        return False

    def get_name(self):
        return "Rejection"

class FalseRejection(BaseMetric):
    """False Rejection (Type I Error / False Positive)"""
    def __call__(self, estimated, truth, fit_out=None):
        # Truth is False (H0), but we Estimated True (Reject H0)
        if truth == 0 and estimated == 1:
            return True
        return False

    def get_name(self):
        return "FalseRejection"

class FalseAcceptance(BaseMetric):
    """False Acceptance (Type II Error / False Negative)"""
    def __call__(self, estimated, truth, fit_out=None):
        # Truth is True (H1), but we Estimated False (Accept H0)
        if truth == 1 and estimated == 0:
            return True
        return False

    def get_name(self):
        return "FalseAcceptance"

class TrueRejection(BaseMetric):
    """True Rejection"""
    def __call__(self, estimated, truth, fit_out=None):
        if truth == 1 and estimated == 1:
            return True
        return False

    def get_name(self):
        return "TrueRejection"

class TrueAcceptance(BaseMetric):
    """True Acceptance"""
    def __call__(self, estimated, truth, fit_out=None):
        if truth == 0 and estimated == 0:
            return True
        return False

    def get_name(self):
        return "TrueAcceptance"

class ComputeAll(BaseMetric):
    """Single class to compute testing and latent position errors"""
    def __call__(self, estimated=None, truth=None, fit_out=None):
        out = {}
        if estimated is not None and truth is not None:
            # compute test metrics
            test_metrics = {
                'Rejection': Rejection()(estimated, truth),
                'FalseRejection': FalseRejection()(estimated, truth),
                'FalseAcceptance': FalseAcceptance()(estimated, truth),
                'TrueRejection': TrueRejection()(estimated, truth),
                'TrueAcceptance': TrueAcceptance()(estimated, truth)
            }
            out.update(test_metrics)

        if fit_out is not None:
            Xhat, Zhat, X, Z = fit_out
            # compute latent position metrics
            latent_metrics = {
                'MSE_x': MSE()(Xhat, X),
                'MSE_z': MSE()(Zhat, Z),
                'RelativeFrobeniusNorm_x': RelativeFrobeniusNorm(gram_matrix=True)(Xhat, X),
                'RelativeFrobeniusNorm_z': RelativeFrobeniusNorm(gram_matrix=True)(Zhat, Z),
            }
            out.update(latent_metrics)

        return out

    def get_name(self):
        return "ComputeAll"
