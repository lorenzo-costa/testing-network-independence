import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import norm
from scipy.linalg import blas
from ._metrics_helper import rv_coefficient, rv_coefficient_adjusted


class BaseMetric:
    def __init__(self):
        pass

    def __call__(self, estimated, truth):
        raise NotImplementedError("Subclasses should implement this!")

    def get_name(self):
        raise NotImplementedError("Subclasses should implement this!")


class RVCoefficient(BaseMetric):
    def __call__(self, results):
        estimated = results["estimated_latent"]
        truth = results["true_latent"]
        return rv_coefficient(estimated, truth)

    def get_name(self):
        return "RV Coefficient"


class AdjustedRVCoefficient(BaseMetric):
    def __call__(self, results):
        estimated = results["estimated_latent"]
        truth = results["true_latent"]
        return rv_coefficient_adjusted(estimated, truth)

    def get_name(self):
        return "Adjusted RV Coefficient"


class MSE(BaseMetric):
    def __call__(self, results):
        estimated = results["estimated_latent"]
        truth = results["true_latent"]
        return ((truth - estimated) ** 2).mean()

    def get_name(self):
        return "Mean Squared Error"


class RelativeFrobeniusNorm(BaseMetric):
    """Relative Frobenius Norm, computed as ||Xhat - X||_F / ||X||_F

    Parameters
    ----------
    gram_matrix : bool
        Whether to compute the Gram matrix of the latent positions.
    results : dict
        The results dictionary containing 'estimated_latent' and 'true_latent' keys.
        If 'estimated_latent' is a list, relative frobenus norm will be applied to all
        elements of the list

    Output
    ------
    A float representing the relative Frobenius norm if 'estimated_latent' is a single array
    A list of floats representing the relative Frobenius norm for each element if 'estimated_latent' is a list
    """

    def __init__(self, gram_matrix=False):
        super().__init__()
        # when feeding the estimate latent positions we compute the gram matrix to
        # get rid of orthogonal invariance
        self.gram_matrix = gram_matrix

    def __call__(self, results):
        estimated = results["estimated_latent"]
        truth = results["true_latent"]

        # handles the case where more than one network's latent pos are returned
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

                num = norm(est - true, "fro")
                den = norm(true, "fro")
                out.append(num / den if den != 0 else 0)
            # returns a list
            return out

        # single output computation
        if self.gram_matrix:
            # Compute the Gram matrix for both estimated and truth
            estimated = estimated @ estimated.T
            truth = truth @ truth.T

        num = norm(estimated - truth, "fro")
        den = norm(truth, "fro")
        return num / den if den != 0 else 0

    def get_name(self):
        return "RelativeFrobeniusNorm"


class Rejection(BaseMetric):
    """Rejection of Null Hypothesis, one if rejected.

    Takes as input a results dictionary containing 'reject_null' key.
    """

    def __call__(self, results):
        reject_null = results["reject_null"]
        if reject_null == True:
            return True
        return False

    def get_name(self):
        return "Rejection"


class FalseRejection(BaseMetric):
    """False Rejection (Type I Error / False Positive)

    Takes as input a results dictionary containing 'reject_null' and 'true_null' keys.
    """

    def __call__(self, results):
        reject_null = results["reject_null"]
        null = results["null"]
        # if null is True, but we reject it.
        if (null is True) and (reject_null is True):
            return True
        return False

    def get_name(self):
        return "FalseRejection"


class FalseAcceptance(BaseMetric):
    """False Acceptance (Type II Error / False Negative)

    Takes as input a results dictionary containing 'reject_null' and 'true_null' keys.
    """

    def __call__(self, results):
        reject_null = results["reject_null"]
        null = results["null"]
        # Null is False (H0), but we do not reject it (i.e accept it)
        if (null is False) and (reject_null is False):
            return True
        return False

    def get_name(self):
        return "FalseAcceptance"


class TrueRejection(BaseMetric):
    """True Rejection (reject H0 when it is False)

    Takes as input a results dictionary with keywords 'reject_null' and 'null'.
    """

    def __call__(self, results):
        reject_null = results["reject_null"]
        null = results["null"]
        # Null is False (H1) and we reject it
        if (null is False) and (reject_null is True):
            return True
        return False

    def get_name(self):
        return "TrueRejection"


class TrueAcceptance(BaseMetric):
    """True Acceptance (accept H0 when it is True)

    Takes as input a results dictionary with keywords 'reject_null' and 'null'.
    """

    def __call__(self, results):
        reject_null = results["reject_null"]
        null = results["null"]
        # Null is True (H0) and we accept it
        if (null is True) and (reject_null is False):
            return True
        return False

    def get_name(self):
        return "TrueAcceptance"


class ComputeAll(BaseMetric):
    """Single class to compute testing and latent position errors

    Parameters
    ----------
    gram_matrix : bool
        Whether to compute the Gram matrix for latent position metrics.
    results : dict
        Takes as input a dictionary containing keywords 'reject_null', 'null', 'true_latent' and 'estimated_latent'
    """

    def __init__(self, gram_matrix=True):
        super().__init__()
        self.gram_matrix = gram_matrix

    def __call__(self, results):
        out = {}
        reject_null = results.get("reject_null", None)
        estimated_latent = results.get("estimated_latent", None)

        if reject_null is not None:
            # compute test metrics
            test_metrics = {
                "Rejection": Rejection()(results),
                "FalseRejection": FalseRejection()(results),
                "FalseAcceptance": FalseAcceptance()(results),
                "TrueRejection": TrueRejection()(results),
                "TrueAcceptance": TrueAcceptance()(results),
            }
            out.update(test_metrics)

        if estimated_latent is not None:
            latent_metrics = {
                "RelativeFrobeniusNorm_x": RelativeFrobeniusNorm(
                    gram_matrix=self.gram_matrix
                )(results),
                "RelativeFrobeniusNorm_z": RelativeFrobeniusNorm(
                    gram_matrix=self.gram_matrix
                )(results),
            }
            out.update(latent_metrics)

        return out

    def get_name(self):
        return "ComputeAll"
