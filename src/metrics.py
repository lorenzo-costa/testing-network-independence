from scipy.linalg import norm
from .test_functions.rv_cca_coefficients import rv_coefficient, rv_coefficient_adjusted
import numpy as np


def _latent_pairs(results):
    """Return corresponding estimated/true latent blocks.

    Linear-model methods return ``[Y, X[0], ..., X[p - 1]]``.  Single-array
    inputs remain supported for metrics used outside that pipeline.
    """

    estimated = results["estimated_latent"]
    truth = results["true_latent"]
    estimated_is_blocked = isinstance(estimated, (list, tuple))
    truth_is_blocked = isinstance(truth, (list, tuple))

    if estimated_is_blocked != truth_is_blocked:
        raise ValueError(
            "estimated_latent and true_latent must both be arrays or both be "
            "lists/tuples of latent blocks"
        )

    if estimated_is_blocked:
        if len(estimated) != len(truth):
            raise ValueError(
                "estimated_latent and true_latent must contain the same number "
                "of blocks"
            )
        if len(estimated) == 0:
            raise ValueError("latent block lists must not be empty")
        raw_pairs = zip(estimated, truth)
    else:
        raw_pairs = [(estimated, truth)]

    pairs = []
    for index, (estimated_block, true_block) in enumerate(raw_pairs):
        estimated_block = np.asarray(estimated_block, dtype=float)
        true_block = np.asarray(true_block, dtype=float)
        if estimated_block.ndim != 2 or true_block.ndim != 2:
            label = f" block {index}" if estimated_is_blocked else ""
            raise ValueError(f"latent{label} values must be two-dimensional")
        if estimated_block.shape[0] != true_block.shape[0]:
            label = f"[{index}]" if estimated_is_blocked else ""
            raise ValueError(
                f"estimated_latent{label} and true_latent{label} must have the "
                "same number of rows"
            )
        pairs.append((estimated_block, true_block))

    return pairs, estimated_is_blocked


def _blockwise_metric(results, function):
    pairs, is_blocked = _latent_pairs(results)
    values = [function(estimated, truth) for estimated, truth in pairs]
    return values if is_blocked else values[0]


class BaseMetric:
    def __init__(self):
        pass

    def __call__(self):
        raise NotImplementedError("Subclasses should implement this!")

    def get_name(self):
        raise NotImplementedError("Subclasses should implement this!")


class ReturnMetric(BaseMetric):
    def __init__(self, only_return=None):
        super().__init__()
        self.only_return = only_return
        
    def __call__(self, results, is_null=None):
        estimated = results["estimated_latent"]
        truth = results["true_latent"]
        test_stat = results.get("test_stat", None)
        p_value = results.get("p-value", None)
        only_return = self.only_return
        
        if only_return is not None:
            if only_return == "estimated":
                return estimated
            elif only_return == "truth":
                return truth
            elif only_return == "Y":
                return results.get("observed_Y")
            elif only_return == "X":
                return results.get("observed_X")
            elif only_return == "test_stat":
                return test_stat
            elif only_return == "p-value":
                return p_value
            elif only_return == "is_null":
                return is_null
            else:
                raise ValueError(f"Invalid value for 'only_return': {only_return}")
        return {
            "estimated": estimated,
            "truth": truth,
            "Y": results.get("observed_Y"),
            "X": results.get("observed_X"),
            "test_stat": test_stat,
            "p-value": p_value,
            "is_null": is_null,
        }

    def get_name(self):
        return "ReturnMetric"


class RVCoefficient(BaseMetric):
    def __call__(self, results, is_null=None):
        return _blockwise_metric(results, rv_coefficient)

    def get_name(self):
        return "RV Coefficient"


class AdjustedRVCoefficient(BaseMetric):
    def __call__(self, results, is_null=None):
        return _blockwise_metric(results, rv_coefficient_adjusted)

    def get_name(self):
        return "Adjusted RV Coefficient"


class MSE(BaseMetric):
    def __call__(self, results, is_null=None):
        def mse(estimated, truth):
            if estimated.shape != truth.shape:
                raise ValueError(
                    "estimated and true latent blocks must have matching shapes "
                    "to compute MSE"
                )
            return np.mean((truth - estimated) ** 2)

        return _blockwise_metric(results, mse)

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

    def __call__(self, results, is_null=None):
        def relative_error(estimated, truth):
            if not np.isfinite(estimated).all() or not np.isfinite(truth).all():
                return np.nan

            if self.gram_matrix:
                estimated = estimated @ estimated.T
                truth = truth @ truth.T
            elif estimated.shape != truth.shape:
                raise ValueError(
                    "estimated and true latent blocks must have matching shapes "
                    "when gram_matrix=False"
                )

            num = norm(estimated - truth, "fro")
            den = norm(truth, "fro")
            return num / den if den != 0 else 0

        return _blockwise_metric(results, relative_error)

    def get_name(self):
        return "RelativeFrobeniusNorm"


class RobustRelativeProcrustesDistance:
    """
    Robust Relative Procrustes Distance for heavy-tailed (Cauchy) data.

    1. Robust to outliers via Median-centering and L1-scaling.
    2. Rotation invariant via SVD-based alignment (Kabsch).
    3. Handles differing feature dimensions (columns) via zero-padding.
    4. Scale invariant (Relative) to handle large matrix entries.
    """

    def __call__(self, results, is_null=None):
        def procrustes(est, true):
            if not np.isfinite(est).all() or not np.isfinite(true).all():
                return np.nan

            n_true, d_true = true.shape
            n_est, d_est = est.shape

            # Procrustes requires 1-to-1 observation mapping (rows must match)
            if n_true != n_est:
                raise ValueError(
                    f"Row counts must match for Procrustes. Got {n_true} and {n_est}."
                )

            # --- 0. Dimensional Padding ---
            # Pad the smaller matrix with zeros so the feature columns match
            max_d = max(d_true, d_est)

            if d_true < max_d:
                true_padded = np.pad(
                    true, ((0, 0), (0, max_d - d_true)), mode="constant"
                )
            else:
                true_padded = true

            if d_est < max_d:
                est_padded = np.pad(est, ((0, 0), (0, max_d - d_est)), mode="constant")
            else:
                est_padded = est

            # --- 1. Robust Centering ---
            true_c = true_padded - np.median(true_padded, axis=0)
            est_c = est_padded - np.median(est_padded, axis=0)

            # --- 2. Alignment (Kabsch Algorithm) ---
            # Compute cross-covariance matrix
            H = est_c.T @ true_c

            # SVD works cleanly now because H is a square matrix (max_d x max_d)
            U, _, Vt = np.linalg.svd(H)
            R_opt = U @ Vt

            # Optional but recommended: Ensure we have a rotation, not a reflection
            if np.linalg.det(R_opt) < 0:
                Vt[-1, :] *= -1
                R_opt = U @ Vt

            est_aligned = est_c @ R_opt

            # --- 3. Robust Relative Distance ---
            # Shapes are now guaranteed to match for subtraction
            abs_error = np.sum(np.abs(true_c - est_aligned))
            abs_truth = np.sum(np.abs(true_c))

            rel_dist = abs_error / abs_truth if abs_truth > 0 else abs_error

            return rel_dist

        return _blockwise_metric(results, procrustes)

    def get_name(self):
        return "RobustRelativeProcrustes"


class Rejection(BaseMetric):
    """Rejection of Null Hypothesis, one if rejected.

    Takes as input a results dictionary containing 'reject_null' key.
    """

    def __call__(self, results, is_null=None):
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

    def __call__(self, results, is_null=None):
        reject_null = results["reject_null"]
        # if null is True, but we reject it.
        if (is_null is True) and (reject_null is True):
            return True
        return False

    def get_name(self):
        return "FalseRejection"


class FalseAcceptance(BaseMetric):
    """False Acceptance (Type II Error / False Negative)

    Takes as input a results dictionary containing 'reject_null' and 'true_null' keys.
    """

    def __call__(self, results, is_null=None):
        reject_null = results["reject_null"]
        # Null is False (H0), but we do not reject it (i.e accept it)
        if (is_null is False) and (reject_null is False):
            return True
        return False

    def get_name(self):
        return "FalseAcceptance"


class TrueRejection(BaseMetric):
    """True Rejection (reject H0 when it is False)

    Takes as input a results dictionary with keywords 'reject_null' and 'null'.
    """

    def __call__(self, results, is_null=None):
        reject_null = results["reject_null"]
        # Null is False (H1) and we reject it
        if (is_null is False) and (reject_null is True):
            return True
        return False

    def get_name(self):
        return "TrueRejection"


class TrueAcceptance(BaseMetric):
    """True Acceptance (accept H0 when it is True)

    Takes as input a results dictionary with keywords 'reject_null' and 'null'.
    """

    def __call__(self, results, is_null=None):
        reject_null = results["reject_null"]
        # Null is True (H0) and we accept it
        if (is_null is True) and (reject_null is False):
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

    def __call__(self, results, is_null=None):
        out = {}
        reject_null = results.get("reject_null", None)
        estimated_latent = results.get("estimated_latent", None)

        if reject_null is not None:
            # compute test metrics
            test_metrics = {
                "Rejection": Rejection()(results),
                "FalseRejection": FalseRejection()(results, is_null=is_null),
                "FalseAcceptance": FalseAcceptance()(results, is_null=is_null),
                "TrueRejection": TrueRejection()(results, is_null=is_null),
                "TrueAcceptance": TrueAcceptance()(results, is_null=is_null),
            }
            out.update(test_metrics)

        if estimated_latent is not None and results.get("true_latent") is not None:
            est = RelativeFrobeniusNorm(gram_matrix=self.gram_matrix)(results)
            latent_metrics = {"RelativeFrobeniusNorm": est}

            est_procrustes = RobustRelativeProcrustesDistance()(results)
            latent_metrics["ProcrustesDistance"] = est_procrustes
            out.update(latent_metrics)

        return out

    def get_name(self):
        return "ComputeAll"
