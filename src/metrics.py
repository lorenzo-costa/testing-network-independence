"""Testing outcomes and latent-recovery metrics for multiple networks."""

import numpy as np
from scipy.linalg import norm

from .test_functions.rv_cca_coefficients import rv_coefficient, rv_coefficient_adjusted


def _x_blocks(latents, name):
    """Read explicit block boundaries; never guess them from a concatenation."""
    blocks = latents.get("X_blocks")
    if blocks is None and isinstance(latents.get("X"), (list, tuple)):
        blocks = latents["X"]
    if blocks is not None and (not isinstance(blocks, (list, tuple)) or not blocks):
        raise ValueError(f"{name}.X_blocks must be a nonempty list of matrices.")
    return blocks


def _concatenate_blocks(blocks):
    if blocks is None or any(block is None for block in blocks):
        return None
    return np.concatenate(blocks, axis=1)


def _latent_pairs(results):
    """Return Y, each X block, and concatenated X for named network results."""
    estimated = results.get("estimated_latent")
    truth = results.get("true_latent")
    if estimated is not None and not isinstance(estimated, dict):
        raise ValueError("estimated_latent must be a mapping or None.")
    if truth is not None and not isinstance(truth, dict):
        raise ValueError("true_latent must be a mapping or None.")
    estimated = {} if estimated is None else estimated
    truth = {} if truth is None else truth
    est_blocks = _x_blocks(estimated, "estimated_latent")
    true_blocks = _x_blocks(truth, "true_latent")
    if (
        est_blocks is not None
        and true_blocks is not None
        and len(est_blocks) != len(true_blocks)
    ):
        raise ValueError("Estimated and true X must have the same number of blocks.")
    count = (
        len(est_blocks)
        if est_blocks is not None
        else (len(true_blocks) if true_blocks is not None else 0)
    )
    pairs = [("Y", estimated.get("Y"), truth.get("Y"))]
    pairs.extend(
        (
            f"X_{i + 1}",
            None if est_blocks is None else est_blocks[i],
            None if true_blocks is None else true_blocks[i],
        )
        for i in range(count)
    )
    est_global = (
        _concatenate_blocks(est_blocks)
        if est_blocks is not None
        else estimated.get("X")
    )
    true_global = (
        _concatenate_blocks(true_blocks) if true_blocks is not None else truth.get("X")
    )
    pairs.append(("X_global", est_global, true_global))
    return pairs


def _optional_boolean(value, name):
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean or missing.")
    return bool(value)


def _testing_outcome(results, is_null, expected_null, expected_rejection):
    null = _optional_boolean(is_null, "is_null")
    rejection = _optional_boolean(results.get("reject_null"), "reject_null")
    if null is None or rejection is None:
        return np.nan
    return null == expected_null and rejection == expected_rejection


class BaseMetric:
    def __call__(self, results, is_null=None):
        raise NotImplementedError("Subclasses should implement this!")

    def get_name(self):
        raise NotImplementedError("Subclasses should implement this!")


class _LatentPairMetric(BaseMetric):
    """Apply a measure to each named latent pair."""

    def __call__(self, results, is_null=None):
        return self._evaluate_pairs(_latent_pairs(results))

    def _evaluate_pairs(self, pairs):
        """Evaluate an already normalized set of latent pairs."""
        return {
            name: self._evaluate_pair(estimated, truth)
            for name, estimated, truth in pairs
        }

    def _evaluate_pair(self, estimated, truth):
        if estimated is None or truth is None:
            return np.nan
        estimated = np.asarray(estimated, dtype=float)
        truth = np.asarray(truth, dtype=float)
        if estimated.ndim != 2 or truth.ndim != 2:
            raise ValueError("Latent positions must be 2D matrices.")
        if estimated.shape[0] != truth.shape[0]:
            raise ValueError(
                "Estimated and true latents must have matching row counts."
            )
        if (
            estimated.size == 0
            or truth.size == 0
            or not np.isfinite(estimated).all()
            or not np.isfinite(truth).all()
        ):
            return np.nan
        return self._compute_pair(estimated, truth)


class ReturnMetric(BaseMetric):
    """Return raw results; Y/X select true Y and concatenated true X.

    Estimated values (including individual X_blocks) are available through
    only_return="estimated". Unlike numeric error metrics, missing raw arrays
    are returned as None.
    """

    def __init__(self, only_return=None):
        self.only_return = only_return

    def __call__(self, results, is_null=None):
        estimated = results.get("estimated_latent")
        truth = results.get("true_latent")
        if estimated is not None and not isinstance(estimated, dict):
            raise ValueError("estimated_latent must be a mapping or None.")
        if truth is not None and not isinstance(truth, dict):
            raise ValueError("true_latent must be a mapping or None.")
        true_values = {} if truth is None else truth
        y = true_values.get("Y")
        blocks = _x_blocks(true_values, "true_latent")
        x = _concatenate_blocks(blocks) if blocks is not None else true_values.get("X")
        values = {
            "estimated": estimated,
            "truth": truth,
            "Y": y,
            "X": x,
            "test_stat": results.get("test_stat"),
            "p-value": results.get("p-value"),
            "is_null": is_null,
        }
        if self.only_return is None:
            return values
        if self.only_return not in values:
            raise ValueError(f"Invalid value for 'only_return': {self.only_return}")
        return values[self.only_return]

    def get_name(self):
        return "ReturnMetric"


class RVCoefficient(_LatentPairMetric):
    """RV similarity between each estimate/truth pair (not an error distance)."""

    def _compute_pair(self, estimated, truth):
        return rv_coefficient(estimated, truth)

    def get_name(self):
        return "RV Coefficient"


class AdjustedRVCoefficient(_LatentPairMetric):
    def _compute_pair(self, estimated, truth):
        return rv_coefficient_adjusted(estimated, truth)

    def get_name(self):
        return "Adjusted RV Coefficient"


class MSE(_LatentPairMetric):
    """Raw coordinate MSE for each pair, without rotation alignment."""

    def _compute_pair(self, estimated, truth):
        if estimated.shape != truth.shape:
            raise ValueError("MSE requires matching estimated and true dimensions.")
        return ((truth - estimated) ** 2).mean()

    def get_name(self):
        return "Mean Squared Error"


class RelativeFrobeniusNorm(_LatentPairMetric):
    """Relative Frobenius error, optionally comparing rotation-invariant Grams."""

    def __init__(self, gram_matrix=False):
        self.gram_matrix = gram_matrix

    def _compute_pair(self, estimated, truth):
        if self.gram_matrix:
            estimated = estimated @ estimated.T
            truth = truth @ truth.T
        elif estimated.shape != truth.shape:
            raise ValueError("Coordinate Frobenius error requires matching dimensions.")
        num = norm(estimated - truth, "fro")
        den = norm(truth, "fro")
        # Preserve the existing convention for a zero-norm truth.
        return num / den if den != 0 else 0

    def get_name(self):
        return "RelativeFrobeniusNorm"


class RobustRelativeProcrustesDistance(_LatentPairMetric):
    """Median-centered, rotation-aligned L1 relative error for each latent pair.

    Different feature dimensions are zero-padded before rotation alignment.
    """

    def _compute_pair(self, estimated, truth):
        d_true, d_est = truth.shape[1], estimated.shape[1]
        max_d = max(d_true, d_est)
        true_padded = np.pad(truth, ((0, 0), (0, max_d - d_true)))
        est_padded = np.pad(estimated, ((0, 0), (0, max_d - d_est)))
        true_c = true_padded - np.median(true_padded, axis=0)
        est_c = est_padded - np.median(est_padded, axis=0)
        U, _, Vt = np.linalg.svd(est_c.T @ true_c)
        R_opt = U @ Vt
        if np.linalg.det(R_opt) < 0:
            Vt[-1, :] *= -1
            R_opt = U @ Vt
        est_aligned = est_c @ R_opt
        abs_error = np.sum(np.abs(true_c - est_aligned))
        abs_truth = np.sum(np.abs(true_c))
        return abs_error / abs_truth if abs_truth > 0 else abs_error

    def get_name(self):
        return "RobustRelativeProcrustes"


class Rejection(BaseMetric):
    def __call__(self, results, is_null=None):
        rejection = _optional_boolean(results.get("reject_null"), "reject_null")
        return np.nan if rejection is None else rejection

    def get_name(self):
        return "Rejection"


class FalseRejection(BaseMetric):
    """Type I error indicator; NaN if the null label or decision is missing."""

    def __call__(self, results, is_null=None):
        return _testing_outcome(results, is_null, True, True)

    def get_name(self):
        return "FalseRejection"


class FalseAcceptance(BaseMetric):
    """Type II error indicator; NaN if the null label or decision is missing."""

    def __call__(self, results, is_null=None):
        return _testing_outcome(results, is_null, False, False)

    def get_name(self):
        return "FalseAcceptance"


class TrueRejection(BaseMetric):
    def __call__(self, results, is_null=None):
        return _testing_outcome(results, is_null, False, True)

    def get_name(self):
        return "TrueRejection"


class TrueAcceptance(BaseMetric):
    def __call__(self, results, is_null=None):
        return _testing_outcome(results, is_null, True, False)

    def get_name(self):
        return "TrueAcceptance"


class ComputeAll(BaseMetric):
    """Testing outcomes plus Frobenius/Procrustes recovery errors.

    Named latent results produce flat fields with suffixes Y, X_1, ...,
    X_global. Global errors compare concatenated matrices directly.
    Missing truth or null labels produce NaN for the affected metrics.
    """

    def __init__(self, gram_matrix=True):
        self.gram_matrix = gram_matrix

    def __call__(self, results, is_null=None):
        out = {
            metric.get_name(): metric(results, is_null=is_null)
            for metric in (
                Rejection(),
                FalseRejection(),
                FalseAcceptance(),
                TrueRejection(),
                TrueAcceptance(),
            )
        }
        if (
            results.get("estimated_latent") is not None
            or results.get("true_latent") is not None
        ):
            pairs = _latent_pairs(results)
            for prefix, metric in (
                ("RelativeFrobeniusNorm", RelativeFrobeniusNorm(self.gram_matrix)),
                ("ProcrustesDistance", RobustRelativeProcrustesDistance()),
            ):
                values = metric._evaluate_pairs(pairs)
                out.update(
                    {f"{prefix}_{name}": value for name, value in values.items()}
                )
        return out

    def get_name(self):
        return "ComputeAll"
