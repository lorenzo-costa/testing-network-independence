import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multiscale_graphcorr
import warnings

from ._base_class import BasePermutationTest


def _distance_correlation(Z, Y):
    """Biased sample distance correlation, used as a permutation statistic."""
    distance_z = squareform(pdist(Z, metric="euclidean"))
    distance_y = squareform(pdist(Y, metric="euclidean"))

    def center(distance):
        return (
            distance
            - distance.mean(axis=0, keepdims=True)
            - distance.mean(axis=1, keepdims=True)
            + distance.mean()
        )

    centered_z = center(distance_z)
    centered_y = center(distance_y)
    covariance = np.mean(centered_z * centered_y)
    variance_z = np.mean(centered_z**2)
    variance_y = np.mean(centered_y**2)
    denominator = np.sqrt(variance_z * variance_y)
    return 0.0 if denominator == 0 else covariance / denominator


def _mgc_statistic(Z, Y):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return multiscale_graphcorr(Z, Y, reps=0, workers=1).statistic


class DistanceCorrelationTest(BasePermutationTest):
    def __init__(
        self,
        k=None,
        test_method="mgc",
        npermutations=100,
        alpha=0.05,
        rng=None,
        solver=None,
        use_true_latent=False,
        permutation_type="covariate",
        n_jobs=1,
        batch_size=32,
        verbose=False,
        **kwargs,
    ):
        if test_method not in {"dcorr", "mgc"}:
            raise ValueError("test_method must be 'dcorr' or 'mgc'.")
        self.test_method = test_method
        test_function = _mgc_statistic if test_method == "mgc" else _distance_correlation
        super().__init__(
            k=k,
            npermutations=npermutations,
            alpha=alpha,
            rng=rng,
            solver=solver,
            test_function=test_function,
            use_true_latent=use_true_latent,
            permutation_type=permutation_type,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def compute_distance_matrix(self, values):
        return squareform(pdist(values, metric="euclidean"))

    def _double_center(self, distance):
        return (
            distance
            - distance.mean(axis=0, keepdims=True)
            - distance.mean(axis=1, keepdims=True)
            + distance.mean()
        )

    def fit(self, data):
        self._process_input(data)
        self._fit_permutation()

    def get_name(self):
        return "DistanceCorrelation_" + self.permutation_type
