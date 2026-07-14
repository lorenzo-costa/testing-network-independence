import numpy as np
import pytest

from src.methods._base_class import BaseEstimationMethod, BaseMethod, BasePermutationTest


class RecordingSolver:
    def __init__(self):
        self.inputs = []

    def __call__(self, matrix, k, rng=None):
        self.inputs.append(np.asarray(matrix).copy())
        return np.asarray(matrix)[:, :k].copy(), np.ones(k)


def test_result_structure_contains_one_latent_and_observed_y():
    method = BaseMethod()
    method.Zhat = np.zeros((3, 1))
    method.Z = np.ones((3, 1))
    method.Y = np.arange(3)[:, None]
    method.pvalue = 0.5
    method.reject_null = False
    method.test_stat_estimate = 1.25

    result = method.get_estimated()
    assert set(result) == {
        "estimated_latent",
        "true_latent",
        "observed_Y",
        "p-value",
        "reject_null",
        "test_stat",
    }
    assert result["estimated_latent"] is method.Zhat
    assert result["true_latent"] is method.Z
    assert result["observed_Y"] is method.Y


def test_true_latent_does_not_require_or_call_solver():
    method = BaseEstimationMethod(use_true_latent=True)
    z = np.arange(8.0).reshape(4, 2)
    method._process_input({"Z": z, "Y": np.arange(4.0)})
    np.testing.assert_array_equal(method.Zhat, z)
    assert method.Y.shape == (4, 1)


def test_estimated_latent_uses_only_a():
    solver = RecordingSolver()
    a = np.arange(16.0).reshape(4, 4)
    method = BaseEstimationMethod(solver=solver, k=2)
    method._process_input({"A": a, "Y": np.ones((4, 1))})
    assert len(solver.inputs) == 1
    np.testing.assert_array_equal(solver.inputs[0], a)


@pytest.mark.parametrize("permutation_type", ["covariate", "latent"])
def test_latent_permutations_do_not_refit(permutation_type):
    solver = RecordingSolver()
    method = BasePermutationTest(
        solver=solver,
        k=2,
        npermutations=4,
        permutation_type=permutation_type,
        test_function=lambda z, y: float(np.sum(z * y)),
        rng=np.random.default_rng(4),
    )
    method._process_input({"A": np.eye(5), "Y": np.ones((5, 2))})
    method._fit_permutation()
    assert len(solver.inputs) == 1


def test_observed_permutation_relabels_a_and_refits():
    solver = RecordingSolver()
    method = BasePermutationTest(
        solver=solver,
        k=1,
        npermutations=3,
        permutation_type="observed",
        test_function=lambda z, y: float(z[:, 0] @ y[:, 0]),
        rng=np.random.default_rng(5),
    )
    method._process_input({"A": np.arange(25.0).reshape(5, 5), "Y": np.ones((5, 1))})
    method._fit_permutation()
    assert len(solver.inputs) == 4


def test_observed_permutation_rejects_true_latent_mode():
    with pytest.raises(ValueError, match="requires use_true_latent=False"):
        BasePermutationTest(
            use_true_latent=True,
            permutation_type="observed",
            test_function=lambda z, y: 0.0,
        )
