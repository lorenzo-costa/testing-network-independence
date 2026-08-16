import numpy as np
import pytest

from src.methods._base_class import (
    BaseEstimationMethod,
    BaseMethod,
    BasePermutationTest,
)


class RecordingSolver:
    def __init__(self):
        self.inputs = []
        self.dimensions = []

    def __call__(self, matrix, k, rng=None):
        self.inputs.append(np.asarray(matrix).copy())
        self.dimensions.append(k)
        return np.asarray(matrix)[:, :k].copy(), np.ones(k)


def network_data(n=5):
    values = np.arange(n * n, dtype=float).reshape(n, n)
    return {
        "Ay": values,
        "Ax": [values + 1, values + 2],
        "Y": np.arange(2 * n, dtype=float).reshape(n, 2),
        "X": [
            np.arange(n, dtype=float).reshape(n, 1),
            np.arange(n, dtype=float).reshape(n, 1) + 10,
        ],
    }


def test_result_structure_contains_all_latent_blocks():
    method = BaseMethod()
    method.Yhat = np.zeros((3, 1))
    method.Xhat_blocks = [np.ones((3, 1)), np.full((3, 2), 2.0)]
    method.Y = np.full((3, 1), 3.0)
    method.X = [np.full((3, 1), 4.0), np.full((3, 2), 5.0)]
    method.pvalue = 0.5
    method.reject_null = False
    method.test_stat_estimate = 1.25

    result = method.get_estimated()

    assert set(result) == {
        "estimated_latent",
        "true_latent",
        "observed_Y",
        "observed_X",
        "p-value",
        "reject_null",
        "test_stat",
    }
    assert result["estimated_latent"] == [method.Yhat, *method.Xhat_blocks]
    assert result["true_latent"] == [method.Y, *method.X]


def test_true_latent_blocks_are_concatenated_for_global_test():
    method = BaseEstimationMethod(k=[2, 1, 1], use_true_latent=True)
    data = network_data()

    method._process_input(data)

    np.testing.assert_array_equal(method.Yhat, data["Y"])
    np.testing.assert_array_equal(
        method.Xhat,
        np.concatenate(data["X"], axis=1),
    )
    assert [block.shape for block in method.Xhat_blocks] == [(5, 1), (5, 1)]


def test_input_processing_rejects_mismatched_x_blocks():
    method = BaseEstimationMethod(k=[2, 1, 1], use_true_latent=True)
    data = network_data()
    data["X"] = [data["X"][0]]

    with pytest.raises(ValueError, match="one latent-position array per Ax"):
        method._process_input(data)


def test_estimated_latents_use_ay_and_every_ax():
    solver = RecordingSolver()
    method = BaseEstimationMethod(solver=solver, k=[2, 1, 1])
    data = network_data()

    method._process_input(data)

    assert len(solver.inputs) == 3
    assert solver.dimensions == [2, 1, 1]
    np.testing.assert_array_equal(solver.inputs[0], data["Ay"])
    np.testing.assert_array_equal(solver.inputs[1], data["Ax"][0])
    np.testing.assert_array_equal(solver.inputs[2], data["Ax"][1])
    assert method.Xhat.shape == (5, 2)


def test_latent_permutations_do_not_refit():
    solver = RecordingSolver()
    method = BasePermutationTest(
        solver=solver,
        k=[2, 1, 1],
        npermutations=4,
        permutation_type="latent",
        test_function=lambda y, x: float(np.sum(y * x)),
        rng=np.random.default_rng(4),
    )
    method._process_input(network_data())

    method._fit_permutation()

    assert len(solver.inputs) == 3
    assert len(method.permutation_indices) == 4


def test_latent_permutation_changes_only_y_positions():
    calls = []

    def record_statistic(y, x):
        calls.append((y.copy(), x.copy()))
        return float(np.sum(y) + np.sum(x))

    method = BasePermutationTest(
        k=[2, 1, 1],
        use_true_latent=True,
        npermutations=3,
        permutation_type="latent",
        test_function=record_statistic,
        rng=np.random.default_rng(8),
    )
    method._process_input(network_data())

    method._fit_permutation()

    np.testing.assert_array_equal(calls[0][0], method.Yhat)
    np.testing.assert_array_equal(calls[0][1], method.Xhat)
    for (permuted_y, unchanged_x), permutation in zip(
        calls[1:], method.permutation_indices
    ):
        np.testing.assert_array_equal(permuted_y, method.Yhat[permutation])
        np.testing.assert_array_equal(unchanged_x, method.Xhat)


def test_covariate_permutation_is_rejected():
    with pytest.raises(ValueError, match="Must be 'latent' or 'observed'"):
        BasePermutationTest(
            k=[1, 1],
            use_true_latent=True,
            permutation_type="covariate",
            test_function=lambda y, x: 0.0,
        )


def test_observed_permutation_relabels_ay_and_refits():
    solver = RecordingSolver()
    method = BasePermutationTest(
        solver=solver,
        k=[1, 1, 1],
        npermutations=3,
        permutation_type="observed",
        test_function=lambda y, x: float(y[:, 0] @ x[:, 0]),
        rng=np.random.default_rng(5),
    )
    data = network_data()
    method._process_input(data)

    method._fit_permutation()

    assert len(solver.inputs) == 6
    for solver_input, permutation in zip(
        solver.inputs[3:], method.permutation_indices
    ):
        np.testing.assert_array_equal(
            solver_input,
            data["Ay"][permutation][:, permutation],
        )


def test_observed_permutation_rejects_true_latent_mode():
    with pytest.raises(ValueError, match="requires use_true_latent=False"):
        BasePermutationTest(
            k=[1, 1],
            use_true_latent=True,
            permutation_type="observed",
            test_function=lambda y, x: 0.0,
        )
