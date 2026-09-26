import os

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
        self.rngs = []

    def __call__(self, matrix, k, rng=None):
        self.inputs.append(np.asarray(matrix).copy())
        self.dimensions.append(k)
        self.rngs.append(rng)
        return np.asarray(matrix)[:, :k].copy(), np.ones(k)


def dot_product_statistic(z, y):
    return float(z[:, 0] @ y[:, 0])


def process_id_statistic(z, y):
    return float(os.getpid())


def stochastic_solver(matrix, k, rng=None):
    values = np.asarray(matrix, dtype=float)[:, :k]
    return values + rng.normal(size=values.shape), np.ones(k)


def network_data(n=8, p=2):
    rng = np.random.default_rng(123)
    matrices = [rng.normal(size=(n, n)) for _ in range(p + 1)]
    matrices = [matrix + matrix.T for matrix in matrices]
    return {"A_Y": matrices[0], "A_X": matrices[1:]}


def latent_data(n=6):
    return {
        "Y": np.arange(n, dtype=float).reshape(n, 1),
        "X": [np.arange(n * 2, dtype=float).reshape(n, 2), np.ones((n, 2))],
    }


def test_result_structure_contains_test_and_latent_outputs():
    method = BaseMethod()
    method.Zhat = np.zeros((3, 1))
    method.Z = np.ones((3, 1))
    method.pvalue = 0.5
    method.reject_null = False
    method.test_stat_estimate = 1.25

    result = method.get_estimated()
    assert set(result) == {
        "estimated_latent",
        "true_latent",
        "p-value",
        "reject_null",
        "test_stat",
    }
    assert result["estimated_latent"] is method.Zhat
    assert result["true_latent"] is method.Z


def test_estimation_base_embeds_y_and_every_x_network():
    solver = RecordingSolver()
    data = network_data(n=6, p=3)
    method = BaseEstimationMethod(solver=solver, d_y=2, d_x=1)
    method._process_input(data)

    assert solver.dimensions == [2, 1, 1, 1]
    assert method.Yhat.shape == (6, 2)
    assert method.Xhat.shape == (6, 3)
    assert len(method.Xhat_blocks) == 3
    np.testing.assert_array_equal(
        method.Xhat, np.concatenate(method.Xhat_blocks, axis=1)
    )


def test_estimation_base_true_latents_bypass_solver():
    solver = RecordingSolver()
    data = latent_data(n=6)
    method = BaseEstimationMethod(use_true_latent=True, solver=solver)
    method._process_input(data)

    assert solver.inputs == []
    np.testing.assert_array_equal(method.Yhat, data["Y"])
    np.testing.assert_array_equal(method.Xhat, np.concatenate(data["X"], axis=1))
    assert (method.d_y, method.d_x) == (1, 2)


def test_estimation_base_rejects_incomplete_network_input():
    method = BaseEstimationMethod(use_true_latent=True)
    with pytest.raises(ValueError, match="Y and X"):
        method._process_input({"Z": np.ones((4, 2)), "Y": np.ones((4, 1))})


def test_latent_permutations_fit_every_network_once_and_keep_x_fixed():
    solver = RecordingSolver()
    calls = []

    def statistic(y, x):
        calls.append((y.copy(), x.copy()))
        return float(y[:, 0] @ x[:, -1])

    method = BasePermutationTest(
        solver=solver,
        d_y=3,
        d_x=2,
        npermutations=4,
        permutation_type="latent",
        test_function=statistic,
        rng=np.random.default_rng(4),
    )
    data = network_data(n=5, p=3)
    method.fit(data)
    assert len(solver.inputs) == 4
    assert solver.dimensions == [3, 2, 2, 2]
    assert all(rng is method.rng for rng in solver.rngs)
    expected_x = np.concatenate([a[:, :2] for a in data["A_X"]], axis=1)
    np.testing.assert_array_equal(calls[0][0], data["A_Y"][:, :3])
    for _, x in calls:
        np.testing.assert_array_equal(x, expected_x)
    for permutation, (y, _) in zip(method.permutation_indices, calls[1:]):
        np.testing.assert_array_equal(y, method.Yhat[permutation])


def test_adjacency_permutation_relabels_both_y_axes_and_refits_only_y():
    solver = RecordingSolver()
    calls = []

    def statistic(y, x):
        calls.append((y.copy(), x.copy()))
        return float(y[:, 0] @ x[:, -1])

    method = BasePermutationTest(
        solver=solver,
        d_y=2,
        d_x=1,
        npermutations=3,
        permutation_type="adjacency",
        test_function=statistic,
        rng=np.random.default_rng(5),
    )
    data = network_data(n=5, p=2)
    method.fit(data)
    assert len(solver.inputs) == 3 + method.npermutations
    assert solver.dimensions == [2, 1, 1, 2, 2, 2]
    for permutation, matrix, (y, x) in zip(
        method.permutation_indices, solver.inputs[3:], calls[1:]
    ):
        expected = data["A_Y"][permutation][:, permutation]
        np.testing.assert_array_equal(matrix, expected)
        np.testing.assert_array_equal(y, expected[:, :2])
        np.testing.assert_array_equal(x, method.Xhat)


def test_adjacency_permutation_rejects_true_latent_mode():
    with pytest.raises(ValueError, match="requires use_true_latent=False"):
        BasePermutationTest(
            use_true_latent=True,
            permutation_type="adjacency",
            test_function=lambda z, y: 0.0,
        )


def test_scalar_permutation_finalizer_uses_finite_permutation_correction():
    method = BasePermutationTest(
        use_true_latent=True,
        test_function=lambda z, y: 0.0,
        one_sided=True,
        alpha=0.4,
    )

    method._compute_pvalue_and_rejection(3.0, np.array([1.0, 2.0]))

    assert method.test_stat_estimate == 3.0
    assert method.permutation_distribution == [1.0, 2.0]
    assert method.pvalue == pytest.approx(1 / 3)
    assert method.reject_null is True


def test_unordered_permutation_results_are_restored_to_task_order():
    method = BasePermutationTest(
        use_true_latent=True,
        npermutations=3,
        test_function=lambda z, y: 0.0,
    )

    collected = method._collect_permutation_statistics(
        [(2, 30.0), (0, 10.0), (1, 20.0)]
    )

    assert collected == [10.0, 20.0, 30.0]


def test_vector_permutation_finalizer_standardizes_and_takes_maximum():
    statistics = np.array(
        [
            [4.0, 2.0, 8.0],
            [1.0, 3.0, 4.0],
            [2.0, 5.0, 3.0],
        ]
    )
    method = BasePermutationTest(
        use_true_latent=True,
        test_function=lambda z, y: [0.0],
    )

    method._compute_pvalue_and_rejection(statistics[0], statistics[1:])

    expected_means = statistics.mean(axis=0)
    expected_stds = statistics.std(axis=0, ddof=1)
    expected_standardized = (statistics - expected_means) / expected_stds
    expected_maxima = expected_standardized.max(axis=1)
    expected_exceedances = np.count_nonzero(expected_maxima[1:] >= expected_maxima[0])

    np.testing.assert_allclose(method.statistic_matrix, statistics)
    np.testing.assert_allclose(method.statistic_means, expected_means)
    np.testing.assert_allclose(method.statistic_standard_deviations, expected_stds)
    np.testing.assert_allclose(method.standardized_statistics, expected_standardized)
    assert method.test_stat_estimate == pytest.approx(expected_maxima[0])
    assert method.permutation_distribution == pytest.approx(expected_maxima[1:])
    assert method.pvalue == pytest.approx((1 + expected_exceedances) / 3)


@pytest.mark.parametrize("n_jobs", [0, -2, 1.5, True])
def test_permutation_test_rejects_invalid_n_jobs(n_jobs):
    with pytest.raises(ValueError, match="n_jobs must be"):
        BasePermutationTest(
            use_true_latent=True,
            test_function=lambda z, y: 0.0,
            n_jobs=n_jobs,
        )


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_permutation_test_rejects_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size must be"):
        BasePermutationTest(
            use_true_latent=True,
            test_function=lambda z, y: 0.0,
            batch_size=batch_size,
        )


def test_all_cpu_n_jobs_resolves_cpu_count(monkeypatch):
    monkeypatch.setattr("src.methods._base_class.os.cpu_count", lambda: 7)
    method = BasePermutationTest(
        use_true_latent=True,
        test_function=lambda z, y: 0.0,
        n_jobs=-1,
    )

    assert method._effective_n_jobs() == 7


def test_parallel_permutations_match_serial_permutations():
    data = latent_data(n=8)
    methods = []
    for n_jobs, batch_size in ((1, 32), (3, 32), (3, 1)):
        method = BasePermutationTest(
            use_true_latent=True,
            npermutations=8,
            test_function=dot_product_statistic,
            rng=np.random.default_rng(42),
            n_jobs=n_jobs,
            batch_size=batch_size,
        )
        method._process_input(data)
        method._fit_permutation()
        methods.append(method)

    serial = methods[0]
    for parallel in methods[1:]:
        for serial_perm, parallel_perm in zip(
            serial.permutation_indices, parallel.permutation_indices
        ):
            np.testing.assert_array_equal(serial_perm, parallel_perm)
        np.testing.assert_allclose(
            serial.permutation_distribution,
            parallel.permutation_distribution,
        )
        assert serial.test_stat_estimate == pytest.approx(parallel.test_stat_estimate)
        assert serial.pvalue == parallel.pvalue


def test_adjacency_parallel_permutations_use_reproducible_child_rngs():
    data = network_data(n=6, p=3)
    methods = []
    for n_jobs in (1, 3):
        method = BasePermutationTest(
            solver=stochastic_solver,
            d_y=2,
            d_x=1,
            npermutations=6,
            permutation_type="adjacency",
            test_function=dot_product_statistic,
            rng=np.random.default_rng(43),
            n_jobs=n_jobs,
            batch_size=2,
        )
        method._process_input(data)
        method._fit_permutation()
        methods.append(method)

    serial, parallel = methods
    np.testing.assert_allclose(serial.permuted_statistics, parallel.permuted_statistics)
    np.testing.assert_allclose(
        serial.permutation_distribution, parallel.permutation_distribution
    )
    assert serial.pvalue == parallel.pvalue


def test_parallel_permutations_execute_in_worker_processes():
    main_process = os.getpid()
    method = BasePermutationTest(
        use_true_latent=True,
        npermutations=6,
        test_function=process_id_statistic,
        rng=np.random.default_rng(44),
        n_jobs=3,
    )
    method._process_input(latent_data())
    method._fit_permutation()

    assert all(process_id != main_process for process_id in method.permuted_statistics)


def test_parallel_permutations_compute_pool_chunk_size():
    method = BasePermutationTest(
        use_true_latent=True,
        npermutations=16,
        test_function=dot_product_statistic,
        rng=np.random.default_rng(46),
        n_jobs=2,
        batch_size=2,
    )
    method._process_input(latent_data())
    method._fit_permutation()

    assert method.permutation_chunk_size == 4


def test_parallel_permutations_reject_nested_daemon_process(monkeypatch):
    class DaemonProcess:
        daemon = True

    monkeypatch.setattr(
        "src.methods._base_class.current_process",
        lambda: DaemonProcess(),
    )
    method = BasePermutationTest(
        use_true_latent=True,
        npermutations=2,
        test_function=dot_product_statistic,
        rng=np.random.default_rng(49),
        n_jobs=2,
    )
    method._process_input(latent_data())

    with pytest.raises(RuntimeError, match="parallel simulation worker"):
        method._fit_permutation()


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_verbose_permutations_show_tqdm_progress(capsys, n_jobs):
    method = BasePermutationTest(
        use_true_latent=True,
        npermutations=3,
        test_function=dot_product_statistic,
        rng=np.random.default_rng(47),
        n_jobs=n_jobs,
        batch_size=2,
        verbose=True,
    )
    method._process_input(latent_data())
    method._fit_permutation()

    captured = capsys.readouterr()
    assert "Permutations" in captured.err
    assert "3/3" in captured.err


def test_nonverbose_permutations_hide_tqdm_progress(capsys):
    method = BasePermutationTest(
        use_true_latent=True,
        npermutations=2,
        test_function=dot_product_statistic,
        rng=np.random.default_rng(48),
        n_jobs=2,
        verbose=False,
    )
    method._process_input(latent_data())
    method._fit_permutation()

    captured = capsys.readouterr()
    assert "Permutations" not in captured.err
    assert captured.out == ""


def test_ordered_network_input_and_result_matrices_preserve_block_order():
    solver = RecordingSolver()
    data = network_data(n=7, p=3)
    matrices = [data["A_Y"], *data["A_X"]]
    method = BasePermutationTest(
        d_y=3,
        d_x=2,
        solver=solver,
        test_function=dot_product_statistic,
        npermutations=2,
        rng=np.random.default_rng(50),
    )
    method.fit(matrices)

    assert method.Yhat.shape == (7, 3)
    assert method.Xhat.shape == (7, 6)
    assert method.p == 3
    assert solver.dimensions == [3, 2, 2, 2]
    for actual, expected in zip(solver.inputs, matrices):
        np.testing.assert_array_equal(actual, expected)
    for k, matrix in enumerate(matrices[1:]):
        np.testing.assert_array_equal(
            method.Xhat[:, 2 * k : 2 * (k + 1)], matrix[:, :2]
        )
        np.testing.assert_array_equal(method.Xhat_blocks[k], matrix[:, :2])
    result = method.get_estimated()
    assert result["estimated_latent"]["Y"] is method.Yhat
    assert result["estimated_latent"]["X"] is method.Xhat
    assert result["estimated_latent"]["X_blocks"] is method.Xhat_blocks
    assert result["true_latent"] == {"Y": None, "X": None, "X_blocks": None}


def test_true_multiple_latents_bypass_solver_without_mutating_inputs():
    data = latent_data()
    solver = RecordingSolver()
    method = BasePermutationTest(
        use_true_latent=True,
        solver=solver,
        test_function=dot_product_statistic,
        npermutations=2,
    )
    method.fit(data)
    assert solver.inputs == []
    assert (method.d_y, method.d_x) == (1, 2)
    result = method.get_estimated()
    np.testing.assert_array_equal(method.Yhat, data["Y"])
    np.testing.assert_array_equal(method.Xhat, np.concatenate(data["X"], axis=1))
    np.testing.assert_array_equal(result["true_latent"]["X"], method.Xhat)
    for estimated, actual, truth in zip(
        result["estimated_latent"]["X_blocks"],
        result["true_latent"]["X_blocks"],
        data["X"],
    ):
        np.testing.assert_array_equal(estimated, truth)
        np.testing.assert_array_equal(actual, truth)
    method.Yhat[:] = -99
    method.Xhat_blocks[0][:] = -99
    np.testing.assert_array_equal(data["Y"], latent_data()["Y"])
    np.testing.assert_array_equal(data["X"][0], latent_data()["X"][0])


def test_supplied_truth_only_infers_dimensions_and_does_not_replace_estimates():
    data = network_data(n=6, p=2)
    data.update(Y=np.zeros((6, 3)), X=[np.zeros((6, 2)) for _ in range(2)])
    solver = RecordingSolver()
    method = BasePermutationTest(solver=solver, test_function=dot_product_statistic)
    method._process_input(data)
    assert solver.dimensions == [3, 2, 2]
    np.testing.assert_array_equal(method.Yhat, data["A_Y"][:, :3])
    np.testing.assert_array_equal(method.Xhat[:, 2:], data["A_X"][1][:, :2])


@pytest.mark.parametrize("name", ["d_y", "d_x"])
@pytest.mark.parametrize("value", [0, -1, True, np.bool_(True), 1.5, "2"])
def test_embedding_dimensions_are_positive_integers(name, value):
    with pytest.raises(ValueError, match=name):
        BasePermutationTest(
            use_true_latent=True, test_function=dot_product_statistic, **{name: value}
        )


@pytest.mark.parametrize("permutation_type", ["covariate", "observed", "invalid"])
def test_only_latent_and_adjacency_modes_are_accepted(permutation_type):
    with pytest.raises(ValueError, match="Must be 'latent' or 'adjacency'"):
        BasePermutationTest(
            solver=RecordingSolver(),
            test_function=dot_product_statistic,
            permutation_type=permutation_type,
        )


@pytest.mark.parametrize(
    "data,message",
    [
        ([], "at least two"),
        ([np.eye(3)], "at least two"),
        (np.eye(3), "dictionary"),
        ({"A_Y": np.eye(3)}, "supplied together"),
        ({"A_X": [np.eye(3)]}, "supplied together"),
        ({"A_Y": np.ones((3, 2)), "A_X": [np.eye(3)]}, "square"),
        ({"A_Y": np.eye(3), "A_X": []}, "nonempty list"),
        ({"A_Y": np.eye(3), "A_X": np.eye(3)}, "nonempty list"),
        ({"A_Y": np.eye(3), "A_X": [np.eye(4)]}, "matching A_Y"),
        ({"A_Y": np.eye(3), "A_X": [np.full((3, 3), np.nan)]}, "finite"),
        (
            {"A_Y": np.eye(3), "A_X": [np.eye(3)], "Y": np.ones((4, 1))},
            "Y must have n rows",
        ),
        (
            {"A_Y": np.eye(3), "A_X": [np.eye(3)], "X": [np.ones((3, 1))] * 2},
            "p latent blocks",
        ),
    ],
)
def test_invalid_network_data_fails_before_fitting(data, message):
    solver = RecordingSolver()
    method = BasePermutationTest(
        d_y=1, d_x=1, solver=solver, test_function=dot_product_statistic
    )
    with pytest.raises(ValueError, match=message):
        method._process_input(data)
    assert solver.inputs == []


def test_missing_dimensions_fail_before_any_solver_calls():
    solver = RecordingSolver()
    method = BasePermutationTest(solver=solver, test_function=dot_product_statistic)
    with pytest.raises(ValueError, match="Specify d_y and d_x"):
        method._process_input(network_data())
    assert solver.inputs == []


@pytest.mark.parametrize(
    "bad_output",
    [np.ones((5, 1)), np.ones((6, 2)), np.ones(6), np.full((6, 1), np.inf)],
)
def test_solver_output_shape_and_finiteness_are_validated(bad_output):
    method = BasePermutationTest(
        d_y=1,
        d_x=1,
        solver=lambda *args, **kwargs: (bad_output, None),
        test_function=dot_product_statistic,
    )
    with pytest.raises(ValueError, match="Yhat"):
        method._process_input(network_data(n=6))
