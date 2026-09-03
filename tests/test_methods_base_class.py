import os

import numpy as np
import pytest

from src.methods._base_class import BaseEstimationMethod, BaseMethod, BasePermutationTest


class RecordingSolver:
    def __init__(self):
        self.inputs = []

    def __call__(self, matrix, k, rng=None):
        self.inputs.append(np.asarray(matrix).copy())
        return np.asarray(matrix)[:, :k].copy(), np.ones(k)


def dot_product_statistic(z, y):
    return float(z[:, 0] @ y[:, 0])


def process_id_statistic(z, y):
    return float(os.getpid())


def stochastic_solver(matrix, k, rng=None):
    values = np.asarray(matrix, dtype=float)[:, :k]
    return values + rng.normal(size=values.shape), np.ones(k)


def test_result_structure_contains_one_latent_and_observed_y():
    method = BaseMethod()
    method.Zhat = np.zeros((3, 1))
    method.Z = np.ones((3, 1))
    method.Y = np.arange(3)[:, None]
    method.X = np.array([[0], [1], [1]])
    method.pvalue = 0.5
    method.reject_null = False
    method.test_stat_estimate = 1.25

    result = method.get_estimated()
    assert set(result) == {
        "estimated_latent",
        "true_latent",
        "observed_Y",
        "conditioning_X",
        "p-value",
        "reject_null",
        "test_stat",
    }
    assert result["estimated_latent"] is method.Zhat
    assert result["true_latent"] is method.Z
    assert result["observed_Y"] is method.Y
    assert result["conditioning_X"] is method.X


def test_input_processing_accepts_and_validates_conditioning_x():
    method = BaseEstimationMethod(use_true_latent=True)
    z = np.arange(8.0).reshape(4, 2)
    method._process_input(
        {"Z": z, "Y": np.arange(4.0), "X": np.array([0, 0, 1, 1])}
    )
    assert method.X.shape == (4, 1)

    with pytest.raises(ValueError, match="X must be a 2D array with n rows"):
        method._process_input(
            {"Z": z, "Y": np.arange(4.0), "X": np.ones((3, 1))}
        )


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


def test_scalar_permutation_finalizer_preserves_existing_pvalue():
    method = BasePermutationTest(
        use_true_latent=True,
        test_function=lambda z, y: 0.0,
        one_sided=True,
        alpha=0.4,
    )

    method._compute_pvalue_and_rejection(3.0, np.array([1.0, 2.0]))

    assert method.test_stat_estimate == 3.0
    assert method.permutation_distribution == [1.0, 2.0]
    assert method.pvalue == 0.0
    assert method.reject_null is True


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
    expected_exceedances = np.count_nonzero(
        expected_maxima[1:] >= expected_maxima[0]
    )

    np.testing.assert_allclose(method.statistic_matrix, statistics)
    np.testing.assert_allclose(method.statistic_means, expected_means)
    np.testing.assert_allclose(
        method.statistic_standard_deviations, expected_stds
    )
    np.testing.assert_allclose(
        method.standardized_statistics, expected_standardized
    )
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
    data = {
        "Z": np.arange(16.0).reshape(8, 2),
        "Y": np.arange(8.0).reshape(8, 1),
    }
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
        assert serial.test_stat_estimate == pytest.approx(
            parallel.test_stat_estimate
        )
        assert serial.pvalue == parallel.pvalue


def test_observed_parallel_permutations_use_reproducible_child_rngs():
    data = {
        "A": np.arange(36.0).reshape(6, 6),
        "Y": np.arange(6.0).reshape(6, 1),
    }
    methods = []
    for n_jobs in (1, 3):
        method = BasePermutationTest(
            solver=stochastic_solver,
            k=1,
            npermutations=6,
            permutation_type="observed",
            test_function=dot_product_statistic,
            rng=np.random.default_rng(43),
            n_jobs=n_jobs,
            batch_size=2,
        )
        method._process_input(data)
        method._fit_permutation()
        methods.append(method)

    serial, parallel = methods
    np.testing.assert_allclose(
        serial.permuted_statistics, parallel.permuted_statistics
    )
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
    method._process_input(
        {
            "Z": np.arange(12.0).reshape(6, 2),
            "Y": np.arange(6.0).reshape(6, 1),
        }
    )
    method._fit_permutation()

    assert all(
        process_id != main_process for process_id in method.permuted_statistics
    )


def test_parallel_permutations_compute_pool_chunk_size():
    method = BasePermutationTest(
        use_true_latent=True,
        npermutations=16,
        test_function=dot_product_statistic,
        rng=np.random.default_rng(46),
        n_jobs=2,
        batch_size=2,
    )
    method._process_input(
        {
            "Z": np.arange(12.0).reshape(6, 2),
            "Y": np.arange(6.0).reshape(6, 1),
        }
    )
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
    method._process_input(
        {
            "Z": np.arange(12.0).reshape(6, 2),
            "Y": np.arange(6.0).reshape(6, 1),
        }
    )

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
    method._process_input(
        {
            "Z": np.arange(12.0).reshape(6, 2),
            "Y": np.arange(6.0).reshape(6, 1),
        }
    )
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
    method._process_input(
        {
            "Z": np.arange(12.0).reshape(6, 2),
            "Y": np.arange(6.0).reshape(6, 1),
        }
    )
    method._fit_permutation()

    captured = capsys.readouterr()
    assert "Permutations" not in captured.err
