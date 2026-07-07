import importlib

import numpy as np
import pytest

from src.methods.cvm_test import ObservedCVM


def matrix_product_statistic(matrix_a, matrix_b):
    return float(np.mean(matrix_a * matrix_b))


def test_module_imports_observed_cvm():
    module = importlib.import_module("src.methods.cvm_test")

    assert module.ObservedCVM is ObservedCVM


def test_observed_cvm_runs_on_adjacency_matrices_by_itself():
    rng = np.random.default_rng(30)
    matrix_a = rng.normal(size=(8, 8))
    matrix_b = rng.normal(size=(8, 8))
    method = ObservedCVM(
        test_function=matrix_product_statistic,
        npermutations=4,
        rng=np.random.default_rng(4),
    )

    method.fit({"A": matrix_a, "B": matrix_b})
    result = method.get_estimated()

    assert method.get_name() == "ObservedCVMPermutationTest"
    assert len(method.permutation_distribution) == 4
    assert np.isfinite(result["test_stat"])
    assert 0.0 <= result["p-value"] <= 1.0
    assert isinstance(result["reject_null"], bool)


def test_observed_cvm_rejects_non_dictionary_input():
    method = ObservedCVM(test_function=matrix_product_statistic)

    with pytest.raises(ValueError, match="Invalid data format"):
        method.fit(np.eye(3))

