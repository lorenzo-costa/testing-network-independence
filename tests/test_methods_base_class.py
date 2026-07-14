import importlib

import numpy as np
import pytest

from src.methods._base_class import BaseEstimationMethod, BaseMethod, BasePermutationTest


def dummy_solver(matrix, k, rng=None):
    values = np.asarray(matrix, dtype=float)
    return values[:, :k].copy(), np.ones(k)


def test_module_imports_base_classes():
    module = importlib.import_module("src.methods._base_class")

    assert module.BaseMethod is BaseMethod
    assert module.BaseEstimationMethod is BaseEstimationMethod
    assert module.BasePermutationTest is BasePermutationTest


def test_base_method_requires_subclasses_to_implement_interface():
    method = BaseMethod()

    with pytest.raises(NotImplementedError, match="Subclasses should implement this"):
        method.fit({})
    with pytest.raises(NotImplementedError, match="Subclasses should implement this"):
        method.get_name()


def test_base_method_returns_standard_result_structure():
    method = BaseMethod()
    method.Xhat = np.ones((3, 1))
    method.Zhat = np.zeros((3, 1))
    method.X = None
    method.Z = None
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
    assert result["p-value"] == 0.5
    assert result["reject_null"] is False


def test_estimation_base_requires_solver():
    with pytest.raises(ValueError, match="Solver must be provided"):
        BaseEstimationMethod()


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"solver": None, "test_function": lambda x, y: 0.0}, "Solver must be provided"),
        ({"solver": dummy_solver, "test_function": None}, "Test function must be provided"),
    ],
)
def test_permutation_base_requires_components(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BasePermutationTest(**kwargs)


def test_estimation_base_rejects_non_dictionary_input():
    method = BaseEstimationMethod(solver=dummy_solver, k=2)

    with pytest.raises(ValueError, match="Invalid data format"):
        method._process_input(np.zeros((3, 3)))

