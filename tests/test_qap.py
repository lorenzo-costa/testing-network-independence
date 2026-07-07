import importlib

import numpy as np
import pytest

from src.methods.QAP import QAP


def symmetric_matrix(seed, n=8):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, n))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    return matrix


def test_module_imports_qap():
    module = importlib.import_module("src.methods.QAP")

    assert module.QAP is QAP


def test_qap_runs_permutations_on_adjacency_matrices_by_itself():
    method = QAP(npermutations=4, rng=np.random.default_rng(8))

    method.fit({"A": symmetric_matrix(60), "B": symmetric_matrix(61)})
    result = method.get_estimated()

    assert method.get_name() == "QAP"
    assert len(method.permutation_distribution) == 4
    assert np.isfinite(result["test_stat"])
    assert 0.0 <= result["p-value"] <= 1.0
    assert isinstance(result["reject_null"], bool)


def test_qap_rejects_non_dictionary_input():
    method = QAP(npermutations=1)

    with pytest.raises(ValueError, match="Invalid data format"):
        method.fit(np.eye(3))

