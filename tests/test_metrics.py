import pytest
import sys
from pathlib import Path
import numpy as np
from src.dgp import GaussianNetwork
from scipy import stats
import sys
from pathlib import Path
from src.metrics import rv_coefficient, Rejection, TrueRejection, FalseRejection


@pytest.mark.parametrize(
    "A, B, expected",
    [
        (np.array([[1, 2, 3], [1, 2, 3]]), np.array([[1, 2, 3], [1, 2, 3]]), 1),
        (np.array([[1, 2, 3], [1, 2, 3]]), np.array([[4, 5, 6], [4, 5, 6]]), 1),
        (np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]), 0),
        (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[4, 5, 6], [1, 2, 3]]), 0.5143766),
    ],
)
def test_rv_coef(A, B, expected):
    assert rv_coefficient(A, B) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize(
    "truth, estimated", [(True, True), (False, True), (True, False), (False, False)]
)
def test_rejection(truth, estimated):
    metric = Rejection()
    assert metric(truth, estimated) == estimated


@pytest.mark.parametrize(
    "truth, estimated", [(True, True), (False, True), (True, False), (False, False)]
)
def test_false_rejection(truth, estimated):
    metric = FalseRejection()
    if estimated == True:
        if truth == False:
            expected = True
        else:
            expected = False
    else:
        expected = False
    assert metric(truth, estimated) == expected


@pytest.mark.parametrize(
    "truth, estimated", [(True, True), (False, True), (True, False), (False, False)]
)
def test_true_rejection(truth, estimated):
    metric = TrueRejection()
    if estimated == True:
        if truth == True:
            expected = True
        else:
            expected = False
    else:
        expected = False
    assert metric(truth, estimated) == expected
