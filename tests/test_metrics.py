import pytest
import sys
from pathlib import Path
import numpy as np
from src.dgp import GaussianNetwork
from scipy import stats
import sys
from pathlib import Path
from src.metrics import rv_coefficient


@pytest.mark.parametrize(
    "A, B, expected",
    [
        (np.array([[1,2,3], [1,2,3]]), np.array([[1,2,3], [1,2,3]]), 1),
        (np.array([[1,2,3], [1,2,3]]), np.array([[4,5,6], [4,5,6]]), 1),
        (np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]), 0),
        (np.array([[1,2, 3], [4,5,6]]), np.array([[4,5,6], [1,2,3]]), 0.5143766)  
    ]
)
def test_rv_coef(A, B, expected):
    assert rv_coefficient(A, B) == pytest.approx(expected, rel=1e-4)
