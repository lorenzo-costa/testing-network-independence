import pytest
import sys
from pathlib import Path
import numpy as np
from src.dgp import GaussianNetwork
from scipy import stats
import sys
from pathlib import Path
from src.metrics import (
    rv_coefficient,
    Rejection,
    TrueRejection,
    FalseRejection,
    RelativeFrobeniusNorm,
)


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
    "reject_null, null", [(True, True), (False, True), (True, False), (False, False)]
)
def test_rejection(reject_null, null):
    results = {"reject_null": reject_null, "null": null}
    assert Rejection()(results) == reject_null


@pytest.mark.parametrize(
    "reject_null, null", [(True, True), (False, True), (True, False), (False, False)]
)
def test_false_rejection(reject_null, null):
    results = {"reject_null": reject_null, "null": null}
    if null == True:
        if reject_null == True:
            expected = True
        else:
            expected = False
    else:
        expected = False
    assert FalseRejection()(results) == expected


@pytest.mark.parametrize(
    "reject_null, null", [(True, True), (False, True), (True, False), (False, False)]
)
def test_true_rejection(reject_null, null):
    results = {"reject_null": reject_null, "null": null}
    if null == False:
        if reject_null == True:
            expected = True
        else:
            expected = False
    else:
        expected = False
    assert TrueRejection()(results) == expected


class TestRelativeFrobeniusNorm:
    @pytest.fixture
    def simple_data(self):
        # Create simple matrices for easy manual verification
        # Truth: Identity matrix (2x2)
        true_latent = np.array([[1.0, 0.0], [0.0, 1.0]])
        # Estimate: Identity + perturbation
        est_latent = np.array([[1.1, 0.0], [0.0, 1.1]])
        return true_latent, est_latent

    def test_single_input_no_gram(self, simple_data):
        """
        Test standard calculation without Gram matrix.
        Formula: ||Est - True|| / ||True||
        """
        true_latent, est_latent = simple_data
        metric = RelativeFrobeniusNorm(gram_matrix=False)

        results = {"estimated_latent": est_latent, "true_latent": true_latent}

        # Manual calculation
        diff = est_latent - true_latent  # [[0.1, 0], [0, 0.1]]
        expected_num = np.linalg.norm(diff, "fro")
        expected_den = np.linalg.norm(true_latent, "fro")
        expected_val = expected_num / expected_den

        calculated_val = metric(results)

        assert np.isclose(calculated_val, expected_val), (
            f"Expected {expected_val}, got {calculated_val}"
        )

    def test_single_input_with_gram(self):
        """
        Test calculation WITH Gram matrix transformation.
        Formula: ||Est@Est.T - True@True.T|| / ||True@True.T||
        """
        # Define matrices where Gram matrix makes a difference
        # X = [[1, 0]], XX^T = [[1]]
        true_latent = np.array([[1.0, 0.0]])
        # Est = [[2, 0]], EstEst^T = [[4]]
        est_latent = np.array([[2.0, 0.0]])

        metric = RelativeFrobeniusNorm(gram_matrix=True)
        results = {"estimated_latent": est_latent, "true_latent": true_latent}

        # Manual Calculation
        # True Gram: [[1]]
        # Est Gram: [[4]]
        # Diff: 3
        # Norm(Diff): 3
        # Norm(True Gram): 1
        # Result: 3.0

        val = metric(results)
        assert np.isclose(val, 3.0)

    def test_tuple_input_multiple_networks(self):
        """Test handling of tuple inputs (looping over multiple latent positions)."""
        # Network 1
        t1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        e1 = np.array([[1.0, 0.0], [0.0, 1.0]])  # Perfect match

        # Network 2
        t2 = np.array([[2.0]])
        e2 = np.array([[4.0]])  # Double the value

        results = {"estimated_latent": (e1, e2), "true_latent": (t1, t2)}

        metric = RelativeFrobeniusNorm(gram_matrix=False)
        output = metric(results)

        assert isinstance(output, list)
        assert len(output) == 2
        assert output[0] == 0.0  # Perfect match should be 0 error
        assert np.isclose(output[1], 1.0)  # (4-2)/2 = 1.0

    def test_tuple_input_with_gram(self):
        """Test tuple inputs combined with gram_matrix=True."""
        t1 = np.array([[1.0, 0.0]])
        e1 = np.array([[2.0, 0.0]])

        results = {
            "estimated_latent": (e1, e1),  # Using same data twice for simplicity
            "true_latent": (t1, t1),
        }

        metric = RelativeFrobeniusNorm(gram_matrix=True)
        output = metric(results)

        # Based on previous Gram test, result should be 3.0 for both
        assert np.allclose(output, [3.0, 3.0])

    def test_zero_denominator_safe_handling(self):
        """
        Test edge case where true_latent is all zeros.
        Code should handle division by zero by returning 0.
        """
        true_latent = np.zeros((2, 2))
        est_latent = np.ones((2, 2))

        results = {"estimated_latent": est_latent, "true_latent": true_latent}

        metric = RelativeFrobeniusNorm()
        val = metric(results)

        assert val == 0

    def test_perfect_match(self):
        """Test that identical matrices return 0.0 error."""
        arr = np.random.rand(5, 5)
        results = {"estimated_latent": arr, "true_latent": arr}

        metric = RelativeFrobeniusNorm()
        assert metric(results) == 0.0
