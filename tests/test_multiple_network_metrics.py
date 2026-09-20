"""Recovery metrics use each network pair and the actual concatenated X pair."""

from copy import deepcopy

import numpy as np
import pytest

from src.dgp import BernoulliNetwork, GaussianNetwork
from src.methods import CanonicalCorrelationTest, DistanceCorrelationTest, RVTest
from src.metrics import (
    AdjustedRVCoefficient,
    ComputeAll,
    FalseAcceptance,
    FalseRejection,
    MSE,
    RVCoefficient,
    Rejection,
    RelativeFrobeniusNorm,
    ReturnMetric,
    RobustRelativeProcrustesDistance,
    TrueAcceptance,
    TrueRejection,
)
from src.solvers.weighted_network import ASE


METRICS = [
    RVCoefficient(),
    AdjustedRVCoefficient(),
    MSE(),
    RelativeFrobeniusNorm(),
    RelativeFrobeniusNorm(gram_matrix=True),
    RobustRelativeProcrustesDistance(),
]
OUTCOMES = [FalseRejection(), FalseAcceptance(), TrueRejection(), TrueAcceptance()]


@pytest.fixture
def results():
    rng = np.random.default_rng(230)
    truth_y = rng.normal(size=(10, 3))
    truth_x = [rng.normal(size=(10, 2)) * scale for scale in (1, 3, 10)]
    est_y = truth_y + rng.normal(size=truth_y.shape)
    est_x = [x + (i + 1) * rng.normal(size=x.shape) for i, x in enumerate(truth_x)]
    return {
        "estimated_latent": {
            "Y": est_y,
            "X": np.concatenate(est_x, axis=1),
            "X_blocks": est_x,
        },
        "true_latent": {
            "Y": truth_y,
            "X": np.concatenate(truth_x, axis=1),
            "X_blocks": truth_x,
        },
        "reject_null": True,
        "test_stat": 1.2,
        "p-value": 0.01,
    }


def single_value(metric, estimated, truth):
    value = metric({"estimated_latent": estimated, "true_latent": truth})
    return value[0] if isinstance(value, list) else value


@pytest.mark.parametrize("metric", METRICS)
def test_each_pair_and_global_match_single_matrix_calculations(metric, results):
    original = deepcopy(results)
    actual = metric(results)
    assert list(actual) == ["Y", "X_1", "X_2", "X_3", "X_global"]
    est, true = results["estimated_latent"], results["true_latent"]
    assert actual["Y"] == pytest.approx(single_value(metric, est["Y"], true["Y"]))
    for i in range(3):
        assert actual[f"X_{i + 1}"] == pytest.approx(
            single_value(metric, est["X_blocks"][i], true["X_blocks"][i])
        )
    assert actual["X_global"] == pytest.approx(
        single_value(
            metric,
            np.concatenate(est["X_blocks"], axis=1),
            np.concatenate(true["X_blocks"], axis=1),
        )
    )
    for key in ("estimated_latent", "true_latent"):
        for name in ("Y", "X"):
            np.testing.assert_array_equal(results[key][name], original[key][name])
        for before, after in zip(original[key]["X_blocks"], results[key]["X_blocks"]):
            np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("gram_matrix", [False, True])
def test_global_frobenius_is_not_average_network_error(results, gram_matrix):
    metric = RelativeFrobeniusNorm(gram_matrix=gram_matrix)
    actual = metric(results)
    est, true = results["estimated_latent"]["X"], results["true_latent"]["X"]
    if gram_matrix:
        est, true = est @ est.T, true @ true.T
    expected = np.linalg.norm(est - true, "fro") / np.linalg.norm(true, "fro")
    assert actual["X_global"] == pytest.approx(expected)
    assert actual["X_global"] != pytest.approx(
        np.mean([actual[f"X_{i}"] for i in (1, 2, 3)])
    )


@pytest.mark.parametrize("metric", METRICS)
def test_named_x_lists_are_accepted_and_keep_network_order(metric, results):
    expected = metric(results)
    for values in (results["estimated_latent"], results["true_latent"]):
        values["X"] = values.pop("X_blocks")
    assert metric(results) == pytest.approx(expected)


@pytest.mark.parametrize("metric", METRICS)
def test_one_x_network_has_both_individual_and_global_errors(metric, results):
    for values in (results["estimated_latent"], results["true_latent"]):
        values["X_blocks"] = values["X_blocks"][:1]
        values["X"] = values["X_blocks"][0]
    actual = metric(results)
    assert set(actual) == {"Y", "X_1", "X_global"}
    assert actual["X_1"] == pytest.approx(actual["X_global"])


@pytest.mark.parametrize("metric", METRICS)
@pytest.mark.parametrize("missing", [None, {"Y": None, "X": None, "X_blocks": None}])
def test_missing_truth_returns_nan_for_every_requested_pair(metric, results, missing):
    results["true_latent"] = missing
    actual = metric(results)
    assert set(actual) == {"Y", "X_1", "X_2", "X_3", "X_global"}
    assert all(np.isnan(value) for value in actual.values())


@pytest.mark.parametrize("metric", METRICS)
def test_missing_one_truth_block_only_invalidates_it_and_global(metric, results):
    results["true_latent"]["X_blocks"][1] = None
    actual = metric(results)
    assert np.isnan(actual["X_2"])
    assert np.isnan(actual["X_global"])
    assert all(np.isfinite(actual[name]) for name in ("Y", "X_1", "X_3"))


@pytest.mark.parametrize("metric", METRICS)
def test_nonfinite_block_is_nan_without_hiding_other_network_errors(metric, results):
    results["estimated_latent"]["X_blocks"][0][0, 0] = np.nan
    actual = metric(results)
    assert np.isnan(actual["X_1"])
    assert np.isnan(actual["X_global"])
    assert all(np.isfinite(actual[name]) for name in ("Y", "X_2", "X_3"))


@pytest.mark.parametrize("metric", METRICS)
def test_block_count_mismatch_is_not_silently_truncated(metric, results):
    results["true_latent"]["X_blocks"].pop()
    with pytest.raises(ValueError, match="same number of blocks"):
        metric(results)


@pytest.mark.parametrize("metric", METRICS)
def test_without_block_metadata_no_network_boundaries_are_guessed(metric, results):
    for values in (results["estimated_latent"], results["true_latent"]):
        del values["X_blocks"]
    assert set(metric(results)) == {"Y", "X_global"}


@pytest.mark.parametrize("gram_matrix", [False, True])
def test_compute_all_has_flat_pair_and_global_fields(results, gram_matrix):
    actual = ComputeAll(gram_matrix)(results, is_null=False)
    expected_keys = {
        "Rejection",
        "FalseRejection",
        "FalseAcceptance",
        "TrueRejection",
        "TrueAcceptance",
    }
    for prefix, metric in (
        ("RelativeFrobeniusNorm", RelativeFrobeniusNorm(gram_matrix)),
        ("ProcrustesDistance", RobustRelativeProcrustesDistance()),
    ):
        for name, value in metric(results).items():
            key = f"{prefix}_{name}"
            expected_keys.add(key)
            assert actual[key] == pytest.approx(value)
    assert set(actual) == expected_keys
    assert all(np.isscalar(value) for value in actual.values())
    assert actual["TrueRejection"] is True
    assert actual["FalseRejection"] is False


def test_compute_all_keeps_known_rejection_when_null_or_truth_is_missing(results):
    results["true_latent"] = None
    actual = ComputeAll()(results)
    assert actual["Rejection"] is True
    assert all(np.isnan(value) for key, value in actual.items() if key != "Rejection")


@pytest.mark.parametrize("metric", OUTCOMES)
@pytest.mark.parametrize("missing", [None, np.nan])
def test_missing_null_or_rejection_returns_nan(metric, missing):
    assert np.isnan(metric({"reject_null": True}, is_null=missing))
    assert np.isnan(metric({"reject_null": missing}, is_null=True))


@pytest.mark.parametrize("is_null", [True, False, np.bool_(True), np.bool_(False)])
@pytest.mark.parametrize("reject", [True, False, np.bool_(True), np.bool_(False)])
def test_testing_indicators_accept_python_and_numpy_booleans(is_null, reject):
    results = {"reject_null": reject}
    assert Rejection()(results, is_null=is_null) == bool(reject)
    for metric, null_value, rejection_value in (
        (FalseRejection(), True, True),
        (FalseAcceptance(), False, False),
        (TrueRejection(), False, True),
        (TrueAcceptance(), True, False),
    ):
        assert metric(results, is_null=is_null) == (
            bool(is_null) == null_value and bool(reject) == rejection_value
        )


def test_return_metric_uses_new_truth_and_preserves_all_estimated_blocks(results):
    actual = ReturnMetric()(results, is_null=False)
    np.testing.assert_array_equal(actual["Y"], results["true_latent"]["Y"])
    np.testing.assert_array_equal(actual["X"], results["true_latent"]["X"])
    assert actual["estimated"] is results["estimated_latent"]
    assert actual["truth"] is results["true_latent"]
    for selector in ("Y", "X", "test_stat", "p-value", "estimated", "truth"):
        value = ReturnMetric(selector)(results)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, actual[selector])
        else:
            assert value == actual[selector]


@pytest.mark.parametrize("network_class", [GaussianNetwork, BernoulliNetwork])
@pytest.mark.parametrize(
    "method_class", [RVTest, CanonicalCorrelationTest, DistanceCorrelationTest]
)
@pytest.mark.parametrize("mode", ["latent", "adjacency"])
def test_dgp_method_and_metrics_integration(network_class, method_class, mode):
    data = network_class(16, 3, 1, 2, rng=np.random.default_rng(231)).generate()
    method = method_class(
        solver=ASE,
        npermutations=2,
        permutation_type=mode,
        rng=np.random.default_rng(232),
    )
    method.fit(data)
    results = method.get_estimated()
    metrics = ComputeAll()(results)
    for prefix in ("RelativeFrobeniusNorm", "ProcrustesDistance"):
        for name in ("Y", "X_1", "X_2", "X_3", "X_global"):
            assert np.isfinite(metrics[f"{prefix}_{name}"])
    assert np.isnan(metrics["TrueRejection"])
    assert metrics["Rejection"] == results["reject_null"]


def test_networks_without_true_positions_produce_nan_recovery_metrics():
    data = GaussianNetwork(12, 2, 1, 2, rng=np.random.default_rng(233)).generate()
    method = RVTest(solver=ASE, d_y=2, d_x=1, npermutations=2)
    method.fit([data["A_Y"], *data["A_X"]])
    actual = ComputeAll()(method.get_estimated())
    assert isinstance(actual["Rejection"], bool)
    assert all(np.isnan(value) for name, value in actual.items() if name != "Rejection")
