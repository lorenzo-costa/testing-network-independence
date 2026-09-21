import numpy as np
import pytest

from src.load_config import _resolve_method
from src.methods import MRQAP, QAP


def _symmetric(array):
    array = np.asarray(array, dtype=float)
    return (array + array.T) / 2


def _network_data(seed=12, n=7):
    rng = np.random.default_rng(seed)
    tested = _symmetric(rng.normal(size=(n, n)))
    control_1 = _symmetric(rng.normal(size=(n, n)))
    control_2 = _symmetric(rng.normal(size=(n, n)))
    noise = _symmetric(rng.normal(scale=0.2, size=(n, n)))
    outcome = 0.7 * tested - 0.3 * control_1 + 0.2 * control_2 + noise
    controls = np.stack((control_1, control_2), axis=2)
    return {"A": tested, "Y": outcome, "X": controls}


def _global_network_data(seed=112, n=8, p=3):
    rng = np.random.default_rng(seed)
    predictors = [_symmetric(rng.normal(size=(n, n))) for _ in range(p)]
    noise = _symmetric(rng.normal(scale=0.3, size=(n, n)))
    outcome = 0.8 * predictors[0] - 0.45 * predictors[-1] + noise
    return {"A_Y": outcome, "A_X": predictors}


def _upper(network):
    indices = np.triu_indices(network.shape[0], k=1)
    return network[indices]


def _manual_omnibus_f(outcome, predictors, include_intercept=True):
    y = _upper(outcome)
    x = np.column_stack([_upper(network) for network in predictors])
    design = np.column_stack((np.ones(y.size), x)) if include_intercept else x
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    full_rss = np.sum((y - design @ coefficients) ** 2)
    reduced_residuals = y - y.mean() if include_intercept else y
    reduced_rss = np.sum(reduced_residuals**2)
    numerator_df = x.shape[1]
    denominator_df = y.size - design.shape[1]
    return ((reduced_rss - full_rss) / numerator_df) / (full_rss / denominator_df)


def test_qap_uses_finite_permutation_pvalue_correction():
    rng = np.random.default_rng(4)
    data = {
        "A": _symmetric(rng.normal(size=(8, 8))),
        "B": _symmetric(rng.normal(size=(8, 8))),
    }
    method = QAP(npermutations=9, rng=np.random.default_rng(18))

    method.fit(data)

    null = np.asarray(method.permutation_distribution)
    extreme = np.count_nonzero(np.abs(null) >= abs(method.test_stat_estimate))
    assert method.pvalue == (extreme + 1) / 10
    assert method.pvalue >= 0.1


def test_y_permutation_matches_manual_ols_and_corrected_pvalue():
    data = _network_data()
    method = MRQAP(
        npermutations=8,
        rng=np.random.default_rng(81),
        permutation_strategy="y_permutation",
    )

    method.fit(data)

    design = np.column_stack(
        (
            np.ones(_upper(data["A"]).size),
            _upper(data["X"][:, :, 0]),
            _upper(data["X"][:, :, 1]),
            _upper(data["A"]),
        )
    )
    expected_observed = np.linalg.lstsq(design, _upper(data["Y"]), rcond=None)[0][-1]
    expected_null = []
    for permutation in method.permutation_indices:
        permuted_y = data["Y"][permutation][:, permutation]
        expected_null.append(
            np.linalg.lstsq(design, _upper(permuted_y), rcond=None)[0][-1]
        )

    np.testing.assert_allclose(method.test_stat_estimate, expected_observed)
    np.testing.assert_allclose(method.permutation_distribution, expected_null)
    extreme = np.count_nonzero(np.abs(expected_null) >= abs(expected_observed))
    assert method.pvalue == (extreme + 1) / 9
    assert isinstance(method.test_stat_estimate, float)


def test_dsp_matches_double_semipartialling_formula():
    data = _network_data(seed=23)
    method = MRQAP(
        npermutations=7,
        rng=np.random.default_rng(91),
        permutation_strategy="dsp",
    )

    method.fit(data)

    controls = np.column_stack(
        (
            np.ones(_upper(data["A"]).size),
            _upper(data["X"][:, :, 0]),
            _upper(data["X"][:, :, 1]),
        )
    )
    full_design = np.column_stack((controls, _upper(data["A"])))
    expected_observed = np.linalg.lstsq(full_design, _upper(data["Y"]), rcond=None)[0][
        -1
    ]

    q, _ = np.linalg.qr(controls, mode="reduced")

    def residualize(value):
        return value - q @ (q.T @ value)

    tested_residuals = residualize(_upper(data["A"]))
    outcome_residuals = residualize(_upper(data["Y"]))
    residual_network = np.zeros_like(data["A"])
    upper_indices = np.triu_indices(data["A"].shape[0], k=1)
    residual_network[upper_indices] = tested_residuals
    residual_network[(upper_indices[1], upper_indices[0])] = tested_residuals

    expected_null = []
    for permutation in method.permutation_indices:
        permuted = residual_network[permutation][:, permutation]
        u = residualize(_upper(permuted))
        expected_null.append((u @ outcome_residuals) / (u @ u))

    np.testing.assert_allclose(method.test_stat_estimate, expected_observed)
    np.testing.assert_allclose(method.permutation_distribution, expected_null)


def test_global_y_permutation_matches_manual_omnibus_f_and_pvalue():
    data = _global_network_data()
    method = MRQAP(
        npermutations=9,
        rng=np.random.default_rng(181),
        permutation_strategy="y_permutation",
    )

    method.fit(data)

    expected_observed = _manual_omnibus_f(data["A_Y"], data["A_X"])
    expected_null = []
    for permutation in method.permutation_indices:
        permuted_y = data["A_Y"][permutation][:, permutation]
        expected_null.append(_manual_omnibus_f(permuted_y, data["A_X"]))

    assert method.global_test is True
    assert method.numerator_df == len(data["A_X"])
    assert method.denominator_df == _upper(data["A_Y"]).size - len(data["A_X"]) - 1
    assert method.observed_coefficients.shape == (len(data["A_X"]),)
    assert method.test_stat_estimate == pytest.approx(expected_observed)
    np.testing.assert_allclose(method.permutation_distribution, expected_null)
    extreme = np.count_nonzero(expected_null >= expected_observed)
    assert method.pvalue == (extreme + 1) / 10


def test_global_test_without_intercept_matches_manual_omnibus_f():
    data = _global_network_data(seed=113, n=7, p=2)
    method = MRQAP(
        npermutations=5,
        include_intercept=False,
        rng=np.random.default_rng(182),
    )

    method.fit(data)

    assert method.test_stat_estimate == pytest.approx(
        _manual_omnibus_f(data["A_Y"], data["A_X"], include_intercept=False)
    )
    assert method.denominator_df == _upper(data["A_Y"]).size - len(data["A_X"])


def test_global_mode_ignores_optional_latent_truth_metadata():
    data = _global_network_data(seed=114, n=7, p=2)
    data["Y"] = np.arange(14, dtype=float).reshape(7, 2)
    data["X"] = [np.ones((7, 1)), np.zeros((7, 1))]
    original_y = data["A_Y"].copy()
    original_x = [network.copy() for network in data["A_X"]]
    method = MRQAP(npermutations=3, rng=np.random.default_rng(183))

    method.fit(data)

    assert method.global_test is True
    assert method.X is None
    np.testing.assert_array_equal(data["A_Y"], original_y)
    for actual, original in zip(data["A_X"], original_x):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("strategy", ["y_permutation", "dsp"])
def test_batching_does_not_change_results(strategy):
    data = _network_data(seed=31, n=8)
    unbatched = MRQAP(
        npermutations=11,
        rng=np.random.default_rng(10),
        permutation_strategy=strategy,
        batch_size=None,
    )
    batched = MRQAP(
        npermutations=11,
        rng=np.random.default_rng(10),
        permutation_strategy=strategy,
        batch_size=4,
    )

    unbatched.fit(data)
    batched.fit(data)

    np.testing.assert_allclose(
        unbatched.permutation_distribution, batched.permutation_distribution
    )
    assert unbatched.test_stat_estimate == pytest.approx(batched.test_stat_estimate)
    assert unbatched.pvalue == batched.pvalue


def test_global_batching_does_not_change_results():
    data = _global_network_data(seed=115, n=8, p=3)
    unbatched = MRQAP(
        npermutations=11,
        rng=np.random.default_rng(184),
        batch_size=None,
    )
    batched = MRQAP(
        npermutations=11,
        rng=np.random.default_rng(184),
        batch_size=4,
    )

    unbatched.fit(data)
    batched.fit(data)

    np.testing.assert_allclose(
        unbatched.permutation_distribution, batched.permutation_distribution
    )
    assert unbatched.test_stat_estimate == pytest.approx(batched.test_stat_estimate)
    assert unbatched.pvalue == batched.pvalue


def test_node_inputs_create_requested_distance_networks_and_a_takes_precedence():
    rng = np.random.default_rng(42)
    n = 6
    A = _symmetric(rng.normal(size=(n, n)))
    Y = rng.normal(size=(n, 2))
    X = rng.normal(size=(n, 3))
    ignored_z = rng.normal(size=(n + 2, 4))
    method = MRQAP(npermutations=3, rng=np.random.default_rng(7))

    method.fit({"A": A, "Z": ignored_z, "Y": Y, "X": X})

    expected_y = np.linalg.norm(Y[:, None, :] - Y[None, :, :], axis=2)
    assert method.test_network is not ignored_z
    np.testing.assert_array_equal(method.test_network, A)
    np.testing.assert_allclose(method.outcome_network, expected_y)
    assert method.control_networks.shape == (n, n, 3)
    for column in range(X.shape[1]):
        expected_x = np.abs(X[:, None, column] - X[None, :, column])
        np.testing.assert_allclose(method.control_networks[:, :, column], expected_x)


def test_node_valued_z_is_used_when_a_is_absent():
    rng = np.random.default_rng(15)
    Z = rng.normal(size=(7, 2))
    Y = rng.normal(size=7)
    method = MRQAP(npermutations=3, rng=np.random.default_rng(16))

    method.fit({"Z": Z, "Y": Y})

    expected_z = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    np.testing.assert_allclose(method.test_network, expected_z)
    assert method.control_networks.shape == (7, 7, 0)


def test_directed_mode_uses_all_ordered_off_diagonal_dyads_without_intercept():
    rng = np.random.default_rng(8)
    n = 5
    A = rng.normal(size=(n, n))
    Y = 1.3 * A + rng.normal(scale=0.1, size=(n, n))
    X = rng.normal(size=(n, n))
    method = MRQAP(
        npermutations=3,
        rng=np.random.default_rng(9),
        symmetric=False,
        include_intercept=False,
    )

    method.fit({"A": A, "Y": Y, "X": X})

    mask = ~np.eye(n, dtype=bool)
    design = np.column_stack((X[mask], A[mask]))
    expected = np.linalg.lstsq(design, Y[mask], rcond=None)[0][-1]
    assert method._row_indices.size == n * (n - 1)
    assert method.test_stat_estimate == pytest.approx(expected)


def test_diagonal_entries_are_excluded():
    data = _network_data(seed=72)
    changed = {key: value.copy() for key, value in data.items()}
    np.fill_diagonal(changed["A"], 1e6)
    np.fill_diagonal(changed["Y"], -1e6)
    for control in range(changed["X"].shape[2]):
        np.fill_diagonal(changed["X"][:, :, control], 5e5)

    original_method = MRQAP(npermutations=5, rng=np.random.default_rng(80))
    changed_method = MRQAP(npermutations=5, rng=np.random.default_rng(80))
    original_method.fit(data)
    changed_method.fit(changed)

    assert original_method.test_stat_estimate == pytest.approx(
        changed_method.test_stat_estimate
    )
    np.testing.assert_allclose(
        original_method.permutation_distribution,
        changed_method.permutation_distribution,
    )


def test_degenerate_regression_raises_clear_error():
    n = 6
    constant_network = np.ones((n, n)) - np.eye(n)
    with pytest.raises(ValueError, match="degenerate"):
        MRQAP(npermutations=3).fit(
            {"A": constant_network, "Y": np.arange(n, dtype=float)}
        )


def test_global_degenerate_regression_raises_clear_error():
    data = _global_network_data(seed=116, n=7, p=2)
    data["A_X"][1] = data["A_X"][0].copy()

    with pytest.raises(
        ValueError, match="global MRQAP regression design is degenerate"
    ):
        MRQAP(npermutations=3).fit(data)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"A_Y": np.eye(4)}, "supplied together"),
        ({"A_X": [np.eye(4)]}, "supplied together"),
        ({"A_Y": np.eye(4), "A_X": []}, "nonempty list"),
        ({"A_Y": np.ones((4, 3)), "A_X": [np.eye(4)]}, "square matrix"),
        (
            {"A_Y": np.eye(4), "A_X": [np.eye(5)]},
            "matching A_Y",
        ),
    ],
)
def test_global_input_validation(data, message):
    with pytest.raises(ValueError, match=message):
        MRQAP(npermutations=3).fit(data)


def test_global_test_rejects_dsp_permutation_strategy():
    with pytest.raises(ValueError, match="only.*y_permutation"):
        MRQAP(npermutations=3, permutation_strategy="dsp").fit(
            _global_network_data(seed=117)
        )


def test_mrqap_is_resolvable_from_simulation_configuration():
    resolved = _resolve_method(
        {
            "name": "MRQAP",
            "kwargs": {
                "permutation_strategy": "dsp",
                "batch_size": 2,
            },
        }
    )

    method = resolved(npermutations=3, rng=np.random.default_rng(2))

    assert isinstance(method, MRQAP)
    assert method.permutation_strategy == "dsp"
    assert method.batch_size == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"permutation_strategy": "unknown"}, "permutation_strategy"),
        ({"batch_size": 0}, "batch_size"),
        ({"npermutations": 0}, "npermutations"),
    ],
)
def test_mrqap_rejects_invalid_constructor_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MRQAP(**kwargs)
