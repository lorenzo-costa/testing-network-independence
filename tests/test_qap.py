import numpy as np
import pytest

from src.dgp import BernoulliNetwork, GaussianNetwork
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


def _upper(network):
    indices = np.triu_indices(network.shape[0], k=1)
    return network[indices]


def _global_qap_data(seed=4, n=14):
    rng = np.random.default_rng(seed)
    ax_1 = _symmetric(rng.normal(size=(n, n)))
    ax_2 = _symmetric(rng.normal(size=(n, n)))
    noise = _symmetric(rng.normal(scale=0.4, size=(n, n)))
    ay = 0.8 * ax_1 - 0.5 * ax_2 + noise
    return {"Ay": ay, "Ax": [ax_1, ax_2]}


def test_qap_fits_all_ax_and_uses_studentized_global_wald_statistic():
    data = _global_qap_data()
    method = QAP(npermutations=5, rng=np.random.default_rng(18))

    method.fit(data)

    mask = ~np.eye(data["Ay"].shape[0], dtype=bool)
    design = np.column_stack([network[mask] for network in data["Ax"]])
    design = np.column_stack((np.ones(mask.sum()), design))
    expected_coefficients = np.linalg.lstsq(
        design, data["Ay"][mask], rcond=None
    )[0][1:]

    np.testing.assert_allclose(method.coefficients, expected_coefficients)
    assert method.coefficient_covariance.shape == (2, 2)
    np.testing.assert_allclose(
        method.coefficient_covariance,
        method.coefficient_covariance.T,
    )
    assert method.test_stat_estimate >= 0


def test_qap_permutes_only_ay_and_uses_finite_pvalue_correction():
    data = _global_qap_data(seed=9)
    method = QAP(npermutations=9, rng=np.random.default_rng(18))

    method.fit(data)

    expected_null = [
        method._compute_studentized_wald(
            data["Ay"][permutation][:, permutation],
            data["Ax"],
        )[0]
        for permutation in method.permutation_indices
    ]
    np.testing.assert_allclose(method.permutation_distribution, expected_null)

    null = np.asarray(method.permutation_distribution)
    extreme = np.count_nonzero(null >= method.test_stat_estimate)
    assert method.pvalue == (extreme + 1) / 10
    assert method.pvalue >= 0.1


@pytest.mark.parametrize("network_type", [GaussianNetwork, BernoulliNetwork])
def test_qap_runs_on_dgp_output(network_type):
    data = network_type(
        n=20,
        k=[2, 1, 3],
        rng=np.random.default_rng(31),
    ).generate()
    method = QAP(npermutations=4, rng=np.random.default_rng(32))

    method.fit(data)

    assert method.coefficients.shape == (2,)
    assert len(method.permutation_distribution) == 4
    assert 0.0 <= method.pvalue <= 1.0
    assert isinstance(method.reject_null, bool)


def test_qap_ignores_diagonal_entries():
    data = _global_qap_data(seed=17)
    changed = {
        "Ay": data["Ay"].copy(),
        "Ax": [network.copy() for network in data["Ax"]],
    }
    np.fill_diagonal(changed["Ay"], 1e8)
    for network in changed["Ax"]:
        np.fill_diagonal(network, -1e8)

    original = QAP(npermutations=4, rng=np.random.default_rng(22))
    modified = QAP(npermutations=4, rng=np.random.default_rng(22))
    original.fit(data)
    modified.fit(changed)

    assert original.test_stat_estimate == pytest.approx(modified.test_stat_estimate)
    np.testing.assert_allclose(
        original.permutation_distribution,
        modified.permutation_distribution,
    )


def test_qap_rejects_directed_or_degenerate_networks():
    data = _global_qap_data(seed=25)
    directed = {"Ay": data["Ay"].copy(), "Ax": data["Ax"]}
    directed["Ay"][0, 1] += 1

    with pytest.raises(ValueError, match="Ay must be symmetric"):
        QAP(npermutations=2).fit(directed)

    with pytest.raises(ValueError, match="degenerate"):
        QAP(npermutations=2).fit(
            {"Ay": data["Ay"], "Ax": [data["Ax"][0], data["Ax"][0]]}
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"alpha": 0}, "alpha"),
        ({"npermutations": 0}, "npermutations"),
    ],
)
def test_qap_rejects_invalid_constructor_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        QAP(**kwargs)


def test_qap_is_resolvable_from_simulation_configuration():
    resolved = _resolve_method({"name": "QAP"})
    method = resolved(npermutations=3, rng=np.random.default_rng(2))

    assert isinstance(method, QAP)


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
