import importlib
from pathlib import Path

import numpy as np
import pytest

from src.methods.ac_test import EstimateAC, MultivariateACTest
from src.methods import ConditionalGenerator


ROOT = Path(__file__).resolve().parents[1]


def dummy_solver(matrix, k, rng=None):
    values = np.asarray(matrix, dtype=float)
    return values[:, :k].copy(), np.ones(k)


def latent_data(seed=10):
    rng = np.random.default_rng(seed)
    return {
        "Y": rng.normal(size=(14, 2)),
        "Z": rng.normal(size=(14, 2)),
    }


def test_module_imports_ac_methods():
    module = importlib.import_module("src.methods.ac_test")

    assert module.EstimateAC is EstimateAC
    assert module.MultivariateACTest is MultivariateACTest


def test_config_resolver_forwards_adaptive_m_option():
    from src.load_config import _resolve_method
    resolved = _resolve_method(
        {
            "name": "MultivariateACTest",
            "kwargs": {
                "adaptive_m": True,
                "M": 99,
                "use_permutation_coeff": True,
            },
        }
    )

    assert resolved.func is MultivariateACTest
    assert resolved.keywords == {
        "adaptive_m": True,
        "M": 99,
        "use_permutation_coeff": True,
    }


def test_old_approximation_key_is_rejected_by_config_resolver():
    from src.load_config import _resolve_method
    with pytest.raises(ValueError, match="use 'test_type'"):
        _resolve_method({"name": "MultivariateACTest", "kwargs": {"approximation": "permutation"}})


@pytest.mark.parametrize(
    ("filename", "expected_ky", "expected_permutation"),
    [
        ("config_gaussian.yaml", [3], True),
        ("config_maema.yaml", [3], True),
        ("config_functionals.yaml", [3], True),
        ("config_null_gaussian.yaml", [3], True),
        ("config_null_bernoulli.yaml", [3], True),
    ],
)
def test_ac_configs_set_expected_response_and_coefficient_permutation(
    filename,
    expected_ky,
    expected_permutation,
):
    from src.load_config import load_config
    config = load_config(ROOT / filename)
    ac_methods = [
        method
        for method in config["methods"]["list"]
        if getattr(method, "func", method) is MultivariateACTest
    ]

    assert config["simulation"]["ky"] == expected_ky
    assert len(ac_methods) == 3
    assert all(
        method.keywords.get("use_permutation_coeff", False) is expected_permutation
        for method in ac_methods
    )


@pytest.mark.parametrize(
    "filename",
    ["config_gaussian.yaml", "config_maema.yaml"],
)
def test_univariate_power_configs_use_requested_grid_and_gaussian_copula(filename):
    from src.load_config import load_config
    config = load_config(ROOT / filename)

    assert config["simulation"]["n"] == [50, 100, 200, 400]
    assert config["simulation"]["rho"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert {
        setup[0].keywords["copula_model"]
        for setup in config["setups"]
    } == {"gaussian"}


def test_estimate_ac_runs_with_true_latent_positions():
    method = EstimateAC(
        solver=dummy_solver,
        use_true_latent=True,
        M=1,
        rng=np.random.default_rng(1),
    )

    method.fit(latent_data())
    result = method.get_estimated()

    assert method.get_name() == "EstimateAC"
    assert np.isfinite(result["test_stat"])
    assert result["p-value"] is None
    assert result["reject_null"] is None


def test_multivariate_ac_test_runs_permutations_by_itself():
    method = MultivariateACTest(
        solver=dummy_solver,
        use_true_latent=True,
        M=1,
        npermutations=3,
        rng=np.random.default_rng(2),
    )

    method.fit(latent_data())
    result = method.get_estimated()

    assert len(method.null_distribution) == 3
    assert np.isfinite(result["test_stat"])
    assert 0.0 <= result["p-value"] <= 1.0
    assert isinstance(result["reject_null"], bool)


def test_fixed_m_ac_test_forwards_multivariate_y_permutation_option(monkeypatch):
    calls = []

    def recording_ac(**kwargs):
        calls.append(kwargs)
        return 0.25

    monkeypatch.setattr("src.methods.ac_test.ac_coefficient", recording_ac)
    data = latent_data()
    method = MultivariateACTest(
        use_true_latent=True,
        M=1,
        npermutations=1,
        use_permutation_coeff=True,
        rng=np.random.default_rng(4),
    )
    method.fit(data)

    assert len(calls) == 2
    assert all(call["permutation"] is True for call in calls)
    assert all(call["Y"].shape[1] == 2 for call in calls)


@pytest.mark.parametrize("y_columns", [1, 2])
def test_adaptive_ac_test_runs_with_real_coefficients(y_columns):
    data = latent_data()
    data["Y"] = data["Y"][:, :y_columns]
    method = MultivariateACTest(
        use_true_latent=True,
        adaptive_m=True,
        npermutations=3,
        rng=np.random.default_rng(3),
    )

    method.fit(data)
    result = method.get_estimated()

    assert len(method.null_distribution) == 3
    assert method.adaptive_s_statistics.shape == (
        4,
        len(method.adaptive_m_values),
    )
    assert np.isfinite(result["test_stat"])
    assert 0.0 <= result["p-value"] <= 1.0
    assert method.get_name() == "MultivariateAC_PermutationTest_covariate_adaptive"


def test_multivariate_ac_requires_observed_y():
    method = MultivariateACTest(
        solver=dummy_solver,
        k=2,
        npermutations=1,
    )

    with pytest.raises(ValueError, match="Observed covariates Y must be provided"):
        method.fit({"A": np.eye(4)})


@pytest.mark.parametrize("method_type", [EstimateAC, MultivariateACTest])
def test_ac_method_passes_x_as_conditioning_variable(monkeypatch, method_type):
    recorded = []

    def recording_ac(*, Y, Z, X=None, **kwargs):
        recorded.append(X)
        return 0.25

    monkeypatch.setattr("src.methods.ac_test.ac_coefficient", recording_ac)
    data = latent_data()
    data["X"] = np.repeat([0, 1], 7).reshape(-1, 1)
    kwargs = {"npermutations": 2} if method_type is MultivariateACTest else {}
    method = method_type(use_true_latent=True, M=1, **kwargs)
    method.fit(data)

    assert recorded[0] is method.X
    np.testing.assert_array_equal(recorded[0], data["X"])
    if method_type is MultivariateACTest:
        assert len(recorded) == 3
        assert all(conditioning is method.X for conditioning in recorded)


def test_conditional_ac_model_x_uses_fixed_x_with_real_coefficient():
    class FixedGenerator(ConditionalGenerator):
        def __init__(self, responses):
            self.responses = iter(responses)
            self.x_seen = []

        def sample(self, X):
            self.x_seen.append(X.copy())
            return next(self.responses)

    data = latent_data()
    data["Y"] = data["Y"][:, :1]
    data["X"] = np.repeat([0, 1], 7).reshape(-1, 1)
    generator = FixedGenerator([data["Y"] + 0.5, data["Y"] - 0.5])
    method = MultivariateACTest(
        use_true_latent=True, M=1, npermutations=2,
        test_type="model_X", conditional_generator=generator,
        rng=np.random.default_rng(23),
    )
    method.fit(data)

    assert method.get_name() == "MultivariateAC_ModelXTest_covariate_1"
    assert len(method.null_distribution) == 2
    assert np.isfinite(method.test_stat_estimate)
    assert 0 <= method.pvalue <= 1
    assert len(generator.x_seen) == 2
    for x in generator.x_seen:
        np.testing.assert_array_equal(x, data["X"])
    np.testing.assert_array_equal(method.X, data["X"])


def test_adaptive_ac_model_x_reuses_vector_finalizer(monkeypatch):
    seen_x = []

    def deterministic_statistic(self, Z, Y, M, rng=None):
        seen_x.append(self.X.copy())
        return float(M * Y[0, 0] + Y[1, 0])

    class FixedGenerator(ConditionalGenerator):
        def __init__(self):
            self.responses = iter([
                np.array([[1.0], [2.0], *([[0.0]] * 8)]),
                np.array([[2.0], [3.0], *([[0.0]] * 8)]),
            ])

        def sample(self, X):
            return next(self.responses)

    monkeypatch.setattr(MultivariateACTest, "_adaptive_s_statistic", deterministic_statistic)
    data = {
        "Z": np.ones((10, 1)),
        "Y": np.array([[3.0], [1.0], *([[0.0]] * 8)]),
        "X": np.arange(10).reshape(-1, 1),
    }
    method = MultivariateACTest(
        use_true_latent=True, adaptive_m=True, npermutations=2,
        test_type="model_X", conditional_generator=FixedGenerator(),
    )
    method.fit(data)
    m_values = method.adaptive_m_values
    matrix = np.stack((3 * m_values + 1, m_values + 2, 2 * m_values + 3)).astype(float)
    standardized = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=1)
    maxima = standardized.max(axis=1)
    np.testing.assert_allclose(method.adaptive_s_statistics, matrix)
    np.testing.assert_allclose(method.null_distribution, maxima[1:])
    assert method.pvalue == pytest.approx((1 + np.count_nonzero(maxima[1:] >= maxima[0])) / 3)
    assert len(seen_x) == 3 * len(m_values)
    for x in seen_x:
        np.testing.assert_array_equal(x, data["X"])


def test_model_x_ac_cannot_ignore_conditioning_x():
    with pytest.raises(ValueError, match="requires the AC statistic to use"):
        MultivariateACTest(
            use_true_latent=True, test_type="model_X", _ignore_X=True,
            conditional_generator=object(),
        )


@pytest.mark.parametrize("permutation_type", ["covariate", "latent", "observed"])
def test_ac_permutations_are_restricted_within_x_strata(
    monkeypatch, permutation_type
):
    monkeypatch.setattr(
        "src.methods.ac_test.ac_coefficient", lambda **kwargs: 0.25
    )
    rng = np.random.default_rng(20)
    data = {
        "A": rng.normal(size=(12, 12)),
        "Z": rng.normal(size=(12, 2)),
        "Y": rng.normal(size=(12, 2)),
        "X": np.repeat([0, 1, 2], 4).reshape(-1, 1),
    }
    method = MultivariateACTest(
        k=2,
        solver=dummy_solver,
        use_true_latent=permutation_type != "observed",
        permutation_type=permutation_type,
        M=1,
        npermutations=5,
        rng=np.random.default_rng(21),
    )
    method.fit(data)

    assert len(method.permutation_indices) == 5
    for permutation in method.permutation_indices:
        np.testing.assert_array_equal(data["X"][permutation], data["X"])


def test_ac_permutations_remain_global_when_x_is_none(monkeypatch):
    monkeypatch.setattr(
        "src.methods.ac_test.ac_coefficient", lambda **kwargs: 0.25
    )
    seed = 22
    data = latent_data()
    method = MultivariateACTest(
        use_true_latent=True,
        M=1,
        npermutations=1,
        rng=np.random.default_rng(seed),
    )
    method.fit(data)

    expected = np.random.default_rng(seed).permutation(data["Y"].shape[0])
    np.testing.assert_array_equal(method.permutation_indices[0], expected)


def test_adaptive_m_grid_is_rounded_deduplicated_and_capped():
    method = MultivariateACTest(use_true_latent=True)
    np.testing.assert_array_equal(
        method._adaptive_m_grid(10),
        [1, 2, 3, 5, 8],
    )
    np.testing.assert_array_equal(method._adaptive_m_grid(2), [1])


def test_adaptive_s_statistic_preserves_options(monkeypatch):
    calls = []

    def recording_ac(**kwargs):
        calls.append(kwargs)
        return float(kwargs["Y"][0, 0])

    monkeypatch.setattr("src.methods.ac_test.ac_coefficient", recording_ac)
    method = MultivariateACTest(
        use_true_latent=True,
        M=99,
        use_permutation_coeff=True,
        use_right_neighbor=True,
        adaptive_m=True,
        rng=np.random.default_rng(30),
    )
    method.X = np.ones((3, 1))
    method._ignore_X = False
    y = np.array([[2.0], [1.0], [0.0]])
    z = np.arange(3, dtype=float).reshape(-1, 1)

    assert method._adaptive_s_statistic(z, y, 2) == 2.0
    assert len(calls) == 1
    assert calls[0]["Y"] is y
    for call in calls:
        assert call["M"] == 2
        assert call["X"] is method.X
        assert call["permutation"] is True
        assert call["right_neighbor"] is True


def test_adaptive_test_uses_sample_standardization_and_corrected_pvalue(
    monkeypatch,
):
    s_statistics = np.array(
        [
            [9.0, 4.0, 2.0, 8.0, 5.0],
            [0.0, 1.0, 3.0, 4.0, 7.0],
            [0.0, 2.0, 5.0, 3.0, 6.0],
        ]
    )
    values = iter(s_statistics.ravel())
    seen_y = []

    def fixed_s_statistic(self, Z, Y, M, rng=None):
        seen_y.append(Y.copy())
        return next(values)

    monkeypatch.setattr(
        MultivariateACTest, "_adaptive_s_statistic", fixed_s_statistic
    )
    data = {
        "Y": np.arange(10, dtype=float).reshape(-1, 1),
        "Z": np.arange(20, dtype=float).reshape(10, 2),
    }
    method = MultivariateACTest(
        use_true_latent=True,
        M=99,
        adaptive_m=True,
        npermutations=2,
        rng=np.random.default_rng(31),
    )
    method.fit(data)

    expected_means = s_statistics.mean(axis=0)
    expected_stds = s_statistics.std(axis=0, ddof=1)
    expected_z = (s_statistics - expected_means) / expected_stds
    expected_a = expected_z.max(axis=1)
    # The one-neighbor candidate determines the observed adaptive maximum.
    assert expected_z[0].argmax() == 0
    expected_pvalue = (
        1 + np.count_nonzero(expected_a[1:] >= expected_a[0])
    ) / 3

    np.testing.assert_array_equal(method.adaptive_m_values, [1, 2, 3, 5, 8])
    np.testing.assert_allclose(method.adaptive_s_statistics, s_statistics)
    np.testing.assert_allclose(method.adaptive_m_means, expected_means)
    np.testing.assert_allclose(method.adaptive_m_stds, expected_stds)
    np.testing.assert_allclose(method.adaptive_z_statistics, expected_z)
    assert method.test_stat_estimate == pytest.approx(expected_a[0])
    assert method.null_distribution == pytest.approx(expected_a[1:])
    assert method.pvalue == pytest.approx(expected_pvalue)
    assert len(method.permutation_indices) == 2

    grid_size = len(method.adaptive_m_values)
    for permutation_number, permutation in enumerate(method.permutation_indices, 1):
        start = permutation_number * grid_size
        stop = start + grid_size
        for passed_y in seen_y[start:stop]:
            np.testing.assert_array_equal(passed_y, data["Y"][permutation])


def test_adaptive_test_passes_multivariate_y_to_each_m_statistic(monkeypatch):
    seen_shapes = []

    def shape_based_s_statistic(self, Z, Y, M, rng=None):
        seen_shapes.append(Y.shape)
        row_weights = np.arange(1, Y.shape[0] + 1)
        return float(M + row_weights @ Y[:, 0])

    monkeypatch.setattr(
        MultivariateACTest,
        "_adaptive_s_statistic",
        shape_based_s_statistic,
    )
    data = latent_data()
    method = MultivariateACTest(
        use_true_latent=True,
        adaptive_m=True,
        npermutations=2,
        rng=np.random.default_rng(32),
    )
    method.fit(data)

    expected_calls = (method.npermutations + 1) * len(method.adaptive_m_values)
    assert seen_shapes == [data["Y"].shape] * expected_calls
    assert np.isfinite(method.test_stat_estimate)


def test_adaptive_test_raises_when_an_m_specific_std_is_zero(monkeypatch):
    monkeypatch.setattr(
        MultivariateACTest,
        "_adaptive_s_statistic",
        lambda self, Z, Y, M, rng=None: 1.0,
    )
    data = {
        "Y": np.arange(10, dtype=float).reshape(-1, 1),
        "Z": np.arange(20, dtype=float).reshape(10, 2),
    }
    method = MultivariateACTest(
        use_true_latent=True,
        adaptive_m=True,
        npermutations=2,
    )

    with pytest.raises(ValueError, match="standard deviation is zero"):
        method.fit(data)


def test_parallel_adaptive_ac_matches_serial():
    data = latent_data()
    methods = []
    for n_jobs in (1, 3):
        method = MultivariateACTest(
            use_true_latent=True,
            adaptive_m=True,
            npermutations=5,
            rng=np.random.default_rng(45),
            n_jobs=n_jobs,
            batch_size=2,
        )
        method.fit(data)
        methods.append(method)

    serial, parallel = methods
    for serial_perm, parallel_perm in zip(
        serial.permutation_indices, parallel.permutation_indices
    ):
        np.testing.assert_array_equal(serial_perm, parallel_perm)
    np.testing.assert_allclose(
        serial.adaptive_s_statistics, parallel.adaptive_s_statistics
    )
    np.testing.assert_allclose(
        serial.null_distribution, parallel.null_distribution
    )
    assert serial.pvalue == parallel.pvalue
