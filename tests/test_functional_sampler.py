import importlib

import numpy as np
import pytest

from src.dgp import BernoulliNetwork
from src.latent_samplers.functional_sampler import FunctionalGenerator


def test_module_imports_functional_generator():
    module = importlib.import_module("src.latent_samplers.functional_sampler")

    assert module.FunctionalGenerator is FunctionalGenerator


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n": 0, "k": 2, "ky": 1}, "n must be a positive integer"),
        ({"n": 5, "k": 0, "ky": 1}, "k must be a positive integer"),
        (
            {"n": 5, "k": 3, "ky": 2},
            r"scalar response \(ky=1\).*component-wise response \(ky=k\)",
        ),
        (
            {"n": 5, "k": 2, "ky": 1, "noise_scale": -0.1},
            "noise_scale must be non-negative",
        ),
        (
            {"n": 5, "k": 2, "ky": 1, "noise_type": "invalid"},
            "noise_type must be 'additive' or 'multiplicative'",
        ),
        (
            {"n": 5, "k": 2, "ky": 1, "predictor_distribution": "invalid"},
            "predictor_distribution must be 'gaussian', 'student_t', or 'uniform_rdpg'",
        ),
    ],
)
def test_constructor_validates_basic_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        FunctionalGenerator(**kwargs)


def test_unknown_functional_is_rejected():
    with pytest.raises(ValueError, match="Unknown functional"):
        FunctionalGenerator(n=5, k=2, ky=1, functional_form="invalid")


def test_interaction_supports_output_standardization():
    generator = FunctionalGenerator(
        n=2,
        k=4,
        functional_form="interaction",
        function_params={"output_center": 2.0, "output_sd": 0.5},
    )
    z = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
    ])

    signal = generator._evaluate_function(z)

    np.testing.assert_allclose(signal, [8.0, -4.0])


@pytest.mark.parametrize(
    "params, message",
    [
        ({"output_center": np.inf}, "output_center must be finite"),
        ({"output_sd": 0.0}, "output_sd must be positive"),
        ({"output_sd": np.inf}, "output_sd must be positive"),
    ],
)
def test_interaction_validates_output_standardization(params, message):
    generator = FunctionalGenerator(
        n=2, k=4, functional_form="interaction", function_params=params
    )

    with pytest.raises(ValueError, match=message):
        generator._evaluate_function(np.ones((2, 4)))


def test_abs_max_supports_centered_scaled_wedge():
    generator = FunctionalGenerator(
        n=3,
        k=1,
        functional_form="abs_max",
        function_params={"center": 0.5, "input_scale": np.sqrt(12.0)},
    )
    z = np.array([[0.0], [0.5], [1.0]])

    signal = generator._evaluate_function(z)

    np.testing.assert_allclose(signal, [np.sqrt(3.0), 0.0, np.sqrt(3.0)])


@pytest.mark.parametrize(
    "params, message",
    [
        ({"center": [0.0, 0.5]}, "center for abs_max"),
        ({"input_scale": 0.0}, "input_scale must be positive"),
        ({"input_scale": np.inf}, "input_scale must be positive"),
    ],
)
def test_abs_max_validates_wedge_parameters(params, message):
    generator = FunctionalGenerator(
        n=3, k=1, functional_form="abs_max", function_params=params
    )

    with pytest.raises(ValueError, match=message):
        generator._evaluate_function(np.array([[0.0], [0.5], [1.0]]))


@pytest.mark.parametrize(
    "covariance, message",
    [
        (np.eye(3), "column_covariance must have shape"),
        (np.array([[1.0, 1.0], [0.0, 1.0]]), "must be symmetric"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "must be positive semidefinite"),
    ],
)
def test_column_covariance_is_validated(covariance, message):
    with pytest.raises(ValueError, match=message):
        FunctionalGenerator(
            n=5,
            k=2,
            ky=1,
            column_covariance=covariance,
        )


def test_uniform_rdpg_predictors_produce_valid_inner_products():
    generator = FunctionalGenerator(
        n=100,
        k=4,
        ky=1,
        predictor_distribution="uniform_rdpg",
        rng=np.random.default_rng(42),
    )

    z, _ = generator.sample_latent()
    probabilities = z @ z.T

    assert np.all(z >= 0)
    assert np.all(np.linalg.norm(z, axis=1) <= 1)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_uniform_rdpg_predictors_generate_binary_rdpg_network():
    result = BernoulliNetwork(
        n=30,
        k=3,
        ky=1,
        rdpg=True,
        functional_form="linear",
        predictor_distribution="uniform_rdpg",
        rng=np.random.default_rng(43),
    ).generate()

    probabilities = result["Z"] @ result["Z"].T
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert set(np.unique(result["A"])).issubset({0, 1})


def test_uniform_rdpg_rejects_centering():
    with pytest.raises(ValueError, match="center_latent must be False"):
        FunctionalGenerator(
            n=5,
            k=2,
            predictor_distribution="uniform_rdpg",
            center_latent=True,
        )


@pytest.mark.parametrize("rho", [0, -0.1, 1.1, np.nan, np.inf, -np.inf])
def test_additive_noise_rejects_invalid_rho(rho):
    with pytest.raises(ValueError, match="rho must satisfy"):
        FunctionalGenerator(n=5, k=2, rho=rho)


@pytest.mark.parametrize("rho", [0.2, 0.8])
@pytest.mark.parametrize("noise_scale", [0.0, 100.0])
def test_rho_targets_signal_fraction_and_overrides_noise_scale(rho, noise_scale):
    generator = FunctionalGenerator(
        n=50_000, k=1, rho=rho, noise_scale=noise_scale,
        rng=np.random.default_rng(44),
    )

    # Reusing the generator checks calibration for each fresh signal sample.
    for signal_scale in (1.0, 7.0):
        generator.functional_form = lambda z: signal_scale * z[:, 0]
        z, y = generator.sample_latent()
        signal = signal_scale * z[:, 0]
        noise = y[:, 0] - signal
        signal_fraction = np.var(signal) / (np.var(signal) + np.var(noise))

        assert signal_fraction == pytest.approx(rho, abs=0.01)
        assert abs(np.corrcoef(signal, noise)[0, 1]) < 0.02
        assert y.shape == (generator.n, 1)


@pytest.mark.parametrize("rho, constant", [(1.0, False), (0.3, True)])
def test_rho_zero_noise_cases_preserve_signal_and_rng(rho, constant):
    rng = np.random.default_rng(45)
    reference_rng = np.random.default_rng(45)
    generator = FunctionalGenerator(
        n=10, k=1, rho=rho, noise_scale=100.0, rng=rng,
        functional_form=(lambda z: np.ones(len(z))) if constant else "linear",
    )
    reference = FunctionalGenerator(
        n=10, k=1, functional_form=generator.functional_form, rng=reference_rng,
    )

    z, y = generator.sample_latent()
    expected_z, expected_y = reference.sample_latent()

    np.testing.assert_array_equal(z, expected_z)
    np.testing.assert_array_equal(y, expected_y)
    assert rng.random() == reference_rng.random()


@pytest.mark.parametrize("noise_type, rho", [
    ("additive", None), ("multiplicative", None), ("multiplicative", 0.2),
])
def test_existing_noise_scale_behavior(noise_type, rho):
    rng = np.random.default_rng(46)
    generator = FunctionalGenerator(
        n=20, k=1, rho=rho, noise_type=noise_type, noise_scale=0.5,
        rng=np.random.default_rng(46),
    )
    expected_z = rng.multivariate_normal(np.zeros(1), np.eye(1), size=20)
    epsilon = rng.standard_normal(20)
    signal = expected_z[:, 0]
    expected_y = (signal + 0.5 * epsilon if noise_type == "additive"
                  else signal * np.exp(0.5 * epsilon))

    z, y = generator.sample_latent()

    np.testing.assert_array_equal(z, expected_z)
    np.testing.assert_array_equal(y[:, 0], expected_y)


def test_network_passes_rho_to_functional_sampler():
    network = BernoulliNetwork(
        n=30, k=1, functional_form="linear", rho=1.0, noise_scale=100.0,
        rng=np.random.default_rng(47),
    )

    result = network.generate()

    np.testing.assert_array_equal(result["Y"], result["Z"])
    assert network.get_name().endswith("_rho1")


@pytest.mark.parametrize("column_result", [False, True])
def test_componentwise_callable_receives_separate_column_slices(monkeypatch, column_result):
    z = np.array([[1.0, 4.0], [-2.0, 5.0], [3.0, -6.0]])
    inputs = []

    def square(x):
        inputs.append(x.copy())
        return x**2 if column_result else x[:, 0]**2

    generator = FunctionalGenerator(n=3, k=2, ky=2, functional_form=square)
    monkeypatch.setattr(generator, "_sample_predictors", lambda: z)

    sampled_z, y = generator.sample_latent()

    np.testing.assert_array_equal(sampled_z, z)
    np.testing.assert_array_equal(y, z**2)
    assert y.shape == (3, 2)
    assert len(inputs) == 2
    for j, x in enumerate(inputs):
        assert x.shape == (3, 1)
        np.testing.assert_array_equal(x, z[:, [j]])


@pytest.mark.parametrize("functional", [
    name for name in FunctionalGenerator.AVAILABLE_FUNCTIONALS
    if name not in {"interaction", "hierarchical"}
])
def test_componentwise_builtins_match_unit_variance_scalar_implementation(monkeypatch, functional):
    z = np.array([[-0.7, 0.3], [0.2, -0.8], [1.1, 0.6]])
    covariance = np.array([[4.0, 1.0], [1.0, 9.0]])
    params = {
        "weights": [2.0], "second_weights": [-3.0], "center": [0.2],
        "active_features": [0], "frequency": 2.3, "threshold": 0.1,
        "input_scale": 0.7, "global_scale": 1.4,
    }
    generator = FunctionalGenerator(
        n=3, k=2, ky=2, functional_form=functional,
        function_params=params, column_covariance=covariance,
    )
    scalar = FunctionalGenerator(
        n=3, k=1, functional_form=functional, function_params=params,
    )
    monkeypatch.setattr(generator, "_sample_predictors", lambda: z)

    _, y = generator.sample_latent()

    for j in range(2):
        np.testing.assert_array_equal(y[:, j], scalar._evaluate_function(z[:, [j]]))
    # Unit normalization leaves each linear column intact, despite unequal
    # marginal variances and a nonzero covariance between predictor columns.
    if functional == "linear":
        np.testing.assert_array_equal(y, z)
    assert generator.kz == generator.ky == 2
    np.testing.assert_array_equal(generator.column_covariance, covariance)
    assert generator.function_params == params


@pytest.mark.parametrize("functional, params, message", [
    ("linear", {"weights": [1, 2]}, "weights must have shape"),
    ("manifold", {"second_weights": [1, 2]}, "second_weights must have shape"),
    ("abs_max", {"center": [0, 1]}, "center for abs_max"),
    ("tanh_product", {"center": [0, 1]}, "center for tanh_product"),
    ("sparse_nonlinear", {"active_features": [1]}, "out-of-range"),
    ("sparse_nonlinear", {"active_count": 2}, "active_count must lie"),
    ("piecewise", {"switch_feature": 1}, "switch_feature must be"),
])
def test_componentwise_parameters_follow_one_dimensional_contract(functional, params, message):
    generator = FunctionalGenerator(
        n=3, k=2, ky=2, functional_form=functional, function_params=params,
    )
    with pytest.raises(ValueError, match=message):
        generator.sample_latent()


@pytest.mark.parametrize("k, ky", [(3, 1), (3, 3), (1, 1)])
def test_supported_response_dimensions(k, ky):
    generator = FunctionalGenerator(n=5, k=k, ky=ky)
    z, y = generator.sample_latent()
    assert generator.ky == ky
    assert z.shape == (5, k)
    assert y.shape == (5, ky)


@pytest.mark.parametrize("functional", ["interaction", "hierarchical", "pairwise_interaction"])
def test_scalar_only_functionals_rejected_only_in_componentwise_mode(functional):
    with pytest.raises(ValueError, match="cannot be applied component-wise to scalar inputs"):
        FunctionalGenerator(n=5, k=4, ky=4, functional_form=functional)
    scalar = FunctionalGenerator(n=5, k=4, ky=1, functional_form=functional)
    _, y = scalar.sample_latent()
    assert y.shape == (5, 1)


@pytest.mark.parametrize("noise_type, rho", [
    ("additive", None), ("multiplicative", None), ("multiplicative", 0.2),
])
def test_componentwise_noise_uses_independent_vectors(monkeypatch, noise_type, rho):
    z = np.tile(np.arange(1.0, 21.0)[:, None], (1, 2))
    reference_rng = np.random.default_rng(51)
    generator = FunctionalGenerator(
        n=20, k=2, ky=2, noise_scale=0.5, noise_type=noise_type, rho=rho,
        rng=np.random.default_rng(51),
    )
    monkeypatch.setattr(generator, "_sample_predictors", lambda: z)
    epsilon = np.column_stack([reference_rng.standard_normal(20) for _ in range(2)])
    expected = (z + 0.5 * epsilon if noise_type == "additive"
                else z * np.exp(0.5 * epsilon))

    _, y = generator.sample_latent()

    np.testing.assert_array_equal(y, expected)
    assert not np.array_equal(y[:, 0], y[:, 1])
    assert generator.rng.random() == reference_rng.random()


def test_componentwise_rho_calibrates_each_column_and_skips_constant(monkeypatch):
    varying = np.arange(8.0)
    z = np.column_stack([varying, np.full(8, 3.0), 7 * varying])
    rho = 0.4
    generator = FunctionalGenerator(
        n=8, k=3, ky=3, rho=rho, noise_scale=100,
        rng=np.random.default_rng(52),
    )
    monkeypatch.setattr(generator, "_sample_predictors", lambda: z)
    reference_rng = np.random.default_rng(52)
    expected = z.copy()
    for j in (0, 2):
        scale = np.sqrt(np.var(z[:, j]) * (1 - rho) / rho)
        expected[:, j] += scale * reference_rng.standard_normal(8)

    _, y = generator.sample_latent()

    np.testing.assert_array_equal(y, expected)
    np.testing.assert_array_equal(y[:, 1], z[:, 1])
    assert generator.rng.random() == reference_rng.random()


@pytest.mark.parametrize("distribution, centered", [
    ("gaussian", False), ("gaussian", True), ("student_t", False),
    ("student_t", True), ("uniform_rdpg", False),
])
def test_componentwise_preserves_joint_predictor_sampling(distribution, centered):
    options = dict(
        n=20, k=2, column_covariance=np.array([[4.0, 1.0], [1.0, 9.0]]),
        predictor_distribution=distribution, center_latent=centered, rho=1,
    )
    generator = FunctionalGenerator(ky=2, rng=np.random.default_rng(53), **options)
    reference = FunctionalGenerator(ky=1, rng=np.random.default_rng(53), **options)

    z, y = generator.sample_latent()
    expected_z, _ = reference.sample_latent()

    np.testing.assert_array_equal(z, expected_z)
    np.testing.assert_array_equal(y, z)
    assert generator.rng.random() == reference.rng.random()


@pytest.mark.parametrize("method", ["sample_latent", "_sample_latent_functional", "_sample_latent"])
def test_componentwise_sampling_aliases(method):
    generator = FunctionalGenerator(n=5, k=2, ky=2)
    z, y = getattr(generator, method)()
    np.testing.assert_array_equal(y, z)


@pytest.mark.parametrize("noise_type, rho, noise_scale", [
    ("additive", None, 0), ("additive", None, 0.5),
    ("additive", 0.4, 100), ("multiplicative", 0.4, 0.5),
])
def test_multivariate_scalar_response_preserves_signal_noise_and_rng(noise_type, rho, noise_scale):
    covariance = np.array([[4.0, 1.0], [1.0, 9.0]])
    weights = np.array([2.0, -1.0])
    generator = FunctionalGenerator(
        n=10, k=2, ky=1, column_covariance=covariance,
        function_params={"weights": weights}, noise_type=noise_type,
        rho=rho, noise_scale=noise_scale, rng=np.random.default_rng(54),
    )
    reference_rng = np.random.default_rng(54)
    expected_z = reference_rng.multivariate_normal(np.zeros(2), covariance, size=10)
    signal = expected_z @ (weights / np.sqrt(weights @ covariance @ weights))
    scale = (np.sqrt(np.var(signal) * (1 - rho) / rho)
             if noise_type == "additive" and rho is not None else noise_scale)
    expected = signal
    if scale:
        epsilon = reference_rng.standard_normal(10)
        expected = (signal + scale * epsilon if noise_type == "additive"
                    else signal * np.exp(scale * epsilon))

    z, y = generator.sample_latent()

    assert y.shape == (10, 1)
    np.testing.assert_array_equal(z, expected_z)
    np.testing.assert_array_equal(y[:, 0], expected)
    assert generator.rng.random() == reference_rng.random()


def test_names_distinguish_response_dimensions_and_preserve_scalar_name():
    scalar = FunctionalGenerator(n=5, k=2, ky=1)
    componentwise = FunctionalGenerator(n=5, k=2, ky=2)
    assert scalar.get_name() == "functional_linear_n5_k2_predictors_gaussian_noise0"
    assert componentwise.get_name() == "functional_linear_n5_k2_ky2_predictors_gaussian_noise0"


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_componentwise_rejects_nonfinite_functional_values(invalid):
    generator = FunctionalGenerator(
        n=3, k=2, ky=2, functional_form=lambda x: np.full(len(x), invalid),
    )
    with pytest.raises(FloatingPointError, match="functional generated non-finite"):
        generator.sample_latent()


def test_componentwise_rejects_invalid_callable_shape():
    generator = FunctionalGenerator(
        n=3, k=2, ky=2, functional_form=lambda x: np.zeros((3, 2)),
    )
    with pytest.raises(ValueError, match="functional must return shape"):
        generator.sample_latent()


def test_componentwise_rejects_nonfinite_noise():
    generator = FunctionalGenerator(n=3, k=2, ky=2, noise_scale=np.nan)
    with pytest.raises(FloatingPointError, match="Noise generation produced non-finite"):
        generator.sample_latent()


def test_componentwise_rejects_nonfinite_predictors(monkeypatch):
    generator = FunctionalGenerator(n=3, k=2, ky=2, functional_form="threshold")
    monkeypatch.setattr(generator, "_sample_predictors", lambda: np.full((3, 2), np.inf))
    with pytest.raises(FloatingPointError, match="Predictor generation produced non-finite"):
        generator.sample_latent()
