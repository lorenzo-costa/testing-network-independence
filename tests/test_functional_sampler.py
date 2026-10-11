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
        ({"n": 5, "k": 2, "ky": 2}, "only supports scalar responses"),
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
