from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dgp import BernoulliNetwork, GaussianNetwork
from src.helper_functions.simulation_functions import run_scenario
from src.load_config import (
    _resolve_linear_model_simulation,
    build_factorial_design,
    flatten_args_columns,
    load_config,
)
from src.methods import CanonicalCorrelationTest, DistanceCorrelationTest, MRQAP, RVTest


ROOT = Path(__file__).resolve().parents[1]


def test_linear_model_config_builds_requested_factorial_sweep():
    config = load_config(ROOT / "linear_model_config.yaml")
    design = build_factorial_design(config)

    assert config["experiment_type"] == "linear_model"
    assert config["simulation"]["n"] == [50, 100, 200]
    assert config["simulation"]["p"] == [1]
    assert config["simulation"]["d_x"] == [5]
    assert config["simulation"]["d_y"] == [5]
    assert config["simulation"]["snr"] == [0, 0.1, 0.25, 0.5, 1]
    assert config["methods"]["use_true_latent"] == [False, True]
    assert len(design) == 2 * 5 * 3 * 1 * 5 * 2

    assert {row["setup"][0].args[0] for row in design} == {
        GaussianNetwork,
        BernoulliNetwork,
    }
    assert {row["method"].func for row in design} == {
        RVTest,
        CanonicalCorrelationTest,
        DistanceCorrelationTest,
        MRQAP,
    }
    for row in design:
        network_kwargs = row["setup"][0].keywords["network_kwargs"]
        assert network_kwargs in ({}, {"rdpg": False})
        assert row["B"] == (0 if row["snr"] == 0 else None)
        assert row["hypothesis"] == ("H0" if row["snr"] == 0 else "H1")

    mgc_methods = [
        row["method"] for row in design if row["method"].func is DistanceCorrelationTest
    ]
    assert mgc_methods
    assert all(method.keywords["test_method"] == "mgc" for method in mgc_methods)


def test_active_fraction_config_builds_requested_factorial_sweep():
    config = load_config(ROOT / "linear_model_active_fraction_config.yaml")
    design = build_factorial_design(config)

    assert config["simulation"]["snr"] == [0.1, 0.25, 0.5]
    assert config["simulation"]["b_active_network_fraction"] == [
        0,
        0.1,
        0.5,
        1,
    ]
    assert len(design) == len(config["setups"]) * 5 * 3 * 4 * 3 * 4 * 2
    for row in design:
        fraction = row["b_active_network_fraction"]
        is_null = int(np.floor(fraction * row["p"] + 0.5)) == 0
        assert row["B"] == (0 if is_null else None)
        assert row["hypothesis"] == ("H0" if is_null else "H1")
        assert row["snr"] == (0 if is_null else row["requested_snr"])

    row = next(
        row
        for row in design
        if row["setup"][0].args[0] is GaussianNetwork
        and row["b_active_network_fraction"] == 0.5
        and row["p"] == 5
    )
    network = row["setup"][0](**row, rng=np.random.default_rng(900))
    blocks = np.split(network.generate()["B"], row["p"], axis=1)
    assert sum(np.any(block) for block in blocks) == 3
    assert network.snr == row["requested_snr"]


def test_linear_model_signal_sweeps_accept_both_and_require_at_least_one():
    base = {
        "nsim": 1,
        "seed": 1,
        "n": [10],
        "p": [5],
        "d_x": [1],
        "d_y": [1],
    }
    with pytest.raises(ValueError, match="at least one"):
        _resolve_linear_model_simulation(base)
    resolved = _resolve_linear_model_simulation(
        {**base, "snr": 1, "b_active_network_fraction": 0.5}
    )
    assert resolved["snr"] == [1]
    assert resolved["b_active_network_fraction"] == [0.5]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _resolve_linear_model_simulation(
            {**base, "b_active_network_fraction": [1.1]}
        )


@pytest.mark.parametrize(
    "parameter,invalid",
    [
        ("snr", -1),
        ("snr", np.inf),
        ("snr", True),
        ("b_active_network_fraction", -0.1),
        ("b_active_network_fraction", 1.1),
        ("b_active_network_fraction", np.nan),
        ("b_active_network_fraction", True),
    ],
)
def test_joint_signal_sweep_validates_each_parameter(parameter, invalid):
    simulation = dict(
        nsim=1, seed=1, n=10, p=5, d_x=1, d_y=1,
        snr=[0.5], b_active_network_fraction=[0.5],
    )
    simulation[parameter] = [invalid]
    with pytest.raises(ValueError, match=f"simulation.{parameter}"):
        _resolve_linear_model_simulation(simulation)


@pytest.mark.parametrize("mode", ["snr", "fraction", "joint"])
def test_signal_sweeps_generate_calibrated_alternatives_and_correct_nulls(mode):
    config = load_config(ROOT / "linear_model_active_fraction_config.yaml")
    for factory, _ in config["setups"]:
        factory.keywords["network_kwargs"].update(
            x_network_correlation=0.25, eps_variance=2
        )
    simulation = dict(nsim=1, seed=1, n=8, p=[1, 5], d_x=2, d_y=3)
    if mode != "fraction":
        simulation["snr"] = [0, 0.25, 1]
    if mode != "snr":
        simulation["b_active_network_fraction"] = [0, 0.1, 0.5, 1]
    config["simulation"] = _resolve_linear_model_simulation(simulation)
    config["methods"]["list"] = config["methods"]["list"][:1]
    config["methods"]["use_true_latent"] = [True]
    design = build_factorial_design(config)
    setup_count = len(config["setups"])
    assert len(design) == setup_count * 2 * (3 if mode != "fraction" else 1) * (
        4 if mode != "snr" else 1
    )

    coordinates = set()
    for index, row in enumerate(design):
        fraction = row.get("b_active_network_fraction", 1)
        count = int(np.floor(fraction * row["p"] + 0.5))
        target = row.get("requested_snr", row.get("snr"))
        is_null = count == 0 or target == 0
        coordinates.add((row["p"], target, fraction))
        assert row["hypothesis"] == ("H0" if is_null else "H1")
        network = row["setup"][0](**row, rng=np.random.default_rng(910 + index))
        assert network.is_null == is_null
        sampler = network.latent_sampler
        coefficients = sampler.sample_latent()["B"]
        blocks = np.split(coefficients, row["p"], axis=1)
        assert sum(np.any(block) for block in blocks) == (0 if is_null else count)
        ratio = np.trace(coefficients @ sampler.x_covariance @ coefficients.T) / np.trace(
            sampler.eps_covariance
        )
        if mode == "fraction":
            assert "snr" not in row
            assert network.snr is None
        else:
            assert ratio == pytest.approx(0 if is_null else target)
            assert network.snr == row["snr"]
        assert ("requested_snr" in row) == (mode == "joint")

    assert len(coordinates) == len(design) // setup_count


@pytest.mark.parametrize("fraction", [0, 0.1, 0.5])
def test_joint_sweep_scenario_preserves_signal_coordinates_and_null_metrics(fraction):
    config = load_config(ROOT / "linear_model_active_fraction_config.yaml")
    config["simulation"] = _resolve_linear_model_simulation(
        dict(
            nsim=1, seed=1, n=12, p=1, d_x=1, d_y=1,
            snr=0.25, b_active_network_fraction=fraction,
        )
    )
    config["methods"]["npermutations"] = [2]
    config["methods"]["use_true_latent"] = [True]
    row = build_factorial_design(config)[0]
    result = run_scenario(config["metrics"], row, seed=np.random.SeedSequence(999))
    frame = flatten_args_columns(pd.DataFrame([result]))
    assert frame.loc[0, "requested_snr"] == 0.25
    assert frame.loc[0, "snr"] == (0 if fraction < 0.5 else 0.25)
    assert frame.loc[0, "b_active_network_fraction"] == fraction
    assert frame.loc[0, "hypothesis"] == ("H0" if fraction < 0.5 else "H1")
    outcome = result["ComputeAll"]
    rejection = bool(outcome["Rejection"])
    assert outcome["FalseRejection"] == (fraction < 0.5 and rejection)
    assert outcome["TrueRejection"] == (fraction >= 0.5 and rejection)


def test_every_linear_model_network_and_method_combination_runs():
    config = load_config(ROOT / "linear_model_config.yaml")
    design = build_factorial_design(config)
    representatives = {}
    for row in design:
        key = (
            row["setup"][0].args[0],
            row["method"].func,
        )
        if row["snr"] == 0.5:
            representatives.setdefault(key, row)

    assert len(representatives) == 8
    for index, row in enumerate(representatives.values()):
        runtime = dict(row)
        runtime.update(n=12, npermutations=2)
        result = run_scenario(
            config["metrics"],
            runtime,
            seed=np.random.SeedSequence(500 + index),
        )

        assert result["args"]["p"] == 1
        assert result["args"]["d_x"] == 5
        assert result["args"]["d_y"] == 5
        assert result["args"]["eps_distribution"] == "multivariate_gaussian"
        assert result["args"]["x_network_correlation"] is None
        assert np.isfinite(result["ComputeAll"]["Rejection"])
        if row["method"].func is CanonicalCorrelationTest:
            assert result["args"]["cca_gamma"] == pytest.approx(np.sqrt(12))


def test_asymptotic_linear_model_config_includes_mrqap_and_both_rv_nulls():
    config = load_config(ROOT / "linear_model_asymptotic_config.yaml")
    design = build_factorial_design(config)

    assert len(design) == 2 * 3 * 5 * 5 * 3
    assert config["methods"]["npermutations"] == [400]
    assert config["methods"]["use_true_latent"] is None
    assert config["output"]["file_prefix"] == "linear_model_asymptotic_results"
    assert {row["method"].func for row in design} == {MRQAP, RVTest}

    mrqap_methods = [row["method"] for row in design if row["method"].func is MRQAP]
    assert all(
        method.keywords["permutation_strategy"] == "y_permutation"
        for method in mrqap_methods
    )
    assert all(method.keywords["batch_size"] == 32 for method in mrqap_methods)
    assert all("use_true_latent" not in row for row in design)

    rv_methods = {
        (method.keywords["approximation"], method.keywords["asymptotic_null"])
        for method in (row["method"] for row in design)
        if method.func is RVTest
    }
    assert rv_methods == {
        ("asymptotic", "independence"),
        ("asymptotic", "zero_covariance"),
    }


def test_asymptotic_linear_model_config_runs_global_mrqap_scenario():
    config = load_config(ROOT / "linear_model_asymptotic_config.yaml")
    design = build_factorial_design(config)
    row = next(
        row
        for row in design
        if row["setup"][0].args[0] is GaussianNetwork
        and row["method"].func is MRQAP
        and row["p"] == 5
        and row["snr"] == 0
    )
    runtime = dict(row)
    runtime["n"] = 30
    runtime["npermutations"] = 2

    result = run_scenario(
        config["metrics"],
        runtime,
        seed=np.random.SeedSequence(700),
    )

    assert result["args"]["method_name"] == "MRQAP"
    assert "asymptotic_null" not in result["args"]
    assert result["args"]["npermutations"] == 2
    assert np.isfinite(result["ComputeAll"]["Rejection"])


def test_linear_model_results_flatten_uses_current_columns():
    args = {
        "n": 200,
        "p": 5,
        "d_x": 5,
        "d_y": 3,
        "snr": 0.25,
        "asymptotic_null": "zero_covariance",
        "x_network_correlation": 0.5,
        "eps_distribution": "student_t_3",
        "hypothesis": "H1",
        "dgp_name": "GaussianNetwork_multiple_networks",
        "method_name": "RV_PermutationTest_latent",
    }
    frame = pd.DataFrame([{"args": args}])

    flatten_args_columns(frame)

    assert "k" not in frame
    assert "rho" not in frame
    assert frame.loc[0, "p"] == 5
    assert frame.loc[0, "d_x"] == 5
    assert frame.loc[0, "d_y"] == 3
    assert frame.loc[0, "snr"] == 0.25
    assert frame.loc[0, "asymptotic_null"] == "zero_covariance"
    assert frame.loc[0, "x_network_correlation"] == 0.5
    assert frame.loc[0, "eps_distribution"] == "student_t_3"
    assert frame.loc[0, "hypothesis"] == "H1"
