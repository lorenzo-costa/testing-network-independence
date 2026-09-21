from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dgp import BernoulliNetwork, GaussianNetwork
from src.helper_functions.simulation_functions import run_scenario
from src.load_config import (
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
    assert config["simulation"]["p"] == [5, 10, 25, 50, 100]
    assert config["simulation"]["d_x"] == [5]
    assert config["simulation"]["d_y"] == [5]
    assert config["simulation"]["snr"] == [0, 0.1, 0.25, 0.5, 1]
    assert config["methods"]["use_true_latent"] == [False, True]
    assert len(design) == 4 * 3 * 3 * 5 * 5 * 2

    assert {row["setup"][0].args[0] for row in design} == {
        GaussianNetwork,
        BernoulliNetwork,
    }
    assert {row["method"].func for row in design} == {
        RVTest,
        CanonicalCorrelationTest,
        DistanceCorrelationTest,
    }
    for row in design:
        network_kwargs = row["setup"][0].keywords["network_kwargs"]
        assert network_kwargs["eps_distribution"] == "student_t_3"
        assert network_kwargs["x_network_correlation"] in {0, 0.5}
        assert row["B"] == (0 if row["snr"] == 0 else None)
        assert row["hypothesis"] == ("H0" if row["snr"] == 0 else "H1")

    assert {
        (
            row["setup"][0].args[0],
            row["setup"][0].keywords["network_kwargs"]["x_network_correlation"],
        )
        for row in design
    } == {
        (GaussianNetwork, 0),
        (GaussianNetwork, 0.5),
        (BernoulliNetwork, 0),
        (BernoulliNetwork, 0.5),
    }

    mgc_methods = [
        row["method"] for row in design if row["method"].func is DistanceCorrelationTest
    ]
    assert mgc_methods
    assert all(method.keywords["test_method"] == "mgc" for method in mgc_methods)


def test_every_linear_model_network_and_method_combination_runs():
    config = load_config(ROOT / "linear_model_config.yaml")
    design = build_factorial_design(config)
    representatives = {}
    for row in design:
        key = (
            row["setup"][0].args[0],
            row["setup"][0].keywords["network_kwargs"]["x_network_correlation"],
            row["method"].func,
        )
        if row["snr"] == 0.5:
            representatives.setdefault(key, row)

    assert len(representatives) == 12
    for index, row in enumerate(representatives.values()):
        runtime = dict(row)
        runtime.update(n=12, npermutations=2)
        result = run_scenario(
            config["metrics"],
            runtime,
            seed=np.random.SeedSequence(500 + index),
        )

        assert result["args"]["p"] == 5
        assert result["args"]["d_x"] == 5
        assert result["args"]["d_y"] == 5
        assert result["args"]["eps_distribution"] == "student_t_3"
        assert result["args"]["x_network_correlation"] in {0, 0.5}
        assert np.isfinite(result["ComputeAll"]["Rejection"])
        if row["method"].func is CanonicalCorrelationTest:
            assert result["args"]["cca_gamma"] == pytest.approx(np.sqrt(12))


def test_asymptotic_linear_model_config_uses_global_mrqap_permutations():
    config = load_config(ROOT / "linear_model_asymptotic_config.yaml")
    design = build_factorial_design(config)

    assert len(design) == 2 * 3 * 5 * 5
    assert config["methods"]["npermutations"] == [400]
    assert config["methods"]["use_true_latent"] is None
    assert config["output"]["file_prefix"] == "linear_model_mrqap_results"
    assert {row["method"].func for row in design} == {MRQAP}

    mrqap_methods = [row["method"] for row in design]
    assert all(
        method.keywords["permutation_strategy"] == "y_permutation"
        for method in mrqap_methods
    )
    assert all(method.keywords["batch_size"] == 32 for method in mrqap_methods)
    assert all("use_true_latent" not in row for row in design)


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


def test_linear_model_results_flatten_without_legacy_k_or_rho():
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

    assert frame.loc[0, "k"] == "NA"
    assert frame.loc[0, "p"] == 5
    assert frame.loc[0, "d_x"] == 5
    assert frame.loc[0, "d_y"] == 3
    assert frame.loc[0, "snr"] == 0.25
    assert frame.loc[0, "asymptotic_null"] == "zero_covariance"
    assert frame.loc[0, "x_network_correlation"] == 0.5
    assert frame.loc[0, "eps_distribution"] == "student_t_3"
    assert frame.loc[0, "hypothesis"] == "H1"
