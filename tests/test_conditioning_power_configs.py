from pathlib import Path

import numpy as np
import pytest

from src.load_config import build_factorial_design, load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "filename, experiment_type, setup_count, design_count, expected_rho",
    [
        (
            "config_conditioning_null.yaml",
            "conditioning_mixed",
            24,
            792,
            {0.0},
        ),
        (
            "config_conditional_copula.yaml",
            "conditional_copula",
            24,
            1944,
            {0.2},
        ),
        (
            "config_postlinearnoise.yaml",
            "post_nonlinear_noise",
            16,
            144,
            {0.2},
        ),
    ],
)
def test_conditioning_power_configs_load_and_build_complete_designs(
    filename, experiment_type, setup_count, design_count, expected_rho
):
    config = load_config(ROOT / filename)
    design = build_factorial_design(config)

    assert config["experiment_type"] == experiment_type
    assert config["simulation"]["nsim"] == 100
    assert config["simulation"]["seed"] == 1
    assert config["simulation"]["n"] == [100, 300, 500]
    assert config["simulation"]["k"] == [3]
    assert config["simulation"]["ky"] == [1]
    assert set(config["simulation"]["rho"]) == expected_rho
    assert config["simulation"]["alpha"] == [0.5]
    assert config["simulation"]["edge_var"] == [3]
    assert config["methods"]["npermutations"] == [100]
    assert config["methods"]["use_true_latent"] == [False]
    assert len(config["setups"]) == setup_count
    assert len(design) == design_count
    assert {row["rho"] for row in design} == expected_rho
    assert {row["method"].keywords["M"] for row in design} == {
        1,
        "sqrt",
        "half",
    }


def test_conditional_copula_config_covers_requested_factorial_settings():
    config = load_config(ROOT / "config_conditional_copula.yaml")
    keywords = [factory.keywords for factory, _ in config["setups"]]

    assert config["simulation"]["marginals_z"] == [
        "gaussian",
        "uniform -1 1",
        "chi 5",
    ]
    assert config["simulation"]["marginals_y"] == [
        "gaussian",
        "cauchy",
        "chi 5",
    ]
    assert {entry["conditional_copula"] for entry in keywords} == {
        "gaussian",
        "student_t",
        "clayton",
    }
    assert {entry["C"] for entry in keywords} == {2, 3}
    assert {
        tuple(np.asarray(entry["column_covariance"]).ravel()) for entry in keywords
    } == {
        tuple(np.eye(3).ravel()),
        tuple(np.full((3, 3), 0.5).ravel() + np.eye(3).ravel() * 0.5),
    }
    student_entries = [
        entry for entry in keywords if entry["conditional_copula"] == "student_t"
    ]
    assert all(entry["copula_params"] == {"df": 3} for entry in student_entries)


def test_postlinear_config_covers_requested_covariance_settings():
    config = load_config(ROOT / "config_postlinearnoise.yaml")
    keywords = [factory.keywords for factory, _ in config["setups"]]

    assert "marginals" not in config["simulation"]
    assert all(entry["post_nonlinear_noise"] is True for entry in keywords)
    assert {entry["C"] for entry in keywords} == {2, 3}
    assert len(
        {
            tuple(np.asarray(entry["column_covariance_z"]).ravel())
            for entry in keywords
        }
    ) == 2
    assert len(
        {
            tuple(np.asarray(entry["stratum_covariance"]).ravel())
            for entry in keywords
        }
    ) == 2


def test_merged_null_config_partitions_sampler_specific_sweeps():
    config = load_config(ROOT / "config_conditioning_null.yaml")
    design = build_factorial_design(config)
    conditional_rows = [
        row
        for row in design
        if row["setup"][0].keywords.get("conditional_copula") is not None
    ]
    post_nonlinear_rows = [
        row
        for row in design
        if row["setup"][0].keywords.get("post_nonlinear_noise") is not None
    ]

    assert len(conditional_rows) == 648
    assert len(post_nonlinear_rows) == 144
    assert all(row["rho"] == 0.0 for row in design)
    assert {
        row["setup"][0].keywords["conditional_copula"]
        for row in conditional_rows
    } == {"gaussian"}
    assert all("marginals" in row for row in conditional_rows)
    assert all("marginals" not in row for row in post_nonlinear_rows)


@pytest.mark.parametrize(
    "filename",
    [
        "config_conditioning_null.yaml",
        "config_conditional_copula.yaml",
        "config_postlinearnoise.yaml",
    ],
)
def test_conditioning_config_representative_scenario_runs(filename):
    config = load_config(ROOT / filename)
    row = build_factorial_design(config)[0]
    dgp_factory, solver = row["setup"]
    runtime = dict(row)
    runtime.update(
        n=24,
        npermutations=2,
        solver=solver,
        rng=np.random.default_rng(100),
    )

    data = dgp_factory(**runtime).generate()
    method = row["method"](**runtime)
    method.fit(data)

    assert data["A"].shape == (24, 24)
    assert data["Z"].shape == (24, 3)
    assert data["Y"].shape == (24, 1)
    assert data["X"].shape == (24, 1)
    assert np.isfinite(method.test_stat_estimate)
    assert len(method.permutation_distribution) == 2
