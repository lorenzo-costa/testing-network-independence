from pathlib import Path

import numpy as np
import pytest

from src import dgp
from src.load_config import build_factorial_design, load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "filename, experiment_type, setup_count, design_count, expected_nsim, expected_rho, expected_npermutations",
    [
        (
            "config_conditioning_null_gaussian.yaml",
            "conditioning_mixed",
            12,
            396,
            50,
            {0.0},
            [300],
        ),
        (
            "config_conditioning_null_bernoulli.yaml",
            "conditioning_mixed",
            12,
            396,
            50,
            {0.0},
            [300],
        ),
        (
            "config_conditional_copula_gaussian.yaml",
            "conditional_copula",
            12,
            972,
            100,
            {0.2},
            [100],
        ),
        (
            "config_conditional_copula_bernoulli.yaml",
            "conditional_copula",
            12,
            972,
            100,
            {0.2},
            [100],
        ),
        (
            "config_postlinearnoise_gaussian.yaml",
            "post_nonlinear_noise",
            8,
            216,
            100,
            {0.5},
            [100],
        ),
        (
            "config_postlinearnoise_bernoulli.yaml",
            "post_nonlinear_noise",
            8,
            216,
            100,
            {0.5},
            [100],
        ),
    ],
)
def test_conditioning_power_configs_load_and_build_complete_designs(
    filename,
    experiment_type,
    setup_count,
    design_count,
    expected_nsim,
    expected_rho,
    expected_npermutations,
):
    config = load_config(ROOT / filename)
    design = build_factorial_design(config)

    assert config["experiment_type"] == experiment_type
    assert config["simulation"]["nsim"] == expected_nsim
    assert config["simulation"]["seed"] == 1
    assert config["simulation"]["n"] == [100, 300, 500]
    assert config["simulation"]["k"] == [3]
    assert config["simulation"]["ky"] == [1]
    assert set(config["simulation"]["rho"]) == expected_rho
    assert config["simulation"]["alpha"] == [0.05]
    assert config["simulation"]["edge_var"] == [3]
    assert config["methods"]["npermutations"] == expected_npermutations
    assert config["methods"]["use_true_latent"] == [False]
    assert len(config["setups"]) == setup_count
    assert len(design) == design_count
    assert {row["rho"] for row in design} == expected_rho
    assert {row["method"].keywords["M"] for row in design} == {
        1,
        "sqrt",
        "half",
    }


@pytest.mark.parametrize(
    "filename",
    [
        "config_conditional_copula_gaussian.yaml",
        "config_conditional_copula_bernoulli.yaml",
    ],
)
def test_conditional_copula_config_covers_requested_factorial_settings(filename):
    config = load_config(ROOT / filename)
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
    assert all(entry["center_latent"] is False for entry in keywords)
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


@pytest.mark.parametrize(
    "filename",
    [
        "config_postlinearnoise_gaussian.yaml",
        "config_postlinearnoise_bernoulli.yaml",
    ],
)
def test_postlinear_config_covers_requested_covariance_settings(filename):
    config = load_config(ROOT / filename)
    design = build_factorial_design(config)
    keywords = [factory.keywords for factory, _ in config["setups"]]

    assert "marginals" not in config["simulation"]
    assert config["simulation"]["function_type"] == [
        "identity",
        "square",
        "tanh",
    ]
    assert {row["function_type"] for row in design} == {
        "identity",
        "square",
        "tanh",
    }
    assert all(entry["post_nonlinear_noise"] is True for entry in keywords)
    assert all(entry["center_latent"] is False for entry in keywords)
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


@pytest.mark.parametrize(
    "filename",
    [
        "config_conditioning_null_gaussian.yaml",
        "config_conditioning_null_bernoulli.yaml",
    ],
)
def test_null_configs_partition_sampler_specific_sweeps(filename):
    config = load_config(ROOT / filename)
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

    assert len(conditional_rows) == 324
    assert len(post_nonlinear_rows) == 72
    assert all(row["rho"] == 0.0 for row in design)
    equicorrelated_z = np.full((3, 3), 0.5) + np.eye(3) * 0.5
    equicorrelated_stratum = np.full((4, 4), 0.5) + np.eye(4) * 0.5
    assert {
        row["setup"][0].keywords["conditional_copula"]
        for row in conditional_rows
    } == {"gaussian"}
    assert all("marginals" in row for row in conditional_rows)
    assert all("marginals" not in row for row in post_nonlinear_rows)
    assert all(
        row["setup"][0].keywords["center_latent"] is False
        for row in conditional_rows + post_nonlinear_rows
    )
    assert {
        tuple(
            np.asarray(row["setup"][0].keywords["column_covariance"]).ravel()
        )
        for row in conditional_rows
    } == {
        tuple(np.eye(3).ravel()),
        tuple(equicorrelated_z.ravel()),
    }
    post_nonlinear_covariance_pairs = {
        (
            tuple(
                np.asarray(
                    row["setup"][0].keywords["column_covariance_z"]
                ).ravel()
            ),
            tuple(
                np.asarray(
                    row["setup"][0].keywords["stratum_covariance"]
                ).ravel()
            ),
        )
        for row in post_nonlinear_rows
    }
    assert post_nonlinear_covariance_pairs == {
        (tuple(column.ravel()), tuple(stratum.ravel()))
        for column in (np.eye(3), equicorrelated_z)
        for stratum in (np.eye(4), equicorrelated_stratum)
    }


@pytest.mark.parametrize(
    "filename, expected_dgp, expected_prefix",
    [
        (
            "config_conditioning_null_gaussian.yaml",
            dgp.GaussianNetwork,
            "conditioning_ac_null_gaussian",
        ),
        (
            "config_conditioning_null_bernoulli.yaml",
            dgp.BernoulliNetwork,
            "conditioning_ac_null_bernoulli",
        ),
    ],
)
def test_split_null_configs_select_one_network_type(
    filename, expected_dgp, expected_prefix
):
    config = load_config(ROOT / filename)
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

    assert len(config["setups"]) == 12
    assert len(conditional_rows) == 324
    assert len(post_nonlinear_rows) == 72
    assert all(factory.func is expected_dgp for factory, _ in config["setups"])
    assert config["output"]["file_prefix"] == expected_prefix


@pytest.mark.parametrize(
    "filename, expected_dgp, setup_count, design_count, expected_prefix",
    [
        (
            "config_conditional_copula_gaussian.yaml",
            dgp.GaussianNetwork,
            12,
            972,
            "conditional_copula_ac_power_alternative_gaussian",
        ),
        (
            "config_conditional_copula_bernoulli.yaml",
            dgp.BernoulliNetwork,
            12,
            972,
            "conditional_copula_ac_power_alternative_bernoulli",
        ),
        (
            "config_postlinearnoise_gaussian.yaml",
            dgp.GaussianNetwork,
            8,
            216,
            "postlinearnoise_ac_power_alternative_gaussian",
        ),
        (
            "config_postlinearnoise_bernoulli.yaml",
            dgp.BernoulliNetwork,
            8,
            216,
            "postlinearnoise_ac_power_alternative_bernoulli",
        ),
    ],
)
def test_split_alternative_configs_select_one_network_type(
    filename, expected_dgp, setup_count, design_count, expected_prefix
):
    config = load_config(ROOT / filename)

    assert len(config["setups"]) == setup_count
    assert len(build_factorial_design(config)) == design_count
    assert all(factory.func is expected_dgp for factory, _ in config["setups"])
    assert config["output"]["file_prefix"] == expected_prefix


@pytest.mark.parametrize(
    "filename",
    [
        "config_conditioning_null_gaussian.yaml",
        "config_conditioning_null_bernoulli.yaml",
        "config_conditional_copula_gaussian.yaml",
        "config_conditional_copula_bernoulli.yaml",
        "config_postlinearnoise_gaussian.yaml",
        "config_postlinearnoise_bernoulli.yaml",
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
    dgp = dgp_factory(**runtime)
    data = dgp.generate()

    method = row["method"](**runtime)
    method.fit(data)

    assert data["A"].shape == (24, 24)
    assert data["Z"].shape == (24, 3)
    assert data["Y"].shape == (24, 1)
    assert data["X"].shape == (24, 1)
    assert np.isfinite(method.test_stat_estimate)
    assert len(method.permutation_distribution) == 2
    if dgp_factory.keywords.get("post_nonlinear_noise") is not None:
        assert dgp.latent_sampler.function_type == row["function_type"]
