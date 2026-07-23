from pathlib import Path

import numpy as np

from src.load_config import build_factorial_design, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_null_config_crosses_networks_marginals_and_column_covariances():
    config = load_config(ROOT / "config_null.yaml")
    design = build_factorial_design(config)

    network_types = {row["setup"][0].func.__name__ for row in design}
    marginal_pairs = {
        (row["marginals"]["y"], row["marginals"]["z"]) for row in design
    }
    covariance_matrices = {
        tuple(np.asarray(row["column_covariance"]).ravel()) for row in design
    }

    assert config["simulation"]["rho"] == [0.0]
    assert network_types == {"GaussianNetwork", "BernoulliNetwork"}
    assert len(marginal_pairs) == 9
    assert len(covariance_matrices) == 3
    assert len(design) == 972


def test_null_config_column_covariance_reaches_the_latent_sampler():
    config = load_config(ROOT / "config_null.yaml")
    row = build_factorial_design(config)[0]
    dgp_factory, _ = row["setup"]
    runtime = dict(row)
    runtime.update(n=12, rng=np.random.default_rng(40))

    dgp = dgp_factory(**runtime)

    np.testing.assert_array_equal(
        dgp.latent_sampler.column_covariance_z,
        row["column_covariance"],
    )
