import numpy as np

from power_temp import (CONDITIONING_DIMENSION, LATENT_DIMENSION, METHODS, RESPONSE_DIMENSION,
                        experiment_design, parse_args)


def test_continuous_copula_power_design():
    args = parse_args(["--nsim", "1", "--n", "20", "--npermutations", "20"])
    design = experiment_design(args)

    assert len(design) == 4 * len(METHODS)
    assert LATENT_DIMENSION == RESPONSE_DIMENSION == CONDITIONING_DIMENSION == 3
    assert {row["correlation_zx"] for row in design} == {0.0, 0.7}
    assert {row["correlation_yx"] for row in design} == {0.0, 0.7}
    assert {(row["marginal_dependence"], row["conditional_dependence"])
            for row in design} == {
        ("independent", "dependent"),
        ("dependent", "independent"),
        ("dependent", "dependent"),
        ("independent", "independent"),
    }
    for row in design:
        dgp = row["setup"][0]
        assert dgp.keywords["continuous_conditional_copula"] == "gaussian"
        assert dgp.keywords["px"] == 3
        assert np.isclose(
            float(dgp.keywords["marginals"]["z"].split()[-1]),
            1 / np.sqrt(LATENT_DIMENSION),
        )
        rho_zx = row["correlation_zx"]
        rho_yx = row["correlation_yx"]
        rho_zy_given_x = row["conditional_correlation_zy"]
        np.testing.assert_allclose(dgp.keywords["correlation_zx"], rho_zx * np.eye(3))
        np.testing.assert_allclose(dgp.keywords["correlation_yx"], rho_yx * np.eye(3))
        np.testing.assert_allclose(
            dgp.keywords["conditional_correlation_zy"], rho_zy_given_x * np.eye(3)
        )
        marginal_rho = (
            rho_zx * rho_yx
            + np.sqrt((1 - rho_zx**2) * (1 - rho_yx**2)) * rho_zy_given_x
        )
        assert np.isclose(marginal_rho, 0) == (
            row["marginal_dependence"] == "independent"
        )
        assert (rho_zy_given_x == 0) == (
            row["conditional_dependence"] == "independent"
        )
        assert row["method"].keywords["conditional_generator"] == "from_dgp"


def test_three_dimensional_scenarios_generate_valid_rdpg_probabilities():
    args = parse_args(["--nsim", "1", "--n", "20", "--npermutations", "20"])
    rows = experiment_design(args)
    scenarios = {row["scenario"]: row for row in rows}

    for row in scenarios.values():
        network = row["setup"][0](
            n=20, k=LATENT_DIMENSION, ky=RESPONSE_DIMENSION,
            rng=np.random.default_rng(7),
        )
        data = network.generate()
        assert data["Z"].shape == (20, 3)
        assert data["Y"].shape == (20, 3)
        assert data["X"].shape == (20, 3)
        assert np.min(data["Z"]) >= 0
        assert np.max(data["Z"] @ data["Z"].T) <= 1
