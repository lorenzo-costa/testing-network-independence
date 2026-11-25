import numpy as np

from power_temp import METHODS, experiment_design, parse_args


def test_continuous_copula_power_design():
    args = parse_args(["--nsim", "1", "--n", "20", "--npermutations", "20"])
    design = experiment_design(args)

    assert len(design) == 4 * len(METHODS)
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
        assert dgp.keywords["marginals"]["z"] == "uniform 0 1"
        rho_zx = row["correlation_zx"]
        rho_yx = row["correlation_yx"]
        rho_zy_given_x = row["conditional_correlation_zy"]
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
