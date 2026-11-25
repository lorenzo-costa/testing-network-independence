from power_temp import METHODS, experiment_design, parse_args


def test_continuous_copula_power_design():
    args = parse_args(["--nsim", "1", "--n", "20", "--npermutations", "20"])
    design = experiment_design(args)

    assert len(design) == 4 * 2 * len(METHODS)
    assert {(row["correlation_zx"], row["correlation_yx"]) for row in design} == {
        (0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)
    }
    for row in design:
        dgp = row["setup"][0]
        assert dgp.keywords["continuous_conditional_copula"] == "gaussian"
        assert dgp.keywords["marginals"]["z"] == "uniform 0 1"
        assert dgp.keywords["conditional_correlation_zy"] == (
            0.0 if row["condition"] == "null" else 0.5
        )
        assert row["method"].keywords["conditional_generator"] == "from_dgp"
