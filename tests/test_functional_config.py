from itertools import product
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.load_config import build_factorial_design, load_config


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config_functionals.yaml"


@pytest.mark.parametrize("snr", [0.25, [0.1, 0.2, 0.3, 0.4, 0.5]])
def test_snr_config_sweeps_predictors_and_reaches_sampler(tmp_path, snr):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["simulation"]["snr"] = snr
    path = tmp_path / "functionals.yaml"
    path.write_text(yaml.safe_dump(raw))

    config = load_config(path)
    design = build_factorial_design(config)
    expected_snrs = snr if isinstance(snr, list) else [snr]
    distributions = raw["simulation"]["predictor_distribution"]
    forms = [entry["name"] for entry in raw["simulation"]["functionals"]]

    assert {
        (row["rho"], row["predictor_distribution"], row["functional_form"])
        for row in design
    } == set(product(expected_snrs, distributions, forms))
    assert len(design) == (
        len(expected_snrs) * len(distributions) * len(forms)
        * len(raw["simulation"]["n"]) * len(raw["methods"]["list"])
    )
    assert all("noise_scale" not in row for row in design)
    for rho in expected_snrs:
        row = next(row for row in design if row["rho"] == rho)
        factory, _ = row["setup"]
        runtime = dict(row, n=12, rng=np.random.default_rng(50))
        network = factory(**runtime)
        assert network.latent_sampler.rho == rho
        assert np.isfinite(network.generate()["Y"]).all()


@pytest.mark.parametrize("snr, error", [
    ([], ValueError), (0, ValueError), (-0.1, ValueError),
    (1.1, ValueError), (float("nan"), ValueError),
    (float("inf"), ValueError), (True, TypeError),
    ("0.25", TypeError), ([0.2, None], TypeError),
])
def test_invalid_snr_is_rejected_when_loading(tmp_path, snr, error):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["simulation"]["snr"] = snr
    path = tmp_path / "functionals.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(error, match="simulation.snr"):
        load_config(path)


def test_snr_and_rho_cannot_conflict(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["simulation"]["rho"] = [0.8]
    path = tmp_path / "functionals.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="Specify only one"):
        load_config(path)


@pytest.mark.parametrize("paired_sweep", [False, True])
def test_piecewise_settings_reach_both_predictors(tmp_path, paired_sweep):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    if paired_sweep:
        distributions = raw["simulation"].pop("predictor_distribution")
        raw["simulation"]["predictor_noise_sweeps"] = [
            {"predictor_distribution": distribution, "noise_scale": 0.0}
            for distribution in distributions
        ]
    path = tmp_path / "functionals.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_config(path)
    design = build_factorial_design(config)
    piecewise_rows = [row for row in design if row["functional_form"] == "piecewise"]

    for distribution, threshold, expected in [
        ("gaussian", 0.5, [-0.25, -0.5, 0.75]),
        ("uniform_rdpg", 0.5, [-0.25, -0.5, 0.75]),
    ]:
        rows = [row for row in piecewise_rows if row["predictor_distribution"] == distribution]
        assert rows
        assert all(row["function_params"]["threshold"] == threshold for row in rows)
        row = rows[0]
        factory, _ = row["setup"]
        sampler = factory(**dict(row, n=3)).latent_sampler
        signal = sampler._evaluate_function(np.array([[0.25], [0.5], [0.75]]))
        np.testing.assert_allclose(signal, expected)

    descriptor = next(
        entry for entry in config["simulation"]["functionals"]
        if entry["functional_form"] == "piecewise"
    )
    assert descriptor["function_params"]["threshold"] == 0.5
    assert all(
        "threshold" not in row["function_params"]
        for row in design if row["functional_form"] not in {"piecewise", "sigmoid"}
    )


def test_abs_max_is_centered_and_scaled_only_for_uniform_predictors():
    design = build_factorial_design(load_config(CONFIG_PATH))
    rows = [row for row in design if row["functional_form"] == "abs_max"]

    for distribution, expected in [
        ("gaussian", [0.0, 0.5, 1.0]),
        ("uniform_rdpg", [3.0, 0.0, 3.0]),
    ]:
        row = next(
            row for row in rows
            if row["predictor_distribution"] == distribution
        )
        factory, _ = row["setup"]
        sampler = factory(**dict(row, n=3)).latent_sampler
        signal = sampler._evaluate_function(np.array([[0.0], [0.5], [1.0]]))
        np.testing.assert_allclose(signal, expected)


@pytest.mark.parametrize("paired_sweep", [False, True])
def test_functional_solver_matches_network_link(tmp_path, paired_sweep):
    from src.solvers.weighted_network import ASE
    from src.solvers.MaMa_uuuuu import pgd_fit_wrapper

    raw = yaml.safe_load(CONFIG_PATH.read_text())
    if paired_sweep:
        distributions = raw["simulation"].pop("predictor_distribution")
        raw["simulation"]["predictor_noise_sweeps"] = [
            {"predictor_distribution": distribution, "noise_scale": 0.0}
            for distribution in distributions
        ]
    path = tmp_path / "functionals.yaml"
    path.write_text(yaml.safe_dump(raw))
    design = build_factorial_design(load_config(path))
    expected = {"uniform_rdpg": ASE, "gaussian": pgd_fit_wrapper}
    assert {row["predictor_distribution"] for row in design} == set(expected)
    for row in design:
        factory, solver = row["setup"]
        distribution = row["predictor_distribution"]
        assert solver is expected[distribution]
        network = factory(**dict(row, n=5))
        assert network._uses_rdpg_link() == (distribution == "uniform_rdpg")


def test_missing_functional_solver_mapping_is_rejected(tmp_path):
    raw = yaml.safe_load(CONFIG_PATH.read_text())
    del raw["setups"][0]["solver"]["gaussian"]
    path = tmp_path / "functionals.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="No functional solver.*gaussian"):
        build_factorial_design(load_config(path))


def test_scalar_functional_solver_remains_supported(tmp_path):
    from src.solvers.weighted_network import ASE

    raw = yaml.safe_load(CONFIG_PATH.read_text())
    raw["setups"][0]["solver"] = "ASE"
    path = tmp_path / "functionals.yaml"
    path.write_text(yaml.safe_dump(raw))
    design = build_factorial_design(load_config(path))
    assert design
    assert all(row["setup"][1] is ASE for row in design)
