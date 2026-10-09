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
