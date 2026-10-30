from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.load_config import build_factorial_design, load_config


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config_functionals_2.yaml"


@pytest.mark.parametrize("paired_sweep", [False, True])
def test_k4_design_preserves_requested_functionals(tmp_path, paired_sweep):
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
    assert len(design) == (
        len(raw["simulation"]["functionals"])
        * len(raw["simulation"]["n"])
        * len(raw["simulation"]["snr"])
        * len(raw["methods"]["list"])
        * 2
    )
    assert all(row["k"] == 4 and row["ky"] == 1 for row in design)

    for distribution, expected_count, threshold in [
        ("gaussian", 11, 0.0), ("uniform_rdpg", 11, 0.5),
    ]:
        rows = [row for row in design if row["predictor_distribution"] == distribution]
        by_form = {row["functional_form"]: row for row in rows}
        assert len(by_form) == expected_count
        assert set(by_form) == {
            "linear", "interaction", "radial", "threshold", "sigmoid",
            "piecewise", "sine", "max", "abs_max", "tanh_product", "pareto",
        }
        assert by_form["threshold"]["function_params"]["threshold"] == threshold
        assert by_form["piecewise"]["function_params"]["threshold"] == 0.5
        interaction_params = by_form["interaction"]["function_params"]
        if distribution == "gaussian":
            assert interaction_params == {
                "output_center": 0.0,
                "output_sd": pytest.approx(np.sqrt(6.0)),
            }
        else:
            interaction_variance = (
                1.0 / 8.0
                + 1.0 / np.pi
                - 7.0 / (2.0 * np.pi**2)
            )
            assert interaction_params == {
                "output_center": pytest.approx(2.0 / np.pi),
                "output_sd": pytest.approx(np.sqrt(interaction_variance)),
            }

        signals = {}
        for form, row in by_form.items():
            factory, _ = row["setup"]
            sampler = factory(**dict(row, n=256, rng=np.random.default_rng(51))).latent_sampler
            z, y = sampler.sample_latent()
            assert z.shape == (256, 4) and y.shape == (256, 1)
            assert np.isfinite(y).all()
            signals[form] = sampler._evaluate_function(z)
            assert np.var(signals[form]) > 0
        for left, right in combinations(signals, 2):
            assert not np.allclose(signals[left], signals[right]), (left, right)
