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
    assert len(design) == 30 * 2 * 3 * 7
    assert all(row["k"] == 4 and row["ky"] == 1 for row in design)

    for distribution, expected_count, threshold in [
        ("gaussian", 15, 0.0), ("uniform_rdpg", 15, 0.5),
    ]:
        rows = [row for row in design if row["predictor_distribution"] == distribution]
        by_form = {row["functional_form"]: row for row in rows}
        assert len(by_form) == expected_count
        assert "max" in by_form
        assert "abs_max" in by_form
        for form in ("threshold", "piecewise"):
            assert by_form[form]["function_params"]["threshold"] == threshold
        assert by_form["sparse_nonlinear"]["function_params"]["active_count"] == 1
        for form in ("nonlinear_additive", "sparse_nonlinear"):
            assert by_form[form]["function_params"]["quadratic_scale"] == 1.0

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
            if distribution == "uniform_rdpg" and {left, right} == {"max", "abs_max"}:
                np.testing.assert_array_equal(signals[left], signals[right])
                continue
            assert not np.allclose(signals[left], signals[right]), (left, right)
