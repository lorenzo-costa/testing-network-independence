from pathlib import Path

import numpy as np

from src.load_config import build_factorial_design, load_config


CONFIG = Path(__file__).resolve().parents[1] / "config_rdpg_continuous_conditional.yaml"


def test_rdpg_continuous_config_resolves_four_three_dimensional_cases():
    config = load_config(CONFIG)
    design = build_factorial_design(config)

    assert len(config["setups"]) == 4
    assert len(design) == 4 * 4 * 3
    marginal_dependence = []
    conditional_dependence = []
    for dgp, _ in config["setups"]:
        kwargs = dgp.keywords
        assert kwargs["px"] == 3
        assert kwargs["continuous_conditional_copula"] == "gaussian"
        loading_z = np.asarray(kwargs["correlation_zx"])
        loading_y = np.asarray(kwargs["correlation_yx"])
        residual = np.asarray(kwargs["conditional_correlation_zy"])
        assert loading_z.shape == loading_y.shape == residual.shape == (3, 3)
        scale_z = np.sqrt(1 - np.sum(loading_z**2, axis=1))
        scale_y = np.sqrt(1 - np.sum(loading_y**2, axis=1))
        marginal_cross = loading_z @ loading_y.T + residual * np.outer(scale_z, scale_y)
        marginal_dependence.append(not np.allclose(marginal_cross, 0))
        conditional_dependence.append(not np.allclose(residual, 0))

    assert list(zip(marginal_dependence, conditional_dependence)) == [
        (False, True), (True, False), (True, True), (False, False)
    ]
