from pathlib import Path

import numpy as np
import pandas as pd

import src.run_simulation_script as script


def test_runner_uses_yaml_parallel_settings_and_flattens_linear_model_columns(
    monkeypatch, tmp_path
):
    config = {
        "simulation": {
            "nsim": 2,
            "parallel": True,
            "n_jobs": -1,
            "batch_size": 7,
            "blas_threads": 1,
        },
        "rng": np.random.default_rng(12),
        "metrics": [],
        "output": {"results_dir": str(tmp_path), "file_prefix": "linear"},
    }
    design = [
        {
            "n": 20,
            "p": 3,
            "d_x": 2,
            "d_y": 4,
            "snr": 0.5,
            "hypothesis": "H1",
        }
    ]
    captured = {}

    monkeypatch.setattr(script, "load_config", lambda path: config)
    monkeypatch.setattr(script, "build_factorial_design_multi", lambda cfgs: design)

    def fake_run_simulation(**kwargs):
        captured.update(kwargs)
        return [
            {
                "args": {
                    **design[0],
                    "dgp_name": "GaussianNetwork_multiple_networks",
                    "method_name": "RV_PermutationTest_latent",
                }
            }
        ]

    monkeypatch.setattr(script, "run_simulation", fake_run_simulation)

    output_path = script.main(["--config", "linear_model_config.yaml"])

    assert captured["nsim"] == 2
    assert captured["parallel"] is True
    assert captured["n_jobs"] is None
    assert captured["batch_size"] == 7
    assert captured["blas_threads"] == 1
    saved = pd.read_csv(output_path)
    assert Path(output_path).parent == tmp_path
    assert saved.loc[0, "p"] == 3
    assert saved.loc[0, "d_x"] == 2
    assert saved.loc[0, "d_y"] == 4
    assert saved.loc[0, "snr"] == 0.5
    assert saved.loc[0, "hypothesis"] == "H1"
