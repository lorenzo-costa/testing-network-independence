from pathlib import Path

import numpy as np
import pandas as pd

from src.helper_functions.simulation_functions import _build_seeded_scenarios
import src.run_sim_script_shard as shard_script


def _seed_keys(tasks):
    return {task[-1].bit_generator.seed_seq.spawn_key for task in tasks}


def test_seeded_scenario_shards_are_disjoint_and_complete():
    factorial = [{"scenario": index} for index in range(5)]
    shards = []
    for shard_index in range(3):
        tasks, global_total = _build_seeded_scenarios(
            4,
            factorial,
            [],
            None,
            np.random.default_rng(18),
            shard_index=shard_index,
            num_shards=3,
            shuffle=True,
        )
        assert global_total == 20
        shards.append(tasks)

    keys = [_seed_keys(tasks) for tasks in shards]
    assert not keys[0] & keys[1]
    assert not keys[0] & keys[2]
    assert not keys[1] & keys[2]
    assert len(set.union(*keys)) == 20
    assert [len(tasks) for tasks in shards] == [7, 7, 6]


def test_shard_runner_uses_slurm_values_and_unique_output(monkeypatch, tmp_path):
    config = {
        "simulation": {
            "nsim": 4,
            "parallel": True,
            "n_jobs": -1,
            "batch_size": 8,
            "blas_threads": 1,
        },
        "rng": np.random.default_rng(4),
        "metrics": [],
        "output": {"results_dir": str(tmp_path), "file_prefix": "linear"},
    }
    design = [{"n": 20}, {"n": 40}]
    captured = {}

    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "12345")
    monkeypatch.setattr(shard_script, "load_config", lambda path: config)
    monkeypatch.setattr(
        shard_script, "build_factorial_design_multi", lambda configs: design
    )

    def fake_run_simulation(**kwargs):
        captured.update(kwargs)
        kwargs["result_callback"](
            {
                "args": {
                    "n": 20,
                    "dgp_name": "GaussianNetwork_multiple_networks",
                    "method_name": "RV_PermutationTest_latent",
                },
            }
        )

    monkeypatch.setattr(shard_script, "run_simulation", fake_run_simulation)
    output = shard_script.main(
        [
            "--config",
            "linear_model_config.yaml",
            "--shard-index",
            "1",
            "--num-shards",
            "3",
            "--n-jobs",
            "32",
        ]
    )

    assert captured["shard_index"] == 1
    assert captured["num_shards"] == 3
    assert captured["n_jobs"] == 32
    assert captured["blas_threads"] == 1
    assert callable(captured["result_callback"])
    assert Path(output).name == "linear_12345_shard-001-of-003.csv"
    saved = pd.read_csv(output)
    assert len(saved) == 1


def test_csv_result_writer_flushes_bounded_batches(tmp_path):
    output = tmp_path / "streamed.csv"
    writer = shard_script._CsvResultWriter(output, batch_size=2)

    for n in (10, 20, 30):
        writer.add({"value": n, "args": {"n": n}})

    assert writer.rows_written == 2
    assert len(writer.buffer) == 1

    writer.close()

    saved = pd.read_csv(output)
    assert saved["value"].tolist() == [10, 20, 30]
    assert saved["n"].tolist() == [10, 20, 30]
