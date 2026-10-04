import csv

import numpy as np

from src.helper_functions import simulation_functions
from src.helper_functions.simulation_output import CsvResultWriter


def _fake_run_scenario(metrics, args, seed, method_params=None):
    return {
        "scenario": args["scenario"],
        "marker": int(np.random.default_rng(seed).integers(0, 2**31)),
    }


def test_shards_cover_the_same_deterministic_scenarios(monkeypatch):
    monkeypatch.setattr(simulation_functions, "run_scenario", _fake_run_scenario)
    design = [{"scenario": "first"}, {"scenario": "second"}]

    whole = simulation_functions.run_simulation(
        nsim=3,
        factorial_design=design,
        metrics=[],
        rng=np.random.default_rng(22),
    )
    shards = [
        simulation_functions.run_simulation(
            nsim=3,
            factorial_design=design,
            metrics=[],
            rng=np.random.default_rng(22),
            shard_index=shard_index,
            num_shards=4,
        )
        for shard_index in range(4)
    ]

    combined = [result for shard in shards for result in shard]
    assert {(row["scenario"], row["marker"]) for row in combined} == {
        (row["scenario"], row["marker"]) for row in whole
    }


def test_result_callback_streams_rows_without_retaining_them(monkeypatch):
    monkeypatch.setattr(simulation_functions, "run_scenario", _fake_run_scenario)
    written = []

    returned = simulation_functions.run_simulation(
        nsim=2,
        factorial_design=[{"scenario": "only"}],
        metrics=[],
        rng=np.random.default_rng(9),
        result_callback=written.append,
    )

    assert returned is None
    assert len(written) == 2


def test_csv_result_writer_batches_rows(tmp_path):
    path = tmp_path / "results.csv"
    writer = CsvResultWriter(path, batch_size=2)
    writer.add({"value": 1, "args": {"n": 10}})
    writer.add({"value": 2, "args": {"n": 20}})
    writer.close()

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert writer.rows_written == 2
    assert [row["value"] for row in rows] == ["1", "2"]
