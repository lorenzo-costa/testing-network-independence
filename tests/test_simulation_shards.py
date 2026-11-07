import csv

import pytest

from src.helper_functions.simulation_output import CsvResultWriter
from src.merge_simulation_shards import merge_shard_outputs
from tests.test_simulation_grouping import row, normalized, run


def test_shards_cover_the_same_deterministic_rows():
    design = [row("first"), row("second"), row("different-data", strength=0.8)]
    whole = run(design, nsim=3, seed=22)
    shards = [run(design, nsim=3, seed=22,
                  shard_index=index, num_shards=4) for index in range(4)]
    combined = [result for shard in shards for result in shard]
    assert len(combined) == len(whole)
    assert normalized(combined) == normalized(whole)


def test_result_callback_streams_rows_without_retaining_them():
    written = []
    returned = run([row("only")], nsim=2, seed=9,
                   result_callback=written.append)
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


def _write_shard(path, values):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["value"])
        writer.writerows([[value] for value in values])


def test_merge_waits_for_closed_shards_even_when_all_files_exist(tmp_path):
    first = tmp_path / "simulation_results_123_shard-000-of-002.csv"
    second = tmp_path / "simulation_results_123_shard-001-of-002.csv"
    _write_shard(first, [1, 2])
    _write_shard(second, [3, 4])

    assert merge_shard_outputs(tmp_path, "simulation_results", "123", [2, 2], 0) is None
    assert first.exists() and second.exists()
    assert not (tmp_path / "simulation_results_123.csv").exists()

    final = merge_shard_outputs(tmp_path, "simulation_results", "123", [2, 2], 1)
    with final.open(newline="", encoding="utf-8") as file:
        assert [row["value"] for row in csv.DictReader(file)] == ["1", "2", "3", "4"]
    assert not first.exists() and not second.exists()


def test_merge_preserves_sources_if_a_completed_shard_changes(tmp_path):
    first = tmp_path / "simulation_results_123_shard-000-of-002.csv"
    second = tmp_path / "simulation_results_123_shard-001-of-002.csv"
    _write_shard(first, [1, 2])
    _write_shard(second, [3, 4])
    merge_shard_outputs(tmp_path, "simulation_results", "123", [2, 2], 0)

    with first.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([5])

    with pytest.raises(ValueError, match="Shard 0 has 3 rows; expected 2"):
        merge_shard_outputs(tmp_path, "simulation_results", "123", [2, 2], 1)
    assert first.exists() and second.exists()
    assert not (tmp_path / "simulation_results_123.csv").exists()
