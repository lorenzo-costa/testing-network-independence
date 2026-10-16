import csv

from src.helper_functions.simulation_output import CsvResultWriter
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
