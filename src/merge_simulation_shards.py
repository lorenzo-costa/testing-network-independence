"""Merge completed simulation shards without reading files still being written."""

import argparse
import csv
import os
from pathlib import Path
import time

from src.helper_functions.simulation_functions import shard_row_count
from src.load_config import build_factorial_design_multi, load_config


def _csv_row_count(path):
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.reader(source)
        header = next(reader, None)
        if header is None:
            return 0
        count = 0
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"Malformed CSV row in {path}")
            count += 1
        return count


def merge_shard_outputs(results_dir, prefix, array_job_id, expected_rows, shard_index):
    """Mark one finished shard, then merge only when every shard is complete.

    The caller must invoke this after the shard writer has closed successfully.
    A completion marker is created only after its CSV has the expected row count.
    The lock serializes validation, marker creation, and the final merge.
    """
    results_dir = Path(results_dir)
    num_shards = len(expected_rows)
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Require a valid shard index and at least one shard")
    if any(count < 0 for count in expected_rows):
        raise ValueError("Expected row counts must be nonnegative")

    stem = f"{prefix}_{array_job_id}"
    final_path = results_dir / f"{stem}.csv"
    shard_paths = [
        results_dir / f"{stem}_shard-{index:03d}-of-{num_shards:03d}.csv"
        for index in range(num_shards)
    ]
    marker_paths = [
        results_dir / f".{stem}_shard-{index:03d}-of-{num_shards:03d}.done"
        for index in range(num_shards)
    ]
    lock_dir = results_dir / f".{stem}.merge.lock"
    temporary_path = results_dir / f".{stem}.tmp-{shard_index}"

    deadline = time.monotonic() + 300
    while True:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for merge lock {lock_dir}")
            time.sleep(0.25)

    try:
        total_expected = sum(expected_rows)
        if final_path.is_file():
            actual = _csv_row_count(final_path)
            if actual != total_expected:
                raise ValueError(
                    f"Existing merged file has {actual} rows; expected {total_expected}: "
                    f"{final_path}"
                )
            print(f"Merged result already exists at {final_path}.")
            return final_path

        own_count = _csv_row_count(shard_paths[shard_index])
        if own_count != expected_rows[shard_index]:
            raise ValueError(
                f"Shard {shard_index} has {own_count} rows; "
                f"expected {expected_rows[shard_index]}"
            )
        marker_paths[shard_index].mkdir(exist_ok=True)

        unfinished = [path for path in marker_paths if not path.is_dir()]
        if unfinished:
            print(f"Merge deferred: {len(unfinished)} of {num_shards} shards have not finished.")
            return None

        header = None
        rows_written = 0
        with temporary_path.open("x", newline="", encoding="utf-8") as destination:
            writer = csv.writer(destination)
            for index, shard_path in enumerate(shard_paths):
                with shard_path.open(newline="", encoding="utf-8") as source:
                    reader = csv.reader(source)
                    shard_header = next(reader, None)
                    if shard_header is None:
                        if expected_rows[index] != 0:
                            raise ValueError(f"Shard {index} is empty")
                        continue
                    if header is None:
                        header = shard_header
                        writer.writerow(header)
                    elif shard_header != header:
                        raise ValueError(f"CSV header differs in {shard_path}")

                    shard_count = 0
                    for row in reader:
                        if len(row) != len(header):
                            raise ValueError(f"Malformed CSV row in {shard_path}")
                        writer.writerow(row)
                        shard_count += 1
                    if shard_count != expected_rows[index]:
                        raise ValueError(
                            f"Shard {index} has {shard_count} rows; "
                            f"expected {expected_rows[index]}"
                        )
                    rows_written += shard_count

        if header is None or rows_written != total_expected:
            raise ValueError(
                f"Merged {rows_written} rows; expected {total_expected}"
            )

        os.replace(temporary_path, final_path)
        for shard_path in shard_paths:
            shard_path.unlink()
        for marker_path in marker_paths:
            marker_path.rmdir()
        print(f"Merged {rows_written} rows into {final_path} and deleted all shards.")
        return final_path
    finally:
        temporary_path.unlink(missing_ok=True)
        lock_dir.rmdir()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file")
    parser.add_argument("array_job_id")
    parser.add_argument("num_shards", type=int)
    parser.add_argument("shard_index", type=int)
    args = parser.parse_args(argv)

    config = load_config(args.config_file)
    factorial = build_factorial_design_multi([config])
    nsim = config["simulation"]["nsim"]
    expected_rows = [
        shard_row_count(nsim, factorial, index, args.num_shards)
        for index in range(args.num_shards)
    ]
    output = config["output"]
    return merge_shard_outputs(
        output["results_dir"],
        output.get("file_prefix", "simulation_results"),
        args.array_job_id,
        expected_rows,
        args.shard_index,
    )


if __name__ == "__main__":
    main()
