"""Validate complete shard sets and read CSV rows in deterministic file order."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from collections.abc import Iterator, Sequence
import re
import pandas as pd

_SHARD_FILENAME = re.compile(
    r"^(?P<run>.+)_shard-(?P<index>\d+)-of-(?P<count>\d+)\.csv$"
)


def _resolve_shard_metadata(
    results_dir: str | PathLike[str],
    filenames: Sequence[str | PathLike[str]],
) -> list[tuple[str, int, int, Path]]:
    """Resolve and validate one complete, non-duplicated shard set."""
    if not filenames:
        raise ValueError("At least one shard filename must be provided.")

    root = Path(results_dir).expanduser().resolve()
    paths = [Path(filename) for filename in filenames]
    paths = [path if path.is_absolute() else root / path for path in paths]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        listed = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing shard result files:\n{listed}")

    metadata = []
    for path in paths:
        match = _SHARD_FILENAME.fullmatch(path.name)
        if match is None:
            raise ValueError(
                f"Shard filename does not match '_shard-III-of-NNN.csv': {path.name}"
            )
        metadata.append(
            (
                match.group("run"),
                int(match.group("index")),
                int(match.group("count")),
                path,
            )
        )

    run_ids = {run_id for run_id, _, _, _ in metadata}
    shard_counts = {count for _, _, count, _ in metadata}
    if len(run_ids) != 1 or len(shard_counts) != 1:
        raise ValueError("Shard files must belong to the same simulation run.")

    shard_count = shard_counts.pop()
    indices = [index for _, index, _, _ in metadata]
    if len(indices) != len(set(indices)):
        raise ValueError("Shard filenames contain duplicate shard indices.")
    expected = set(range(shard_count))
    actual = set(indices)
    if actual != expected:
        missing_indices = sorted(expected - actual)
        extra_indices = sorted(actual - expected)
        raise ValueError(
            "Incomplete shard set: "
            f"missing={missing_indices}, unexpected={extra_indices}."
        )

    return sorted(metadata, key=lambda item: item[1])


def iter_shard_outputs(
    results_dir: str | PathLike[str],
    filenames: Sequence[str | PathLike[str]],
    *,
    chunksize: int = 10_000,
    usecols: Sequence[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield validated shard rows in bounded-memory chunks."""
    if isinstance(chunksize, bool) or not isinstance(chunksize, int) or chunksize < 1:
        raise ValueError("chunksize must be a positive integer.")

    metadata = _resolve_shard_metadata(results_dir, filenames)
    for _, shard_index, shard_count, path in metadata:
        for frame in pd.read_csv(path, chunksize=chunksize, usecols=usecols):
            frame["source_file"] = path.name
            frame["shard_index"] = shard_index
            frame["num_shards"] = shard_count
            yield frame


def combine_shard_outputs(
    results_dir: str | PathLike[str],
    filenames: Sequence[str | PathLike[str]],
) -> pd.DataFrame:
    """Read, validate, and concatenate every CSV shard from one Slurm run.

    Filenames must use the shard runner's ``_shard-III-of-NNN.csv`` suffix.
    The function rejects mixed runs, duplicate shard indices, and incomplete
    shard sets so power and type-I-error estimates cannot silently use a
    partial simulation batch.
    """
    metadata = _resolve_shard_metadata(results_dir, filenames)
    frames = []
    for _, shard_index, shard_count, path in metadata:
        frame = pd.read_csv(path)
        frame["source_file"] = path.name
        frame["shard_index"] = shard_index
        frame["num_shards"] = shard_count
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
