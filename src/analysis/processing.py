"""Expand, filter and aggregate current simulation results.

Public functions preserve caller data; owned intermediate frames can be updated
in place. Grouping dimensions are always explicit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from .io import iter_shard_outputs
from .parsing import parse_config_string, parse_result_string

PLOT_CHUNKSIZE = 5_000

DEFAULT_COLUMN_ALIASES = {
    "dx": "d_x",
    "dy": "d_y",
    "rejection": "Rejection",
}

DEFAULT_NUMERIC_COLUMNS = (
    "n",
    "p",
    "d_x",
    "d_y",
    "snr",
    "alpha",
    "Rejection",
)


def merge_result_shards(
    results_dir: str | Path,
    filenames: Sequence[str | Path],
    *,
    chunksize: int = PLOT_CHUNKSIZE,
    usecols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Validate, read, and concatenate one complete simulation shard set.

    This is the staged loading step. The returned dataframe is
    deliberately raw: serialized configuration and metric columns are left
    untouched for :func:`preprocess_results` to expand in a separate step.
    ``source_file``, ``shard_index``, and ``num_shards`` identify each row's
    origin.
    """
    chunks = list(
        iter_shard_outputs(
            results_dir,
            filenames,
            chunksize=chunksize,
            usecols=usecols,
        )
    )
    if not chunks:
        raise ValueError("The shard files contain no result rows.")
    return pd.concat(chunks, ignore_index=True)


def merge_result_shard_sets(
    results_dir: str | Path,
    shard_sets: Mapping[str, Sequence[str | Path]],
    *,
    chunksize: int = PLOT_CHUNKSIZE,
    usecols: Sequence[str] | None = None,
    set_column: str = "result_set",
) -> pd.DataFrame:
    """Merge several independently validated shard sets into one dataframe.

    Each mapping value must be a complete shard run. The mapping key is added
    in ``set_column``, which makes it possible to filter primary, asymptotic,
    MRQAP, or other result families after preprocessing.
    """
    frames = []
    for name, filenames in shard_sets.items():
        if not filenames:
            continue
        frame = merge_result_shards(
            results_dir,
            filenames,
            chunksize=chunksize,
            usecols=usecols,
        )
        frame[set_column] = name
        frames.append(frame)
    if not frames:
        raise ValueError("At least one non-empty shard set must be provided.")
    return pd.concat(frames, ignore_index=True)


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "na", "nan", "none"}
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if np.isscalar(missing) else False


def _expand_mapping_column(
    values: pd.Series,
    parser: Callable[[Any], Mapping[str, Any]],
    fields: Sequence[str] | None,
) -> pd.DataFrame:
    records = values.map(parser)
    if fields is None:
        keys = sorted(
            {key for record in records if isinstance(record, Mapping) for key in record}
        )
    else:
        keys = list(dict.fromkeys(fields))
    return pd.DataFrame(
        [
            {
                key: record.get(key) if isinstance(record, Mapping) else None
                for key in keys
            }
            for record in records
        ],
        index=values.index,
    )


def _overlay_expanded_columns(
    base: pd.DataFrame,
    expanded: pd.DataFrame,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    result = base.copy() if copy else base
    for column in expanded:
        parsed = expanded[column]
        available = ~parsed.map(_is_missing_value)
        if column not in result:
            result[column] = parsed
            continue
        original = result[column].astype(object)
        original.loc[available] = parsed.loc[available]
        result[column] = original
    return result


def preprocess_results(
    raw_results: pd.DataFrame,
    *,
    config_column: str | None = "args",
    metric_column: str | None = "ComputeAll",
    config_fields: Sequence[str] | None = None,
    metric_fields: Sequence[str] | None = None,
    column_aliases: Mapping[str, str] | None = None,
    numeric_columns: Sequence[str] = DEFAULT_NUMERIC_COLUMNS,
    method_labeler: Callable[[Mapping[str, Any]], Any] | None = None,
    required_columns: Sequence[str] = (),
    keep_serialized: bool = False,
) -> pd.DataFrame:
    """Expand raw simulation rows into a tidy, analysis-ready dataframe.

    The function also accepts an already-tabular dataframe without serialized
    columns. With ``config_fields=None`` and ``metric_fields=None`` every
    discovered key is retained, so experiment-specific attributes such as
    ``approximation`` or ``asymptotic_null`` remain available for filtering,
    faceting, or defining separate plotted series.

    Parameters
    ----------
    raw_results
        Raw merged shard rows or a similarly structured dataframe.
    config_column, metric_column
        Columns containing serialized dictionaries. Pass ``None`` when that
        source is absent.
    config_fields, metric_fields
        Keys to extract. ``None`` discovers and retains every key.
    column_aliases
        Source-to-canonical column names. The defaults normalize ``dx``,
        ``dy``, and lowercase ``rejection``.
    method_labeler
        Optional callable receiving the complete processed row as a mapping.
    required_columns
        Columns whose presence is required after preprocessing.
    """
    if not isinstance(raw_results, pd.DataFrame):
        raise TypeError("raw_results must be a pandas DataFrame")

    serialized = {
        column
        for column in (config_column, metric_column)
        if column is not None and column in raw_results
    }
    result = raw_results.copy()
    if not keep_serialized:
        result = result.drop(columns=serialized)

    if config_column is not None and config_column in raw_results:
        config_values = _expand_mapping_column(
            raw_results[config_column],
            parse_config_string,
            config_fields,
        )
        result = _overlay_expanded_columns(result, config_values, copy=False)
    if metric_column is not None and metric_column in raw_results:
        metric_values = _expand_mapping_column(
            raw_results[metric_column],
            parse_result_string,
            metric_fields,
        )
        result = _overlay_expanded_columns(result, metric_values, copy=False)

    aliases = DEFAULT_COLUMN_ALIASES if column_aliases is None else column_aliases
    for source, target in aliases.items():
        if source not in result or source == target:
            continue
        if target not in result:
            result = result.rename(columns={source: target})
            continue
        missing = result[target].map(_is_missing_value)
        result.loc[missing, target] = result.loc[missing, source]
        result = result.drop(columns=source)

    if method_labeler is not None:
        result["method"] = result.apply(
            lambda row: method_labeler(row.to_dict()),
            axis=1,
        )

    for column in numeric_columns:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    missing_columns = [column for column in required_columns if column not in result]
    if missing_columns:
        raise ValueError(f"Processed results are missing columns: {missing_columns}")
    return result.reset_index(drop=True)


def filter_results(
    results: pd.DataFrame,
    filters: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return rows matching scalar, collection, or callable filters."""
    if not filters:
        return results.copy()
    selected = pd.Series(True, index=results.index)
    for column, criterion in filters.items():
        if column not in results:
            raise KeyError(f"Filter column is not present: {column}")
        values = results[column]
        if callable(criterion):
            mask = criterion(values)
        elif _is_missing_value(criterion):
            mask = values.map(_is_missing_value)
        elif isinstance(criterion, Sequence) and not isinstance(
            criterion,
            (str, bytes),
        ):
            mask = values.isin(criterion)
        else:
            mask = values == criterion
        selected &= pd.Series(mask, index=results.index).fillna(False).astype(bool)
    return results.loc[selected].copy()


def aggregate_metric(
    results: pd.DataFrame,
    value: str,
    groupby: str | Sequence[str],
    *,
    mean_column: str | None = None,
    error: str | None = "sem",
    error_column: str | None = None,
    count_column: str = "replicates",
) -> pd.DataFrame:
    """Aggregate any numeric metric over explicitly chosen dimensions.

    Keeping ``groupby`` explicit prevents optional experiment attributes from
    being silently pooled. For example, include both ``approximation`` and
    ``asymptotic_null`` when those define distinct procedures.
    """
    groups = [groupby] if isinstance(groupby, str) else list(groupby)
    groups = list(dict.fromkeys(groups))
    required = [*groups, value]
    missing = [column for column in required if column not in results]
    if missing:
        raise ValueError(f"Cannot aggregate missing columns: {missing}")
    if error not in {None, "sem", "std"}:
        raise ValueError("error must be None, 'sem', or 'std'.")

    mean_column = mean_column or f"{value}_mean"
    error_column = error_column or (f"{value}_{error}" if error else None)
    data = results[required].copy()
    data[value] = pd.to_numeric(data[value], errors="coerce")
    if data[value].notna().sum() == 0:
        raise ValueError(f"Metric column contains no numeric values: {value}")

    aggregations: dict[str, str] = {
        mean_column: "mean",
        count_column: "count",
    }
    if error is not None and error_column is not None:
        aggregations[error_column] = error

    if groups:
        aggregated = (
            data.groupby(groups, dropna=False)[value].agg(**aggregations).reset_index()
        )
    else:
        values = data[value]
        record = {
            mean_column: values.mean(),
            count_column: values.count(),
        }
        if error is not None and error_column is not None:
            record[error_column] = getattr(values, error)()
        aggregated = pd.DataFrame([record])
    if error_column is not None and error_column in aggregated:
        aggregated[error_column] = aggregated[error_column].fillna(0.0)
    return aggregated
