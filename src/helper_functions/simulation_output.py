"""Streaming CSV output for long-running simulation studies."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path


CSV_BATCH_SIZE = 100


class CsvResultWriter:
    """Append result mappings to a CSV file without retaining all rows in memory."""

    def __init__(self, path, batch_size=CSV_BATCH_SIZE):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = None
        self._buffer = []
        self._batch_size = batch_size
        self.rows_written = 0

    def add(self, row):
        if not isinstance(row, Mapping):
            raise TypeError("CSV result rows must be mappings")
        self._buffer.append(dict(row))
        if len(self._buffer) >= self._batch_size:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        if self._writer is None:
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=list(self._buffer[0]),
                extrasaction="raise",
            )
            self._writer.writeheader()
        self._writer.writerows(self._buffer)
        self.rows_written += len(self._buffer)
        self._buffer.clear()
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self.flush()
            self._file.close()
