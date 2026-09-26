"""Bounded CSV output for simulation callbacks."""

from pathlib import Path
import pandas as pd
from src.load_config import flatten_args_columns

CSV_BATCH_SIZE = 1000


class CsvResultWriter:
    """Write simulation results in bounded batches while preserving CSV shape."""

    def __init__(self, output_path, batch_size=CSV_BATCH_SIZE):
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self.buffer = []
        self.rows_written = 0
        self._wrote_header = False

    def add(self, result):
        self.buffer.append(result)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        frame = pd.DataFrame(self.buffer)
        flatten_args_columns(frame)
        frame.to_csv(
            self.output_path,
            mode="a" if self._wrote_header else "w",
            header=not self._wrote_header,
            index=False,
        )
        self.rows_written += len(frame)
        self._wrote_header = True
        self.buffer.clear()

    def close(self):
        self.flush()
