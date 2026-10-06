"""Turning a debug session's `data` argument into a `LoadedData`."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from decider.lab.data import LoadedData, read
from decider.lab.data._keys import suggest_key


def load_rows(data: Any) -> LoadedData:
    """`data` as `LoadedData`: a file path through `read()`, inline rows wrapped as-is."""
    if data is None:
        raise ValueError("no data: pass `data` (rows or a JSON/CSV/Parquet file) or define SAMPLE in the module")
    if isinstance(data, str):
        frame = read(data)
        return LoadedData.from_frame(frame, dataset=Path(data).name, suggested_key=suggest_key(frame.columns))
    if isinstance(data, dict):
        data = [data]  # sample_request.json can be one record, not a list of them
    return LoadedData.from_frame(pl.DataFrame(data))
