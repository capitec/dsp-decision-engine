"""Turning a debug session's `data` argument into a `LoadedData`."""
from __future__ import annotations

from typing import Any

import polars as pl

from decider.data import LoadedData, load


def load_rows(data: Any) -> LoadedData:
    """`data` as `LoadedData`: a file path through `load()`, inline rows wrapped as-is."""
    if data is None:
        raise ValueError("no data: pass `data` (rows or a JSON/CSV/Parquet file) or define SAMPLE in the module")
    if isinstance(data, str):
        return load(data)
    if isinstance(data, dict):
        data = [data]  # sample_request.json can be one record, not a list of them
    return LoadedData.from_frame(pl.DataFrame(data))
