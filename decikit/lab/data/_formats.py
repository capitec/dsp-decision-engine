"""The one table from a file's suffix to the polars reader."""
from __future__ import annotations

import polars as pl


def format_of(path: str) -> str | None:
    lower = path.lower()
    if lower.endswith((".json", ".jsonl", ".ndjson")):
        return "json"
    if lower.endswith(".csv"):
        return "csv"
    if lower.endswith(".parquet"):
        return "parquet"
    return None


def scan(path: str, format: str | None = None) -> pl.LazyFrame:
    fmt = format or format_of(path)
    if fmt == "parquet":
        return pl.scan_parquet(path)
    if fmt == "csv":
        return pl.scan_csv(path)
    if fmt == "json":
        if path.lower().endswith((".jsonl", ".ndjson")):
            return pl.scan_ndjson(path)
        # A .json file is an ndjson export or an array of objects; neither can be told apart without reading.
        try:
            return pl.read_ndjson(path).lazy()
        except Exception:
            return pl.read_json(path).lazy()
    raise ValueError(f"no data format for {path!r}; expected one of json, csv, parquet")
