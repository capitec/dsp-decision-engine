import json
import typing as t

import polars as pl


def parse_application_json(data: bytes) -> t.Union[t.Dict[str, t.Any], pl.DataFrame]:
    # An object is one record for the single-record path; an array of objects is a frame.
    value = json.loads(data)
    if isinstance(value, dict):
        return value
    return pl.from_dicts(value, infer_schema_length=None)


DEFAULT_INPUT_HANDLERS = {
    "application/json": parse_application_json,
    "application/jsonl": pl.read_ndjson,
    "application/x-parquet": pl.read_parquet,
    "text/csv": pl.read_csv,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": pl.read_excel,
    "application/vnd.ms-excel": pl.read_excel,
}
