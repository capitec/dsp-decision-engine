import json
import typing as t

import polars as pl

from .media_types import MediaType


def parse_application_json(data: bytes) -> t.Union[t.Dict[str, t.Any], pl.DataFrame]:
    # An object is one record for the single-record path; an array of objects is a frame.
    value = json.loads(data)
    if isinstance(value, dict):
        return value
    return pl.from_dicts(value, infer_schema_length=None)


DEFAULT_INPUT_HANDLERS = {
    MediaType.APPLICATION_JSON.value: parse_application_json,
    MediaType.APPLICATION_JSONL.value: pl.read_ndjson,
    MediaType.APPLICATION_X_PARQUET.value: pl.read_parquet,
    MediaType.TEXT_CSV.value: pl.read_csv,
    MediaType.APPLICATION_EXCEL.value: pl.read_excel,
    MediaType.APPLICATION_VND_MS_EXCEL.value: pl.read_excel,
}
