import datetime as dt
import functools
import json
import typing as t

import polars as pl
import typing_extensions as te
from pydantic import TypeAdapter, ValidationError

from decider.engine.ir.decls import base_annotation
from decider.exceptions import InputParsingError


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


def has_date(annotation: t.Any) -> bool:
    if isinstance(annotation, type) and issubclass(annotation, dt.date):
        return True
    # A TypedDict stays a dict for the step, so its date fields can be coerced in place.
    if te.is_typeddict(annotation):
        return any(has_date(h) for h in te.get_type_hints(annotation).values())
    return any(has_date(a) for a in t.get_args(annotation))


_adapter = functools.cache(TypeAdapter)


def coerce_record(record: t.Dict[str, t.Any], dates: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
    # JSON has no dates, so ISO strings become the declared date/datetime. Only inputs holding dates are
    # validated: JSON already delivers everything else as the step expects, and validation costs per request.
    # A bare `dict` says nothing about which strings are dates, so its contents are left as sent.
    out = dict(record)
    for name, annotation in dates.items():
        if out.get(name) is not None:
            try:
                out[name] = _adapter(annotation).validate_python(out[name])
            except ValidationError as e:
                raise InputParsingError(
                    f"input {name!r} is declared {annotation} but got {out[name]!r}; "
                    f"send dates as ISO strings like \"2026-01-31\". ({e.errors()[0]['msg']})") from None
    return out


def coerce_frame(df: pl.DataFrame, dates: t.Dict[str, t.Any]) -> pl.DataFrame:
    # ponytail: only top-level date/datetime string columns; cast nested list/struct fields when a frame needs them.
    casts = []
    for name, annotation in dates.items():
        base = base_annotation(annotation)
        if name in df.columns and df.schema[name] == pl.String and base in (dt.date, dt.datetime):
            col = pl.col(name).str
            casts.append(col.to_datetime() if base is dt.datetime else col.to_date())
    if not casts:
        return df
    try:
        return df.with_columns(casts)
    except pl.exceptions.PolarsError as e:
        raise InputParsingError(f"a date column isn't ISO formatted like \"2026-01-31\": {e}") from None


_DUMMY = {bool: False, int: 1, str: "", bytes: "", dt.date: dt.date(2000, 1, 1),
          dt.datetime: dt.datetime(2000, 1, 1)}


def dummy(annotation: t.Any) -> t.Any:
    # 1 rather than 0 so an ordinary ratio doesn't divide by zero; one element so a list has a dtype.
    a = base_annotation(annotation)
    origin = t.get_origin(a) or a
    if origin is list:
        return [dummy(t.get_args(a)[0])] if t.get_args(a) else [1.0]
    if te.is_typeddict(a):
        return {k: dummy(h) for k, h in te.get_type_hints(a).items()}
    if origin is dict:
        return {}
    return _DUMMY.get(origin, 1.0)
