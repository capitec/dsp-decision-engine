import datetime as dt
import functools
import typing as t

import polars as pl
import typing_extensions as te
from pydantic import ConfigDict, TypeAdapter, ValidationError
from pydantic_core import from_json

from decider.serving._errors import InputParsingError
from decider.ops.nested import struct_item
from decider.ir import base_annotation


def parse_application_json(data: bytes) -> t.Union[t.Dict[str, t.Any], pl.DataFrame]:
    # An object is one record for the single-record path; an array of objects is a frame.
    # pydantic_core's parser answers the same values as `json.loads`, about 4x faster.
    value = from_json(data)
    if isinstance(value, dict):
        return value
    if not value:
        raise InputParsingError("request body is an empty array; send at least one record or a single object")
    return pl.from_dicts(value, infer_schema_length=None)


DEFAULT_INPUT_HANDLERS = {
    "application/json": parse_application_json,
    "application/vnd.apache.arrow.stream": pl.read_ipc_stream,
    "application/vnd.apache.arrow.file": pl.read_ipc,
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


@functools.cache
def _adapter(annotation: t.Any) -> TypeAdapter:
    # Keeps the keys a TypedDict doesn't declare: one declaring only the date fields must not drop the rest.
    # pydantic takes a config only for a container, hence the one-element list.
    return TypeAdapter(list[annotation], config=ConfigDict(extra="allow"))


def coerce_record(record: t.Dict[str, t.Any], dates: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
    # JSON has no dates, so ISO strings become the declared date/datetime. Only inputs holding dates are
    # validated: JSON already delivers everything else as the step expects, and validation costs per request.
    # A bare `dict` says nothing about which strings are dates, so its contents are left as sent.
    out = dict(record)
    for name, annotation in dates.items():
        if out.get(name) is not None:
            try:
                out[name] = _adapter(annotation).validate_python([out[name]])[0]
            except ValidationError as e:
                raise InputParsingError(
                    f"input {name!r} is declared {annotation} but got {out[name]!r}; "
                    f"send dates as ISO strings like \"2026-01-31\". ({e.errors()[0]['msg']})") from None
    return out


def coerce_frame(df: pl.DataFrame, dates: t.Dict[str, t.Any]) -> pl.DataFrame:
    # Plain date columns cast in polars; dates nested in lists or structs go through the same
    # validation as a single record, once per column.
    casts = []
    for name, annotation in dates.items():
        if name not in df.columns:
            continue
        base = base_annotation(annotation)
        if df.schema[name] == pl.String and base in (dt.date, dt.datetime):
            col = pl.col(name).str
            casts.append(col.to_datetime() if base is dt.datetime else col.to_date())
        elif isinstance(df.schema[name], (pl.List, pl.Struct)):
            casts.append(_coerce_nested(df.get_column(name), annotation))
    if not casts:
        return df
    try:
        return df.with_columns(casts)
    except pl.exceptions.PolarsError as e:
        raise InputParsingError(f"a date column isn't ISO formatted like \"2026-01-31\": {e}") from None


def _coerce_nested(column: pl.Series, annotation: t.Any) -> pl.Series:
    rows = column.to_list()
    present = [i for i, row in enumerate(rows) if row is not None]
    try:
        coerced = _adapter(annotation).validate_python([rows[i] for i in present])
    except ValidationError as e:
        raise InputParsingError(
            f"input {column.name!r} is declared {annotation} but got dates that don't validate; "
            f"send dates as ISO strings like \"2026-01-31\". ({e.errors()[0]['msg']})") from None
    for i, value in zip(present, coerced):
        rows[i] = value
    return pl.Series(column.name, rows)


_DUMMY = {bool: False, int: 1, str: "", bytes: "", dt.date: dt.date(2000, 1, 1),
          dt.datetime: dt.datetime(2000, 1, 1)}


def dummy(annotation: t.Any) -> t.Any:
    # 1 rather than 0 so an ordinary ratio doesn't divide by zero; one element so a list has a dtype.
    a = base_annotation(annotation)
    a = struct_item(a) or a   # `Struct[Item]` is Item's dict at this boundary
    origin = t.get_origin(a) or a
    if origin is list:
        return [dummy(t.get_args(a)[0])] if t.get_args(a) else [1.0]
    if te.is_typeddict(a):
        return {k: dummy(h) for k, h in te.get_type_hints(a).items()}
    if origin is dict:
        return {}
    return _DUMMY.get(origin, 1.0)
