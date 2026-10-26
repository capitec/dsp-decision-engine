"""Custom and nested output types, resolved into the flat columns and list columns a tree returns."""
from __future__ import annotations

import typing as t

from decikit._serde.schema import ExplicitType
from decikit.steps.trees.schema.conditions import get_field

LISTS = ("list", "set", "array")


def resolve(dtypes: t.Any, data: list[dict], default: t.Optional[dict],
            type_defs: dict) -> tuple[list, list[dict], t.Optional[dict]]:
    """`(dtypes, data, default)` with every Custom type resolved through `type_defs`.

    A struct column (Custom or inline) becomes one `column.field` column per
    field. A list keeps one column; its struct items become records of just
    the struct's fields. A struct value is the record itself, `{"$key": key}`
    or the bare key, matched on the struct's `display_field`. A struct's
    fields are values, categoricals or lists: one level of struct only.
    """
    # (output column, its type, the column it is read from, how its value is read)
    columns: list[tuple[str, t.Any, str, t.Callable]] = []
    for column, dtype in dtypes.items() if isinstance(dtypes, dict) else dtypes:
        where = f"output column {column!r}"
        struct = _struct(dtype, type_defs, where)
        if struct is None:
            dtype, read = _column(dtype, type_defs, where)
            columns.append((column, dtype, column, read))
            continue
        fields, records = struct
        for name, field_type in fields:
            field_type, read = _column(field_type, type_defs, f"{where} field {name!r}")
            columns.append((f"{column}.{name}", field_type, column,
                            lambda v, n=name, r=records, w=where, f=read: f(get_field(_record(v, r, w), n))))
    names = [name for name, *_ in columns]
    if clash := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(f"output columns {clash} appear twice once structs are flattened; rename one")

    def row(r: t.Optional[dict]) -> t.Optional[dict]:
        # A column a row leaves out stays out, so it is still null there.
        return None if r is None else {name: read(r[src]) for name, _, src, read in columns if src in r}

    return [(name, dtype) for name, dtype, *_ in columns], [row(r) for r in data], row(default)


def _column(dtype: t.Any, type_defs: dict, where: str) -> tuple[t.Any, t.Callable]:
    # A column that isn't a struct: its resolved type, and how a value of it is read.
    # A struct's field types come from `type_defs` as raw JSON.
    dtype = ExplicitType.model_validate(dtype) if isinstance(dtype, dict) and "type" in dtype else dtype
    if not (isinstance(dtype, ExplicitType) and dtype.type.lower() in LISTS and "inner" in dtype.model_extra):
        return _plain(dtype, type_defs, where), lambda v: v
    item = _struct(dtype.model_extra["inner"], type_defs, f"{where} item")
    if item is None:
        inner, read = _plain(dtype.model_extra["inner"], type_defs, where), (lambda v: v)
    else:
        fields, records = item
        inner = [list(f) for f in fields]
        read = (lambda v, n=[f for f, _ in fields], r=records, w=where: None if v is None else
                [{k: get_field(_record(x, r, w), k) for k in n} for x in v])
    return ExplicitType.model_validate({**dtype.model_extra, "type": dtype.type, "inner": inner}), read


def _is_custom(dtype: t.Any) -> bool:
    return not isinstance(dtype, (str, list)) and str(get_field(dtype, "type", "")).lower() == "custom"


def _plain(dtype: t.Any, type_defs: dict, where: str) -> t.Any:
    # A Custom categorical is a string column; a type that isn't Custom passes through.
    if not _is_custom(dtype):
        return dtype
    kind, _ = _custom(dtype, type_defs, where)
    if kind != "categorical":
        raise ValueError(f"{where}: a struct's fields must be plain values or categoricals, not a nested {kind}")
    return "String"


def _struct(dtype: t.Any, type_defs: dict, where: str) -> t.Optional[tuple[list, t.Optional[dict]]]:
    # `(fields, records by key)` of a Custom struct, `(fields, None)` of an inline one, else None.
    if isinstance(dtype, list) or isinstance(dtype, dict) and "type" not in dtype:
        fields, records = list(dtype.items()) if isinstance(dtype, dict) else list(dtype), None
    elif _is_custom(dtype):
        kind, definition = _custom(dtype, type_defs, where)
        if kind != "struct":
            return None
        fields = get_field(definition, "fields") or []
        fields = list(fields.items()) if isinstance(fields, dict) else [tuple(f) for f in fields]
        key = get_field(definition, "display_field")
        records = {get_field(r, key): r for r in get_field(definition, "data") or []}
    else:
        return None
    return [(name, _plain(ft, type_defs, f"{where} field {name!r}")) for name, ft in fields], records


def _custom(dtype: t.Any, type_defs: dict, where: str) -> tuple[str, t.Any]:
    type_id = get_field(dtype, "type_id")
    type_def = type_defs.get(type_id)
    kind = get_field(type_def, "type")
    if kind not in ("categorical", "struct"):
        raise ValueError(f"{where}: custom type {type_id!r} is "
                         f"{'not in type_defs' if kind is None else repr(kind)}; use a categorical or a struct")
    return kind, get_field(type_def, "definition")


def _record(value: t.Any, records: t.Optional[dict], where: str) -> t.Optional[dict]:
    if value is None or isinstance(value, dict) and "$key" not in value:
        return value
    if records is None:
        raise ValueError(f"{where}: {value!r} is not a record")
    key = value["$key"] if isinstance(value, dict) else value
    if key not in records:
        raise ValueError(f"{where}: {key!r} is not a record of its struct type")
    return records[key]
