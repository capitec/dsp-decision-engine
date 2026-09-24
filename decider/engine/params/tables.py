from __future__ import annotations

import keyword
from functools import cache
from typing import Annotated, Any, Mapping

import numpy as np
from pydantic import AfterValidator, ConfigDict, Field, TypeAdapter
from typing_extensions import TypedDict

from decider.engine.params.bundles import bundle_class
from decider.engine.params.declare import _NO_DEFAULT, ParamSpec, _carrier_class, _mark

Table = Any
"""The annotation of a `param_table()` argument: a namedtuple with one read-only numpy array per column."""

_TYPES = {"int": (int, np.int64), "float": (float, np.float64), "bool": (bool, np.bool_)}


def param_table(
    columns: Mapping[str, type],
    default: Any = _NO_DEFAULT,
    *,
    required: bool = False,
    shared_key: str | None = None,
) -> Any:
    """Declare a table-valued param as a function argument's default: rows from the params document.

    `columns` maps each column to `int`, `float` or `bool`. In the params
    document the rows sit where a `param()` value would, as a list with one
    dict per row and a value for every column; they are checked when the
    document arrives. The step receives the table as a namedtuple with one
    read-only numpy array per column (`rates.band[i]`, `len(rates.band)`),
    the same in every mode, so a compiled kernel reads it too. Editing the
    rows or how many there are never recompiles; only changing `columns`
    does.

    Args:
        default: the rows used when the params document doesn't set them.
        required: declare a table with no default rows; a document that
            doesn't set it is invalid.
        shared_key: read the rows from the document's top-level `"shared"`
            entry under this key, so several steps use one table.

    Example::

        def rate(score: int, rates: Table = param_table({"floor": int, "rate": float},
                                                         default=[{"floor": 0, "rate": 0.2}])) -> float:
            r = 0.0
            for i in range(len(rates.floor)):
                if score >= rates.floor[i]:
                    r = rates.rate[i]
            return r

        # params document: {"rate": {"rates": [{"floor": 0, "rate": 0.2}, {"floor": 600, "rate": 0.1}]}}
    """
    schema = []
    for column, kind in columns.items():
        if not column.isidentifier() or keyword.iskeyword(column) or column.startswith("_"):
            raise TypeError(f"param_table: column {column!r} must be a Python identifier not starting with '_'; "
                            f"try {column.replace('-', '_').lstrip('_')!r}")
        name = getattr(kind, "__name__", None)
        if name not in _TYPES:
            raise TypeError(f"param_table: column {column!r} is {kind!r}; columns are int, float or bool "
                            "(code a string key as an int, or use a DecisionTableConfig)")
        schema.append((column, name))
    if not schema:
        raise TypeError("param_table needs at least one column")
    if required == (default is not _NO_DEFAULT):
        raise TypeError("param_table() takes either default rows or required=True, not both or neither")
    schema = tuple(schema)
    attrs = dict(required=required, shared_key=shared_key, on_invalid="error", schema=schema)
    if required:
        return _mark(ParamSpec, None, default=None, field_info=Field(), **attrs)
    rows = list(default)
    columns = TypeAdapter(table_type(schema)).validate_python(rows)
    # The marker is the default table itself, so a direct call of the function gets its columns.
    spec = _carrier_class(ParamSpec, type(columns))(*columns)
    spec.__dict__.update(default=rows, field_info=Field(rows), **attrs)
    return spec


@cache
def table_type(schema: tuple[tuple[str, str], ...]) -> Any:
    """The pydantic type that checks a table's rows and returns them as columns; one object per schema.

    Example::

        TypeAdapter(table_type((("band", "int"),))).validate_python([{"band": 1}]).band   # array([1])
    """
    row = TypedDict("Row", {c: _TYPES[k][0] for c, k in schema})
    row.__pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    cls = bundle_class(tuple(c for c, _ in schema))

    def columns(rows: list) -> tuple:
        arrays = []
        for c, k in schema:
            try:
                a = np.array([r[c] for r in rows], _TYPES[k][1])
            except OverflowError:
                raise ValueError(f"column {c!r} holds an int that doesn't fit in int64") from None
            # Every call sharing the cached params bundle sees these arrays.
            a.flags.writeable = False
            arrays.append(a)
        return cls(*arrays)

    return Annotated[list[row], AfterValidator(columns)]


def rows_like(schema: Mapping[str, Any]) -> str:
    """What a table's rows look like, for error messages: `expected a list of rows like [{"band": int}, ...]`."""
    return "expected a list of rows like [{" + ", ".join(f'"{c}": {d}' for c, d in schema.items()) + "}, ...]"
