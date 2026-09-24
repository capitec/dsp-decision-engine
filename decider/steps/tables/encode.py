"""A decision table as data: its conditions as a program, its rows as packed arrays the matcher reads."""
from __future__ import annotations

import collections
import typing as t
from dataclasses import dataclass

import numpy as np
import polars as pl

from decider.engine.params.tables import rows_like
from decider.serializable.schema import PolarsSchema
from decider.steps.tables.schema import (
    AndExpression,
    BetweenExpression,
    BoundMode,
    EqExpression,
    InExpression,
    IsTrueExpression,
    OrExpression,
    bounds,
    check_rows,
    groups,
    leaves,
)
from decider.steps.trees.ops import GE, GT, LE, LT

# Condition kinds, the columns of a program row, and the two ends of a program.
BETWEEN, EQ_F, EQ_B, EQ_S, IN_F, IN_S, TRUE = range(7)
KIND, VAR, LEAF, LO_OP, HI_OP, THEN, ELSE = range(7)
WIDTH = 7
MATCH, FAIL = -1, -2


def kind(dtype: t.Any, column: str) -> str:
    """`float`, `int`, `bool`, `str`, `list[float]` or `list[str]` for a polars dtype."""
    base = dtype.base_type()
    if base is pl.Boolean:
        return "bool"
    if base.is_numeric():
        return "int" if base.is_integer() else "float"
    if base in (pl.String, pl.Categorical, pl.Enum):
        return "str"
    if base is pl.List:
        inner = kind(dtype.inner, column)
        if inner in ("float", "int", "str"):
            return "list[str]" if inner == "str" else "list[float]"
    raise ValueError(f"column {column!r} is {dtype}; a decision table column is a number, Boolean, String "
                     "or a List of numbers or strings")


def dtypes(columns: t.Mapping[str, t.Any]) -> dict[str, t.Any]:
    """The polars dtype of each column, from `"Float64"` or `{"type": "List", "inner": "String"}` spellings."""
    return dict(PolarsSchema.model_validate(dict(columns)).schema)


class Rows(collections.namedtuple("Rows", ["ints", "floats", "chars", "n"])):
    """A table's rows packed for the matcher: three array addresses and the row count.

    `data` holds the rows as dicts (what the Python matcher reads) and
    `arrays` the arrays the addresses point into, which must outlive every call.
    """


def _variable_kind(leaf: t.Any, types: dict[str, str]) -> str:
    if isinstance(leaf, BetweenExpression):
        return "float"
    if isinstance(leaf, IsTrueExpression):
        return "bool"
    column = leaf.value_column if isinstance(leaf, EqExpression) else leaf.values_column
    k = types[column]
    return "str" if k in ("str", "list[str]") else "bool" if k == "bool" else "float"


def variables(expression: t.Any, types: dict[str, str]) -> dict[str, str]:
    """Each input the table reads and its kind (`float`, `bool` or `str`), from how the conditions use it."""
    found: dict[str, str] = {}
    for leaf in leaves(expression):
        k = _variable_kind(leaf, types)
        if found.setdefault(leaf.variable, k) != k:
            raise ValueError(f"variable {leaf.variable!r} is compared as {found[leaf.variable]} and as {k}; "
                             "one variable has one type")
    return found


def check_columns(expression: t.Any, types: dict[str, str]) -> None:
    """Raise `ValueError` if a condition names a column the table lacks or one of the wrong type."""
    for leaf in leaves(expression):
        if isinstance(leaf, BetweenExpression):
            for col, label in ((leaf.lower_bound_column, "Lower"), (leaf.upper_bound_column, "Upper")):
                if col is not None and col not in types:
                    raise ValueError(f"{label} bound column '{col}' not found in parameters columns")
                if col is not None and types[col] not in ("float", "int"):
                    raise ValueError(f"{label} bound column '{col}' must be numeric, got {types[col]}")
        elif isinstance(leaf, InExpression):
            if leaf.values_column not in types:
                raise ValueError(f"Values column '{leaf.values_column}' not found in parameters columns")
            if not types[leaf.values_column].startswith("list"):
                raise ValueError(f"Values column '{leaf.values_column}' must be a list type, "
                                 f"got {types[leaf.values_column]}")
        elif isinstance(leaf, EqExpression) and leaf.value_column not in types:
            raise ValueError(f"Value column '{leaf.value_column}' not found in parameters columns")


def _leaf_kind(leaf: t.Any, types: dict[str, str]) -> int:
    if isinstance(leaf, BetweenExpression):
        return BETWEEN
    if isinstance(leaf, IsTrueExpression):
        return TRUE
    if isinstance(leaf, InExpression):
        return IN_S if types[leaf.values_column] == "list[str]" else IN_F
    k = types[leaf.value_column]
    return EQ_S if k == "str" else EQ_B if k == "bool" else EQ_F


def program(expression: t.Any, types: dict[str, str], slots: dict[str, int]) -> tuple[np.ndarray, int]:
    """The conditions as jump rows `(kind, variable slot, condition, lower op, upper op, then, else)` and the entry row.

    A target of `MATCH` means the row matches, `FAIL` that it doesn't.
    """
    numbered = {id(leaf): k for k, leaf in enumerate(leaves(expression))}
    rows: list[list[int]] = []

    def build(e: t.Any, then: int, other: int) -> int:
        if isinstance(e, (AndExpression, OrExpression)):
            for sub in reversed(e.expressions):
                if isinstance(e, AndExpression):
                    then = build(sub, then, other)
                else:
                    other = build(sub, then, other)
            return then if isinstance(e, AndExpression) else other
        lo, hi = (GE, LT) if getattr(e, "mode", BoundMode.lower_inclusive) is BoundMode.lower_inclusive else (GT, LE)
        rows.append([_leaf_kind(e, types), slots[e.variable], numbered[id(e)], lo, hi, then, other])
        return len(rows) - 1

    entry = build(expression, MATCH, FAIL)
    return np.array(rows, np.int64).ravel(), entry


@dataclass
class Shape:
    """What packing rows needs: the conditions, each output's `(column, kind, string choices)`, the column dtypes."""

    expression: t.Any
    outputs: list[tuple[str, str, t.Optional[tuple[str, ...]]]]
    dtypes: dict[str, t.Any]

    def arrive(self, frame: t.Any) -> Rows:
        """Rows from a params document, checked against the declared columns and packed."""
        return pack(typed(frame.data, self.dtypes), self)

    def given(self, value: t.Any) -> t.Any:
        """Refuse a params value that isn't a list of rows before it is read as one."""
        data = value.get("data") if isinstance(value, dict) else value
        if not isinstance(data, list):
            raise ValueError(f"a table param takes its rows as a list, one dict per row; {expected(self.dtypes)}")
        for r, row in enumerate(data):
            if not isinstance(row, dict):
                raise ValueError(f"row {r} is {row!r}, not a dict of column to value; {expected(self.dtypes)}")
        return value


def expected(types: t.Mapping[str, t.Any]) -> str:
    """What a table's rows look like, for error messages: `[{"lo": Float64, "band": String}, ...]`."""
    return rows_like(types)


def typed(data: list[dict], types: dict[str, t.Any]) -> list[dict]:
    """`data` cast to the declared column dtypes; `ValueError` naming the row and column that doesn't fit."""
    for r, row in enumerate(data):
        if unknown := sorted(set(row) - set(types)):
            raise ValueError(f"row {r} has column(s) {unknown} the table does not declare; {expected(types)}")
    given = set().union(*map(set, data)) if data else set(types)
    if missing := sorted(set(types) - given):
        raise ValueError(f"rows lack the declared column(s) {missing}; {expected(types)}")
    try:
        cast = pl.DataFrame(data, schema=types, strict=True).to_dicts()
    except Exception as e:  # polars raises several unrelated types for a value that doesn't fit
        raise ValueError(f"rows don't fit the declared columns: {_misfit(data, types) or e}; "
                         f"{expected(types)}") from None
    # An Enum column turns a value outside its categories into a null rather than failing.
    for r, (raw, row) in enumerate(zip(data, cast)):
        for column, value in row.items():
            if value is None and raw.get(column) is not None:
                raise ValueError(f"rows don't fit the declared columns: row {r}, column {column!r}: "
                                 f"{raw[column]!r} is not a {types[column]}; {expected(types)}")
    return cast


def _misfit(data: list[dict], types: dict[str, t.Any]) -> t.Optional[str]:
    for r, row in enumerate(data):
        for column, value in row.items():
            try:
                pl.DataFrame([{column: value}], schema={column: types[column]}, strict=True)
            except Exception:
                return f"row {r}, column {column!r}: {value!r} is not a {types[column]}"
    return None


def pack(data: list[dict], shape: Shape) -> Rows:
    """`data` as the arrays the matcher reads: a header of two starts per condition and output, then each one's data."""
    check_rows(shape.expression, data)
    n = len(data)
    conds = list(leaves(shape.expression))
    ladders = groups(shape.expression, data)
    ints: list[int] = [0] * (2 * (len(conds) + len(shape.outputs)))
    floats: list[float] = []
    chars = bytearray()

    def put(seq: t.Iterable, into: list) -> int:
        start = len(into)
        into.extend(seq)
        return start

    def text(values: t.Iterable[t.Any]) -> list[int]:
        # Absolute offsets into `chars`: value k is chars[offsets[k]:offsets[k + 1]].
        offsets = [len(chars)]
        for v in values:
            chars.extend(str(v).encode())
            offsets.append(len(chars))
        return offsets

    def head(k: int, i: int, f: int) -> None:
        ints[2 * k], ints[2 * k + 1] = i, f

    # ponytail: numbers compare as float64, exact up to 2**53; add an int64 lane, as trees have, if a table keys on
    # bigger ints.
    for k, leaf in enumerate(conds):
        if isinstance(leaf, BetweenExpression):
            bands = bounds(leaf, data, ladders)
            head(k, put([lo is not None for lo, _ in bands] + [hi is not None for _, hi in bands], ints),
                 put([0.0 if lo is None else float(lo) for lo, _ in bands]
                     + [0.0 if hi is None else float(hi) for _, hi in bands], floats))
        elif isinstance(leaf, EqExpression):
            col = [r[leaf.value_column] for r in data]
            has = [v is not None for v in col]
            if shape.dtypes[leaf.value_column].base_type() in (pl.String, pl.Categorical, pl.Enum):
                head(k, put(text("" if v is None else v for v in col) + has, ints), 0)
            else:
                head(k, put(has, ints), put([0.0 if v is None else float(v) for v in col], floats))
        elif isinstance(leaf, InExpression):
            lists = [[v for v in (r[leaf.values_column] or ()) if v is not None] for r in data]
            counts = np.cumsum([0] + [len(v) for v in lists]).tolist()
            items = [v for values in lists for v in values]
            if kind(shape.dtypes[leaf.values_column], leaf.values_column) == "list[str]":
                start = len(ints) + n + 1
                head(k, put([start + c for c in counts] + text(items), ints), 0)
            else:
                start = len(floats)
                head(k, put([start + c for c in counts], ints), put(map(float, items), floats))
    for k, (column, out, choices) in enumerate(shape.outputs, len(conds)):
        col = [r[column] for r in data]
        valid = [v is not None for v in col]
        if out == "float":
            head(k, put(valid, ints), put([0.0 if v is None else float(v) for v in col], floats))
        else:
            codes = ([-1 if v is None else choices.index(v) for v in col] if choices is not None
                     else [0 if v is None else int(v) for v in col])
            head(k, put(valid + codes, ints), 0)
    arrays = [np.array(ints or [0], np.int64), np.array(floats or [0.0], np.float64),
              np.frombuffer(bytes(chars) or bytes(1), np.uint8).copy()]
    rows = Rows(arrays[0].ctypes.data, arrays[1].ctypes.data, arrays[2].ctypes.data, n)
    rows.data, rows.arrays = data, arrays
    return rows
