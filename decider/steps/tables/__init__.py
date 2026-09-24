"""Decision tables: `DecisionTableConfig` finds the first row a record matches, as one row node."""
from __future__ import annotations

import math
import typing as t
from dataclasses import replace

import polars as pl
from pydantic import AfterValidator, BeforeValidator, Field, model_validator

from decider.engine.ir.decls import Input, NullPolicy, Output
from decider.engine.ir.nodes import CallNode
from decider.steps.configurable import ConfigurableStep
from decider.steps.tables.encode import Shape, check_columns, dtypes, kind, pack, program, typed, variables
from decider.steps.tables.matcher import match
from decider.steps.tables.reference import reference
from decider.steps.tables.schema import (
    AndExpression,
    BetweenExpression,
    BoundMode,
    EqExpression,
    Expression,
    InExpression,
    IsTrueExpression,
    OrExpression,
    check_rows,
    leaves,
)
from decider.steps.values import TableRef, TableValue

__all__ = [
    "DecisionTableConfig", "AndExpression", "BetweenExpression", "BoundMode", "EqExpression", "Expression",
    "InExpression", "IsTrueExpression", "OrExpression",
]

_TYPES = {"float": float, "int": int, "bool": bool}
_ORDER = {"float": 0, "bool": 1, "str": 2}


class DecisionTableConfig(ConfigurableStep):
    """A decision table as a step: each record takes the outputs of the first row it matches, else `default`.

    `columns` declares every column of the table (polars dtype names, or
    `{"type": "List", "inner": "String"}`); `expression` says how a record
    matches a row; `outputs` are the columns written, one output column each.
    A null input never matches. `rows` are the rows inline, or
    `{"table": "bands"}` to read them from the params document, where they
    are checked against `columns` when the document arrives. Editing rows,
    or how many there are, never recompiles. decider 0.3's
    `"parameters": {"data": [...], "dtypes": {...}}` spelling loads too.

    Row `r` is the locator `<name>#<r>` a session breaks on.

    Example::

        bands = DecisionTableConfig.load({
            "type": "decision_table", "name": "bands",
            "columns": {"lo": "Float64", "hi": "Float64", "band": "String"},
            "rows": [{"lo": None, "hi": 30.0, "band": "low"}, {"lo": 30.0, "hi": None, "band": "high"}],
            "expression": {"type": "between", "variable": "score",
                           "lower_bound_column": "lo", "upper_bound_column": "hi"},
            "outputs": ["band"], "default": ["other"],
        })
        bands.run(df)                                       # writes band

    Rows from params (a parameter table): write `"rows": {"table": "<param>"}`
    and the rows become a required param of the step, named `<param>`. The
    params document holds them under the step's node path, as a list with
    one dict per row, a value for every declared column (`None` for null)::

        pricing = DecisionTableConfig.load({
            "type": "decision_table", "name": "pricing",
            "columns": {"product": "String", "lo": "Float64", "hi": "Float64", "rate": "Float64"},
            "rows": {"table": "prices"},
            "expression": {"type": "and", "expressions": [
                {"type": "eq", "variable": "product", "value_column": "product"},
                {"type": "between", "variable": "amount", "lower_bound_column": "lo", "upper_bound_column": "hi"}]},
            "outputs": ["rate"],
        })
        pipeline = flow(pricing, name="loans")
        pipeline.parameters()
        # {'loans/pricing': {'prices': {'type': 'table', 'schema':
        #     {'product': 'String', 'lo': 'Float64', 'hi': 'Float64', 'rate': 'Float64'}}}}
        params = {"loans": {"pricing": {"prices": [
            {"product": "loan", "lo": None, "hi": 10000.0, "rate": 0.20},
            {"product": "loan", "lo": 10000.0, "hi": None, "rate": 0.15},
        ]}}}
        pipeline.run(df, params=params)       # df has product and amount; writes rate

    `pipeline.parameters().defaults()` leaves the table out: it has no
    default, so a run without it is a `ParamsError` ("param 'prices' is
    required but missing"). The rows are checked against `columns` when the
    document arrives: an undeclared or missing column, or a value of the
    wrong type, is a `ParamsError` naming the step, the param, the row and
    the column. A table on its own (not inside `flow`) is keyed by its name
    alone: `{"pricing": {"prices": [...]}}`. Changing the rows, or how many
    there are, never recompiles. A String output of such a table must be
    an `{"type": "Enum", "categories": [...]}` column, so its values are
    known before the rows arrive.

    In a `between` condition, rows with the same `eq` values (here, one
    product's bands) form one ladder: its first row may leave its lower
    bound `None` and its last its upper, and a missing bound between is the
    neighbouring band's.
    """

    type: t.Literal["decision_table"] = "decision_table"
    columns: t.Dict[str, t.Any]
    rows: TableValue
    expression: Expression
    outputs: t.List[str]
    default: t.Optional[t.List[t.Any]] = None

    @model_validator(mode="before")
    @classmethod
    def _legacy(cls, data: t.Any) -> t.Any:
        # decider 0.3 kept columns and rows in one `parameters` frame and could unnest a struct output.
        if not isinstance(data, dict) or "parameters" not in data:
            return data
        frame = data["parameters"]
        frame = {"data": frame} if isinstance(frame, list) else dict(frame)
        rows = frame.get("data", [])
        types = frame.get("dtypes") or {k: _dtype_name(v) for k, v in pl.DataFrame(rows).schema.items()}
        rest = {k: v for k, v in data.items() if k not in ("parameters", "unnest_output")}
        return {"rows": rows, "columns": dict(types), **rest}

    @model_validator(mode="after")
    def _check(self) -> DecisionTableConfig:
        types = self._types()
        for output in self.outputs:
            if output not in types:
                raise ValueError(f"Output column '{output}' not found in parameters columns")
            if types[output].startswith("list"):
                raise ValueError(f"Output column '{output}' is a list; outputs are single values")
        if self.default is not None and len(self.default) != len(self.outputs):
            raise ValueError(f"Default values length ({len(self.default)}) must match outputs "
                             f"length ({len(self.outputs)})")
        check_columns(self.expression, types)
        variables(self.expression, types)
        if not isinstance(self.rows, TableRef):
            check_rows(self.expression, typed(self.rows.data, dtypes(self.columns)))
        return self

    def _types(self) -> dict[str, str]:
        return {c: kind(d, c) for c, d in dtypes(self.columns).items()}

    def to_ir(self, ctx: t.Any) -> CallNode:
        dt, types = dtypes(self.columns), self._types()
        kinds = variables(self.expression, types)
        names = sorted(kinds, key=lambda v: (_ORDER[kinds[v]], v))
        slots = {v: [n for n in names if kinds[n] == kinds[v]].index(v) for v in names}
        inputs = tuple(_input(v, kinds[v]) for v in names)
        ref = self.rows if isinstance(self.rows, TableRef) else None
        data = None if ref else typed(self.rows.data, dt)
        defaults = list(self.default or [None] * len(self.outputs))
        outputs, specs, columns = [], [], []
        for k, (column, d) in enumerate(zip(self.outputs, defaults), len(list(leaves(self.expression)))):
            out = types[column]
            if out == "str":
                choices = _choices(column, dt[column], data, d, self.name)
                outputs.append(Output(column, t.Literal[choices or ("",)]))
                specs.append((k, -1 if d is None else choices.index(str(d))))
                columns.append((column, out, choices))
                continue
            py = _TYPES[out]
            nullable = ref is not None or d is None or any(r[column] is None for r in data)
            outputs.append(Output(column, py | None if nullable else py))
            specs.append((k, py() if d is None else py(d), *((int(d is not None),) if nullable else ())))
            columns.append((column, out, None))
        shape = Shape(self.expression, columns, dt)
        code, entry = program(self.expression, types, slots)
        consts = (("program", code.ctypes.data), ("entry", entry), ("outputs", tuple(specs)))
        params = ()
        rows = None
        if ref is None:
            rows = pack(data, shape)
            consts += (("rows", rows),)
        else:
            decl = ctx.table(ref, {c: str(d) for c, d in dt.items()})
            checked = Field()
            checked.metadata += [BeforeValidator(shape.given), AfterValidator(shape.arrive)]
            params = (replace(decl, field_info=checked),)
        py_defaults = [None if d is None else str(d) if types[c] == "str" else _TYPES[types[c]](d)
                       for c, d in zip(self.outputs, defaults)]
        fn = reference(self.expression, names, self.outputs, py_defaults, ref.table if ref else None, rows)
        # The node's consts are addresses into the program; the node holds its reference, which holds it.
        fn.arrays = [code]
        return CallNode(ctx.origin(self), "row", match, inputs, tuple(outputs), params, reference=fn, consts=consts)


def _input(name: str, kind: str) -> Input:
    # A null number is NaN, which no comparison matches: a fill costs a launch less than an Optional's mask.
    if kind == "float":
        return Input(name, float, NullPolicy.MISSING_AS, math.nan)
    return Input(name, (bytes if kind == "str" else bool) | None, NullPolicy.OPTIONAL)


def _choices(column: str, dtype: t.Any, data: t.Optional[list[dict]], default: t.Any, name: str) -> tuple[str, ...]:
    extra = () if default is None else (str(default),)
    if data is not None:
        return tuple(dict.fromkeys([str(r[column]) for r in data if r[column] is not None] + list(extra)))
    if dtype.base_type() is not pl.Enum:
        raise ValueError(f"{name}: output column {column!r} is a String and its rows come from params, so its "
                         "values aren't known up front; declare it {\"type\": \"Enum\", \"categories\": [...]}")
    return tuple(dict.fromkeys([*dtype.categories, *extra]))


def _dtype_name(dtype: pl.DataType) -> t.Any:
    if isinstance(dtype, pl.List):
        return {"type": "List", "inner": _dtype_name(dtype.inner)}
    return type(dtype).__name__
