"""The decision-table document — decider 1's vocabulary, kept as-is.

Every type name, field name and default below is decider 1's, from
`decider/modules/credit/decision_table/config.py` and `module.py`:
`AndExpression`/`OrExpression`/`BetweenExpression`/`InExpression`/
`IsTrueExpression`/`EqExpression`, `variable`, `lower_bound_column`,
`upper_bound_column`, `values_column`, `value_column`, `mode`, `allow_gaps`,
`outputs`, `default`. A `DecisionTableModule` config that decider 1 accepts
parses here with only its wrapper renamed.

**Each expression class owns its own kernel emission.** decider 1's
`decision_table/config.py` gives every `BaseExpression` a `__call__` that
builds its own `pl.Expr` — the class *is* the behaviour, not a schema that a
separate dispatcher pattern-matches on. `codegen.py` used to be that
dispatcher: one `isinstance(expr, BetweenExpression)`/`isinstance(expr,
EqExpression)`/... chain to decide how to flatten And/Or and another to
decide what source a condition emits. Both chains are gone. `to_dnf()`
(And/Or flattening) and `emit()` (one condition's source lines plus the
`shared` arrays it reads them from) are methods on these classes instead,
so `codegen.py` only *orchestrates* — walk the DNF groups, ask each leaf to
emit itself, stitch the pieces into one kernel file. Adding a new
expression kind is one class here, in this file: fields, `get_variables`,
`validate_parameters`, `emit` (and `to_dnf` only if it is itself a
connective like `AndExpression`/`OrExpression`) — plus one line adding it to
the `Expression` union below. Nothing in `codegen.py` changes, because
nothing in `codegen.py` knows the expression kinds by name any more.

`emit()` needs a little more from the table's emitter than a bare
`DecisionTable` gives it — a place to register which input variables the
kernel signature needs, hoisted string-matcher bookkeeping, a unique array
prefix per condition. `ConditionContext` (below `EmittedCondition`) is the
small structural protocol that names exactly that, and
`tables.codegen._TableEmitter` satisfies it without this module importing
codegen — the one import that must not happen, or `codegen.py` (schema
classes -> codegen orchestrator) and `schema.py` (orchestrator ->
schema classes) would import each other.

Two divergences:

1. **`parameters` is rows + dtypes, not a polars `DataFrame` subclass.**
   decider 1's `ParametersConfig` extends `decider.serializable.dataframe.
   DataFrame`, which drags polars into the document type. Here the document
   is plain data (`rows: list[dict]`, `dtypes: list[tuple[str, str]]`) —
   which is the same JSON decider 1's `DataFrame` serialises to, and keeps
   the schema importable without the heavy stack (doc 02 §2, "the graph is
   data"). The `data=`/`dtypes=` spelling decider 1's tests use is accepted
   verbatim.

2. **`BoundMode` *is* `decider2.trees.schema.RangeEndLogic`.** decider 1 has
   two identical two-value enums under two names — `BoundMode` in the
   decision table, `RangeEndLogic` in the tree — with the same members and
   the same rationale. They are aliased here so a fix to one cannot leave
   the other behind. decider 1's own comment, kept because it is the reason
   the enum has exactly two members:

       ``both_inclusive`` / ``both_exclusive`` are intentionally omitted:
       they create overlaps or gaps at shared boundaries, making contiguous
       range tables ambiguous.
"""
from __future__ import annotations

import typing as t
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field, model_validator

from decider2.trees.codegen import LINE_CAP, TreeTooLarge
from decider2.trees.schema import RangeEndLogic

__all__ = [
    "BoundMode",
    "ParametersConfig",
    "EmittedCondition",
    "ConditionContext",
    "TableTooComplex",
    "AndExpression",
    "OrExpression",
    "BetweenExpression",
    "InExpression",
    "IsTrueExpression",
    "EqExpression",
    "Expression",
    "DecisionTable",
]


class TableTooComplex(TreeTooLarge):
    """The table's flattened expression exceeds the emitted-line cap.

    Raised from `AndExpression.to_dnf()` when distributing And over Or
    would blow past doc 05 §7's line cap before a single line is emitted —
    the only place the expression *shape* can explode, so the only place
    that needs to guard against it. `codegen.py` raises the sibling check
    on the kernel's total emitted line count, once the whole table (every
    condition's `emit()`, every output column) is known.
    """


def _f64(values: t.Sequence[t.Any], fill: float = 0.0) -> np.ndarray:
    return np.array([fill if v is None else float(v) for v in values], dtype=np.float64)


def _bool(values: t.Sequence[t.Any]) -> np.ndarray:
    return np.array([bool(v) for v in values], dtype=np.bool_)


@dataclass
class EmittedCondition:
    """One condition, already resolved to `shared` arrays plus source lines.

    What `Expression.emit()` returns — decider2's analogue of decider 1's
    `BaseExpression.__call__` returning a `pl.Expr`. `arrays` is this
    condition's *data* (the table's rows: free to change, doc 08 §3.4);
    `lines` is its *shape* (the expression: a compile if it changes).
    `kind`/`variable` are carried through for introspection (`TableModule.
    explain()`'s callers) but are not otherwise read by `codegen.py`.
    """

    kind: str
    variable: str
    arrays: t.Dict[str, np.ndarray] = field(default_factory=dict)
    lines: t.List[str] = field(default_factory=list)


class ConditionContext(t.Protocol):
    """What a leaf expression's `emit()` needs from the table's emitter.

    Structural, not a base class: `tables.codegen._TableEmitter` satisfies
    this without either module importing the other. `parameters` is the
    table's rows; `use_var` registers an input variable and returns its
    kernel identifier; `matcher_name` and `literal_index` are the hoisted
    string-matcher bookkeeping a string-valued `eq`/`in` condition needs
    (doc 05 §1.5) — the same mechanism `decider2.trees.codegen` uses.
    """

    parameters: "ParametersConfig"

    def use_var(self, variable: str) -> str: ...
    def matcher_name(self, variable: str) -> str: ...
    def literal_index(self, variable: str, literal: str) -> int: ...


BoundMode = RangeEndLogic
"""decider 1's `BoundMode`, aliased onto the tree's `RangeEndLogic`.

    lower_inclusive  ->  [lower, upper)   i.e.  lower <= x < upper
    upper_inclusive  ->  (lower, upper]   i.e.  lower <  x <= upper
"""


class ParametersConfig(BaseModel):
    """The table itself: N rows of bound/value/output columns.

    decider 1's `ParametersConfig(data=..., dtypes=...)`, same constructor
    keywords, without the polars `DataFrame` base class.
    """

    data: t.List[t.Dict[str, t.Any]] = Field(default_factory=list)
    dtypes: t.Union[t.Dict[str, str], t.List[t.Tuple[str, t.Any]]] = Field(
        default_factory=dict
    )

    @property
    def columns(self) -> list[str]:
        if isinstance(self.dtypes, dict):
            return list(self.dtypes)
        return [name for name, _ in self.dtypes]

    @property
    def dtype_map(self) -> dict[str, t.Any]:
        if isinstance(self.dtypes, dict):
            return dict(self.dtypes)
        return {name: dt for name, dt in self.dtypes}

    def column(self, name: str) -> list[t.Any]:
        return [row.get(name) for row in self.data]

    def is_string_column(self, column: str) -> bool:
        """Whether `column` holds strings (or lists of strings) — a kernel
        cannot compare a string against an array of strings (doc 05 §1.5),
        so `eq`/`in` conditions on a string column route through the
        hoisted matcher instead of a plain numeric compare."""
        dtype = self.dtype_map.get(column)
        if isinstance(dtype, str) and dtype in ("String", "Utf8"):
            return True
        for row in self.data:
            value = row.get(column)
            if isinstance(value, str):
                return True
            if isinstance(value, (list, tuple)) and any(isinstance(v, str) for v in value):
                return True
        return False

    def __len__(self) -> int:
        return len(self.data)


class _BaseExpression(BaseModel):
    """decider 1's `BaseExpression` surface, minus the polars half.

    `get_variables()` is kept by name because it is the method decider 1's
    `DecisionTableModule.expand_nodes` calls to discover what the table
    reads; here it is what the generated kernel's signature is built from.

    `to_dnf()` and `emit()` are decider2's own additions, in decider 1's
    spirit (`BaseExpression.__call__` builds its own `pl.Expr`): every leaf
    condition (`between`/`in`/`is_true`/`eq`) overrides `emit()` and takes
    the inherited `to_dnf()`, which says "I am already one AND-of-leaves
    group" — `[[self]]`. The two connectives, `AndExpression`/
    `OrExpression`, override `to_dnf()` instead, to flatten themselves away,
    and never override `emit()`, because after `to_dnf()` runs, no And/Or
    node is left in any group for `codegen.py` to call it on.
    """

    def get_variables(self) -> list[str]:  # pragma: no cover - overridden
        raise NotImplementedError

    def validate_parameters(self, parameters: ParametersConfig) -> None:  # pragma: no cover
        raise NotImplementedError

    def to_dnf(self) -> list[list["Expression"]]:
        """This expression, flattened into OR-of-ANDs.

        The leaf default: an expression that is not itself a connective is
        already a one-condition AND-group. `AndExpression`/`OrExpression`
        override this to actually flatten; nothing else needs to.
        """
        return [[t.cast("Expression", self)]]

    def emit(self, ctx: ConditionContext, prefix: str) -> EmittedCondition:  # pragma: no cover
        """This leaf condition's `shared` arrays and kernel source lines.

        Only a leaf (never `AndExpression`/`OrExpression`, which flatten
        away in `to_dnf()` before `emit()` is ever called) overrides this.
        """
        raise NotImplementedError


def _dedup(names: t.Iterable[str]) -> list[str]:
    """Order-preserving de-dup — decider 1's `seen`/`result` walk in
    `AndExpression.get_variables`, and the determinism doc 05 §4.2 needs."""
    return list(dict.fromkeys(names))


class AndExpression(_BaseExpression):
    type: t.Literal["and"]
    expressions: t.List["Expression"]

    def get_variables(self) -> list[str]:
        return _dedup(v for e in self.expressions for v in e.get_variables())

    def validate_parameters(self, parameters: ParametersConfig) -> None:
        for e in self.expressions:
            e.validate_parameters(parameters)

    def to_dnf(self) -> list[list["Expression"]]:
        """Distribute AND over each child's OR-of-ANDs (cartesian product).

        decider 1's table vocabulary has no NOT, so every expression is
        already monotone and this terminates without negation pushing.
        Distributing And over Or is the one place this can grow, which is
        why the guard is here rather than in `OrExpression.to_dnf`: an Or
        alone only concatenates, it never multiplies.
        """
        groups: list[list[Expression]] = [[]]
        for e in self.expressions:
            sub_groups = e.to_dnf()
            groups = [g + s for g in groups for s in sub_groups]
            if len(groups) > 64:
                raise TableTooComplex(
                    "this table's And/Or nesting expands to more than 64 "
                    "disjuncts in normal form, which would emit more source "
                    f"than doc 05 §7's {LINE_CAP}-line cap allows. Split it "
                    "into two tables composed with `|`, or lift the shared "
                    "conditions out of the Or."
                )
        return groups


class OrExpression(_BaseExpression):
    type: t.Literal["or"]
    expressions: t.List["Expression"]

    def get_variables(self) -> list[str]:
        return _dedup(v for e in self.expressions for v in e.get_variables())

    def validate_parameters(self, parameters: ParametersConfig) -> None:
        for e in self.expressions:
            e.validate_parameters(parameters)

    def to_dnf(self) -> list[list["Expression"]]:
        """Concatenate each child's OR-of-ANDs — an Or of Ors is one Or."""
        out: list[list[Expression]] = []
        for e in self.expressions:
            out.extend(e.to_dnf())
        return out


class BetweenExpression(_BaseExpression):
    """A banded lookup — decider 1's `BetweenExpression`, same fields.

    The bound-resolution rule is decider 1's and is reproduced exactly,
    because it is load-bearing for real contiguous tables: a row's missing
    lower bound is taken from the *previous* row's upper, and a missing
    upper from the *next* row's lower. Only row 0 may have an open lower
    edge and only the last row an open upper edge; `allow_gaps=False` (the
    default) additionally requires each row's upper to equal the next row's
    lower.
    """

    type: t.Literal["between"]
    variable: str
    lower_bound_column: t.Optional[str] = None
    upper_bound_column: t.Optional[str] = None
    mode: BoundMode = BoundMode.lower_inclusive
    allow_gaps: bool = False

    def get_variables(self) -> list[str]:
        return [self.variable]

    def resolved_bounds(
        self, parameters: ParametersConfig
    ) -> list[tuple[t.Optional[float], t.Optional[float]]]:
        """decider 1's `_safe_list_get` neighbour fill, verbatim."""
        n = len(parameters)
        lower = parameters.column(self.lower_bound_column) if self.lower_bound_column else [None] * n
        upper = parameters.column(self.upper_bound_column) if self.upper_bound_column else [None] * n

        def at(lst: list, i: int) -> t.Any:
            return lst[i] if 0 <= i < len(lst) else None

        out = []
        for i in range(n):
            lo = lower[i] if lower[i] is not None else at(upper, i - 1)
            hi = upper[i] if upper[i] is not None else at(lower, i + 1)
            if lo is None and hi is None:
                raise ValueError(f"Row {i} has no lower or upper bound after resolution")
            out.append((lo, hi))
        return out

    def validate_parameters(self, parameters: ParametersConfig) -> None:
        if not self.lower_bound_column and not self.upper_bound_column:
            raise ValueError(
                "At least one of lower_bound_column or upper_bound_column must be specified"
            )
        columns = parameters.columns
        for col, label in ((self.lower_bound_column, "Lower"), (self.upper_bound_column, "Upper")):
            if col is not None and col not in columns:
                raise ValueError(f"{label} bound column '{col}' not found in parameters columns")

        n = len(parameters)
        lower = parameters.column(self.lower_bound_column) if self.lower_bound_column else [None] * n
        upper = parameters.column(self.upper_bound_column) if self.upper_bound_column else [None] * n

        def at(lst: list, i: int) -> t.Any:
            return lst[i] if 0 <= i < len(lst) else None

        for i in range(n):
            lo = lower[i] if lower[i] is not None else at(upper, i - 1)
            hi = upper[i] if upper[i] is not None else at(lower, i + 1)
            if lo is None and hi is None:
                raise ValueError(
                    f"Row {i}: both bounds are unresolvable. Only row 0's lower and the "
                    "last row's upper may be None (open edges)."
                )
            if lo is None and i > 0:
                raise ValueError(
                    f"Row {i}: lower bound unresolvable — only row 0 may have an open lower edge."
                )
            if hi is None and i < n - 1:
                raise ValueError(
                    f"Row {i}: upper bound unresolvable — only row {n - 1} may have an "
                    "open upper edge."
                )
            if not self.allow_gaps and i < n - 1:
                next_lower = lower[i + 1] if lower[i + 1] is not None else hi
                if hi is not None and next_lower is not None and hi != next_lower:
                    raise ValueError(
                        f"Row {i} upper ({hi}) != row {i + 1} lower ({next_lower}): ranges "
                        "are not contiguous. Set allow_gaps=True to permit this."
                    )

    def emit(self, ctx: ConditionContext, prefix: str) -> EmittedCondition:
        var = ctx.use_var(self.variable)
        bounds = self.resolved_bounds(ctx.parameters)
        lo_op, hi_op = (
            (">=", "<") if self.mode is BoundMode.lower_inclusive else (">", "<=")
        )
        arrays = {
            f"{prefix}_lo": _f64([lo for lo, _ in bounds]),
            f"{prefix}_hi": _f64([hi for _, hi in bounds]),
            f"{prefix}_has_lo": _bool([lo is not None for lo, _ in bounds]),
            f"{prefix}_has_hi": _bool([hi is not None for _, hi in bounds]),
        }
        lines = [
            f"if ok and {prefix}_has_lo[r] and not ({var} {lo_op} {prefix}_lo[r]): ok = False",
            f"if ok and {prefix}_has_hi[r] and not ({var} {hi_op} {prefix}_hi[r]): ok = False",
        ]
        return EmittedCondition("between", self.variable, arrays, lines)


class InExpression(_BaseExpression):
    """Set membership against a per-row list column — decider 1's
    `InExpression`."""

    type: t.Literal["in"]
    variable: str
    values_column: str

    def get_variables(self) -> list[str]:
        return [self.variable]

    def validate_parameters(self, parameters: ParametersConfig) -> None:
        if self.values_column not in parameters.columns:
            raise ValueError(
                f"Values column '{self.values_column}' not found in parameters columns"
            )
        for i, row in enumerate(parameters.data):
            value = row.get(self.values_column)
            if value is not None and not isinstance(value, (list, tuple)):
                raise ValueError(
                    f"Values column '{self.values_column}' must be a list type, row {i} "
                    f"holds {type(value).__name__}"
                )

    def emit(self, ctx: ConditionContext, prefix: str) -> EmittedCondition:
        n = len(ctx.parameters)
        column_values = ctx.parameters.column(self.values_column)
        per_row = [column_values[i] or [] for i in range(n)]
        is_string = ctx.parameters.is_string_column(self.values_column)
        if is_string:
            var = ctx.matcher_name(self.variable)
            ctx.use_var(self.variable)
            flat = [
                ctx.literal_index(self.variable, str(v))
                for values in per_row
                for v in values
            ]
        else:
            var = ctx.use_var(self.variable)
            flat = [float(v) for values in per_row for v in values]
        offsets = np.zeros(n + 1, dtype=np.int64)
        for i, values in enumerate(per_row):
            offsets[i + 1] = offsets[i] + len(values)
        arrays = {
            f"{prefix}_off": offsets,
            f"{prefix}_vals": np.array(
                flat, dtype=np.int64 if is_string else np.float64
            ).reshape(-1),
        }
        # CSR membership: the set for row r is vals[off[r]:off[r+1]].
        # Variable-length sets are exactly why the table's contents can
        # stay data while a tree's would have to be unrolled.
        lines = [
            "if ok:",
            "    hit = False",
            f"    for j in range({prefix}_off[r], {prefix}_off[r + 1]):",
            f"        if {var} == {prefix}_vals[j]:",
            "            hit = True",
            "            break",
            f"    if not hit and {prefix}_off[r + 1] > {prefix}_off[r]: ok = False",
            f"    if {prefix}_off[r + 1] == {prefix}_off[r]: ok = False",
        ]
        return EmittedCondition("in", self.variable, arrays, lines)


class IsTrueExpression(_BaseExpression):
    """A boolean gate that consults no table column — decider 1's
    `IsTrueExpression` returns `[expr] * len(parameters)`, i.e. the same
    condition for every row."""

    type: t.Literal["is_true"]
    variable: str

    def get_variables(self) -> list[str]:
        return [self.variable]

    def validate_parameters(self, parameters: ParametersConfig) -> None:
        return None

    def emit(self, ctx: ConditionContext, prefix: str) -> EmittedCondition:
        var = ctx.use_var(self.variable)
        return EmittedCondition(
            "is_true", self.variable, {}, [f"if not ({var} != 0): ok = False"]
        )


class EqExpression(_BaseExpression):
    """Equality against a per-row scalar column — decider 1's
    `EqExpression`, including its docstring's example config::

        {"type": "eq", "variable": "BureauKey", "value_column": "key"}
    """

    type: t.Literal["eq"]
    variable: str
    value_column: str

    def get_variables(self) -> list[str]:
        return [self.variable]

    def validate_parameters(self, parameters: ParametersConfig) -> None:
        if self.value_column not in parameters.columns:
            raise ValueError(
                f"Value column '{self.value_column}' not found in parameters columns"
            )

    def emit(self, ctx: ConditionContext, prefix: str) -> EmittedCondition:
        values = ctx.parameters.column(self.value_column)
        if ctx.parameters.is_string_column(self.value_column):
            var = ctx.matcher_name(self.variable)
            ctx.use_var(self.variable)
            codes = [
                -1 if v is None else ctx.literal_index(self.variable, str(v))
                for v in values
            ]
            arrays = {
                f"{prefix}_code": np.array(codes, dtype=np.int64),
                f"{prefix}_has": _bool([v is not None for v in values]),
            }
            lines = [
                f"if ok and {prefix}_has[r] and not ({var} == {prefix}_code[r]): ok = False"
            ]
            return EmittedCondition("eq", self.variable, arrays, lines)
        var = ctx.use_var(self.variable)
        arrays = {
            f"{prefix}_val": _f64(values),
            f"{prefix}_has": _bool([v is not None for v in values]),
        }
        lines = [f"if ok and {prefix}_has[r] and not ({var} == {prefix}_val[r]): ok = False"]
        return EmittedCondition("eq", self.variable, arrays, lines)


Expression = t.Annotated[
    t.Union[
        AndExpression,
        OrExpression,
        BetweenExpression,
        InExpression,
        IsTrueExpression,
        EqExpression,
    ],
    Field(discriminator="type"),
]

AndExpression.model_rebuild()
OrExpression.model_rebuild()


class DecisionTable(BaseModel):
    """decider 1's `DecisionTableModule`, as a document.

    Same fields: `parameters`, `expression`, `outputs`, `default`. The
    `type`/`unnest_output` fields are dropped — `type` discriminated
    decider 1's module registry, which decider2 does not have, and
    `unnest_output` existed because decider 1 returned a struct column;
    decider2's output frame is flat by construction (doc 03 §7), so every
    output column is already a top-level column.

    First match wins, and a row that matches nothing takes `default` —
    decider 1's `calculate_decision_table_output` chains `when/then` in row
    order and closes with `otherwise(default)`.
    """

    name: str = "decision_table"
    parameters: ParametersConfig
    expression: Expression
    outputs: t.List[str]
    default: t.Optional[t.List[t.Any]] = None

    @model_validator(mode="after")
    def validate_config(self) -> "DecisionTable":
        for output in self.outputs:
            if output not in self.parameters.columns:
                raise ValueError(
                    f"Output column '{output}' not found in parameters columns"
                )
        if self.default is not None and len(self.default) != len(self.outputs):
            raise ValueError(
                f"Default values length ({len(self.default)}) must match outputs "
                f"length ({len(self.outputs)})"
            )
        self.expression.validate_parameters(self.parameters)
        return self

    def default_for(self, column: str) -> t.Any:
        if self.default is None:
            return None
        return self.default[self.outputs.index(column)]
