"""Decision table -> a generic row-scan kernel (doc 08 §3.4).

**A table is not codegen'd the way a tree is, and that is doc 08 §3.4's
call**, by its own test: "can one compiled loop evaluate every instance of
this kind, with the instance supplied as arrays? If yes, generic kernel. If
it needs a `switch` over node types, codegen." A table is "N rows x M
condition columns, uniform operators", so it gets the generic kernel, and
§3.4's table lists its interior change as **free**.

Concretely, the split is:

* **Rows are data.** Every bound, every value, every set lives in a numpy
  array passed through `shared` (doc 03 §4.2). Adding a row, deleting a row
  or retuning a bound changes an array, not a line of source — no
  recompile, not even a staged one. This is the strongest form of doc 08
  §2's "values ... free — swap a bundle, no compile", and it is why a table
  and a tree are different engines rather than one.
* **The expression shape is code.** How many conditions there are, which
  operator each one is and which variable it reads decide the emitted loop
  body and the kernel's signature. Changing those recompiles, correctly:
  it is doc 08 §2's "interiors — shape" class.

The emitted body is one `for r in range(n_rows)` scan, first match wins,
returning the matched row index or -1 — which is exactly decider 1's
semantics (`calculate_decision_table_output` chains `when/then` in row order
and closes with `otherwise(default)`), and the row index doubles as the
table's path capture, the same way a tree's `result_idx` does.

**Strings.** A string variable cannot be compared against an array of
strings in a kernel (doc 05 §1.5). Each *distinct literal* a string
condition mentions becomes a `str` `param()` on a hoisted one-input matcher
step — the same mechanism `decider2.trees.codegen` uses, imported from
there so the two cannot drift — and the table's arrays hold that literal's
*index*, not its text. The consequence is honest and worth stating: for a
string column, retuning a literal's text is free, but adding a new distinct
literal is a shape change. Numeric tables keep the full free-interior
property.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from decider2.tables.schema import (
    AndExpression,
    BetweenExpression,
    BoundMode,
    DecisionTable,
    EqExpression,
    InExpression,
    IsTrueExpression,
    OrExpression,
)
from decider2.trees.codegen import LINE_CAP, TreeTooLarge, safe_ident

__all__ = ["EmittedTable", "emit_table", "TableTooComplex"]


class TableTooComplex(TreeTooLarge):
    """The table's flattened expression exceeds the emitted-line cap."""


@dataclass
class _Cond:
    """One condition column, already resolved to arrays plus a test."""

    kind: str                     # between | eq | in | is_true
    variable: str
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)


@dataclass
class EmittedTable:
    source: str
    row_fn_name: str
    matcher_fn_names: tuple[str, ...]
    output_fn_names: tuple[str, ...]
    variables: tuple[str, ...]
    string_variables: tuple[str, ...]
    shared: dict[str, np.ndarray]
    n_rows: int
    n_conditions: int
    emitted_lines: int


def _to_dnf(expr: Any) -> list[list[Any]]:
    """Flatten an expression into OR-of-ANDs.

    decider 1's table vocabulary has no NOT, so every expression is already
    monotone and this terminates without negation pushing. `And` distributes
    over `Or`, which is the one place this can grow: the guard below refuses
    a distribution that would blow past the line cap rather than emitting it.
    """
    if isinstance(expr, OrExpression):
        out: list[list[Any]] = []
        for sub in expr.expressions:
            out.extend(_to_dnf(sub))
        return out
    if isinstance(expr, AndExpression):
        groups: list[list[Any]] = [[]]
        for sub in expr.expressions:
            sub_groups = _to_dnf(sub)
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
    return [[expr]]


def _f64(values: Sequence[Any], fill: float = 0.0) -> np.ndarray:
    return np.array([fill if v is None else float(v) for v in values], dtype=np.float64)


def _bool(values: Sequence[Any]) -> np.ndarray:
    return np.array([bool(v) for v in values], dtype=np.bool_)


class _TableEmitter:
    def __init__(self, table: DecisionTable, name: str) -> None:
        self.table = table
        self.name = safe_ident(name)
        self.shared: dict[str, np.ndarray] = {}
        self.variables: list[str] = []
        self._var_set: set[str] = set()
        # variable -> ordered distinct literals, for a string condition
        self.string_literals: dict[str, list[str]] = {}
        self._cond_seq = 0

    def _use_var(self, name: str) -> str:
        if name not in self._var_set:
            self._var_set.add(name)
            self.variables.append(name)
        return safe_ident(name)

    def _is_string_column(self, column: str) -> bool:
        dtype = self.table.parameters.dtype_map.get(column)
        if isinstance(dtype, str) and dtype in ("String", "Utf8"):
            return True
        for row in self.table.parameters.data:
            value = row.get(column)
            if isinstance(value, str):
                return True
            if isinstance(value, (list, tuple)) and any(isinstance(v, str) for v in value):
                return True
        return False

    def _literal_index(self, variable: str, literal: str) -> int:
        """Register a distinct string literal for `variable`, returning the
        code the table arrays will store."""
        literals = self.string_literals.setdefault(variable, [])
        if literal not in literals:
            literals.append(literal)
        return literals.index(literal)

    def _matcher_name(self, variable: str) -> str:
        return f"{self.name}__m_{safe_ident(variable)}"

    def _emit_condition(self, expr: Any) -> _Cond:
        k = self._cond_seq
        self._cond_seq += 1
        prefix = f"{self.name}__c{k}"
        rows = self.table.parameters
        n = len(rows)

        if isinstance(expr, IsTrueExpression):
            var = self._use_var(expr.variable)
            return _Cond("is_true", expr.variable, {}, [f"if not ({var} != 0): ok = False"])

        if isinstance(expr, BetweenExpression):
            var = self._use_var(expr.variable)
            bounds = expr.resolved_bounds(rows)
            lo_op, hi_op = (
                (">=", "<") if expr.mode is BoundMode.lower_inclusive else (">", "<=")
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
            return _Cond("between", expr.variable, arrays, lines)

        if isinstance(expr, EqExpression):
            column = expr.value_column
            values = rows.column(column)
            if self._is_string_column(column):
                var = self._matcher_name(expr.variable)
                self._use_var(expr.variable)
                codes = [
                    -1 if v is None else self._literal_index(expr.variable, str(v))
                    for v in values
                ]
                arrays = {
                    f"{prefix}_code": np.array(codes, dtype=np.int64),
                    f"{prefix}_has": _bool([v is not None for v in values]),
                }
                lines = [
                    f"if ok and {prefix}_has[r] and not ({var} == {prefix}_code[r]): ok = False"
                ]
                return _Cond("eq", expr.variable, arrays, lines)
            var = self._use_var(expr.variable)
            arrays = {
                f"{prefix}_val": _f64(values),
                f"{prefix}_has": _bool([v is not None for v in values]),
            }
            lines = [
                f"if ok and {prefix}_has[r] and not ({var} == {prefix}_val[r]): ok = False"
            ]
            return _Cond("eq", expr.variable, arrays, lines)

        if isinstance(expr, InExpression):
            column = expr.values_column
            per_row = [rows.column(column)[i] or [] for i in range(n)]
            is_string = self._is_string_column(column)
            if is_string:
                var = self._matcher_name(expr.variable)
                self._use_var(expr.variable)
                flat = [
                    self._literal_index(expr.variable, str(v))
                    for values in per_row
                    for v in values
                ]
            else:
                var = self._use_var(expr.variable)
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
            return _Cond("in", expr.variable, arrays, lines)

        raise AssertionError(f"unhandled expression {type(expr)!r}")  # pragma: no cover


def emit_table(table: DecisionTable, *, name: str | None = None) -> EmittedTable:
    """Render one decision table as importable kernel source."""
    name = safe_ident(name or table.name or "decision_table")
    emitter = _TableEmitter(table, name)
    groups = _to_dnf(table.expression)
    emitted_groups = [[emitter._emit_condition(e) for e in group] for group in groups]

    for group in emitted_groups:
        for cond in group:
            emitter.shared.update(cond.arrays)

    n_rows = len(table.parameters)
    emitter.shared[f"{name}__n_rows"] = np.array([n_rows], dtype=np.int64)

    lines: list[str] = []
    lines.append(f'"""Generated decision-table kernel for {table.name!r}.')
    lines.append("")
    lines.append("Generic over ROWS: every bound and value below is read from an")
    lines.append("array passed through `shared`, never written into this file, so")
    lines.append("adding a row or retuning a bound is a value change with no")
    lines.append("recompile at all (doc 08 §2, §3.4). The loop shape is fixed by the")
    lines.append("expression, which is the part that is code.")
    lines.append('"""')
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from decider2.params import param")
    lines.append("")

    # Hoisted string matchers — same mechanism as a tree's.
    matcher_names: list[str] = []
    for variable, literals in emitter.string_literals.items():
        fn_name = emitter._matcher_name(variable)
        matcher_names.append(fn_name)
        ident = safe_ident(variable)
        param_names = [f"{ident}_lit_{i}" for i in range(len(literals))]
        args = ", ".join(
            [f"{ident}: str"]
            + [f"{p}: str = param({lit!r})" for p, lit in zip(param_names, literals)]
        )
        lines.append("")
        lines.append(f"def {fn_name}({args}) -> int:")
        lines.append(f'    """Which distinct literal `{variable}` equals, or -1.')
        lines.append("")
        lines.append("    Hoisted so this step reads exactly one `str` input, which is")
        lines.append("    what runtime.invoke._resolve_str_param_code requires to turn")
        lines.append("    each literal into that column's int32 dictionary code")
        lines.append("    (doc 05 §1.5, EXPERIMENTS.md §O).")
        lines.append('    """')
        for i, p in enumerate(param_names):
            branch = "if" if i == 0 else "elif"
            lines.append(f"    {branch} {ident} == {p}:")
            lines.append(f"        return {i}")
        lines.append("    return -1")
        lines.append("")

    # The scan itself.
    sig_parts: list[str] = []
    for variable in emitter.variables:
        if variable in emitter.string_literals:
            sig_parts.append(f"{emitter._matcher_name(variable)}: int")
        else:
            sig_parts.append(f"{safe_ident(variable)}: float")
    sig_parts.append("shared")

    row_fn = f"{name}_row"
    lines.append("")
    lines.append(f"def {row_fn}({', '.join(sig_parts)}) -> int:")
    lines.append(f'    """Index of the first row of {table.name!r} that matches, or -1."""')
    for key in emitter.shared:
        lines.append(f"    {key} = shared.{key}")
    lines.append(f"    n = {name}__n_rows[0]")
    lines.append("    for r in range(n):")
    lines.append("        matched = False")
    for gi, group in enumerate(emitted_groups):
        lines.append("        if not matched:")
        lines.append("            ok = True")
        for cond in group:
            for ln in cond.lines:
                lines.append(f"            {ln}")
        lines.append("            if ok:")
        lines.append("                matched = True")
    lines.append("        if matched:")
    lines.append("            return r")
    lines.append("    return -1")
    lines.append("")

    # One step per numeric/boolean output column — and its values are an
    # ARRAY, not emitted source.
    #
    # This is the difference between a table and a tree, and it is the whole
    # of doc 08 §3.4's "free" interior. A tree's leaf values are emitted
    # (`trees.codegen._literal`: what a leaf returns is that tree's shape),
    # so editing one costs a staged compile. A table's output column is a
    # column of the table — as much data as its bounds are — so it rides in
    # `shared` and editing it costs nothing at all. An earlier draft of this
    # module emitted them the tree's way; `test_editing_rows_never_
    # recompiles` caught that the source then changed with the data, which
    # would have silently downgraded the table to the tree's guarantee.
    output_fns: list[str] = []
    for column in table.outputs:
        values = table.parameters.column(column)
        if any(isinstance(v, str) for v in values):
            continue
        declared = table.parameters.dtype_map.get(column)
        is_bool = all(isinstance(v, bool) or v is None for v in values)
        is_int = (not is_bool) and (
            declared in ("Int64", "Int32")
            or all(isinstance(v, int) or v is None for v in values)
        )
        py_type = "bool" if is_bool else ("int" if is_int else "float")
        np_dtype = np.bool_ if is_bool else (np.int64 if is_int else np.float64)

        fn_name = safe_ident(column)
        values_key = f"{name}__out_{fn_name}"
        default_key = f"{values_key}_default"
        fill = False if is_bool else 0
        emitter.shared[values_key] = np.array(
            [fill if v is None else v for v in values], dtype=np_dtype
        )
        default = table.default_for(column)
        emitter.shared[default_key] = np.array(
            [fill if default is None else default], dtype=np_dtype
        )

        lines.append("")
        lines.append(f"def {fn_name}({row_fn}: int, shared) -> {py_type}:")
        lines.append(f'    """`{column}` for the matched row, read from the table.')
        lines.append("")
        lines.append("    The values are an array in `shared`, never source, so")
        lines.append("    editing this column is free (doc 08 §3.4).")
        lines.append('    """')
        lines.append(f"    if {row_fn} < 0:")
        lines.append(f"        return shared.{default_key}[0]")
        lines.append(f"    return shared.{values_key}[{row_fn}]")
        lines.append("")
        output_fns.append(fn_name)

    source = "\n".join(lines) + "\n"
    emitted = sum(1 for ln in lines if ln.strip() and not ln.strip().startswith(("#", '"""')))
    if emitted > LINE_CAP:
        raise TableTooComplex(
            f"decision table {table.name!r} emits {emitted} lines, over the "
            f"{LINE_CAP}-line cap (doc 05 §7). Its ROWS are free — they are arrays — "
            "so this is its expression shape or its output columns, not its size. "
            "Split it into two tables composed with `|`."
        )

    return EmittedTable(
        source=source,
        row_fn_name=row_fn,
        matcher_fn_names=tuple(matcher_names),
        output_fn_names=tuple(output_fns),
        variables=tuple(emitter.variables),
        string_variables=tuple(emitter.string_literals),
        shared=emitter.shared,
        n_rows=n_rows,
        n_conditions=sum(len(g) for g in emitted_groups),
        emitted_lines=emitted,
    )
