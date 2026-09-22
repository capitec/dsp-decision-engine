"""Decision table -> a generic row-scan kernel (doc 08 §3.4).

**A table was already not codegen'd the way a tree was, and that is doc
08 §3.4's call**, by its own test: "can one compiled loop evaluate every
instance of this kind, with the instance supplied as arrays? If yes,
generic kernel. If it needs a `switch` over node types, codegen." A table
is "N rows x M condition columns, uniform operators", so it gets the
generic kernel, and §3.4's table lists its interior change as **free**.
Every bound, value and set already lived in an array passed through
`shared` (doc 03 §4.2) before this migration touched this module.

**What this migration removes is the last piece that was still text: the
EXPRESSION shape.** How many conditions there are, which operator each one
is and how And/Or groups them used to decide an emitted `if ok and
...: ok = False` chain — source, recompiled on a structural edit. Doc 08
§3.4's own "generic kernel" test says a table doesn't need that: one
compiled loop (`decider2.tables.interpreter.scan_table`) can evaluate any
instance of "N rows x M conditions, four fixed operator shapes" with the
instance supplied as arrays — the CONDITIONS are as much data as the ROWS
already were. `schema.Expression.to_dnf()` (And/Or flattening) and
`schema.Expression.emit()` (one condition's `shared` arrays plus its
`CondOp` — decider2's analogue of a source line, now data) stay methods on
those classes, the same way decider 1's `BaseExpression.__call__` builds
its own `pl.Expr` (`decider/modules/credit/decision_table/config.py`). What
is left here is pure orchestration that does not care which expression
kinds exist: call `to_dnf()`, hand each leaf an emitter and a fresh array
prefix, ask it to `emit()`, flatten the returned `CondOp`s into
`scan_table`'s flat op arrays, and stitch a tiny wrapper function around one
call into it — alongside the string matchers and output steps, neither of
which is expression-shaped, so neither belongs on an expression class.

Concretely, the split is now:

* **Rows AND the expression are data.** Every bound, every value, every
  set, and now every condition's kind/operator/variable too, lives in a
  numpy array passed through `shared`. Adding a row, retuning a bound,
  adding a condition or restructuring the And/Or nesting all change an
  array, not a line of source — no recompile, not even a staged one.
* **The variable and output-column SET is still code**, because it decides
  the wrapper function's own SIGNATURE (which columns the boundary layer
  must supply) — the same reason a tree's feature/param SET (not their
  values) is the one thing that still costs a rebuild.

The scan is one `for r in range(n_rows)` walk over `scan_table`'s flat op
arrays, first match wins, returning the matched row index or -1 — exactly
decider 1's semantics (`calculate_decision_table_output` chains `when/then`
in row order and closes with `otherwise(default)`), and the row index
doubles as the table's path capture, the same way a tree's `result_idx`
does.

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

from dataclasses import dataclass

import numpy as np

from decider2.tables.interpreter import BETWEEN, EQ, IN, IS_TRUE
from decider2.tables.schema import CondOp, DecisionTable, ParametersConfig, TableTooComplex
from decider2.trees.codegen import safe_ident

__all__ = ["EmittedTable", "emit_table", "TableTooComplex"]


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


class _TableEmitter:
    """The one piece of per-table state `Expression.emit()` needs.

    This is `schema.ConditionContext`'s implementation — structurally, not
    by inheritance, so `schema.py` never imports this module. Everything
    here is bookkeeping that has to be shared *across* conditions (which
    input variables the kernel signature ends up with, which distinct
    string literals a variable's matcher hoists, a strictly-increasing
    array-name prefix) rather than anything specific to one expression
    kind, which is why it stayed here instead of moving with `emit()`.
    """

    def __init__(self, table: DecisionTable, name: str) -> None:
        self.table = table
        self.name = safe_ident(name)
        self.shared: dict[str, np.ndarray] = {}
        self.variables: list[str] = []
        self._var_set: set[str] = set()
        # variable -> ordered distinct literals, for a string condition
        self.string_literals: dict[str, list[str]] = {}
        self._cond_seq = 0

    @property
    def parameters(self) -> ParametersConfig:
        return self.table.parameters

    def use_var(self, name: str) -> str:
        if name not in self._var_set:
            self._var_set.add(name)
            self.variables.append(name)
        return safe_ident(name)

    def literal_index(self, variable: str, literal: str) -> int:
        """Register a distinct string literal for `variable`, returning the
        code the table arrays will store."""
        literals = self.string_literals.setdefault(variable, [])
        if literal not in literals:
            literals.append(literal)
        return literals.index(literal)

    def matcher_name(self, variable: str) -> str:
        return f"{self.name}__m_{safe_ident(variable)}"

    def next_prefix(self) -> str:
        """A fresh, strictly-increasing array-name prefix for one condition
        (`{table}__c0`, `{table}__c1`, ...) — assigned in emission order, so
        two runs over the same table shape assign the same prefixes and
        therefore emit identical source (`test_editing_rows_never_
        recompiles`)."""
        k = self._cond_seq
        self._cond_seq += 1
        return f"{self.name}__c{k}"


# CondOp.kind (a plain string, decider2.tables.schema) -> scan_table's
# integer opcode (decider2.tables.interpreter). The one place a kind name
# is translated to an opcode — everything downstream is an int.
_OPCODE = {"between": BETWEEN, "eq": EQ, "is_true": IS_TRUE, "in": IN}


def emit_table(table: DecisionTable, *, name: str | None = None) -> EmittedTable:
    """Render one decision table as an importable wrapper around the
    generic row scanner."""
    name = safe_ident(name or table.name or "decision_table")
    emitter = _TableEmitter(table, name)
    groups = table.expression.to_dnf()
    emitted_groups = [
        [e.emit(emitter, emitter.next_prefix()) for e in group] for group in groups
    ]

    for group in emitted_groups:
        for cond in group:
            emitter.shared.update(cond.arrays)

    n_rows = len(table.parameters)
    emitter.shared[f"{name}__n_rows"] = np.array([n_rows], dtype=np.int64)

    # -- flatten the DNF groups into scan_table's flat op arrays ----------
    group_start: list[int] = []
    group_end: list[int] = []
    op_kind: list[int] = []
    op_var_idx: list[int] = []
    op_local: list[int] = []
    op_lo_op: list[int] = []
    op_hi_op: list[int] = []
    # per-kind CondOp lists, in encounter order — position within its own
    # list is the op_local slot a CMP-style opcode resolves against
    # (decider2.trees.codegen's "no deferred resolution needed" note: unlike
    # a tree's threshold space, nothing here is anonymous-vs-named, so a
    # condition's own local index is stable the instant it is appended).
    between_ops: list[CondOp] = []
    eq_ops: list[CondOp] = []
    in_ops: list[CondOp] = []

    var_index = {v: i for i, v in enumerate(emitter.variables)}
    for group in emitted_groups:
        group_start.append(len(op_kind))
        for cond in group:
            op = cond.op
            assert op is not None  # every leaf Expression.emit() sets one
            op_kind.append(_OPCODE[op.kind])
            op_var_idx.append(var_index[op.variable])
            op_lo_op.append(op.lo_op)
            op_hi_op.append(op.hi_op)
            if op.kind == "between":
                op_local.append(len(between_ops))
                between_ops.append(op)
            elif op.kind == "eq":
                op_local.append(len(eq_ops))
                eq_ops.append(op)
            elif op.kind == "in":
                op_local.append(len(in_ops))
                in_ops.append(op)
            else:  # is_true — reads no per-row-of-table data at all
                op_local.append(0)
        group_end.append(len(op_kind))

    lines: list[str] = []
    lines.append(f'"""Generated decision-table kernel for {table.name!r}.')
    lines.append("")
    lines.append("Generic over ROWS AND CONDITIONS: every bound, value, set and")
    lines.append("op-shape below is read from an array passed through `shared`,")
    lines.append("never written into this file, so adding a row, retuning a bound")
    lines.append("or restructuring the And/Or expression is a value change with no")
    lines.append("recompile at all (doc 08 §2, §3.4). decider2.tables.interpreter.")
    lines.append("scan_table is the one generic loop that walks them; nothing here")
    lines.append("is per-condition branching logic any more.")
    lines.append('"""')
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("import numpy as np")
    lines.append("")
    lines.append("from decider2.params import param")
    lines.append("from decider2.tables.interpreter import scan_table")
    lines.append("")

    # Hoisted string matchers — same mechanism as a tree's, unchanged.
    matcher_names: list[str] = []
    for variable, literals in emitter.string_literals.items():
        fn_name = emitter.matcher_name(variable)
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

    prefix = f"_{name}"
    lines.append("")
    lines.append(f"{prefix}__group_start = np.array({group_start!r}, dtype=np.int32)")
    lines.append(f"{prefix}__group_end = np.array({group_end!r}, dtype=np.int32)")
    lines.append(f"{prefix}__op_kind = np.array({op_kind!r}, dtype=np.int32)")
    lines.append(f"{prefix}__op_var_idx = np.array({op_var_idx!r}, dtype=np.int32)")
    lines.append(f"{prefix}__op_local = np.array({op_local!r}, dtype=np.int32)")
    lines.append(f"{prefix}__op_lo_op = np.array({op_lo_op!r}, dtype=np.int32)")
    lines.append(f"{prefix}__op_hi_op = np.array({op_hi_op!r}, dtype=np.int32)")
    lines.append(f"{prefix}__dummy_f = np.zeros(1)")
    lines.append(f"{prefix}__dummy_i = np.zeros(1, dtype=np.int64)")

    # The scan itself.
    sig_parts: list[str] = []
    for variable in emitter.variables:
        if variable in emitter.string_literals:
            sig_parts.append(f"{emitter.matcher_name(variable)}: int")
        else:
            sig_parts.append(f"{safe_ident(variable)}: float")
    sig_parts.append("shared")

    row_fn = f"{name}_row"
    lines.append("")
    lines.append("")
    lines.append(f"def {row_fn}({', '.join(sig_parts)}) -> int:")
    lines.append(f'    """Index of the first row of {table.name!r} that matches, or -1."""')
    for key in emitter.shared:
        lines.append(f"    {key} = shared.{key}")

    vars_items = []
    for variable in emitter.variables:
        if variable in emitter.string_literals:
            vars_items.append(f"float({emitter.matcher_name(variable)})")
        else:
            vars_items.append(safe_ident(variable))
    lines.append(f"    vars_ = {_tuple_expr(vars_items)}")

    def _quad(op: CondOp) -> str:
        return f"({op.lo_key}, {op.hi_key}, {op.has_lo_key}, {op.has_hi_key})"

    def _pair(op: CondOp) -> str:
        return f"({op.val_key}, {op.has_key})"

    def _csr_off(op: CondOp) -> str:
        return op.off_key

    def _csr_vals(op: CondOp) -> str:
        return op.vals_key

    between_expr = _tuple_expr([_quad(o) for o in between_ops], dummy=f"({prefix}__dummy_f, {prefix}__dummy_f, {prefix}__dummy_f, {prefix}__dummy_f)")
    eq_expr = _tuple_expr([_pair(o) for o in eq_ops], dummy=f"({prefix}__dummy_f, {prefix}__dummy_f)")
    in_off_expr = _tuple_expr([_csr_off(o) for o in in_ops], dummy=f"{prefix}__dummy_i")
    in_vals_expr = _tuple_expr([_csr_vals(o) for o in in_ops], dummy=f"{prefix}__dummy_f")

    lines.append(f"    between_bounds = {between_expr}")
    lines.append(f"    eq_bounds = {eq_expr}")
    lines.append(f"    in_offsets = {in_off_expr}")
    lines.append(f"    in_values = {in_vals_expr}")
    lines.append(f"    n = int({name}__n_rows[0])")
    lines.append(
        f"    return scan_table(vars_, n, {prefix}__group_start, {prefix}__group_end, "
        f"{prefix}__op_kind, {prefix}__op_var_idx, {prefix}__op_local, "
        f"{prefix}__op_lo_op, {prefix}__op_hi_op, between_bounds, eq_bounds, "
        "in_offsets, in_values)"
    )
    lines.append("")

    # One step per numeric/boolean output column — and its values are an
    # ARRAY, not emitted source. Unchanged by this migration: a table's
    # output column was already as free as its bounds.
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


def _tuple_expr(items: list[str], *, dummy: str = "0.0") -> str:
    """A numba-safe homogeneous-tuple literal — never truly empty for the
    same reason `decider2.trees.codegen._tuple_expr` isn't: `scan_table`'s
    unreachable-for-THIS-table branch (e.g. no `in` condition anywhere)
    still has to TYPE-CHECK its `in_offsets[local]`/`in_values[local]`
    reads, so a table with zero conditions of some kind gets one inert
    dummy entry no op ever indexes into."""
    if not items:
        return f"({dummy},)"
    if len(items) == 1:
        return f"({items[0]},)"
    return "(" + ", ".join(items) + ")"
