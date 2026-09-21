"""Decision table -> a generic row-scan kernel (doc 08 §3.4).

**A table is not codegen'd the way a tree is, and that is doc 08 §3.4's
call**, by its own test: "can one compiled loop evaluate every instance of
this kind, with the instance supplied as arrays? If yes, generic kernel. If
it needs a `switch` over node types, codegen." A table is "N rows x M
condition columns, uniform operators", so it gets the generic kernel, and
§3.4's table lists its interior change as **free**.

**This module has no `switch` over expression types either.** It used to:
a `_to_dnf` dispatching on `isinstance(expr, OrExpression)`/`AndExpression`,
and `_TableEmitter._emit_condition` dispatching on
`isinstance(expr, BetweenExpression | InExpression | IsTrueExpression |
EqExpression)`. Both are gone. `schema.Expression.to_dnf()` and
`schema.Expression.emit()` are methods on those classes now — the And/Or
flattening and the per-condition source-and-arrays are each expression's own
behaviour, the same way decider 1's `BaseExpression.__call__` builds its own
`pl.Expr` (`decider/modules/credit/decision_table/config.py`). What is left
here is pure orchestration that does not care which expression kinds exist:
call `to_dnf()`, hand each leaf an emitter and a fresh array prefix, ask it
to `emit()`, and stitch the returned lines and arrays into one kernel file
alongside the string matchers and output steps — neither of which is
expression-shaped, so neither belongs on an expression class.

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

from dataclasses import dataclass

import numpy as np

from decider2.tables.schema import DecisionTable, ParametersConfig, TableTooComplex
from decider2.trees.codegen import LINE_CAP, safe_ident

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


def emit_table(table: DecisionTable, *, name: str | None = None) -> EmittedTable:
    """Render one decision table as importable kernel source."""
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
