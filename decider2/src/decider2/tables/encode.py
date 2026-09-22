"""Decision table -> `Step`s built directly, `fn` a pre-built njit closure
(doc 08 §3.4). Renamed from `codegen.py` — see `decider2.trees.encode`'s
module docstring for why the old name told a reader the opposite of the
truth.

**A table was already not codegen'd the way a tree was, and that stays
true.** Doc 08 §3.4's own test — "can one compiled loop evaluate every
instance of this kind, with the instance supplied as arrays?" — a table
answers "yes" for both its ROWS (always arrays, via `shared`, before this
migration touched anything) and its CONDITIONS (`schema.Expression.to_dnf()`
+ `.emit()`, unchanged by this pass — those methods already returned DATA,
an `EmittedCondition` of arrays plus a `CondOp`, never source text).

**What this pass removes is the wrapper this module used to render as a
`.py` file.** `_TableEmitter`'s bookkeeping — which input variables the
kernel needs, hoisted string-matcher registration, a unique array prefix per
condition — is unchanged; only the tail that used to write
`def {name}_row(...): ... return scan_table(...)` as text now builds a
`Step` directly, with `fn` a closure over the table's SHAPE arrays (group
boundaries, op kinds — genuinely fixed at build time, captured directly)
and, for the table's ROW data (bounds/values/sets — the "free interior" doc
08 §3.4 promises stays retunable without recompiling), a closure that reads
it off the `shared` bundle **by name, at call time** — `getattr(shared,
key)` inline in the closure body, with `key` a closure-captured string
constant, never spliced into source (numba accepts `getattr` with a
constant-string argument; validated empirically before relying on it
here). This is load-bearing, not cosmetic:
`test_a_rebuilt_table_answers_differently_with_no_new_compile`
reuses the SAME compiled driver across two `table_module()` builds with
different row data, passing only a new `shared=` — which only works if row
data is a runtime argument, never a captured constant.

**Strings.** A string variable cannot be compared against an array of
strings in a kernel (doc 05 §1.5). Each *distinct literal* a string
condition mentions becomes a `str`-typed `ParamDecl` on a hoisted
one-input matcher step — the same mechanism `decider2.trees.encode` uses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from pydantic import Field

from decider2.tables.interpreter import BETWEEN, EQ, IN, IS_TRUE, scan_table
from decider2.tables.schema import CondOp, DecisionTable, ParametersConfig, TableTooComplex
from decider2.trees.encode import safe_ident
from decider2.types import Input, NullPolicy, ParamDecl, Step

__all__ = ["EncodedTable", "encode_table", "TableTooComplex"]


@dataclass
class EncodedTable:
    row_step: Step
    matcher_steps: tuple[Step, ...]
    output_steps: tuple[Step, ...]
    row_column: str
    variables: tuple[str, ...]
    string_variables: tuple[str, ...]
    shared: dict[str, np.ndarray]
    n_rows: int
    n_conditions: int


class _TableEmitter:
    """The one piece of per-table state `Expression.emit()` needs.

    Unchanged from the previous migration — see `schema.ConditionContext`'s
    docstring. `shared` is still every row-data array this table's
    conditions read, keyed by name; `decider2.tables.build.TableModule.
    shared` still hands the same dict back for introspection AND for a
    caller composing several tables (`shared={**a.shared, **b.shared}`) —
    unchanged, because the runtime read-by-name mechanism below depends on
    exactly these keys still being present on whatever `shared` bundle
    reaches the kernel.
    """

    def __init__(self, table: DecisionTable, name: str) -> None:
        self.table = table
        self.name = safe_ident(name)
        self.shared: dict[str, np.ndarray] = {}
        self.variables: list[str] = []
        self._var_set: set[str] = set()
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
        literals = self.string_literals.setdefault(variable, [])
        if literal not in literals:
            literals.append(literal)
        return literals.index(literal)

    def matcher_name(self, variable: str) -> str:
        return f"{self.name}__m_{safe_ident(variable)}"

    def next_prefix(self) -> str:
        k = self._cond_seq
        self._cond_seq += 1
        return f"{self.name}__c{k}"


_OPCODE = {"between": BETWEEN, "eq": EQ, "is_true": IS_TRUE, "in": IN}


# ---------------------------------------------------------------------------
# Row-data assembly — no per-condition-COUNT closure family any more
# (replaces `_compose0`..`_compose8`, a wall at 8 conditions of one kind in
# a whole table). `shared`'s row data is still read BY NAME at call time
# (`getattr(shared, key)`, doc 08 §3.4's free interior: a table rebuilt with
# new row data answers differently through the SAME compiled `row_fn`) — what
# changes is what gets STORED there: one 2D array per (condition kind,
# bound role), shape `(conditions of that kind, n_rows)`, built ONCE, in
# plain Python, by stacking each condition's own 1D array (`decider2.
# tables.schema.Expression.emit()`'s `EmittedCondition.arrays`, unchanged)
# rather than composed by N closures called at scan time. `local` (`op_
# local`, in `encode_table` below) becomes an ordinary array index into
# dimension 0, so there is no condition count this raises past — see
# `decider2.tables.interpreter.scan_table`'s own docstring for the read
# side.
# ---------------------------------------------------------------------------


# Row data is read off `shared` by name at call time — `getattr(shared,
# key)` INLINE in `row_fn`/`out_fn` below, with `key` a closure-captured
# string constant (numba types a captured `str` as a literal, which is what
# its `getattr` overload needs) — never a captured array, so a table rebuilt
# with new row data (same shape) answers differently through the SAME
# compiled closure (doc 08 §3.4's free interior).
#
# This used to go through a separate `@njit(cache=True) def _shared_get(
# shared, key)` helper. Measured while fixing the cross-process cache: each
# of its call sites became its OWN numba specialisation (12 per single-
# output table, one per (bundle class, key) pair), each re-typing the whole
# `shared` namedtuple from scratch. Inlining them took a 3-band table's
# cold first `apply()` in a fresh process from ~5.6s to ~3.1s and its
# on-disk cache entries from 18 to 6 (`tests/test_shared_bundle_cache.py`'s
# child script, `NUMBA_DEBUG_CACHE=1`): the bundle is typed once per
# function, and the helper's 12 compile pipelines do not exist.


def _stack2d(shared: dict, ops: list, key_attr: str, n_rows: int) -> np.ndarray:
    """`shared[getattr(op, key_attr)]` for each `op` in `ops`, stacked into
    ONE 2D float64 array of shape `(len(ops), n_rows)` — a build-time,
    plain-numpy operation (never inside njit: `ops`/`key_attr` are fixed
    once the table's conditions are known, so nothing here needs to re-run
    per call the way reading `shared` by name at scan time does)."""
    if not ops:
        return np.empty((0, n_rows), dtype=np.float64)
    return np.stack([shared[getattr(op, key_attr)] for op in ops])


def _stack_in(shared: dict, in_ops: list, n_rows: int) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """The three arrays `scan_table`'s IN handling needs. `in_off`: each
    condition's own CSR offsets array is `n_rows + 1` long (UNIFORM width
    across conditions, unlike its values), so these stack into one 2D
    array the same way `_stack2d` does. `in_vals`: every condition's own
    (ragged-length) admitted-value array, concatenated end to end — a
    per-condition value COUNT varies, and a 2D array can't hold a ragged
    width. `in_vals_start`: condition `local`'s own base offset into that
    flat array — a SECOND level of CSR (one layer already exists per row
    within one condition, `decider2.tables.schema.InExpression`'s own
    docstring; this adds one more, across conditions)."""
    if not in_ops:
        return (
            np.empty((0, n_rows + 1), dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.zeros(1, dtype=np.int64),
        )
    off_2d = np.stack([shared[op.off_key] for op in in_ops]).astype(np.int64)
    vals_list = [shared[op.vals_key] for op in in_ops]
    starts = np.zeros(len(in_ops) + 1, dtype=np.int64)
    for j, vals in enumerate(vals_list):
        starts[j + 1] = starts[j] + len(vals)
    flat = np.concatenate(vals_list).astype(np.float64)
    return off_2d, flat, starts


def _build_matcher_fn():
    """Same shape as `decider2.trees.encode._build_matcher_fn` — `(args,
    params) -> int`, args[0] the column's own int32 code, params the string
    literal codes."""

    @njit(cache=True)
    def matcher_fn(args, params):
        code = args[0]
        for i in range(len(params)):
            if params[i] == code:
                return i
        return -1

    return matcher_fn


def _build_output_fn(shared_key: str, default_key: str):
    @njit(cache=True)
    def out_fn(args, params, shared):
        row = args[0]
        if row < 0:
            return getattr(shared, default_key)[0]
        return getattr(shared, shared_key)[row]

    return out_fn


def encode_table(table: DecisionTable, *, name: str | None = None) -> EncodedTable:
    """Walk one decision table and build its `Step`s directly."""
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
    # `n_rows` is ROW data — how many rows this table has — not SHAPE (the
    # condition structure), so it goes through `shared` like every other
    # per-row array, never captured into the closure: two tables sharing
    # the SAME conditions but a different ROW COUNT (this pass's own
    # `test_a_rebuilt_table_answers_differently_with_no_new_compile`) must
    # answer correctly through the SAME compiled `row_fn`, fed a `shared`
    # whose arrays (and this count) are simply shorter or longer.
    n_rows_key = f"{name}__n_rows"
    emitter.shared[n_rows_key] = np.array([n_rows], dtype=np.int64)

    group_start: list[int] = []
    group_end: list[int] = []
    op_kind: list[int] = []
    op_var_idx: list[int] = []
    op_local: list[int] = []
    op_lo_op: list[int] = []
    op_hi_op: list[int] = []
    between_ops: list[CondOp] = []
    eq_ops: list[CondOp] = []
    in_ops: list[CondOp] = []

    var_index = {v: i for i, v in enumerate(emitter.variables)}
    for group in emitted_groups:
        group_start.append(len(op_kind))
        for cond in group:
            op = cond.op
            assert op is not None
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
            else:
                op_local.append(0)
        group_end.append(len(op_kind))

    group_start_arr = np.array(group_start, dtype=np.int32)
    group_end_arr = np.array(group_end, dtype=np.int32)
    op_kind_arr = np.array(op_kind, dtype=np.int32)
    op_var_idx_arr = np.array(op_var_idx, dtype=np.int32)
    op_local_arr = np.array(op_local, dtype=np.int32)
    op_lo_op_arr = np.array(op_lo_op, dtype=np.int32)
    op_hi_op_arr = np.array(op_hi_op, dtype=np.int32)

    # One 2D array per (condition kind, bound role) — `_stack2d`/`_stack_in`
    # above — stored into `shared` under a fixed, table-instance-qualified
    # key, so `row_fn` below reads each with exactly one `getattr` call
    # regardless of how many conditions of that kind this table has.
    between_lo_key = f"{name}__between_lo"
    between_hi_key = f"{name}__between_hi"
    between_has_lo_key = f"{name}__between_has_lo"
    between_has_hi_key = f"{name}__between_has_hi"
    eq_val_key = f"{name}__eq_val"
    eq_has_key = f"{name}__eq_has"
    in_off_key = f"{name}__in_off"
    in_vals_key = f"{name}__in_vals"
    in_vals_start_key = f"{name}__in_vals_start"

    emitter.shared[between_lo_key] = _stack2d(emitter.shared, between_ops, "lo_key", n_rows)
    emitter.shared[between_hi_key] = _stack2d(emitter.shared, between_ops, "hi_key", n_rows)
    emitter.shared[between_has_lo_key] = _stack2d(emitter.shared, between_ops, "has_lo_key", n_rows)
    emitter.shared[between_has_hi_key] = _stack2d(emitter.shared, between_ops, "has_hi_key", n_rows)
    emitter.shared[eq_val_key] = _stack2d(emitter.shared, eq_ops, "val_key", n_rows)
    emitter.shared[eq_has_key] = _stack2d(emitter.shared, eq_ops, "has_key", n_rows)
    in_off_2d, in_vals_flat, in_vals_start = _stack_in(emitter.shared, in_ops, n_rows)
    emitter.shared[in_off_key] = in_off_2d
    emitter.shared[in_vals_key] = in_vals_flat
    emitter.shared[in_vals_start_key] = in_vals_start

    # The per-condition arrays `cond.arrays` deposited into `shared` above
    # (`{prefix}_lo`, `{prefix}_hi`, ...) are now redundant: every one of
    # them was read exactly once, by `_stack2d`/`_stack_in`, to build the
    # 9 consolidated arrays just above. Drop them.
    #
    # This is not just tidying: EVERY key in `shared` becomes a FIELD of
    # the runtime namedtuple `runtime.invoke._bundle_class` builds from it
    # (`fields = tuple(raw_shared.keys())`) — regardless of whether any
    # step's `row_fn` ever reads that particular field. Measured: with
    # 1 field per condition still present (the pre-2D-array behaviour,
    # unchanged by that migration step alone), a 16-BETWEEN-condition
    # table's first `row_fn`/`kernel` compile took ~26s, almost all of it
    # `_shared_get`'s 9 call sites each re-typing a 76-field namedtuple
    # (~1.5s each); at 400 conditions (1600+ fields) it did not finish in
    # any reasonable time and drove memory into the gigabytes — the SAME
    # class of super-linear numba cost `_compose0`..`_compose8` existed to
    # avoid by capping condition count, just moved from "one closure body
    # per getter count" to "one struct field per condition". Dropping the
    # redundant fields here keeps the namedtuple at a FIXED ~12 fields
    # regardless of table width, so table width stops driving this cost at
    # all — see this module's report.
    for ops, key_attrs in (
        (between_ops, ("lo_key", "hi_key", "has_lo_key", "has_hi_key")),
        (eq_ops, ("val_key", "has_key")),
        (in_ops, ("off_key", "vals_key")),
    ):
        for op in ops:
            for key_attr in key_attrs:
                emitter.shared.pop(getattr(op, key_attr), None)

    # -- matcher steps ------------------------------------------------------
    matcher_names: list[str] = []
    matcher_steps: list[Step] = []
    for variable, literals in emitter.string_literals.items():
        fn_name = emitter.matcher_name(variable)
        matcher_names.append(fn_name)
        ident = safe_ident(variable)
        param_names = [f"{ident}_lit_{i}" for i in range(len(literals))]
        fn = _build_matcher_fn()
        inputs = (Input(name=variable, annotation=str, null_policy=NullPolicy.REQUIRED),)
        params = tuple(
            ParamDecl(name=pn, annotation=str, default=lit, field_info=Field(lit))
            for pn, lit in zip(param_names, literals)
        )
        matcher_steps.append(
            Step(
                name=fn_name, fn=fn, inputs=inputs, params=params,
                doc=f"Which distinct literal `{variable}` equals, or -1.",
                packed=True, output_annotation=int,
            )
        )

    # -- row step ------------------------------------------------------
    row_fn_inputs: list[Input] = []
    for variable in emitter.variables:
        if variable in emitter.string_literals:
            row_fn_inputs.append(
                # `annotation=float`, not `int`: the matcher's own output
                # really is an int, but `vars_` (`scan_table`'s own
                # homogeneous array) needs every entry AS a float64 —
                # `decider2.compile.driver._packed_input_arrays` reads this
                # annotation to cast when gathering a column.
                Input(name=emitter.matcher_name(variable), annotation=float, null_policy=NullPolicy.REQUIRED)
            )
        else:
            row_fn_inputs.append(Input(name=variable, annotation=float, null_policy=NullPolicy.REQUIRED))

    @njit(cache=True)
    def row_fn(args, params, shared):
        vars_ = args
        n = int(getattr(shared, n_rows_key)[0])
        return scan_table(
            vars_, n, group_start_arr, group_end_arr, op_kind_arr, op_var_idx_arr,
            op_local_arr, op_lo_op_arr, op_hi_op_arr,
            getattr(shared, between_lo_key), getattr(shared, between_hi_key),
            getattr(shared, between_has_lo_key), getattr(shared, between_has_hi_key),
            getattr(shared, eq_val_key), getattr(shared, eq_has_key),
            getattr(shared, in_off_key), getattr(shared, in_vals_key),
            getattr(shared, in_vals_start_key),
        )

    row_name = f"{name}_row"
    row_step = Step(
        name=row_name, fn=row_fn, inputs=tuple(row_fn_inputs), params=(),
        doc=f"Index of the first row of {table.name!r} that matches, or -1.",
        packed=True, output_annotation=int, reads_shared=True,
        # Exactly the keys `row_fn` reads, in the order it reads them — the
        # runtime builds this step's bundle from these alone (`types.Step.
        # shared_fields`), so its numba type, compile cost and cache entry
        # are this table's own, whatever else `shared=` carries.
        shared_fields=(
            n_rows_key,
            between_lo_key, between_hi_key, between_has_lo_key, between_has_hi_key,
            eq_val_key, eq_has_key,
            in_off_key, in_vals_key, in_vals_start_key,
        ),
    )

    # -- output steps ------------------------------------------------------
    output_steps: list[Step] = []
    for column in table.outputs:
        values = table.parameters.column(column)
        if any(isinstance(v, str) for v in values):
            continue
        declared = table.parameters.dtype_map.get(column)
        is_bool = all(isinstance(v, bool) or v is None for v in values)
        is_int = (not is_bool) and (
            declared in ("Int64", "Int32") or all(isinstance(v, int) or v is None for v in values)
        )
        py_type = "bool" if is_bool else ("int" if is_int else "float")
        output_annotation = {"float": float, "int": int, "bool": bool}[py_type]
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

        out_fn = _build_output_fn(values_key, default_key)
        output_steps.append(
            Step(
                name=fn_name, fn=out_fn,
                inputs=(Input(name=row_name, annotation=int, null_policy=NullPolicy.REQUIRED),),
                params=(),
                doc=f"`{column}` for the matched row, read from the table.",
                packed=True, output_annotation=output_annotation, reads_shared=True,
                shared_fields=(values_key, default_key),
            )
        )

    return EncodedTable(
        row_step=row_step,
        matcher_steps=tuple(matcher_steps),
        output_steps=tuple(output_steps),
        row_column=row_name,
        variables=tuple(emitter.variables),
        string_variables=tuple(emitter.string_literals),
        shared=emitter.shared,
        n_rows=n_rows,
        n_conditions=sum(len(g) for g in emitted_groups),
    )
