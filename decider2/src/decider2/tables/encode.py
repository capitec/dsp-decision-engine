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
it off the `shared` bundle **by name, at call time** — `_shared_get`,
below: `getattr(shared, key)` with `key` a closure-captured string
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
# Closure builders. Real, hand-written functions — the only "variable" is
# which of a small, fixed family gets picked, decided in Python at build
# time from the walk's own counts.
# ---------------------------------------------------------------------------


@njit(cache=True)
def _shared_get(shared, key):
    """One row-data array, read off `shared` by name at call time — never a
    captured constant, so a table rebuilt with new row data (same shape)
    answers differently through the SAME compiled closure (doc 08 §3.4's
    free interior), with `key` a closure-captured string constant (real
    Python `getattr`, not text)."""
    return getattr(shared, key)


# `_composeN`: real code, one body per getter-count 0..12 — comfortably
# above any realistic table's per-condition-kind count in this codebase's
# own examples and ported test suite. Each closes over N single-argument
# getters (`shared -> array` or `shared -> tuple-of-arrays`, built by
# `_make_group_getter`/`_shared_get` below) and returns their results as one
# tuple, in order — the exact shape `scan_table` needs for
# `between_bounds`/`eq_bounds`/`in_offsets`/`in_values`.
def _compose0(getters, dummy):
    @njit(cache=True)
    def gather(shared):
        return (dummy,)
    return gather


def _compose1(getters):
    g0, = getters

    @njit  # not cache=True -- see decider2.compile.driver._row_getter
    # (module docstring note): a factory-built closure capturing another
    # Dispatcher, called multiple times per process (once per condition),
    # was measured to grow numba's on-disk cache index unboundedly instead
    # of hitting it. Tiny function, cheap to recompile.
    def gather(shared):
        return (g0(shared),)
    return gather


def _compose2(getters):
    g0, g1 = getters

    @njit  # not cache=True -- see decider2.compile.driver._row_getter
    # (module docstring note): a factory-built closure capturing another
    # Dispatcher, called multiple times per process (once per condition),
    # was measured to grow numba's on-disk cache index unboundedly instead
    # of hitting it. Tiny function, cheap to recompile.
    def gather(shared):
        return (g0(shared), g1(shared))
    return gather


def _compose3(getters):
    g0, g1, g2 = getters

    @njit  # not cache=True -- see decider2.compile.driver._row_getter
    # (module docstring note): a factory-built closure capturing another
    # Dispatcher, called multiple times per process (once per condition),
    # was measured to grow numba's on-disk cache index unboundedly instead
    # of hitting it. Tiny function, cheap to recompile.
    def gather(shared):
        return (g0(shared), g1(shared), g2(shared))
    return gather


def _compose4(getters):
    g0, g1, g2, g3 = getters

    @njit  # not cache=True -- see decider2.compile.driver._row_getter
    # (module docstring note): a factory-built closure capturing another
    # Dispatcher, called multiple times per process (once per condition),
    # was measured to grow numba's on-disk cache index unboundedly instead
    # of hitting it. Tiny function, cheap to recompile.
    def gather(shared):
        return (g0(shared), g1(shared), g2(shared), g3(shared))
    return gather


def _compose5(getters):
    g0, g1, g2, g3, g4 = getters

    @njit  # not cache=True -- see decider2.compile.driver._row_getter
    # (module docstring note): a factory-built closure capturing another
    # Dispatcher, called multiple times per process (once per condition),
    # was measured to grow numba's on-disk cache index unboundedly instead
    # of hitting it. Tiny function, cheap to recompile.
    def gather(shared):
        return (g0(shared), g1(shared), g2(shared), g3(shared), g4(shared))
    return gather


def _compose6(getters):
    g0, g1, g2, g3, g4, g5 = getters

    @njit  # not cache=True -- see decider2.compile.driver._row_getter
    # (module docstring note): a factory-built closure capturing another
    # Dispatcher, called multiple times per process (once per condition),
    # was measured to grow numba's on-disk cache index unboundedly instead
    # of hitting it. Tiny function, cheap to recompile.
    def gather(shared):
        return (g0(shared), g1(shared), g2(shared), g3(shared), g4(shared), g5(shared))
    return gather


def _compose7(getters):
    g0, g1, g2, g3, g4, g5, g6 = getters

    @njit  # not cache=True -- see decider2.compile.driver._row_getter
    # (module docstring note): a factory-built closure capturing another
    # Dispatcher, called multiple times per process (once per condition),
    # was measured to grow numba's on-disk cache index unboundedly instead
    # of hitting it. Tiny function, cheap to recompile.
    def gather(shared):
        return (g0(shared), g1(shared), g2(shared), g3(shared), g4(shared), g5(shared), g6(shared))
    return gather


def _compose8(getters):
    g0, g1, g2, g3, g4, g5, g6, g7 = getters

    @njit  # not cache=True -- see decider2.compile.driver._row_getter
    # (module docstring note): a factory-built closure capturing another
    # Dispatcher, called multiple times per process (once per condition),
    # was measured to grow numba's on-disk cache index unboundedly instead
    # of hitting it. Tiny function, cheap to recompile.
    def gather(shared):
        return (
            g0(shared), g1(shared), g2(shared), g3(shared),
            g4(shared), g5(shared), g6(shared), g7(shared),
        )
    return gather


_COMPOSE_BUILDERS = (
    _compose0, _compose1, _compose2, _compose3, _compose4,
    _compose5, _compose6, _compose7, _compose8,
)
_MAX_CONDITIONS_PER_KIND = len(_COMPOSE_BUILDERS) - 1


def _make_scalar_getter(key: str):
    @njit  # not cache=True -- see decider2.compile.driver._row_getter
    # (module docstring note): a factory-built closure capturing another
    # Dispatcher, called multiple times per process (once per condition),
    # was measured to grow numba's on-disk cache index unboundedly instead
    # of hitting it. Tiny function, cheap to recompile.
    def getter(shared):
        return _shared_get(shared, key)
    return getter


def _make_pair_getter(key0: str, key1: str):
    @njit  # not cache=True -- see decider2.compile.driver._row_getter
    # (module docstring note): a factory-built closure capturing another
    # Dispatcher, called multiple times per process (once per condition),
    # was measured to grow numba's on-disk cache index unboundedly instead
    # of hitting it. Tiny function, cheap to recompile.
    def getter(shared):
        return (_shared_get(shared, key0), _shared_get(shared, key1))
    return getter


def _make_quad_getter(key0: str, key1: str, key2: str, key3: str):
    @njit  # not cache=True -- see decider2.compile.driver._row_getter
    # (module docstring note): a factory-built closure capturing another
    # Dispatcher, called multiple times per process (once per condition),
    # was measured to grow numba's on-disk cache index unboundedly instead
    # of hitting it. Tiny function, cheap to recompile.
    def getter(shared):
        return (
            _shared_get(shared, key0), _shared_get(shared, key1),
            _shared_get(shared, key2), _shared_get(shared, key3),
        )
    return getter


def _compose(getters: list, dummy):
    n = len(getters)
    if n > _MAX_CONDITIONS_PER_KIND:
        raise ValueError(
            f"table has {n} conditions of one kind, over this build's "
            f"{_MAX_CONDITIONS_PER_KIND}-per-kind limit "
            "(decider2.tables.encode's fixed closure-arity family)."
        )
    if n == 0:
        return _compose0(getters, dummy)
    return _COMPOSE_BUILDERS[n](tuple(getters))


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
            return _shared_get(shared, default_key)[0]
        return _shared_get(shared, shared_key)[row]

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

    between_getters = [
        _make_quad_getter(op.lo_key, op.hi_key, op.has_lo_key, op.has_hi_key) for op in between_ops
    ]
    eq_getters = [_make_pair_getter(op.val_key, op.has_key) for op in eq_ops]
    in_off_getters = [_make_scalar_getter(op.off_key) for op in in_ops]
    in_vals_getters = [_make_scalar_getter(op.vals_key) for op in in_ops]

    dummy_f = np.zeros(1)
    dummy_i = np.zeros(1, dtype=np.int64)
    between_gather = _compose(between_getters, (dummy_f, dummy_f, dummy_f, dummy_f))
    eq_gather = _compose(eq_getters, (dummy_f, dummy_f))
    in_off_gather = _compose(in_off_getters, dummy_i)
    in_vals_gather = _compose(in_vals_getters, dummy_f)

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
                # homogeneous tuple) needs every entry AS a float64 —
                # `decider2.compile.driver._row_getter` reads this
                # annotation to cast when gathering a row.
                Input(name=emitter.matcher_name(variable), annotation=float, null_policy=NullPolicy.REQUIRED)
            )
        else:
            row_fn_inputs.append(Input(name=variable, annotation=float, null_policy=NullPolicy.REQUIRED))

    @njit(cache=True)
    def row_fn(args, params, shared):
        vars_ = args
        n = int(_shared_get(shared, n_rows_key)[0])
        between_bounds = between_gather(shared)
        eq_bounds = eq_gather(shared)
        in_offsets = in_off_gather(shared)
        in_values = in_vals_gather(shared)
        return scan_table(
            vars_, n, group_start_arr, group_end_arr, op_kind_arr, op_var_idx_arr,
            op_local_arr, op_lo_op_arr, op_hi_op_arr, between_bounds, eq_bounds, in_offsets, in_values,
        )

    row_name = f"{name}_row"
    row_step = Step(
        name=row_name, fn=row_fn, inputs=tuple(row_fn_inputs), params=(),
        doc=f"Index of the first row of {table.name!r} that matches, or -1.",
        packed=True, output_annotation=int, reads_shared=True,
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
