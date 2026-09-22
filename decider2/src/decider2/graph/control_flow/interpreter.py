"""The generic Branch/Loop interpreter (doc 03 §8.2/§8.3): ONE hand-written
`@njit` walker over a DATA program, plus the closure builders that turn a
construct's condition/arm/body steps into something that walker can call
— with no Python source generated anywhere along the way.

**What changed from the previous build, and why.** `branch.py`/`loop.py`
used to write a real, content-addressed `.py` file per construct
(the since-deleted `compile.cache.get_or_build`) holding (a) a by-name `import` of
every condition/arm/body function, (b) a hand-rendered `call_step`/
`call_cond` `if`/`elif` switch casting each of a FIXED `CALL_ARITY` float
arguments down to the callee's declared types, and (c) one wrapper `def`
per output with one named parameter per leaf/param — the last dynamically
generated Python source left in `decider2`. All three are now built
directly, as real closures:

- **(a)/(b) — calling a step by index, with its own declared types.** Each
  `STEP`/`COND` node of a construct's programs gets ONE `adapter(regs) ->
  float64` closure (`make_call_adapter`/`make_packed_adapter`; a callee
  used at the same registers twice shares one). The construct's adapters
  live in one heterogeneous tuple, walked with `numba.literal_unroll` and
  selected by the node's runtime `step_idx` (`make_walker`'s `call`): each
  unrolled version is a DIRECT, statically-typed call to one specific
  compiled callee — what the rendered `elif step_idx == N: return
  _armN_njit(...)` line used to be. The callee's argument list — however
  long, whatever mix of `float`/`int`/`bool`/`str` it declares — is built
  by `literal_unroll`-ing over a tuple of LITERAL slot indices and, at
  each, `_set_static`-inserting `regs[<that slot's register, itself a
  build-time literal>]` CAST to the slot's declared type into a template
  tuple captured at build time (`(0.0, 0, False)` for `(float, int, bool)`),
  then star-called `fn(*args)`. `_set_static` is a ~15-line `numba.
  extending.intrinsic` — an LLVM `insert_value` at a compile-time-constant
  index, which is legal on a heterogeneous struct precisely because the
  index is a literal, plus `context.cast` — the one piece of numba
  extension code this package needs. There is therefore no `CALL_ARITY`
  any more, and no per-arity code: a 30-argument callee and a 1-argument
  callee are the same two closures.

  **Why the register indices are literals captured per NODE, not an
  `argidx[pc, k]` array read per call.** Measured on a realistic body
  (`instalment_fits`, a `pow` and a divide): an adapter reading its
  registers at literal indices and taking ONLY `regs` costs exactly a
  direct call (26.4 vs 26.5 ns); the same adapter handed `(regs, argidx,
  pc)` costs 36.5 ns, and a per-call `argidx[pc]` row view ~10 ns more
  again — array structs crossing a non-inlined call boundary are the
  cost, the template is free (36.5 vs 37.0 for six hand-written reads).
  A two-way closure CHAIN for the dispatch, tried as a way to avoid the
  experimental first-class-function typing a tuple of Dispatchers gets,
  measured 112 ns against the tuple's 59 (and 73 vs 38 with `regs`-only
  adapters), so the tuple stays.

- **(c) — the wrapper.** A `types.Step` with `packed=True`: `fn(args,
  params)` where `args[i]` is `Step.inputs[i]` and `params[i]` is
  `Step.params[i]` (`types.Step.packed`, the convention `decider2.trees.
  encode`/`decider2.tables.encode` already use and `decider2.compile.
  driver` already calls). `_seed_regs` scatters them into registers
  `0..N-1` and `N..N+P-1` with one runtime loop (a homogeneous array
  indexes at a runtime position) and one `literal_unroll` (params may mix
  `float` and int32 dictionary codes) — never one closure per input count.

**Caching, stated once.** Every closure here captures another njit
Dispatcher (an adapter captures its callee; `call` captures the adapters;
`walk` captures `call`; the step `fn` captures `walk`) and is therefore
deliberately NOT `cache=True` — a closure capturing a Dispatcher was
measured (`decider2.compile.driver`'s row-gather closures, `decider2.
trees.encode._build_path_fn`, and this module's own predecessor's
`run_program` comment) to grow numba's on-disk index unboundedly without
ever hitting it, and its predecessor's own attempt to cache a per-construct
specialisation produced a real `ReferenceError` from numba's cache loader.
The USER's own condition/arm/body functions are wrapped `njit(cache=True)`
exactly as `decider2.compile.driver._try_njit` wraps every other step, and
cache normally; `_seed_regs`/`_seed_regs_no_params` are plain module-level
`@njit(cache=True)` functions. What is recompiled per process is the small
per-construct walker + adapters (measured: see this migration's report) —
the same trade `build_packed_kernel`'s per-step row loop already makes.
Nothing here reads a raw address from a global, which is the thing that
actually segfaulted on a warm second process (this module's predecessor
docstring, EXPERIMENTS.md §W and its correction).

**Registers.** Every value a program touches — a leaf input, a `param()`
value, a carry, `loop_idx`, an arm's result — lives in one per-call float64
`regs` array at an index decided once when the program is built
(`_engine.RegisterMap`). Retuning a param only changes what `regs` holds,
never what any specialisation compiled against (doc 08 §2).

**`CMP_LIT`.** The one opcode with no call: `compare(cmp, regs[arg], lit)`
— `decider2.trees.interpreter.compare`'s six comparisons, reused — for
Branch's "which arm produced this value" test and Loop's `loop_idx <
max_iterations` bound.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from numba import literal_unroll, njit
from numba.core import types as nbtypes
from numba.core.dispatcher import Dispatcher
from numba.extending import intrinsic

from decider2.trees.interpreter import compare

__all__ = [
    "STEP", "COND", "CMP_LIT", "SET_ZERO", "SET_LIT", "INCR", "JUMP", "LEAF",
    "NO_ARG",
    "Program", "ProgramBuilder",
    "sentinel_for", "make_call_adapter", "make_packed_adapter",
    "make_walker", "make_step_fn",
]

# --- opcodes -----------------------------------------------------------
STEP = 0
COND = 1
CMP_LIT = 2
SET_ZERO = 3
SET_LIT = 4
INCR = 5
JUMP = 6
LEAF = 7

# --- `Program.code` columns ----------------------------------------------
_OP = 0
_STEP_IDX = 1
_DEST = 2
_NEXT = 3
_ALT = 4
_CMP = 5
_ARG = 6
_N_COLS = 7

NO_ARG = -1


@dataclass(frozen=True)
class Program:
    """One Branch/Loop output's control-flow program, entirely as flat
    arrays. `code[pc]` is `(op, step_idx, dest, next, alt, cmp, arg)`:
    `step_idx` the construct's node-adapter index a `STEP`/`COND` calls
    (`make_walker`), `arg` the one register a `CMP_LIT` reads, `lit[pc]` a
    `SET_LIT`/`CMP_LIT` literal. `n_regs` sizes the per-call register
    scratch."""

    code: np.ndarray    # int32 [n_nodes, 7]
    lit: np.ndarray     # float64 [n_nodes]
    n_regs: int
    start_pc: int


class ProgramBuilder:
    """Appends one node at a time, returns its index, lets `_engine.py`'s
    encoders wire jump targets by index (`set_next`) — the role a
    generated `if`/`while`'s source text used to play, as data."""

    def __init__(self) -> None:
        self._rows: list[list] = []
        self._lit: list[float] = []

    def add(
        self, op: int, *, step_idx: int = NO_ARG, arg: int = NO_ARG, lit: float = 0.0,
        dest: int = NO_ARG, next_: int = NO_ARG, alt: int = NO_ARG, cmp: int = NO_ARG,
    ) -> int:
        self._rows.append([op, step_idx, dest, next_, alt, cmp, arg])
        self._lit.append(float(lit))
        return len(self._rows) - 1

    def set_next(self, pc: int, next_: "int | None" = None, alt: "int | None" = None) -> None:
        if next_ is not None:
            self._rows[pc][_NEXT] = next_
        if alt is not None:
            self._rows[pc][_ALT] = alt

    def build(self, *, start_pc: int, n_regs: int) -> Program:
        return Program(
            code=np.array(self._rows, dtype=np.int32).reshape(len(self._rows), _N_COLS),
            lit=np.array(self._lit, dtype=np.float64),
            n_regs=max(int(n_regs), 1),
            start_pc=int(start_pc),
        )


# ---------------------------------------------------------------------------
# Typed-argument construction. `_set_static(tup, k, val)` returns `tup` with
# element `k` (a compile-time LITERAL, which is what makes an LLVM
# `insert_value` on a heterogeneous struct legal) replaced by `val` cast to
# that element's own type. Driven by `literal_unroll` over a tuple of
# literal slot indices, so an N-argument callee needs no N-specific code.
# ---------------------------------------------------------------------------


@intrinsic
def _set_static(typingctx, tup, idx, val):
    if not isinstance(tup, nbtypes.BaseTuple) or not isinstance(idx, nbtypes.IntegerLiteral):
        return None
    k = idx.literal_value
    # The result is typed with NON-literal elements, whatever `tup` came in
    # as. numba types a captured homogeneous tuple of Python `int`/`bool`
    # constants as `UniTuple(Literal[int](0) x n)`, and a cast from a
    # Literal type back to its base type re-materialises the CONSTANT —
    # silently discarding the runtime value just inserted (found while
    # building this: a bool condition read `False` on every row). The
    # Literal and its base share one LLVM representation, so the insert
    # itself is unaffected; only the declared result type changes.
    out_ty = nbtypes.BaseTuple.from_types([nbtypes.unliteral(t) for t in tup.types])
    elem_ty = out_ty.types[k]
    sig = out_ty(tup, idx, val)

    def codegen(context, builder, signature, args):
        t, _i, v = args
        v_cast = context.cast(builder, v, signature.args[2], elem_ty)
        return builder.insert_value(t, v_cast, k)

    return sig, codegen


# One sentinel per declared annotation: its VALUE is irrelevant (every slot
# is overwritten before the call), its TYPE is the type the callee's
# argument is cast to — the same table `decider2.compile.driver.
# _NUMBA_BY_ANNOTATION` compiles a step against (`str` is an int32
# dictionary code at a kernel boundary, doc 05 §1.5). numpy scalars, never
# Python `0`/`False`: numba gives a captured tuple of Python int/bool
# constants LITERAL element types (`_set_static` above); a numpy scalar is
# only ever its plain type.
_SENTINELS: dict[Any, Any] = {
    float: np.float64(0.0),
    int: np.int64(0),
    bool: np.bool_(False),
    str: np.int32(0),
}


def sentinel_for(annotation: Any) -> Any:
    """Unannotated (`typing.Any`) falls back to float64, the same default
    `decider2.compile.driver._numba_type` gives an undeclared value."""
    return _SENTINELS.get(annotation, np.float64(0.0))


def _as_dispatcher(fn: Callable) -> Callable:
    """The user's own function, `njit(cache=True)`-wrapped exactly as
    `decider2.compile.driver._try_njit` wraps every other step — or left
    alone if it already is a Dispatcher (a nested construct's packed `fn`;
    re-`njit`-ing one is a hard `TypeError`)."""
    if isinstance(fn, Dispatcher):
        return fn
    return njit(cache=True)(fn)


def make_call_adapter(fn: Callable, arg_annotations: Sequence[Any], arg_regs: Sequence[int]) -> Callable:
    """`adapter(regs) -> float64` for a plain (hand-written) callee at one
    program node: `fn(*args)` where `args[k]` is `regs[arg_regs[k]]` cast
    to `arg_annotations[k]`'s type, in the callee's own declared parameter
    order, `arg_regs[k]` a build-time literal. The result is widened back
    to float64 (a bool `-> 1.0/0.0`, an int exactly), the register array's
    one type."""
    compiled = _as_dispatcher(fn)
    template = tuple(sentinel_for(a) for a in arg_annotations)
    regs_t = tuple(int(r) for r in arg_regs)
    slots = tuple(range(len(template)))
    if len(regs_t) != len(slots):
        raise ValueError("make_call_adapter: one register per declared argument")

    if not slots:
        # `literal_unroll(())` does not type-check (an empty tuple has no
        # element type) — the zero-argument callee is its own fixed shape,
        # not the first member of a per-arity family.
        @njit
        def adapter0(regs):
            return np.float64(compiled())
        return adapter0

    @njit
    def adapter(regs):
        args = template
        for k in literal_unroll(slots):
            args = _set_static(args, k, regs[regs_t[k]])
        return np.float64(compiled(*args))
    return adapter


def make_packed_adapter(
    fn: Callable, input_annotations: Sequence[Any], param_annotations: Sequence[Any],
    args_kind: str, arg_regs: Sequence[int],
) -> Callable:
    """`adapter(regs) -> float64` for a PACKED callee at one program node
    — a nested `Branch`/`Loop`'s own step, or any other `types.Step.
    packed` step used as a condition/arm/body. `arg_regs` holds the
    callee's inputs' registers first, then its params' (`_engine.
    call_arg_regs`).

    `args` is built as a homogeneous float64 TUPLE for the `"array"` kind
    (`decider2.compile.driver._packed_args_kind`, passed in as `args_kind`)
    rather than the float64 array the driver hands the same `fn` from a
    frame: `_seed_regs` indexes `args` at a runtime position, which a
    homogeneous tuple supports exactly as an array does, and a tuple lives
    in registers — no heap allocation per nested call (measured: the array
    form cost a nested Loop(Branch(Loop)) ~25% per row). The `"raw1"` kind
    is a 1-tuple preserving the one non-float input's own dtype, as the
    driver's `_gather1_raw` builds it; `params` is a tuple typed per
    declared annotation.
    """
    compiled = _as_dispatcher(fn)
    n_inputs = len(input_annotations)
    regs_t = tuple(int(r) for r in arg_regs)
    if len(regs_t) != n_inputs + len(param_annotations):
        raise ValueError("make_packed_adapter: one register per input and per param")
    a_regs = regs_t[:n_inputs]
    p_regs = regs_t[n_inputs:]
    p_template = tuple(sentinel_for(a) for a in param_annotations)
    p_slots = tuple(range(len(p_template)))
    if args_kind == "raw1":
        a_template = (sentinel_for(input_annotations[0]),)
    else:
        a_template = tuple(sentinel_for(float) for _ in range(n_inputs))
    a_slots = tuple(range(len(a_template)))

    if not a_slots:
        @njit
        def gather_args(regs):
            return np.empty(0, dtype=np.float64)  # `driver._gather0`'s own empty shape
    else:
        @njit
        def gather_args(regs):
            args = a_template
            for k in literal_unroll(a_slots):
                args = _set_static(args, k, regs[a_regs[k]])
            return args

    if not p_slots:
        @njit
        def adapter0(regs):
            return np.float64(compiled(gather_args(regs), ()))
        return adapter0

    @njit
    def adapter(regs):
        params = p_template
        for k in literal_unroll(p_slots):
            params = _set_static(params, k, regs[p_regs[k]])
        return np.float64(compiled(gather_args(regs), params))
    return adapter


# ---------------------------------------------------------------------------
# THE walker. One Python-level definition per CONSTRUCT (a closure over that
# construct's own node-adapter tuple), shared by every program in the
# construct (a Branch's `_path` and each of its `modifies` outputs; a
# Loop's carry).
# ---------------------------------------------------------------------------


def make_walker(adapters: Sequence[Callable]) -> Callable:
    """`walk(code, lit, start_pc, regs) -> float64` over `adapters` — one
    construct's node adapters, indexed by a program node's `step_idx`.
    `call` below is the by-index dispatch: `literal_unroll` versions its
    body once per adapter, so the selected version is a direct call to
    that one compiled callee — the same shape the rendered `if step_idx
    == 0: ... elif step_idx == 1: ...` switch compiled to."""
    adapters_t = tuple(adapters)
    if not adapters_t:
        raise ValueError("make_walker: a Branch/Loop always has at least one callee")

    @njit
    def call(k, regs):
        i = 0
        r = 0.0
        for a in literal_unroll(adapters_t):
            if i == k:
                r = a(regs)
            i += 1
        return r

    @njit
    def walk(code, lit, start_pc, regs):
        pc = start_pc
        while True:
            op = code[pc, _OP]
            if op == STEP:
                regs[code[pc, _DEST]] = call(code[pc, _STEP_IDX], regs)
                pc = code[pc, _NEXT]
            elif op == COND:
                if call(code[pc, _STEP_IDX], regs) != 0.0:
                    pc = code[pc, _NEXT]
                else:
                    pc = code[pc, _ALT]
            elif op == CMP_LIT:
                if compare(code[pc, _CMP], regs[code[pc, _ARG]], lit[pc]):
                    pc = code[pc, _NEXT]
                else:
                    pc = code[pc, _ALT]
            elif op == SET_ZERO:
                regs[code[pc, _DEST]] = 0.0
                pc = code[pc, _NEXT]
            elif op == SET_LIT:
                regs[code[pc, _DEST]] = lit[pc]
                pc = code[pc, _NEXT]
            elif op == INCR:
                regs[code[pc, _DEST]] += 1.0
                pc = code[pc, _NEXT]
            elif op == JUMP:
                pc = code[pc, _NEXT]
            else:  # LEAF
                return regs[code[pc, _DEST]]

    return walk


# ---------------------------------------------------------------------------
# The packed wrapper: `fn(args, params)` -> this program's one output.
# ---------------------------------------------------------------------------


@njit(cache=True)
def _seed_regs(args, params, n_regs):
    """Registers `0..len(args)-1` from `args` (a float64 array, the `raw1`
    1-tuple, or a nested call's homogeneous tuple — all index at a runtime
    position), then `len(args)..` from `params` (`literal_unroll`: a params
    tuple may mix float64 and an int32 dictionary code). Everything else
    starts at 0.0 — `loop_idx` (owned by `SET_ZERO`/`INCR`), a non-carry
    body output (written by its own `STEP` before anything reads it), a
    construct-internal slot."""
    regs = np.zeros(n_regs)
    n = len(args)
    for i in range(n):
        regs[i] = args[i]
    j = n
    for p in literal_unroll(params):
        regs[j] = p
        j += 1
    return regs


@njit(cache=True)
def _seed_regs_no_params(args, params, n_regs):
    """`_seed_regs` for a step with no `param()` fields — `literal_unroll`
    over an empty tuple does not type-check, so this is its own fixed
    shape (the second of exactly two, never a per-count family)."""
    regs = np.zeros(n_regs)
    n = len(args)
    for i in range(n):
        regs[i] = args[i]
    return regs


def make_step_fn(walk: Callable, program: Program, output_annotation: Any, n_params: int) -> Callable:
    """One packed `Step.fn` for `program`: seed, walk, and cast the float64
    result to the step's declared output type — three fixed bodies keyed on
    that type (the same shape `decider2.trees.encode._build_output_fn`
    uses), never one per program."""
    code, lit = program.code, program.lit
    start_pc, n_regs = program.start_pc, program.n_regs
    seed = _seed_regs if n_params else _seed_regs_no_params

    if output_annotation is int:
        @njit
        def fn_int(args, params):
            return int(walk(code, lit, start_pc, seed(args, params, n_regs)))
        return fn_int
    if output_annotation is bool:
        @njit
        def fn_bool(args, params):
            return walk(code, lit, start_pc, seed(args, params, n_regs)) != 0.0
        return fn_bool

    @njit
    def fn_float(args, params):
        return walk(code, lit, start_pc, seed(args, params, n_regs))
    return fn_float
