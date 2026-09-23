"""The fused per-row kernel, built without generating Python source — doc 05
§4.3 (driver shape), §4.2 (determinism), §7 (fusion).

**What replaced `compile/codegen.py`.** That module rendered, per fuse
group, a real `.py` file containing the row loop and the argument plumbing
doc 05 §4.3 shows — `v_x = _compiled_x(arr_a[i], arr_b[i], p_cap)` — and
imported it by content hash. This module builds the *same* kernel directly:

- **One fixed kernel signature for every group**, `kernel(n, cols, valids,
  params_all, outs)`, where every count that used to be a distinct
  positional argument (one per input column, per OPTIONAL validity array,
  per param scalar, per bare `params`/`shared` bundle, per output array)
  is now the length of a *tuple*. A tuple's length and element types are
  part of its numba type, so a group with 3 inputs and one with 300 are two
  ordinary specialisations of one function — there is no width past which
  anything here raises, and nothing here is a family keyed on a count.
- **Params stay kernel arguments, never constants** (doc 05 §4.2, doc 08
  §2). `params_all` is a heterogeneous tuple typed element-by-element from
  the values a caller passed — `48.0` types as `float64`, a bare bundle as
  its NamedTuple, an `Optional` param left at `None` as `none` — which is
  exactly the type information the old kernel's separate positional params
  carried. A value-only retune is therefore the same tuple type and reuses
  the compiled kernel (`Driver.signatures` unchanged); changing a field's
  declared *type* is a new tuple type and recompiles, doc 05 §9 criterion
  5's own negative control.
- **The row body is one `@numba.extending.intrinsic` per group.** Its
  typing sees the real numba types of the tuples above; its lowering walks
  the group's steps in order and, for each, loads that step's arguments
  (a column element, an `Optional` built from a column and its validity
  array, an earlier step's still-in-register result, or a `params_all`
  element), resolves and calls the step's *own* njit dispatcher through
  the same `get_call_type`/`get_function`/`cast` sequence numba's lowering
  uses for a direct call in jitted source, and keeps the result as an SSA
  value for the next step. No buffers, no per-step function pointers, no
  loop over steps at run time: the emitted IR is what the generated source
  compiled to, arrived at through numba's extension API instead of text.
  Measured (this module's report): 0.96x the hand-written kernel's ns/row
  on 24 realistic step bodies, bit-identical output.

**A value crosses between fused steps as its declared dtype.** The result
of each step is cast once, at production, to `numpy_dtype(<its return
annotation>)` — the dtype `stepped`/`interpreted` mode would have
materialised it into before the next step read it. The generated kernel
passed the *native* return type instead, so an unannotated step that
happened to return an `int64` fed the next step an `int64` under `fused`
and a `float64` under `stepped`. Now the three rungs of doc 02 §3.1's
ladder agree by construction on what a value's type is between steps,
not only on its value.

**Not `cache=True`, deliberately.** The kernel closes over the group's
step dispatchers and its intrinsic; the measured facts behind
`compile.driver.build_packed_kernel` apply here unchanged (a closure over
a dispatcher must not be disk-cached: it never hits and grows the index
without bound). Each *step* is still `njit(cache=True)` in its own real
source file and loads from disk in a fresh process; only this glue
compiles per process, once per group, and `Pipeline.precompile()`/
`.warm()` exist to take that off the request path (doc 05 §8).

**`parallel` is authored, never inferred, and has one real limit.** A
`parallel=True` group compiles its loop as `prange`; numba's parfor pass
then expands exactly ONE level of every tuple argument into separate
gufunc parameters and rejects a tuple element that is itself a tuple —
verified against plain `prange` source too, not only this kernel — so a
parallel group cannot carry a bare `params`/`shared` bundle. That case is
refused here at build time with a message, not left to a numba traceback.

Everything here is a pure function of the plan and the step dispatchers:
no addresses, no timestamps, no set/dict iteration-order dependence (doc 05
§4.2) — iteration is always over tuples whose order the plan fixed.
"""
from __future__ import annotations

import inspect
import operator
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import numpy as np
from numba import from_dtype, njit, prange, typeof
from numba.core import types
from numba.core.typing import signature as nb_signature
from numba.extending import intrinsic

from decider2.types import Input, NullPolicy, Step

__all__ = ["ArgRole", "KernelPlan", "kernel_signature", "build_fused_kernel"]


@dataclass(frozen=True)
class ArgRole:
    """One value the fused kernel is handed, and what it means.

    `kernel_signature` lists these in the one true order; `decider2.compile.
    driver._build_call_args` walks the same list to build the actual call,
    so the two can never silently disagree on what goes where. `array`/
    `valid`/`output` roles become elements of the kernel's `cols`/`valids`/
    `outs` tuples, in order; `param_scalar`/`params_bundle`/`shared` roles
    become elements of `params_all`, in order.
    """

    kind: Literal["array", "valid", "param_scalar", "params_bundle", "shared", "output"]
    input_name: str | None = None
    step_name: str | None = None
    param_name: str | None = None
    output_name: str | None = None
    owner: str | None = None          # the step's owning module instance
    # (doc 03 §4.1/§10: params are namespaced by MODULE instance, not step
    # name — a step's OUTPUT name is not unique across modules, doc 03 §3.2's
    # waterfall idiom. `decider2.compile.driver` keys its params lookup by
    # `(owner, step_name, param_name)` to survive that collision.)


@dataclass(frozen=True)
class KernelPlan:
    """Everything `build_fused_kernel` needs, already decided.

    Building one of these is `decider2.compile.driver`'s job — it knows
    which steps survived njit, which values cross a segment boundary and
    which stay in registers. This module only turns an already-decided plan
    into a kernel; it makes no decisions of its own.
    """

    group_name: str
    steps: tuple[Step, ...]                 # already in execution order
    external_inputs: tuple[Input, ...]      # inputs this kernel reads from outside
    required_outputs: tuple[str, ...]       # step names this kernel must write out
    owners: tuple[str, ...] = ()            # module instance name per `steps` entry,
    # parallel to `steps` (doc 03 §4.1/§10) — empty means "not supplied",
    # which `kernel_signature` treats as "each step is its own owner" so a
    # caller bypassing the graph layer (a scratch test, `decider2.compile`
    # used directly) keeps working exactly as before this field existed.
    reads_shared: bool = False
    parallel: bool = False                  # authored via parallel(...), never inferred
    fastmath: bool = False                  # authored per kernel, doc 05 §5.2


def kernel_signature(plan: KernelPlan) -> list[ArgRole]:
    """The kernel's values, in the one true order.

    Row arrays first (one per external input, first-seen order), then a
    validity array for every OPTIONAL one (doc 05 §2 tier 3), then each
    step's bare `params` bundle and own param()-declared scalars, then
    `shared` at most once, then one output array per required output.
    """
    roles: list[ArgRole] = []
    for inp in plan.external_inputs:
        roles.append(ArgRole("array", input_name=inp.name))
    for inp in plan.external_inputs:
        if inp.null_policy is NullPolicy.OPTIONAL:
            roles.append(ArgRole("valid", input_name=inp.name))
    # `plan.owners[i]` is the module instance that owns `plan.steps[i]`; a
    # plan built without that info (owners left at its default `()`) falls
    # back to "each step owns itself", matching the pre-owners behaviour
    # exactly for every caller that never supplied one.
    owners = plan.owners if len(plan.owners) == len(plan.steps) else tuple(s.name for s in plan.steps)
    for step, owner in zip(plan.steps, owners):
        if step.reads_params:
            roles.append(ArgRole("params_bundle", step_name=step.name, owner=owner))
        for decl in step.params:
            roles.append(
                ArgRole("param_scalar", step_name=step.name, param_name=decl.name, owner=owner)
            )
    if plan.reads_shared:
        roles.append(ArgRole("shared"))
    for name in plan.required_outputs:
        roles.append(ArgRole("output", output_name=name))
    return roles


# ---------------------------------------------------------------------------
# Argument sources. One per parameter of one step's own Python signature,
# decided once at build time from the plan — never per row.
# ---------------------------------------------------------------------------

_COL = 0      # (kind, j):     external input column j, element at row i
_OPT = 1      # (kind, j, vj): column j if valids[vj][i] else None — an Optional
_RESULT = 2   # (kind, s):     the result of step s of this same group
_PARAM = 3    # (kind, k):     params_all[k]

_ArgSource = tuple[int, ...]


def _param_indices(plan: KernelPlan) -> tuple[dict[tuple[int, str], int], int | None]:
    """`(step_index, "params")`/`(step_index, <param name>)` -> position in
    `params_all`, plus `shared`'s position (or None). Positions are the
    ranks of the `param_scalar`/`params_bundle`/`shared` roles among
    themselves, in `kernel_signature` order — the same order `driver.
    _build_call_args` packs the tuple in."""
    by_key: dict[tuple[int, str], int] = {}
    k = 0
    for si, step in enumerate(plan.steps):      # the same loop `kernel_signature` runs
        if step.reads_params:
            by_key[(si, "params")] = k
            k += 1
        for decl in step.params:
            by_key[(si, decl.name)] = k
            k += 1
    shared_at = k if plan.reads_shared else None
    return by_key, shared_at


def _arg_sources(plan: KernelPlan) -> tuple[tuple[_ArgSource, ...], ...]:
    """For every step, for every parameter of its Python signature, where
    that argument comes from. Mirrors `driver._row_kwargs`' resolution
    order exactly — `params`, then `shared`, then a declared param, then
    an input by name — so `fused` mode binds arguments the way `stepped`/
    `interpreted` do."""
    col_at = {inp.name: j for j, inp in enumerate(plan.external_inputs)}
    valid_at: dict[str, int] = {}
    for inp in plan.external_inputs:
        if inp.null_policy is NullPolicy.OPTIONAL:
            valid_at[inp.name] = len(valid_at)
    param_at, shared_at = _param_indices(plan)

    per_step: list[tuple[_ArgSource, ...]] = []
    result_at: dict[str, int] = {}
    for si, step in enumerate(plan.steps):
        declared = {d.name for d in step.params}
        sources: list[_ArgSource] = []
        for pname in inspect.signature(step.fn).parameters:
            if pname == "params":
                sources.append((_PARAM, param_at[(si, "params")]))
            elif pname == "shared":
                if shared_at is None:
                    raise ValueError(
                        f"step {step.name!r} reads `shared` but kernel group "
                        f"{plan.group_name!r} was planned without it"
                    )
                sources.append((_PARAM, shared_at))
            elif pname in declared:
                sources.append((_PARAM, param_at[(si, pname)]))
            elif pname in result_at:
                sources.append((_RESULT, result_at[pname]))
            elif pname in valid_at:
                sources.append((_OPT, col_at[pname], valid_at[pname]))
            elif pname in col_at:
                sources.append((_COL, col_at[pname]))
            else:
                raise ValueError(
                    f"step {step.name!r} reads {pname!r}, which kernel group "
                    f"{plan.group_name!r} neither receives as an input nor "
                    "produces earlier in the group"
                )
        per_step.append(tuple(sources))
        result_at[step.name] = si
    return tuple(per_step)


def _make_row_body(
    fns: tuple[Callable, ...],
    sources: tuple[tuple[_ArgSource, ...], ...],
    result_types: tuple[Any, ...],
    out_steps: tuple[int, ...],
):
    """The per-row intrinsic: `row_body(cols, valids, params_all, outs, i)`.

    Everything varying by group is closed over here, at build time, as
    plain Python data; the intrinsic's lowering walks it once per kernel
    specialisation and emits straight-line IR for the row. See the module
    docstring for the call-lowering sequence and why it matches numba's
    own.
    """

    @intrinsic
    def row_body(typingctx, cols, valids, params_all, outs, i):
        sig = types.none(cols, valids, params_all, outs, i)

        def codegen(context, builder, signature, args):
            cols_v, valids_v, params_v, outs_v, i_v = args
            typingctx_ = context.typing_context

            def load(tuple_ty, tuple_v, j):
                arr_ty = tuple_ty.types[j]
                getitem = context.get_function(
                    operator.getitem, nb_signature(arr_ty.dtype, arr_ty, i)
                )
                return getitem(builder, (builder.extract_value(tuple_v, j), i_v)), arr_ty.dtype

            results: list[tuple[Any, Any]] = []
            for fn, step_sources, result_ty in zip(fns, sources, result_types):
                argvals: list[Any] = []
                argtys: list[Any] = []
                for src in step_sources:
                    kind = src[0]
                    if kind == _COL:
                        v, t = load(cols, cols_v, src[1])
                    elif kind == _OPT:
                        v, t = load(cols, cols_v, src[1])
                        ok, _ = load(valids, valids_v, src[2])
                        v = builder.select(
                            ok,
                            context.make_optional_value(builder, t, v),
                            context.make_optional_none(builder, t),
                        )
                        t = types.Optional(t)
                    elif kind == _RESULT:
                        v, t = results[src[1]]
                    else:
                        v, t = builder.extract_value(params_v, src[1]), params_all.types[src[1]]
                    argvals.append(v)
                    argtys.append(t)
                fnty = typeof(fn)
                callsig = fnty.get_call_type(typingctx_, tuple(argtys), {})
                impl = context.get_function(fnty, callsig)
                cast_args = [
                    context.cast(builder, v, t, want)
                    for v, t, want in zip(argvals, argtys, callsig.args)
                ]
                res = impl(builder, cast_args)
                res = context.cast(builder, res, callsig.return_type, result_ty)
                results.append((res, result_ty))
            for k, si in enumerate(out_steps):
                arr_ty = outs.types[k]
                v, t = results[si]
                setitem = context.get_function(
                    operator.setitem, nb_signature(types.none, arr_ty, i, t)
                )
                setitem(builder, (builder.extract_value(outs_v, k), i_v, v))
            return context.get_dummy_value()

        return sig, codegen

    return row_body


def build_fused_kernel(
    plan: KernelPlan,
    compiled: Sequence[Callable],
    result_dtypes: Sequence[np.dtype],
) -> Callable:
    """One `@njit` kernel for `plan`, `kernel(n, cols, valids, params_all,
    outs)`:

    - `n` — the row count;
    - `cols` — a tuple of the external input columns, `plan.external_inputs`
      order, each in its own boundary dtype;
    - `valids` — a tuple of boolean validity arrays, one per OPTIONAL
      external input, in the same order (often `()`);
    - `params_all` — a tuple of every `param_scalar`/`params_bundle`/
      `shared` value, `kernel_signature` order (often `()`);
    - `outs` — a tuple of output arrays, `plan.required_outputs` order, each
      already of that step's `result_dtypes` entry.

    `compiled[i]` is `plan.steps[i]`'s njit dispatcher (`driver._try_njit`);
    `result_dtypes[i]` is the numpy dtype `plan.steps[i]`'s result crosses
    to the next step (and any output array) as — `driver._return_dtype`.
    Flags come from the plan: `nogil` only when every step in the group
    asked for it (doc 00 §2c: one call, so one step holding the GIL holds it
    for all), `parallel`/`fastmath` as authored.
    """
    if len(compiled) != len(plan.steps) or len(result_dtypes) != len(plan.steps):
        raise ValueError("compiled and result_dtypes must be parallel to plan.steps")
    if plan.parallel and (plan.reads_shared or any(s.reads_params for s in plan.steps)):
        raise ValueError(
            f"kernel group {plan.group_name!r} is parallel and reads a bare "
            "`params`/`shared` bundle: numba's parfor pass expands one level of "
            "tuple arguments and cannot pass a bundle nested in the kernel's "
            "params tuple (decider2.compile.kernel). Author the group without "
            "parallel(...), or declare its knobs as param() scalars."
        )
    step_index = {s.name: i for i, s in enumerate(plan.steps)}
    out_steps = tuple(step_index[name] for name in plan.required_outputs)
    row_body = _make_row_body(
        tuple(compiled),
        _arg_sources(plan),
        tuple(from_dtype(np.dtype(d)) for d in result_dtypes),
        out_steps,
    )

    if plan.parallel:
        def kernel(n, cols, valids, params_all, outs):
            for i in prange(n):
                row_body(cols, valids, params_all, outs, i)
    else:
        def kernel(n, cols, valids, params_all, outs):
            for i in range(n):
                row_body(cols, valids, params_all, outs, i)

    # Doc 00 §2b: authored per step, never inferred, off by default. A group
    # releases the GIL only when every step in it asked to — one step that
    # did not is enough to keep it held, because the kernel is one call.
    nogil = bool(plan.steps) and all(s.nogil for s in plan.steps)
    return njit(nogil=nogil, parallel=plan.parallel, fastmath=plan.fastmath)(kernel)
