"""Deterministic driver-source generation — doc 05 §4 (codegen), §4.3
(driver shape), §4.2 (determinism).

Only the glue between individually-njit'd steps is generated here; step
*bodies* stay in whatever real source file the author wrote them in and get
njit'd there (`decider2.compile.driver`). What this module writes is exactly
the row loop and the argument plumbing doc 05 §4.3 shows:

    def _driver(inp, params_m1, params_m2, shared, out):
        for i in range(inp.shape[0]):
            v_disposable = _disposable_income(inp.net_income[i], inp.expenses[i])
            ...
            out.score[i] = v_score

Every function here is a pure function of its arguments -> source text: no
addresses, no timestamps, no `id()`, no PID, and no set/dict iteration-order
dependence (doc 05 §4.2). Iteration in this module is always over a tuple
whose order was fixed by the caller (`Step.inputs` / `Step.params`, both
declared as tuples in `decider2.types`) or over a plain `list`/`dict` that
*this* module builds by inserting keys in an order derived from those same
tuples — never from hashing.
"""
from __future__ import annotations

import inspect
import keyword
import re
from dataclasses import dataclass
from typing import Literal

from decider2.types import Input, NullPolicy, ParamDecl, Step

_IDENT_BAD = re.compile(r"[^0-9A-Za-z_]")


def safe_ident(name: str) -> str:
    """A name guaranteed to be a valid Python identifier.

    Value/step names are already meant to be identifiers (they come off
    parameter and function names), but generated source must not break on
    the one that isn't quite — doc 03 §7's qualified emit names
    (`term_cap@sector_cap`) contain `@` and never reach here as bare Python
    identifiers, but nothing stops a relabelled name doing something
    similar, and this stays defensive rather than assuming it.
    """
    ident = _IDENT_BAD.sub("_", name)
    if not ident or ident[0].isdigit():
        ident = f"v_{ident}"
    if keyword.iskeyword(ident):
        ident = f"{ident}_"
    return ident


@dataclass(frozen=True)
class ArgRole:
    """One argument of a generated kernel function, and what it means.

    Shared between `emit_kernel_source` (which only needs `.text`, to write
    the `def kernel(...)` line and the njit-wrap lines) and
    `decider2.runtime.modes` (which needs the rest, to build the *actual*
    call at run time in exactly the same order) — factored out so the two
    can never silently disagree on argument order.
    """

    kind: Literal["array", "valid", "param_scalar", "params_bundle", "shared", "output"]
    text: str
    input_name: str | None = None
    step_name: str | None = None
    param_name: str | None = None
    output_name: str | None = None
    owner: str | None = None          # the step's owning module instance
    # (doc 03 §4.1/§10: params are namespaced by MODULE instance, not step
    # name — a step's OUTPUT name is not unique across modules, doc 03 §3.2's
    # waterfall idiom. `decider2.runtime.modes` keys its params lookup by
    # `(owner, step_name, param_name)` to survive that collision.)


def _step_ident(step: Step) -> str:
    return safe_ident(step.name)


def _param_arg_name(step: Step, decl: ParamDecl) -> str:
    # Namespaced by step == module-instance name (doc 03 §4.1): two steps
    # each with a param called "cap" must not collide in one kernel's args.
    return f"p_{_step_ident(step)}_{safe_ident(decl.name)}"


def _params_arg_name(step: Step) -> str:
    return f"params_{_step_ident(step)}"


@dataclass(frozen=True)
class KernelPlan:
    """Everything `emit_kernel_source` needs, already decided.

    Building one of these is `decider2.compile.driver`'s job — it knows
    which steps survived njit, which values cross a segment boundary and
    which stay in registers. This module only turns an already-decided plan
    into text; it makes no decisions of its own.
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
    """The kernel function's argument list, in the one true order.

    Row arrays first (one per external input, first-seen order), then a
    validity array for every OPTIONAL one (doc 05 §2 tier 3), then each
    step's own param()-declared scalars, then any bare `params` bundles,
    then `shared` at most once, then one output array per required output.
    """
    roles: list[ArgRole] = []
    for inp in plan.external_inputs:
        roles.append(ArgRole("array", f"arr_{safe_ident(inp.name)}", input_name=inp.name))
    for inp in plan.external_inputs:
        if inp.null_policy is NullPolicy.OPTIONAL:
            roles.append(ArgRole("valid", f"valid_{safe_ident(inp.name)}", input_name=inp.name))
    # `plan.owners[i]` is the module instance that owns `plan.steps[i]`; a
    # plan built without that info (owners left at its default `()`) falls
    # back to "each step owns itself", matching the pre-owners behaviour
    # exactly for every caller that never supplied one.
    owners = plan.owners if len(plan.owners) == len(plan.steps) else tuple(s.name for s in plan.steps)
    for step, owner in zip(plan.steps, owners):
        if step.reads_params:
            roles.append(
                ArgRole("params_bundle", _params_arg_name(step), step_name=step.name, owner=owner)
            )
        for decl in step.params:
            roles.append(
                ArgRole(
                    "param_scalar",
                    _param_arg_name(step, decl),
                    step_name=step.name,
                    param_name=decl.name,
                    owner=owner,
                )
            )
    if plan.reads_shared:
        roles.append(ArgRole("shared", "shared"))
    for name in plan.required_outputs:
        roles.append(ArgRole("output", f"out_{safe_ident(name)}", output_name=name))
    return roles


def emit_kernel_source(plan: KernelPlan) -> str:
    """Render one fused kernel's real source text (doc 05 §4.1, §4.3).

    Real file, never `exec` — this text is only ever written to disk by
    `decider2.compile.cache.get_or_build` and imported by module name.
    Nothing in this function's output depends on when or where it runs.
    """
    if not plan.external_inputs:
        raise ValueError(f"kernel group {plan.group_name!r} has no external inputs")

    roles = kernel_signature(plan)
    lines: list[str] = []
    lines.append(f'"""Generated kernel for group {plan.group_name!r}.')
    lines.append("")
    lines.append("Content-addressed: this file's name is a hash of its own bytes")
    lines.append("(decider2.compile.cache), so an edit here always produces a new")
    lines.append("file rather than a stale cache hit at either the numba or the")
    lines.append('CPython .pyc layer (doc 05 §4.2). Do not hand-edit — regenerate')
    lines.append('from the pipeline instead.')
    lines.append('"""')
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from numba import njit, prange")
    lines.append("")

    for step in plan.steps:
        ident = _step_ident(step)
        lines.append(f"from {step.fn.__module__} import {step.fn.__name__} as _src_{ident}")
    lines.append("")
    for step in plan.steps:
        ident = _step_ident(step)
        lines.append(f"_compiled_{ident} = njit(cache=True)(_src_{ident})")
    lines.append("")

    kernel_args = [r.text for r in roles]
    flags = ["cache=True"]
    # Doc 00 §2b: authored per step, never inferred, off by default. A group
    # releases the GIL only when every step in it asked to — one step that
    # did not is enough to keep it held, because the kernel is one call.
    if plan.steps and all(s.nogil for s in plan.steps):
        flags.append("nogil=True")
    if plan.parallel:
        flags.append("parallel=True")
    if plan.fastmath:
        flags.append("fastmath=True")
    lines.append(f"@njit({', '.join(flags)})")
    lines.append(f"def kernel({', '.join(kernel_args)}):")

    first_array = next(r for r in roles if r.kind == "array")
    lines.append(f"    n = {first_array.text}.shape[0]")
    loop_call = "prange" if plan.parallel else "range"
    lines.append(f"    for i in {loop_call}(n):")

    for inp in plan.external_inputs:
        vname = f"v_{safe_ident(inp.name)}"
        arr = f"arr_{safe_ident(inp.name)}"
        if inp.null_policy is NullPolicy.OPTIONAL:
            valid = f"valid_{safe_ident(inp.name)}"
            lines.append(f"        {vname} = {arr}[i] if {valid}[i] else None")
        else:
            lines.append(f"        {vname} = {arr}[i]")

    for step in plan.steps:
        sig = inspect.signature(step.fn)
        param_by_name = {d.name: d for d in step.params}
        call_args: list[str] = []
        for pname in sig.parameters:
            if pname == "params":
                call_args.append(_params_arg_name(step))
            elif pname == "shared":
                call_args.append("shared")
            elif pname in param_by_name:
                call_args.append(_param_arg_name(step, param_by_name[pname]))
            else:
                call_args.append(f"v_{safe_ident(pname)}")
        ident = _step_ident(step)
        vname = f"v_{safe_ident(step.name)}"
        lines.append(f"        {vname} = _compiled_{ident}({', '.join(call_args)})")

    for name in plan.required_outputs:
        lines.append(f"        out_{safe_ident(name)}[i] = v_{safe_ident(name)}")

    lines.append("")
    return "\n".join(lines)
