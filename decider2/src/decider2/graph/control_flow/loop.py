"""`Loop` — doc 03 §8.3.

```python
BestOffer = Loop(
    should_continue,            # (carried..., loop_idx) -> bool
    OfferStep,                  # body module
    carries=["best_offer", "best_score"],
    max_iterations=511,         # REQUIRED
    name="best_offer_loop",
)
```

Returns a plain `types.Module`, exactly like `Branch` — see `branch.py`'s
docstring and `decider2.graph.control_flow.interpreter`'s module docstring
for why (doc 03 §8.4) and how (a real, bounded loop walked in compiled
code by ONE shared `@njit` kernel over a DATA program — `run_program`,
lifted from EXPERIMENTS.md §W's own verified experiment — never a Python-
level delegate: doc 03 §8.3's whole point is that this must be a REAL
early exit in compiled code, the thing a previous polars port destroyed).

`max_iterations` is a required, keyword-only, no-default parameter of this
Python function — omitting it is a `TypeError` from Python itself before a
single line of this module's own code runs, which is the loudest possible
rejection doc 03 §8.3 asks for ("An unbounded loop inside compiled code
cannot be interrupted").

`loop_idx` (doc 03 §2's wiring table, per this agent's report: not actually
listed there yet) is reserved the same way this build treats it everywhere
else a Loop touches: `should_continue`/`body` may declare an ordinary
`loop_idx: int` parameter, and the generated program's own `SET_ZERO`/
`INCR` opcodes supply the current iteration count for it directly, in a
register — it is never read from the outer pipeline's frame or from
`carries`.

**In practice this build supports exactly one `carries` name.** Doc 03
§8.3's own worked example carries two (`["best_offer", "best_score"]`);
this agent's final report explains why that could not be built safely.
Short version: every carry becomes its own generated step, and
`should_continue` runs every iteration regardless of which carry is the
current target, so EVERY carry's generated function ends up needing
EVERY OTHER carry's pre-loop value too. `decider2.graph.interface.
raw_interface`'s "same name read by one step, produced by another in the
same module = a real dependency" rule then sees an unbreakable cycle for
any 2+ carries — raised here as a clear `ValueError`, deliberately, over
the alternative that was tried and reverted: slicing each target down to
only the steps it transitively needs avoids the cycle, but then a
one-directional cross-carry read silently resolves against a SIBLING
step's already-computed POST-loop value instead of the true pre-loop one
(decider2's ordinary "most recent wins" waterfall, applied somewhere it
does not mean what it looks like it means) — a silent wrong answer, which
is worse than the loud error this module raises instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from decider2.compile import cache
from decider2.graph.control_flow._engine import (
    RegisterMap,
    encode_call,
    merge_inputs,
    normalize,
    render_leaf_param,
    render_prefixed_params,
    render_step_thunk,
    render_switch_branch,
    safe_ident,
    single_step,
)
from decider2.graph.control_flow.interpreter import CALL_ARITY, CMP_LIT, INCR, LEAF, SET_ZERO, ProgramBuilder
from decider2.graph.interface import effective_interface, topological_steps
from decider2.params import build_params_model, harvest_step
from decider2.trees.interpreter import LT
from decider2.types import Module

__all__ = ["Loop"]

# Same default as decider2.runtime.invoke.DEFAULT_BUILD_DIR — see
# branch.py's DEFAULT_BUILD_DIR for why.
DEFAULT_BUILD_DIR = Path(".decider2_cache")

_LOOP_IDX = "loop_idx"


def Loop(
    should_continue: Any,
    body: Any,
    *,
    carries: Sequence[str],
    max_iterations: int,
    name: str,
    build_dir: "str | Path | None" = None,
) -> Module:
    """Doc 03 §8.3. `should_continue` is a bare function/`Step`/`Module`/
    `Pipeline` producing exactly one bool, and — in this build — exactly
    ONE step (see `_engine.py`'s module docstring). `body` may be an
    arbitrary multi-step `Module`/`Pipeline` (no sibling to collide with,
    unlike a Branch arm) — nesting works because `Loop(...)` itself returns
    a plain `Module`, a valid `body` (or a valid arm of a `Branch`) for
    another `Loop`/`Branch` in turn: the nested construct's own single
    generated step is just another by-name case in `call_step`'s switch,
    exactly like a hand-written step (see `branch.py`'s `_register_thunk`/
    this module's `_register_thunk`). Nesting a construct this way loses
    numba's disk cache for that ONE wrapper specifically — a real, reported
    narrowing found while building this, not a correctness gap; see this
    agent's report and `decider2.graph.control_flow.interpreter`'s module
    docstring.

    `carries` values are read by `body` (and typically by `should_continue`)
    as ordinary parameters and MUST be produced by `body` at iteration end
    (checked here, at build time, naming the missing one) — the self-read
    waterfall idiom doc 03 §3.2 already describes for a plain module
    (`def best_offer(best_offer: float, ...) -> float: ...`), one level up.

    Real early exit: `should_continue` is checked BEFORE each iteration (an
    actual conditional back-edge, not a fixed-trip-count loop that always
    runs `max_iterations` times), so a condition that turns false at
    iteration 3 genuinely stops the loop at iteration 3, in compiled code.
    """
    if not name:
        raise ValueError(
            "Loop(...) needs name= — there is no single function a "
            "multi-carry node could derive one from (doc 03 §5, §8.3)."
        )
    carries = tuple(carries)
    if not carries:
        raise ValueError("Loop(...) needs carries=[...] — a loop that carries nothing is not a loop.")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 1:
        raise ValueError(
            f"Loop '{name}': max_iterations must be a positive int, got "
            f"{max_iterations!r} (doc 03 §8.3: REQUIRED, and it bounds a "
            "loop that cannot be interrupted once compiled)."
        )

    sc_pipeline = normalize(should_continue)
    sc_step = single_step(sc_pipeline, role="Loop should_continue")
    body_pipeline = normalize(body)
    body_steps, _group_ids, body_owners, _param_spaces = body_pipeline.flatten_for_runtime()

    body_outputs = {s.name for s in body_steps}
    body_input_names = {inp.name for s in body_steps for inp in s.inputs}
    for cname in carries:
        if cname not in body_outputs:
            raise ValueError(
                f"Loop '{name}': carries names '{cname}', but the body "
                f"never produces it (it produces {sorted(body_outputs)}). "
                "Doc 03 §8.3: a carries value MUST be produced by the body "
                "at iteration end."
            )
        if cname not in body_input_names:
            raise ValueError(
                f"Loop '{name}': carries names '{cname}', but the body "
                "does not read it as an input at iteration start (doc 03 "
                f"§8.3). Add a parameter named '{cname}' to whichever body "
                f"step should see the previous iteration's value — e.g. "
                f"`def {cname}({cname}: float, ...) -> float: ...` (doc 03 "
                "§3.2's self-read waterfall, one level up)."
            )

    # should_continue is checked BEFORE the body's first iteration ever
    # runs (doc 03 §8.3's real "while", not a "do-while") — so a body
    # output it reads that ISN'T a carry (never given an initial value)
    # would read an undefined register on that very first check. A carry
    # is exempt because it always gets one (the caller's own leaf argument).
    for inp in sc_step.inputs:
        if inp.name != _LOOP_IDX and inp.name in body_outputs and inp.name not in carries:
            raise ValueError(
                f"Loop '{name}': should_continue reads '{inp.name}', which "
                "the body produces but which is not one of carries="
                f"{list(carries)} — it has no value before the body's "
                "first iteration ever runs. Add it to carries=[...] (and "
                "have the body self-read it), or have should_continue "
                f"read something else instead."
            )

    resolved_build_dir: "str | Path" = build_dir if build_dir is not None else DEFAULT_BUILD_DIR

    lines: list[str] = []
    lines.append(f'"""Generated Loop kernel for {name!r} (doc 03 §8.3).')
    lines.append("")
    lines.append("Content-addressed, real source — decider2.compile.cache. Do not")
    lines.append("hand-edit; regenerate by rebuilding the Loop(...) call instead.")
    lines.append("")
    lines.append("This file holds ONLY per-step by-name imports (bounded by step COUNT)")
    lines.append("and this Loop's DATA program arrays — the loop itself (head-check,")
    lines.append("back-edge, early exit) is walked by ONE shared, hand-written kernel,")
    lines.append("decider2.graph.control_flow.interpreter.run_program, which calls the")
    lines.append("should_continue/body steps through THIS file's own call_step/")
    lines.append("call_cond switch (passed in as ordinary njit function arguments —")
    lines.append("never a global pointer table: see decider2.graph.control_flow.")
    lines.append("interpreter's module docstring for why that distinction is")
    lines.append("load-bearing here).")
    lines.append('"""')
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("import numpy as np")
    lines.append("from numba import njit, types")
    lines.append("")
    lines.append("from decider2.graph.control_flow.interpreter import run_program")
    lines.append("from decider2.params import param")
    lines.append("")

    step_thunk_local: dict[int, tuple[str, int]] = {}
    step_thunks: list[tuple[object, str]] = []
    cond_thunk_local: dict[int, tuple[str, int]] = {}
    cond_thunks: list[tuple[object, str]] = []

    def _register_thunk(step, role_prefix: str, *, is_cond: bool) -> int:
        table_local = cond_thunk_local if is_cond else step_thunk_local
        table = cond_thunks if is_cond else step_thunks
        key = id(step.fn)
        if key in table_local:
            return table_local[key][1]
        local = f"{role_prefix}_{len(table)}"
        lines.extend(render_step_thunk(step, local))
        idx = len(table)
        table.append((step, local))
        table_local[key] = (local, idx)
        return idx

    sc_slot = _register_thunk(sc_step, "sc", is_cond=True)
    body_slots = [
        _register_thunk(s, f"body{i}", is_cond=False) for i, s in enumerate(body_steps)
    ]
    lines.append("")

    def _emit_switch(fn_name: str, table: list, *, is_cond: bool) -> None:
        args = ", ".join(f"a{i}: float" for i in range(CALL_ARITY))
        ret_type = "types.boolean" if is_cond else "types.float64"
        lines.append("@njit(cache=True)")
        lines.append(f"def {fn_name}(step_idx: int, {args}) -> {ret_type}:")
        for i, (step, local) in enumerate(table):
            branch_kw = "if" if i == 0 else "elif"
            lines.append(f"    {branch_kw} step_idx == {i}:")
            lines.append(render_switch_branch(step, local, i, is_cond=is_cond))
        default = "False" if is_cond else "0.0"
        lines.append(f"    return {default}")
        lines.append("")

    _emit_switch("call_step", step_thunks, is_cond=False)
    _emit_switch("call_cond", cond_thunks, is_cond=True)

    # doc 03 §8.3's own worked example carries TWO names sharing one loop
    # run. This build cannot express that safely — see the module
    # docstring and this agent's final report: a per-target SLICE (only
    # run the body steps a target transitively needs) avoids the cycle
    # below but silently MISWIRES a one-directional cross-carry read (doc
    # 03 §2.1's waterfall resolves it against a SIBLING carry's own
    # already-produced, POST-loop value instead of the true pre-loop one —
    # wrong, and quiet about it). The uniform signature below is
    # deliberately the unsliced, always-everything one: for 2+ carries it
    # ALWAYS hits the cycle check below and fails loudly, which is the
    # correct trade against a sliced version that sometimes computes a
    # silently wrong answer.
    sc_leaf = [i for i in sc_step.inputs if i.name != _LOOP_IDX and i.name not in carries]
    body_leaf = [
        i for s in body_steps for i in s.inputs
        if i.name != _LOOP_IDX and i.name not in carries and i.name not in body_outputs
    ]
    merged_inputs = merge_inputs([("should_continue", sc_leaf), ("body", body_leaf)])

    leaf_sig = [render_leaf_param(i) for i in merged_inputs]
    leaf_sig += [f"{c}: float" for c in carries]  # initial values
    param_sig, _ = render_prefixed_params(sc_step, sc_step.name)
    for i, s in enumerate(body_steps):
        p, _ = render_prefixed_params(s, body_owners[i])
        param_sig += p

    # Every register that genuinely corresponds to a WRAPPER ARGUMENT
    # (never a non-carry body output, which has no value until its own
    # STEP node computes it, and never loop_idx, which SET_ZERO/INCR own
    # outright) — `_emit_regs_build` seeds exactly this set from `_regs`,
    # nothing else, so it never references an identifier the wrapper
    # function does not actually have.
    wrapper_arg_names = {safe_ident(i.name) for i in merged_inputs}
    wrapper_arg_names |= {safe_ident(c) for c in carries}
    wrapper_arg_names |= {safe_ident(f"{sc_step.name}__{d.name}") for d in sc_step.params}
    for i, s in enumerate(body_steps):
        wrapper_arg_names |= {safe_ident(f"{body_owners[i]}__{d.name}") for d in s.params}

    fn_names: dict[str, str] = {}
    for target in carries:
        fn_name = f"{name}_{target}"
        fn_names[target] = fn_name

        rmap = RegisterMap()
        builder = ProgramBuilder(n_regs=0)
        loop_idx_reg = rmap.index_of(safe_ident(_LOOP_IDX))
        # Every carry's own register, pre-registered (in carries= order) so
        # its slot is stable regardless of which body step's dest touches
        # it first below.
        for c in carries:
            rmap.index_of(safe_ident(c))

        n_setzero = builder.add(op=SET_ZERO, dest=loop_idx_reg)
        # doc 03 §8.3: max_iterations bounds the loop REGARDLESS of what
        # should_continue itself checks — the old generated `while _i <
        # {max_iterations}:` was a bound around should_continue's own
        # check, never a substitute for it. `CMP_LIT` reused from
        # decider2.trees.interpreter's six comparisons (`LT`) gives the
        # exact same "loop_idx < max_iterations" test as data, checked
        # first each iteration, before should_continue ever runs.
        n_maxcheck = builder.add(op=CMP_LIT, arg0=loop_idx_reg, arg1=LT, lit=float(max_iterations))
        n_cond = encode_call(builder, rmap, sc_step, sc_step.name, sc_slot, is_cond=True)
        target_reg = rmap.index_of(safe_ident(target))
        n_leaf = builder.add(op=LEAF, dest=target_reg)

        body_pcs: list[int] = []
        for i, s in enumerate(body_steps):
            dest_reg = rmap.index_of(safe_ident(s.name))
            n_body = encode_call(
                builder, rmap, s, body_owners[i], body_slots[i], is_cond=False, dest=dest_reg,
            )
            body_pcs.append(n_body)
        for j in range(len(body_pcs) - 1):
            builder.set_next(body_pcs[j], next_=body_pcs[j + 1])

        n_incr = builder.add(op=INCR, dest=loop_idx_reg)

        builder.set_next(n_setzero, next_=n_maxcheck)
        builder.set_next(n_maxcheck, next_=n_cond, alt=n_leaf)
        builder.set_next(n_cond, next_=(body_pcs[0] if body_pcs else n_incr), alt=n_leaf)
        if body_pcs:
            builder.set_next(body_pcs[-1], next_=n_incr)
        builder.set_next(n_incr, next_=n_maxcheck)

        builder.n_regs = rmap.n_regs
        prog = builder.build(start_pc=n_setzero)
        _emit_program_arrays(lines, f"_{target}", prog)

        lines.append(f"def {fn_name}({', '.join(leaf_sig + param_sig)}) -> float:")
        lines.append(f'    """`{target}` after `{name}` runs to its bound (doc 03 §8.3)."""')
        lines += _emit_regs_build(rmap, prog.n_regs, wrapper_arg_names)
        arr_prefix = f"_{target}"
        lines.append(
            f"    return run_program({arr_prefix}__op, {arr_prefix}__step_idx, "
            f"{arr_prefix}__arg0, {arr_prefix}__arg1, {arr_prefix}__arg2, {arr_prefix}__arg3, "
            f"{arr_prefix}__arg4, {arr_prefix}__arg5, {arr_prefix}__lit, {arr_prefix}__dest, "
            f"{arr_prefix}__next, {arr_prefix}__alt, {n_setzero}, call_step, call_cond, _regs)"
        )
        lines.append("")

    source = "\n".join(lines) + "\n"
    cached = cache.get_or_build(source, resolved_build_dir)

    steps = [harvest_step(getattr(cached.module, fn_names[c]), name=c) for c in carries]
    param_decls = {}
    for s in steps:
        for decl in s.params:
            param_decls[decl.name] = decl
    params_model = build_params_model(name, tuple(param_decls.values())) if param_decls else None

    built = Module(name=name, steps=tuple(steps), params_model=params_model)
    effective_interface(built)  # eager build-time validation, matching module()
    try:
        topological_steps(built.steps)
    except ValueError as exc:
        raise ValueError(
            f"Loop '{name}': carries={list(carries)} has 2+ names, which "
            "this build does not support when should_continue/body reads "
            "cross carry names — a genuine, unresolved gap (see "
            "decider2.graph.control_flow.loop's module docstring): every "
            "carry becomes its own generated step, needing every OTHER "
            "carry's pre-loop value too (should_continue runs every "
            "iteration regardless of target), which decider2.graph."
            "interface's \"same name read by one step, produced by "
            "another in the same module\" rule sees as an unbreakable "
            "cycle — correctly: the alternative (only running the steps "
            "one target's OWN computation needs) was tried and rejected "
            "because it silently wires a cross-carry read to a SIBLING "
            "step's post-loop output instead of the true pre-loop value. "
            f"Split into separate Loop()s if the carries are genuinely "
            f"independent. Original error: {exc}"
        ) from exc
    return built


def _emit_program_arrays(lines: list[str], prefix: str, prog) -> None:
    lines.append(f"{prefix}__op = np.array({list(int(v) for v in prog.op)!r}, dtype=np.int32)")
    lines.append(f"{prefix}__step_idx = np.array({list(int(v) for v in prog.step_idx)!r}, dtype=np.int32)")
    lines.append(f"{prefix}__arg0 = np.array({list(int(v) for v in prog.arg0)!r}, dtype=np.int32)")
    lines.append(f"{prefix}__arg1 = np.array({list(int(v) for v in prog.arg1)!r}, dtype=np.int32)")
    lines.append(f"{prefix}__arg2 = np.array({list(int(v) for v in prog.arg2)!r}, dtype=np.int32)")
    lines.append(f"{prefix}__arg3 = np.array({list(int(v) for v in prog.arg3)!r}, dtype=np.int32)")
    lines.append(f"{prefix}__arg4 = np.array({list(int(v) for v in prog.arg4)!r}, dtype=np.int32)")
    lines.append(f"{prefix}__arg5 = np.array({list(int(v) for v in prog.arg5)!r}, dtype=np.int32)")
    lines.append(f"{prefix}__lit = np.array({list(float(v) for v in prog.lit)!r}, dtype=np.float64)")
    lines.append(f"{prefix}__dest = np.array({list(int(v) for v in prog.dest)!r}, dtype=np.int32)")
    lines.append(f"{prefix}__next = np.array({list(int(v) for v in prog.next_)!r}, dtype=np.int32)")
    lines.append(f"{prefix}__alt = np.array({list(int(v) for v in prog.alt)!r}, dtype=np.int32)")
    lines.append("")


def _emit_regs_build(rmap: "RegisterMap", n_regs: int, wrapper_arg_names: "set[str]") -> list[str]:
    """Build `_regs`, seeding every register that is genuinely one of the
    wrapper function's own arguments (a carry's initial value, a plain
    leaf input, or a `param()` field). Every other register — `loop_idx`
    (owned by `SET_ZERO`/`INCR`) and every NON-carry body output (which has
    no value until its own `STEP` node computes it, guaranteed by `Loop`'s
    own validation above) — is deliberately left at its `np.zeros` default,
    never referenced as if it were a local the wrapper actually has.
    """
    out = [f"    _regs = np.zeros({n_regs})"]
    for reg_name, idx in rmap.names.items():
        if reg_name in wrapper_arg_names:
            out.append(f"    _regs[{idx}] = {reg_name}")
    return out
