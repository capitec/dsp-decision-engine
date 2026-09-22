"""`Loop` — doc 03 §8.3.

```python
BestOffer = Loop(
    should_continue,            # (carried..., loop_idx) -> bool
    OfferStep,                  # body module
    carries=["best_offer"],     # exactly one, in this build — see below
    max_iterations=511,         # REQUIRED
    name="best_offer_loop",
)
```

Returns a plain `types.Module`, exactly like `Branch` — see `branch.py`'s
docstring and `decider2.graph.control_flow.interpreter`'s module docstring
for why (doc 03 §8.4) and how: a real, bounded loop with a real early exit,
walked in compiled code by the construct's one shared `@njit` kernel over a
DATA program — never a Python-level delegate, and (as of this build) never
a generated source file either. Doc 03 §8.3's whole point is that the
early exit must be real in compiled code, the thing a previous polars port
destroyed.

`max_iterations` is a required, keyword-only, no-default parameter of this
Python function — omitting it is a `TypeError` from Python itself before a
single line of this module's own code runs, which is the loudest possible
rejection doc 03 §8.3 asks for ("An unbounded loop inside compiled code
cannot be interrupted").

`loop_idx` (doc 03 §2's wiring table, per this agent's report: not actually
listed there yet) is reserved the same way this build treats it everywhere
else a Loop touches: `should_continue`/`body` may declare an ordinary
`loop_idx: int` parameter, and the program's own `SET_ZERO`/`INCR` opcodes
supply the current iteration count for it directly, in a register — it is
never read from the outer pipeline's frame or from `carries`.

**In practice this build supports exactly one `carries` name.** Doc 03
§8.3's own worked example carries two (`["best_offer", "best_score"]`);
this agent's final report explains why that could not be built safely.
Short version: every carry becomes its own step, and `should_continue`
runs every iteration regardless of which carry is the current target, so
EVERY carry's step ends up needing EVERY OTHER carry's pre-loop value too.
`decider2.graph.interface.raw_interface`'s "same name read by one step,
produced by another in the same module = a real dependency" rule then sees
an unbreakable cycle for any 2+ carries — raised here as a clear
`ValueError`, deliberately, over the alternative that was tried and
reverted: slicing each target down to only the steps it transitively needs
avoids the cycle, but then a one-directional cross-carry read silently
resolves against a SIBLING step's already-computed POST-loop value instead
of the true pre-loop one (decider2's ordinary "most recent wins" waterfall,
applied somewhere it does not mean what it looks like it means) — a silent
wrong answer, which is worse than the loud error this module raises
instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from decider2.graph.control_flow._engine import (
    NodeTable,
    RegisterMap,
    boundary_inputs,
    build_packed_step,
    encode_call,
    merge_inputs,
    normalize,
    output_annotation,
    prefixed_param_decls,
    single_step,
)
from decider2.graph.control_flow.interpreter import CMP_LIT, INCR, LEAF, SET_ZERO, ProgramBuilder
from decider2.graph.interface import effective_interface, topological_steps
from decider2.params import build_params_model
from decider2.trees.interpreter import LT
from decider2.types import Input, Module, NullPolicy

__all__ = ["Loop"]

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
    another `Loop`/`Branch` in turn: the nested construct's own packed step
    is just another callee in this construct's adapter tuple
    (`_engine.NodeTable`, `interpreter.make_packed_adapter`), exactly
    like a hand-written step.

    `carries` values are read by `body` (and typically by `should_continue`)
    as ordinary parameters and MUST be produced by `body` at iteration end
    (checked here, at build time, naming the missing one) — the self-read
    waterfall idiom doc 03 §3.2 already describes for a plain module
    (`def best_offer(best_offer: float, ...) -> float: ...`), one level up.

    Real early exit: `should_continue` is checked BEFORE each iteration (an
    actual conditional back-edge, not a fixed-trip-count loop that always
    runs `max_iterations` times), so a condition that turns false at
    iteration 3 genuinely stops the loop at iteration 3, in compiled code.

    `build_dir` is accepted for call-site compatibility and ignored:
    nothing a Loop builds is written to disk any more.
    """
    del build_dir
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

    body_by_output = {s.name: s for s in body_steps}
    body_input_names = {inp.name for s in body_steps for inp in s.inputs}
    for cname in carries:
        if cname not in body_by_output:
            raise ValueError(
                f"Loop '{name}': carries names '{cname}', but the body "
                f"never produces it (it produces {sorted(body_by_output)}). "
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
        if inp.name != _LOOP_IDX and inp.name in body_by_output and inp.name not in carries:
            raise ValueError(
                f"Loop '{name}': should_continue reads '{inp.name}', which "
                "the body produces but which is not one of carries="
                f"{list(carries)} — it has no value before the body's "
                "first iteration ever runs. Add it to carries=[...] (and "
                "have the body self-read it), or have should_continue "
                f"read something else instead."
            )

    # Every carry program's nodes register their adapters here as they
    # are encoded (a body step at the same registers in two carry programs
    # shares one); the construct's one walker is built over them at the
    # end, and the steps after it (`pending`).
    nodes = NodeTable()
    sc_role = f"Loop '{name}' should_continue"
    body_role = f"Loop '{name}' body"
    pending: list[dict] = []

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
        if i.name != _LOOP_IDX and i.name not in carries and i.name not in body_by_output
    ]
    # The wrapper signature, identical for every carry target: every leaf
    # (first-seen order), then each carry's initial value, then every
    # `param()` field — should_continue's, then each body step's,
    # namespaced by its own module instance name.
    inputs: list[Input] = list(merge_inputs([("should_continue", sc_leaf), ("body", body_leaf)]))
    for cname in carries:
        inputs.append(
            Input(name=cname, annotation=output_annotation(body_by_output[cname]), null_policy=NullPolicy.REQUIRED)
        )
    params = prefixed_param_decls(sc_step, sc_step.name)
    for s, owner in zip(body_steps, body_owners):
        params += prefixed_param_decls(s, owner)

    for target in carries:
        step_inputs = boundary_inputs(inputs, construct=f"Loop '{name}'", output=target)
        rmap = RegisterMap(step_inputs, params)
        loop_idx_reg = rmap.internal(_LOOP_IDX)
        # Every non-carry body output is a construct-internal register,
        # written by its own STEP node before anything reads it (the
        # should_continue check above guarantees the one reader that could
        # run first never names one).
        for s in body_steps:
            rmap.internal(s.name)

        builder = ProgramBuilder()
        n_setzero = builder.add(SET_ZERO, dest=loop_idx_reg)
        # doc 03 §8.3: max_iterations bounds the loop REGARDLESS of what
        # should_continue itself checks — a bound around should_continue's
        # own check, never a substitute for it. `CMP_LIT` reused from
        # decider2.trees.interpreter's six comparisons (`LT`) gives the
        # exact "loop_idx < max_iterations" test as data, checked first
        # each iteration, before should_continue ever runs.
        n_maxcheck = builder.add(CMP_LIT, arg=loop_idx_reg, cmp=LT, lit=float(max_iterations))
        n_cond = encode_call(builder, rmap, nodes, sc_step, sc_step.name, role=sc_role, is_cond=True)
        n_leaf = builder.add(LEAF, dest=rmap.lookup(target))

        body_pcs: list[int] = []
        for s, owner in zip(body_steps, body_owners):
            body_pcs.append(
                encode_call(
                    builder, rmap, nodes, s, owner, role=body_role, is_cond=False, dest=rmap.lookup(s.name),
                )
            )
        for j in range(len(body_pcs) - 1):
            builder.set_next(body_pcs[j], next_=body_pcs[j + 1])

        n_incr = builder.add(INCR, dest=loop_idx_reg)

        builder.set_next(n_setzero, next_=n_maxcheck)
        builder.set_next(n_maxcheck, next_=n_cond, alt=n_leaf)
        builder.set_next(n_cond, next_=(body_pcs[0] if body_pcs else n_incr), alt=n_leaf)
        if body_pcs:
            builder.set_next(body_pcs[-1], next_=n_incr)
        builder.set_next(n_incr, next_=n_maxcheck)

        pending.append(
            dict(
                name=target,
                program=builder.build(start_pc=n_setzero, n_regs=rmap.n_regs),
                inputs=step_inputs, params=params,
                output=output_annotation(body_by_output[target]),
                doc=f"`{target}` after `{name}` runs to its bound (doc 03 §8.3).",
            )
        )

    walk = nodes.walker()
    steps = [build_packed_step(walk=walk, **entry) for entry in pending]

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
            "carry becomes its own step, needing every OTHER carry's "
            "pre-loop value too (should_continue runs every iteration "
            "regardless of target), which decider2.graph.interface's "
            "\"same name read by one step, produced by another in the "
            "same module\" rule sees as an unbreakable cycle — correctly: "
            "the alternative (only running the steps one target's OWN "
            "computation needs) was tried and rejected because it silently "
            "wires a cross-carry read to a SIBLING step's post-loop output "
            "instead of the true pre-loop value. Split into separate "
            f"Loop()s if the carries are genuinely independent. Original "
            f"error: {exc}"
        ) from exc
    return built
