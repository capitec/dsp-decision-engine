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
docstring and `_engine.py`'s module docstring for why (doc 03 §8.4) and how
(inline codegen: a real, bounded `while`/`break`, not a Python-level
delegate — doc 03 §8.3's whole point is that this must be a REAL early
exit in compiled code, the thing a previous polars port destroyed).

`max_iterations` is a required, keyword-only, no-default parameter of this
Python function — omitting it is a `TypeError` from Python itself before a
single line of this module's own code runs, which is the loudest possible
rejection doc 03 §8.3 asks for ("An unbounded loop inside compiled code
cannot be interrupted").

`loop_idx` (doc 03 §2's wiring table, per this agent's report: not actually
listed there yet) is reserved the same way this build treats it everywhere
else a Loop touches: `should_continue`/`body` may declare an ordinary
`loop_idx: int` parameter, and the generated `while` supplies the current
iteration count for it directly, as a local variable — it is never read
from the outer pipeline's frame or from `carries`.

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
    import_and_wrap,
    merge_inputs,
    normalize,
    render_call_args,
    render_leaf_param,
    render_prefixed_params,
    single_step,
)
from decider2.graph.interface import effective_interface, topological_steps
from decider2.params import build_params_model, harvest_step
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
    another `Loop`/`Branch` in turn.

    `carries` values are read by `body` (and typically by `should_continue`)
    as ordinary parameters and MUST be produced by `body` at iteration end
    (checked here, at build time, naming the missing one) — the self-read
    waterfall idiom doc 03 §3.2 already describes for a plain module
    (`def best_offer(best_offer: float, ...) -> float: ...`), one level up.

    Real early exit: `should_continue` is checked BEFORE each iteration (a
    `while`, not a `for` that always runs `max_iterations` times), so a
    condition that turns false at iteration 3 genuinely stops the loop at
    iteration 3, in compiled code.
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
    # would read an undefined local on that very first check. A carry is
    # exempt because it always gets one (the caller's own leaf argument).
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
    lines.append('"""')
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from numba import njit")
    lines.append("from decider2.params import param")
    lines.append("")
    lines += import_and_wrap(sc_step, "_sc_c")
    for i, s in enumerate(body_steps):
        lines += import_and_wrap(s, f"_body{i}_c")
    lines.append("")

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
    # An input name also produced by ANOTHER body step (e.g. a nested
    # Branch's own `_path` output, read by its `modifies` step) is an
    # internal wire, not a genuine leaf — excluded here, and threaded via
    # the `_v_<name>` local the loop body below already builds for every
    # name in `body_outputs`.
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

    fn_names: dict[str, str] = {}
    for target in carries:
        fn_name = f"{name}_{target}"
        fn_names[target] = fn_name
        lines.append(f"def {fn_name}({', '.join(leaf_sig + param_sig)}) -> float:")
        lines.append(f'    """`{target}` after `{name}` runs to its bound (doc 03 §8.3)."""')
        for c in carries:
            lines.append(f"    _v_{c} = {c}")
        lines.append("    _i = 0")
        lines.append(f"    while _i < {max_iterations}:")
        lines.append("        _loop_idx = _i")
        # loop_idx and carries read the LOCAL, per-iteration variables;
        # everything else reads the function's own leaf/param argument.
        sc_args = [
            "_loop_idx" if orig == _LOOP_IDX else (f"_v_{orig}" if orig in carries else a)
            for a, orig in zip(render_call_args(sc_step, sc_step.name), _sig_names(sc_step))
        ]
        lines.append(f"        _cont = _sc_c({', '.join(sc_args)})")
        lines.append("        if not _cont:")
        lines.append("            break")
        for i, s in enumerate(body_steps):
            b_args = render_call_args(s, body_owners[i])
            b_args = [
                "_loop_idx" if orig == _LOOP_IDX else (f"_v_{orig}" if orig in body_outputs or orig in carries else a)
                for a, orig in zip(b_args, _sig_names(s))
            ]
            lines.append(f"        _v_{s.name} = _body{i}_c({', '.join(b_args)})")
        lines.append("        _i += 1")
        lines.append(f"    return _v_{target}")
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


def _sig_names(step) -> list[str]:
    """`step`'s own parameter names, in declared order — used to line up
    `render_call_args`'s output (which already resolved param() names to
    their prefixed form) against the ORIGINAL name, so the loop body knows
    which of `loop_idx`/a carry/a plain leaf each positional slot means."""
    import inspect

    try:
        sig = inspect.signature(step.fn, eval_str=True)
    except (NameError, TypeError):
        sig = inspect.signature(step.fn)
    return list(sig.parameters)
