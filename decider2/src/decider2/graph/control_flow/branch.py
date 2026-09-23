"""`Branch` — doc 03 §8.2.

```python
TermCapRules = Branch(
    is_private_sector,          # condition step -> bool
    CapForPrivate,               # arm for True
    CapForPublic,                # arm for False
    modifies=["term_cap"],
    name="term_cap_rules",
)
```

or, the routing form:

```python
PriceByBand = Branch(risk_band_index, [BandA, BandB, BandC, BandD],
                     modifies=["price_category"], name="price_by_band")
```

Returns a plain `types.Module` (doc 03 §8.4: "there is one type — Module —
and three combinators over it", taken literally) whose steps are built
directly as `types.Step(packed=True)` — one for `<name>_path`, one per
`modifies` name — each a closure over its own DATA program, walked by the
construct's one shared `@njit` kernel (`decider2.graph.control_flow.
interpreter`; `_engine.py`'s module docstring for the layout). No source
file is generated or imported: the condition and arms are called through
a `literal_unroll`-dispatched tuple of typed adapters, so only the taken
arm executes, in compiled code, exactly as §8.2 asks. See `_engine.py` for
the scope cut that makes it possible (an arm must be a single step) and
what doc 03 §8.2 left under-specified against a real implementation (arm
order -> path index, `name=` not being optional despite the doc's own
examples omitting it).
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
from decider2.graph.control_flow.interpreter import CMP_LIT, LEAF, SET_LIT, ProgramBuilder
from decider2.graph.interface import effective_interface, topological_steps
from decider2.params import build_params_model
from decider2.trees.interpreter import EQ
from decider2.types import Input, Module, NullPolicy

__all__ = ["Branch"]

_RESULT = "#result"
_ZERO = "#zero"


def Branch(
    condition: Any,
    true_or_arms: Any,
    false_arm: Any = None,
    *,
    modifies: Sequence[str],
    name: str,
    build_dir: "str | Path | None" = None,
) -> Module:
    """Doc 03 §8.2. `condition` is a bare function, `Step`, `Module` or
    `Pipeline` producing exactly one value (bool for the two-arm form, an
    int index for the routing form) — and, in this build, exactly ONE step
    (see `_engine.py`'s module docstring). `true_or_arms` is either the
    True arm (two-arm form, paired with `false_arm`) or a list of arms
    (routing form, `false_arm` omitted); each arm is likewise exactly one
    step.

    **`name=` is required.** Doc 03 §8.2's own examples never show it, but
    unlike a single-step `module(...)` there is no single function a
    multi-arm Branch could derive a name from.

    **Arm index / path convention** (also under-specified in §8.2, resolved
    here and reported): path is the arm's position in the order given to
    `Branch` — for the two-arm form that is `0` for `true_or_arms` (the arm
    taken when `condition` is truthy) and `1` for `false_arm`; for the
    routing form it is whatever int `condition` itself returns.

    Doc 03 §7: `f"{name}_path"` is an ordinary emittable value (not a
    keyword, not a special case) reporting which arm ran, as int64 — the
    exact convention `decider2.trees.encode`'s `result_idx` already uses,
    mirrored rather than reinvented.

    `build_dir` is accepted for call-site compatibility and ignored:
    nothing a Branch builds is written to disk any more.
    """
    del build_dir
    if not modifies:
        raise ValueError("Branch(...) needs modifies=[...] — a Branch that changes nothing is not a Branch.")
    if not name:
        raise ValueError(
            "Branch(...) needs name= — there is no single function a "
            "multi-arm node could derive one from (doc 03 §5, §8.2)."
        )

    if isinstance(true_or_arms, (list, tuple)):
        if false_arm is not None:
            raise TypeError(
                "Branch(condition, [arms...], modifies=...) is the routing "
                "form and takes no third positional argument; "
                "Branch(condition, true_arm, false_arm, modifies=...) is "
                "the two-arm form and takes no list (doc 03 §8.2)."
            )
        raw_arms = list(true_or_arms)
        routing = True
        if len(raw_arms) < 2:
            raise ValueError("Branch(...) routing form needs at least two arms.")
    else:
        if false_arm is None:
            raise TypeError(
                "Branch(condition, true_arm, false_arm, modifies=...) needs "
                "both arms, or Branch(condition, [arms...], modifies=...) "
                "for the routing form (doc 03 §8.2)."
            )
        raw_arms = [true_or_arms, false_arm]
        routing = False

    cond_pipeline = normalize(condition)
    cond_step = single_step(cond_pipeline, role="Branch condition")
    arm_pipelines = [normalize(a) for a in raw_arms]
    arm_steps = [single_step(ap, role=f"Branch arm {i}") for i, ap in enumerate(arm_pipelines)]
    arm_owner = [ap.elements[0].name for ap in arm_pipelines]

    modifies = tuple(modifies)
    result_types: dict[str, Any] = {}
    producers_by_name: dict[str, list[int]] = {}
    for mname in modifies:
        producers = [i for i, s in enumerate(arm_steps) if s.name == mname]
        if not producers:
            raise ValueError(
                f"Branch '{name}': modifies=[...] names '{mname}', but no "
                "arm produces it (an arm's OWN output name, since arms are "
                "single-step in this build). modifies declares what an "
                "arm MAY change (doc 03 §8.2) — at least one arm must "
                "actually produce it."
            )
        first_ann = output_annotation(arm_steps[producers[0]])
        for i in producers[1:]:
            ann = output_annotation(arm_steps[i])
            if ann != first_ann:
                raise ValueError(
                    f"Branch '{name}': arms disagree on the type of "
                    f"'{mname}' — arm {producers[0]} produces "
                    f"{first_ann.__name__!r}, arm {i} produces "
                    f"{ann.__name__!r}. Doc 03 §8.2: an arm that produces "
                    "a modifies value must agree on type with every other "
                    "arm that does."
                )
        result_types[mname] = first_ann
        producers_by_name[mname] = producers

    # Every program's nodes register their adapters here as they are
    # encoded; the construct's one walker is built over them at the end,
    # and the steps after it (`pending`).
    nodes = NodeTable()
    cond_role = f"Branch '{name}' condition"
    pending: list[dict] = []

    # -- the path step -----------------------------------------------------
    path_name = f"{name}_path"
    cond_leaf = merge_inputs([("condition", cond_step.inputs)])
    path_inputs = boundary_inputs(cond_leaf, construct=f"Branch '{name}'", output=path_name)
    path_params = prefixed_param_decls(cond_step, cond_step.name)
    path_rmap = RegisterMap(path_inputs, path_params)
    path_builder = ProgramBuilder()
    result_reg = path_rmap.internal(_RESULT)
    if routing:
        n_call = encode_call(
            path_builder, path_rmap, nodes, cond_step, cond_step.name, role=cond_role,
            is_cond=False, dest=result_reg,
        )
        n_leaf = path_builder.add(LEAF, dest=result_reg)
        path_builder.set_next(n_call, next_=n_leaf)
        path_start = n_call
    else:
        n_cond = encode_call(
            path_builder, path_rmap, nodes, cond_step, cond_step.name, role=cond_role, is_cond=True,
        )
        n_true = path_builder.add(SET_LIT, lit=0.0, dest=result_reg)
        n_true_leaf = path_builder.add(LEAF, dest=result_reg)
        n_false = path_builder.add(SET_LIT, lit=1.0, dest=result_reg)
        n_false_leaf = path_builder.add(LEAF, dest=result_reg)
        path_builder.set_next(n_true, next_=n_true_leaf)
        path_builder.set_next(n_false, next_=n_false_leaf)
        path_builder.set_next(n_cond, next_=n_true, alt=n_false)
        path_start = n_cond
    path_pending = dict(
        name=path_name,
        program=path_builder.build(start_pc=path_start, n_regs=path_rmap.n_regs),
        inputs=path_inputs, params=path_params, output=int,
        doc=f"Which arm {name!r} took, as its 0-based index (doc 03 §7).",
    )

    # -- one step per modifies name -----------------------------------------
    for mname in modifies:
        producers = producers_by_name[mname]
        needs_passthrough = len(producers) < len(arm_steps)

        # The wrapper signature first (registers 0..N-1 / N..N+P-1), then
        # the program against it — see `_engine.py`'s module docstring.
        inputs: list[Input] = [Input(name=path_name, annotation=int, null_policy=NullPolicy.REQUIRED)]
        seen = {path_name}
        if needs_passthrough:
            inputs.append(Input(name=mname, annotation=result_types[mname], null_policy=NullPolicy.REQUIRED))
            seen.add(mname)
        for inp in merge_inputs([(f"arm {i}", arm_steps[i].inputs) for i in producers]):
            if inp.name not in seen:
                seen.add(inp.name)
                inputs.append(inp)
        params = [d for i in producers for d in prefixed_param_decls(arm_steps[i], arm_owner[i])]
        step_inputs = boundary_inputs(inputs, construct=f"Branch '{name}'", output=mname)

        rmap = RegisterMap(step_inputs, params)
        builder = ProgramBuilder()
        result_reg = rmap.internal(_RESULT)
        path_reg = rmap.lookup(path_name)
        # An arm index no producer matches can never happen (the path IS an
        # arm index, and every arm that produces `mname` is tested below);
        # the fall-through leaf still needs a register to name: the
        # passthrough input when one exists (doc 03 §8.2's "an arm may stay
        # silent, and then that name passes through unchanged"), else a
        # zero-initialised internal.
        fallthrough = builder.add(LEAF, dest=rmap.lookup(mname) if needs_passthrough else rmap.internal(_ZERO))

        entry = fallthrough
        # Build backward — the same "highest index first" order
        # `decider2.trees.schema._encode_isin` uses — so producers[0] ends
        # up tested first, matching decider 1's own first-match-wins order
        # (here: arm order, which IS the path index).
        for j in reversed(range(len(producers))):
            i = producers[j]
            n_call = encode_call(
                builder, rmap, nodes, arm_steps[i], arm_owner[i], role=f"Branch '{name}' arm {i}",
                is_cond=False, dest=result_reg,
            )
            n_leaf = builder.add(LEAF, dest=result_reg)
            builder.set_next(n_call, next_=n_leaf)
            n_test = builder.add(CMP_LIT, arg=path_reg, cmp=EQ, lit=float(i))
            builder.set_next(n_test, next_=n_call, alt=entry)
            entry = n_test

        pending.append(
            dict(
                name=mname,
                program=builder.build(start_pc=entry, n_regs=rmap.n_regs),
                inputs=step_inputs, params=params, output=result_types[mname],
                doc=f"`{mname}` for whichever arm {name!r} took.",
            )
        )

    walk = nodes.walker()
    all_steps = tuple(build_packed_step(walk=walk, **entry) for entry in pending + [path_pending])
    param_decls = {}
    for s in all_steps:
        for decl in s.params:
            param_decls[decl.name] = decl
    params_model = build_params_model(name, tuple(param_decls.values())) if param_decls else None

    built = Module(name=name, steps=all_steps, params_model=params_model)
    effective_interface(built)  # eager build-time validation, matching module()
    try:
        topological_steps(built.steps)
    except ValueError as exc:
        raise ValueError(
            f"Branch '{name}': two or more of modifies={list(modifies)} (or "
            f"{path_name}) mutually need each other's value — a genuine, "
            "unresolved gap of this build: an arm reading another "
            "modifies name that a DIFFERENT arm produces creates a real, "
            f"unbreakable cycle once both live in one node. Original "
            f"error: {exc}"
        ) from exc
    return built
