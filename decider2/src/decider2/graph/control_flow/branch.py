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
and three combinators over it", taken literally) whose steps are REAL,
content-addressed, njit-able source — see `_engine.py`'s module docstring
for why (inline codegen, not a `Pipeline.score()` delegate) and this
agent's final report for the scope cut that makes it possible (an arm must
be a single step) and what doc 03 §8.2 left under-specified against a real
implementation (arm order -> path index, `name=` not being optional despite
the doc's own examples omitting it).
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
    return_type_word,
    single_step,
)
from decider2.graph.interface import effective_interface, topological_steps
from decider2.params import build_params_model, harvest_step
from decider2.types import Module

__all__ = ["Branch"]

# Same default as decider2.runtime.invoke.DEFAULT_BUILD_DIR — Branch's
# generated source lands in the SAME on-disk cache a kernel's own generated
# source does (doc 05 §4.1).
DEFAULT_BUILD_DIR = Path(".decider2_cache")


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
    exact convention `decider2.trees.codegen`'s `result_idx` already uses,
    mirrored rather than reinvented.
    """
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
        first_ann = return_type_word(arm_steps[producers[0]].fn)
        for i in producers[1:]:
            ann = return_type_word(arm_steps[i].fn)
            if ann != first_ann:
                raise ValueError(
                    f"Branch '{name}': arms disagree on the type of "
                    f"'{mname}' — arm {producers[0]} produces {first_ann!r}, "
                    f"arm {i} produces {ann!r}. Doc 03 §8.2: an arm that "
                    "produces a modifies value must agree on type with "
                    "every other arm that does."
                )
        result_types[mname] = first_ann
        producers_by_name[mname] = producers

    # doc 05 §1.5's dictionary-code boundary now round-trips correctly:
    # the condition/arm dispatchers below receive the SAME kernel argument
    # the outer driver already extracted (an int32 code for a str column),
    # never re-encoded through a second, independent path.
    cond_leaf = [i for i in cond_step.inputs]
    merge_inputs([("condition", cond_leaf)])  # validates: required-only

    resolved_build_dir: "str | Path" = build_dir if build_dir is not None else DEFAULT_BUILD_DIR

    lines: list[str] = []
    lines.append(f'"""Generated Branch kernel for {name!r} (doc 03 §8.2).')
    lines.append("")
    lines.append("Content-addressed, real source — decider2.compile.cache. Do not")
    lines.append("hand-edit; regenerate by rebuilding the Branch(...) call instead.")
    lines.append('"""')
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("from numba import njit")
    lines.append("from decider2.params import param")
    lines.append("")
    lines += import_and_wrap(cond_step, "_cond_c")
    for i, s in enumerate(arm_steps):
        lines += import_and_wrap(s, f"_arm{i}_c")
    lines.append("")

    path_fn = f"{name}_path"
    cond_leaf_sig = [render_leaf_param(i) for i in cond_leaf]
    cond_param_sig, _ = render_prefixed_params(cond_step, cond_step.name)
    lines.append(f"def {path_fn}({', '.join(cond_leaf_sig + cond_param_sig)}) -> int:")
    lines.append(f'    """Which arm {name!r} took, as its 0-based index (doc 03 §7).')
    lines.append('    """')
    cond_call_args = render_call_args(cond_step, cond_step.name)
    lines.append(f"    _cond_v = _cond_c({', '.join(cond_call_args)})")
    if routing:
        lines.append("    return int(_cond_v)")
    else:
        lines.append("    if _cond_v:")
        lines.append("        return 0")
        lines.append("    return 1")
    lines.append("")

    mod_fn_names: dict[str, str] = {}
    for mname in modifies:
        producers = producers_by_name[mname]
        needs_passthrough = len(producers) < len(arm_steps)
        fn_name = f"{name}_{mname}" if mname != path_fn else f"{name}__{mname}"
        mod_fn_names[mname] = fn_name

        sig_required = [f"{path_fn}: int"]
        sig_defaulted: list[str] = []
        seen_leaf: set[str] = set()
        if needs_passthrough:
            sig_required.append(f"{mname}: {result_types[mname]}")
            # An arm's own leaf input is very often the SAME name as the
            # modifies value it narrows (e.g. cap_for_private reads
            # `term_cap` too) — already declared above, so it must not be
            # re-declared as a duplicate parameter below.
            seen_leaf.add(mname)
        for i in producers:
            s = arm_steps[i]
            for inp in s.inputs:
                if inp.name in seen_leaf:
                    continue
                seen_leaf.add(inp.name)
                sig_required.append(render_leaf_param(inp))
            p_sig, _ = render_prefixed_params(s, arm_owner[i])
            sig_defaulted.extend(p_sig)

        lines.append(f"def {fn_name}({', '.join(sig_required + sig_defaulted)}) -> {result_types[mname]}:")
        lines.append(f'    """`{mname}` for whichever arm {name!r} took."""')
        for j, i in enumerate(producers):
            branch_kw = "if" if j == 0 else "elif"
            lines.append(f"    {branch_kw} {path_fn} == {i}:")
            call_args = render_call_args(arm_steps[i], arm_owner[i])
            lines.append(f"        return _arm{i}_c({', '.join(call_args)})")
        if needs_passthrough:
            lines.append(f"    return {mname}  # doc 03 §8.2: a silent arm passes it through")
        else:
            zero = {"float": "0.0", "int": "0", "bool": "False", "str": '""'}[result_types[mname]]
            lines.append(f"    return {zero}  # unreachable: every arm produces {mname}")
        lines.append("")

    source = "\n".join(lines) + "\n"
    cached = cache.get_or_build(source, resolved_build_dir)

    path_step = harvest_step(getattr(cached.module, path_fn), name=path_fn)
    modifies_steps = [
        harvest_step(getattr(cached.module, mod_fn_names[m]), name=m) for m in modifies
    ]

    all_steps = tuple(modifies_steps) + (path_step,)
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
            f"{name}_path) mutually need each other's value — a genuine, "
            "unresolved gap of this build: an arm reading another "
            "modifies name that a DIFFERENT arm produces creates a real, "
            f"unbreakable cycle once both live in one node. Original "
            f"error: {exc}"
        ) from exc
    return built
