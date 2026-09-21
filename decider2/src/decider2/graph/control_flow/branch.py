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
and three combinators over it", taken literally) whose steps are thin
wrappers around `decider2.graph.control_flow.interpreter.run_program` — see
that module's docstring for why (a generic `@njit` walker over a DATA
program; the condition/arms are dispatched by a small, by-name `call_step`/
`call_cond` switch passed to `run_program` as an ordinary argument, NOT a
raw pointer table — a correction to this stage's original brief, found and
explained in that module's own docstring and this agent's final report)
and this agent's report for the scope cut that makes it possible (an arm
must be a single step, `CALL_ARITY` args each) and what doc 03 §8.2 left
under-specified against a real implementation (arm order -> path index,
`name=` not being optional despite the doc's own examples omitting it).
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
    return_type_word,
    safe_ident,
    single_step,
)
from decider2.graph.control_flow.interpreter import CALL_ARITY, CMP_LIT, LEAF, SET_LIT, ProgramBuilder
from decider2.graph.interface import effective_interface, topological_steps
from decider2.params import build_params_model, harvest_step
from decider2.trees.interpreter import EQ
from decider2.types import Module

__all__ = ["Branch"]

# Same default as decider2.runtime.invoke.DEFAULT_BUILD_DIR — Branch's
# generated source lands in the SAME on-disk cache a kernel's own generated
# source does (doc 05 §4.1).
DEFAULT_BUILD_DIR = Path(".decider2_cache")

_RESULT = "#result"


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
    lines.append("")
    lines.append("This file holds ONLY per-step by-name imports (bounded by step COUNT)")
    lines.append("and this Branch's DATA program arrays — the routing itself is walked")
    lines.append("by ONE shared, hand-written kernel, decider2.graph.control_flow.")
    lines.append("interpreter.run_program, which calls a condition/arm through THIS")
    lines.append("file's own call_step/call_cond switch (passed in as ordinary njit")
    lines.append("function arguments — never a global pointer table: see decider2.")
    lines.append("graph.control_flow.interpreter's module docstring for why that")
    lines.append("distinction is load-bearing here).")
    lines.append('"""')
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("import numpy as np")
    lines.append("from numba import njit, types")
    lines.append("")
    lines.append("from decider2.graph.control_flow.interpreter import run_program")
    lines.append("from decider2.params import param")
    lines.append("")

    # -- per-step by-name imports, one per DISTINCT step used, and the two
    # switches (`call_step`/`call_cond`) every program in this file shares.
    step_thunk_local: dict[int, tuple[str, int]] = {}  # id(fn) -> (local, step_idx)
    step_thunks: list[tuple[object, str]] = []          # (Step, local) in step_idx order
    cond_thunk_local: dict[int, tuple[str, int]] = {}
    cond_thunks: list[tuple[object, str]] = []

    def _register_thunk(step, role_prefix: str, *, is_cond: bool) -> int:
        """Returns this step's `step_idx` in `call_cond`'s switch (`is_cond`)
        or `call_step`'s (otherwise) — a step used twice (e.g. the same arm
        function reused across two Branch calls, or `is_true` reused as
        both `condition` and `should_continue` in a hand-written pipeline)
        gets exactly one by-name import and one switch case."""
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

    cond_slot = _register_thunk(cond_step, "cond", is_cond=not routing)
    arm_slots = [
        _register_thunk(s, f"arm{i}", is_cond=False) for i, s in enumerate(arm_steps)
    ]
    lines.append("")

    def _emit_switch(fn_name: str, table: list, *, is_cond: bool) -> None:
        """`call_step`/`call_cond`: this construct's own small, closed
        `step_idx` switch over DIRECT, by-name calls (`render_switch_
        branch`) — bounded by this construct's own step count, passed into
        `run_program` as an ordinary njit-function ARGUMENT (see this
        module's own docstring, and decider2.graph.control_flow.
        interpreter's, for why that is load-bearing)."""
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

    # -- the path program -----------------------------------------------
    path_fn = f"{name}_path"
    path_rmap = RegisterMap()
    path_builder = ProgramBuilder(n_regs=0)
    if routing:
        result_reg = path_rmap.index_of(_RESULT)
        n_call = encode_call(
            path_builder, path_rmap, cond_step, cond_step.name, cond_slot,
            is_cond=False, dest=result_reg,
        )
        n_leaf = path_builder.add(op=LEAF, dest=result_reg)
        path_builder.set_next(n_call, next_=n_leaf)
        path_start = n_call
    else:
        result_reg = path_rmap.index_of(_RESULT)
        n_cond = encode_call(
            path_builder, path_rmap, cond_step, cond_step.name, cond_slot, is_cond=True,
        )
        n_true = path_builder.add(op=SET_LIT, lit=0.0, dest=result_reg)
        n_true_leaf = path_builder.add(op=LEAF, dest=result_reg)
        n_false = path_builder.add(op=SET_LIT, lit=1.0, dest=result_reg)
        n_false_leaf = path_builder.add(op=LEAF, dest=result_reg)
        path_builder.set_next(n_true, next_=n_true_leaf)
        path_builder.set_next(n_false, next_=n_false_leaf)
        path_builder.set_next(n_cond, next_=n_true, alt=n_false)
        path_start = n_cond
    path_builder.n_regs = path_rmap.n_regs
    path_prog = path_builder.build(start_pc=path_start)

    _emit_program_arrays(lines, "_path", path_prog)

    cond_leaf_sig = [render_leaf_param(i) for i in cond_leaf]
    cond_param_sig, _ = render_prefixed_params(cond_step, cond_step.name)
    lines.append(f"def {path_fn}({', '.join(cond_leaf_sig + cond_param_sig)}) -> int:")
    lines.append(f'    """Which arm {name!r} took, as its 0-based index (doc 03 §7)."""')
    lines += _emit_regs_build(path_rmap, path_prog.n_regs)
    lines.append(
        "    _r = run_program(_path__op, _path__step_idx, _path__arg0, _path__arg1, "
        "_path__arg2, _path__arg3, _path__arg4, _path__arg5, _path__lit, _path__dest, "
        f"_path__next, _path__alt, {path_start}, call_step, call_cond, _regs)"
    )
    lines.append("    return int(_r)")
    lines.append("")

    # -- one program per modifies name -----------------------------------
    mod_fn_names: dict[str, str] = {}
    for mi, mname in enumerate(modifies):
        producers = producers_by_name[mname]
        needs_passthrough = len(producers) < len(arm_steps)
        fn_name = f"{name}_{mname}" if mname != path_fn else f"{name}__{mname}"
        mod_fn_names[mname] = fn_name

        rmap = RegisterMap()
        builder = ProgramBuilder(n_regs=0)
        result_reg = rmap.index_of(_RESULT)
        path_reg = rmap.index_of(path_fn)

        if needs_passthrough:
            passthrough_reg = rmap.index_of(mname)
            fallthrough = builder.add(op=LEAF, dest=passthrough_reg)
        else:
            zero_reg = rmap.index_of("#zero")
            n_zero = builder.add(op=SET_LIT, lit=0.0, dest=zero_reg)
            fallthrough_leaf = builder.add(op=LEAF, dest=zero_reg)
            builder.set_next(n_zero, next_=fallthrough_leaf)
            fallthrough = n_zero

        entry = fallthrough
        # Build backward — the same "highest index first" order
        # `decider2.trees.schema._encode_isin` uses — so producers[0] ends
        # up tested first, matching decider 1's own first-match-wins order
        # (here: arm order, which IS the path index).
        for j in reversed(range(len(producers))):
            arm_idx = producers[j]
            n_call = encode_call(
                builder, rmap, arm_steps[arm_idx], arm_owner[arm_idx], arm_slots[arm_idx],
                is_cond=False, dest=result_reg,
            )
            n_leaf = builder.add(op=LEAF, dest=result_reg)
            builder.set_next(n_call, next_=n_leaf)
            n_test = builder.add(op=CMP_LIT, arg0=path_reg, arg1=EQ, lit=float(arm_idx))
            builder.set_next(n_test, next_=n_call, alt=entry)
            entry = n_test
        builder.n_regs = rmap.n_regs
        prog = builder.build(start_pc=entry)

        arr_prefix = f"_m{mi}"
        _emit_program_arrays(lines, arr_prefix, prog)

        sig_required = [f"{path_fn}: int"]
        sig_defaulted: list[str] = []
        seen_leaf: set[str] = {path_fn}
        if needs_passthrough:
            sig_required.append(f"{mname}: {result_types[mname]}")
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
        lines += _emit_regs_build(rmap, prog.n_regs)
        lines.append(
            f"    _r = run_program({arr_prefix}__op, {arr_prefix}__step_idx, {arr_prefix}__arg0, "
            f"{arr_prefix}__arg1, {arr_prefix}__arg2, {arr_prefix}__arg3, {arr_prefix}__arg4, "
            f"{arr_prefix}__arg5, {arr_prefix}__lit, {arr_prefix}__dest, {arr_prefix}__next, "
            f"{arr_prefix}__alt, {entry}, call_step, call_cond, _regs)"
        )
        lines.append(f"    {_return_cast(result_types[mname])}")
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


def _return_cast(result_type: str) -> str:
    if result_type == "bool":
        return "return _r != 0.0"
    if result_type == "int":
        return "return int(_r)"
    return "return _r"


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


def _emit_regs_build(rmap: "RegisterMap", n_regs: int) -> list[str]:
    out = [f"    _regs = np.zeros({n_regs})"]
    for reg_name, idx in rmap.names.items():
        if reg_name.startswith("#"):
            continue  # a reserved, program-internal slot — never a wrapper arg
        out.append(f"    _regs[{idx}] = {reg_name}")
    return out
