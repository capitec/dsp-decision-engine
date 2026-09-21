"""Shared machinery for `Branch` (`branch.py`) and `Loop` (`loop.py`) — doc
03 §8.2/§8.3, §8.4 ("There is one type — Module — and three combinators over
it").

**Inline codegen, not delegation.** Each construct emits ONE real, content-
addressed source file (`decider2.compile.cache.get_or_build`, the same
mechanism `decider2.trees.codegen` writes kernel source through) containing
plain, undecorated Python functions — never a closure that calls
`Pipeline.score()` per row. That earlier design was tried and rejected: doc
03 §8.2's entire motivation for `Branch` is "only the taken arm executes in
compiled machine code", and a per-row `.score()` call (EXPERIMENTS.md §R/§S:
350-1500 µs/call — dominated by marshalling, not the kernel) is ~1000×
slower than the sequential-both-arms code it is supposed to beat. Emitting
real source instead means the generated functions njit-compile normally
through the SAME `decider2.compile.driver._try_njit`/`build_driver` path
every other step goes through — no changes to `decider2.compile`/
`decider2.runtime` — and get real branching (`if`/`elif`) and a real
bounded `while`/`break` in compiled machine code, exactly like
`decider2.trees.codegen` already does for a tree's nested conditions.

**Scope, stated once here.** Two arms with colliding internal names (doc 03
§8.2: "each arm is its own scope") is the risk `decider2.trees.codegen`
solves by qualifying every emitted identifier. This build takes the
narrower, still-useful cut of that fix: a `Branch` arm and a `Loop`
`should_continue` must each be exactly ONE step (matching every example in
doc 03 §8.2/§8.3 — `CapForPrivate`/`CapForPublic`/`is_private_sector` are
all single-step). A single step cannot collide with itself, so no
qualification scheme is needed for them at all; `Loop`'s `body`, which has
no sibling to collide with, is NOT under this limit and may be an arbitrary
multi-step `Module`/`Pipeline` — its own steps already form a valid,
collision-free `decider2` DAG by construction (doc 03 §3.1). Multi-step
arms/should_continue are a real, reportable narrowing from full generality;
see this agent's final report.

Every generated step's `param()` fields are namespaced by the arm/body's own
module instance name, prefixed onto the field (`arm0__cap`), so two arms
declaring a same-named-but-different-meaning tunable never collide —
doc 03 §4.1's "params are namespaced by module instance name", applied one
level up, the same fix `decider2.trees.codegen.EmitContext.threshold`
applies via `{node_id}_{role}`.
"""
from __future__ import annotations

import inspect
import keyword
import re
from typing import Any, Sequence

from decider2.graph.pipeline import Pipeline, flow
from decider2.types import Input, NullPolicy, Step

__all__ = [
    "normalize",
    "single_step",
    "single_terminal",
    "safe_ident",
    "type_word",
    "return_type_word",
    "merge_inputs",
    "render_leaf_param",
    "render_prefixed_params",
    "render_call_args",
    "import_and_wrap",
]

_IDENT_BAD = re.compile(r"[^0-9A-Za-z_]")


def safe_ident(name: str) -> str:
    """Same job as `decider2.compile.codegen.safe_ident`/
    `decider2.trees.codegen.safe_ident`; a third, tiny copy rather than an
    import across layers this package (the graph layer) does not otherwise
    depend on."""
    ident = _IDENT_BAD.sub("_", name)
    if not ident or ident[0].isdigit():
        ident = f"v_{ident}"
    if keyword.iskeyword(ident):
        ident = f"{ident}_"
    return ident


def normalize(element: Any) -> Pipeline:
    """A Branch/Loop condition, arm or body may be a bare function, a
    `Step`, a `Module` or a `Pipeline` — the same `Element` union `flow()`
    itself accepts (doc 03 §8.4: Branch/Loop compose the same machinery
    everything else does). This is also how nesting works: `Branch(...)`/
    `Loop(...)` return a plain `Module`, itself a valid `element` here, so
    `Loop(Branch(...))` needs nothing special.
    """
    if isinstance(element, Pipeline):
        return element
    return flow(element)


def single_step(pipeline: Pipeline, *, role: str) -> Step:
    """This build's scope cut (see module docstring): `role` must be
    exactly one step — one module, one step in it."""
    steps = [s for m in pipeline.elements for s in m.steps]
    if len(steps) != 1:
        names = [s.name for s in steps]
        raise ValueError(
            f"{role} must be exactly one step in this build (doc 03 §8.2/"
            f"§8.3's own examples are all single-step); got {len(steps)}: "
            f"{names or '(none)'}. Wrap it as `module(one_fn)` — a "
            "multi-step arm/should_continue is a real, reported limitation "
            "of this pass, not a spec requirement."
        )
    return steps[0]


def single_terminal(pipeline: Pipeline, *, role: str) -> str:
    terminals = pipeline.interface.terminals
    if len(terminals) != 1:
        raise ValueError(
            f"{role} must produce exactly one value nothing else consumes "
            f"(doc 03 §8.2/§8.3's \"condition step -> bool\"); got "
            f"{len(terminals)}: {sorted(terminals)}."
        )
    return terminals[0]


_TYPE_WORDS: dict[Any, str] = {float: "float", int: "int", bool: "bool", str: "str"}


def type_word(annotation: Any) -> str:
    """The generated source's spelling for a declared annotation.

    Unannotated (`typing.Any`) falls back to `"float"` — the same default
    `decider2.compile.driver.numpy_dtype`/`_numba_type` already give an
    undeclared column (doc 05 §1.5).
    """
    return _TYPE_WORDS.get(annotation, "float")


def _return_annotation(fn: Any) -> Any:
    try:
        sig = inspect.signature(fn, eval_str=True)
    except (NameError, TypeError):
        sig = inspect.signature(fn)
    ann = sig.return_annotation
    return float if ann is inspect.Signature.empty else ann


def return_type_word(fn: Any) -> str:
    return type_word(_return_annotation(fn))


def _require_required(inp: Input, *, role: str) -> None:
    if inp.null_policy is not NullPolicy.REQUIRED:
        raise ValueError(
            f"{role}: input '{inp.name}' declares a null policy other than "
            "required (missing_as()/not_applicable_as()/Optional). This "
            "codegen pass only supports plain required leaf inputs across "
            "a Branch/Loop node boundary — a real, reported scope cut, not "
            "a spec requirement."
        )


def merge_inputs(groups: Sequence[tuple[str, Sequence[Input]]]) -> list[Input]:
    """`groups` is `[(role_name, inputs), ...]`. The union, deduplicated by
    name — same name in two groups must agree on type (doc 03 §2's
    ordinary "same name, same meaning" convention, applied one level up).
    Every input must be a plain required leaf (see `_require_required`).
    """
    merged: dict[str, Input] = {}
    origin: dict[str, str] = {}
    for role, inputs in groups:
        for inp in inputs:
            _require_required(inp, role=role)
            existing = merged.get(inp.name)
            if existing is None:
                merged[inp.name] = inp
                origin[inp.name] = role
                continue
            if existing.annotation != inp.annotation:
                raise ValueError(
                    f"input '{inp.name}' is declared differently by "
                    f"'{origin[inp.name]}' ({existing.annotation}) and "
                    f"'{role}' ({inp.annotation}). Same name must mean the "
                    "same thing (doc 03 §2) — rename one."
                )
    return list(merged.values())


def render_leaf_param(inp: Input) -> str:
    """One required leaf as a `def` parameter — no default."""
    return f"{safe_ident(inp.name)}: {type_word(inp.annotation)}"


def render_prefixed_params(step: Step, prefix: str) -> tuple[list[str], list[str]]:
    """A step's own `param()` fields, namespaced by `prefix` (the arm/body's
    module instance name) so two arms' same-named-but-different tunables
    never collide in one generated function (see module docstring).

    Returns `(sig_parts, call_arg_names)` — `sig_parts` for the outer
    generated function's own signature (`arm0__cap: float = param(48.0)`),
    `call_arg_names` the LOCAL names to pass positionally into the arm's
    own compiled dispatcher, in the step's own declared parameter order.
    """
    sig_parts: list[str] = []
    call_names: list[str] = []
    for decl in step.params:
        local = safe_ident(f"{prefix}__{decl.name}")
        sig_parts.append(f"{local}: {type_word(decl.annotation)} = param({decl.default!r})")
        call_names.append(local)
    return sig_parts, call_names


def render_call_args(step: Step, prefix: str) -> list[str]:
    """`step`'s own function's parameters, in its OWN declared order, as
    the local names a generated caller should pass positionally: a leaf
    input by its plain (shared-by-name) identifier, a `param()` field by
    its `prefix`-qualified one (matching `render_prefixed_params`)."""
    param_names = {p.name for p in step.params}
    try:
        sig = inspect.signature(step.fn, eval_str=True)
    except (NameError, TypeError):
        sig = inspect.signature(step.fn)
    out: list[str] = []
    for pname in sig.parameters:
        if pname in param_names:
            out.append(safe_ident(f"{prefix}__{pname}"))
        else:
            out.append(safe_ident(pname))
    return out


def import_and_wrap(step: Step, local_name: str) -> list[str]:
    """Two lines: import a step's own function by its real module path and
    wrap it with `njit(cache=True)` under `local_name` — the exact pattern
    `decider2.compile.codegen.emit_kernel_source` already uses to call one
    step's compiled dispatcher from another njit'd function. A step
    defined as a closure (no module-level name) fails this import; that is
    the same, pre-existing limitation `compile.codegen` has for a fused
    kernel, not something new here.
    """
    return [
        f"from {step.fn.__module__} import {step.fn.__name__} as _{local_name}_src",
        f"{local_name} = njit(cache=True)(_{local_name}_src)",
    ]
