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
`decider2.runtime` — and get real branching and a real bounded loop with
early exit in compiled machine code. What each construct's own file
CONTAINS is no longer nested `if`/`elif`/`while` source, though (see
`decider2.graph.control_flow.interpreter`'s module docstring): it is a flat
DATA program, walked by one shared kernel — the same shape `decider2.trees.
interpreter`/`decider2.tables.interpreter` give a tree's or table's own
condition tests, generalised here to arbitrary condition/arm/body STEPS by
a small, by-name `call_step`/`call_cond` switch (bounded by this
construct's own step count) rather than an inline comparison.

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
from dataclasses import dataclass
from typing import Any, Sequence

from decider2.graph.control_flow.interpreter import CALL_ARITY, COND, STEP, ProgramBuilder
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
    "RegisterMap",
    "call_arg_regs",
    "encode_call",
    "render_step_thunk",
    "render_switch_branch",
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


@dataclass
class RegisterMap:
    """One Branch/Loop construct's name -> register-index assignment,
    built incrementally as the construct is encoded (`encode_call` below) —
    the data-walker's analogue of the local variable names a generated
    `if`/`while`'s own source text used to carry every value under.
    De-duplicates by name: a leaf input and a carry sharing one name (the
    self-read waterfall idiom, doc 03 §3.2, one level up) share one
    register, the same way they used to share one Python local.
    """

    names: "dict[str, int]" = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.names is None:
            self.names = {}

    def index_of(self, name: str) -> int:
        if name not in self.names:
            self.names[name] = len(self.names)
        return self.names[name]

    def __contains__(self, name: str) -> bool:
        return name in self.names

    @property
    def n_regs(self) -> int:
        return max(len(self.names), 1)  # never zero (see interpreter.py's tuple note)


def call_arg_regs(step: Step, prefix: str, rmap: RegisterMap) -> list[int]:
    """`step`'s own function's parameters, in its OWN declared order, as
    REGISTER INDICES: a leaf input (or `loop_idx`) by its plain
    (shared-by-name) identifier, a `param()` field by its `prefix`-
    qualified one (matching `render_prefixed_params`) — the register-array
    analogue of the old `render_call_args`'s local-name list.

    Raises when `step` needs more than `decider2.graph.control_flow.
    interpreter.CALL_ARITY` arguments — `call_step`/`call_cond`
    (`render_switch_branch`) share one fixed-width signature per
    construct, a real, reported scope cut mirroring `single_step`'s
    "exactly one step" limit.
    """
    param_names = {p.name for p in step.params}
    try:
        sig = inspect.signature(step.fn, eval_str=True)
    except (NameError, TypeError):
        sig = inspect.signature(step.fn)
    out: list[int] = []
    for pname in sig.parameters:
        if pname in param_names:
            out.append(rmap.index_of(safe_ident(f"{prefix}__{pname}")))
        else:
            out.append(rmap.index_of(safe_ident(pname)))
    if len(out) > CALL_ARITY:
        raise ValueError(
            f"step '{step.name}' (as used in this Branch/Loop) takes "
            f"{len(out)} arguments (leaf inputs plus param() fields "
            f"combined), over this build's {CALL_ARITY}-argument limit per "
            "condition/arm/body step — decider2.graph.control_flow."
            "interpreter's call_step/call_cond dispatch shares one fixed-"
            "width signature per construct. Split the step, or give it "
            "fewer inputs/params."
        )
    return out


def encode_call(
    builder: ProgramBuilder, rmap: RegisterMap, step: Step, prefix: str, step_idx: int,
    *, is_cond: bool, dest: "int | None" = None,
) -> int:
    """Append one `STEP`/`COND` node calling `step` (already wired as case
    `step_idx` of this construct's `call_step`/`call_cond` switch — see
    `render_switch_branch`), reading its args from `rmap` and — for a
    `STEP`, never a `COND`, which has no result to store — writing to
    register `dest`. Args are padded to `CALL_ARITY` with register 0 (never
    read: `render_switch_branch`'s own case only forwards as many
    positional arguments as the step declares). Returns the new node's
    index, with `next_`/`alt` left for the caller to wire via
    `ProgramBuilder.set_next` (no back-patching needed the way
    `decider2.trees.codegen` avoids it either: caller decides `next_`/`alt`
    before or after this call as convenient, since a `Branch`/`Loop`
    program's shape is built top-down from data already in hand, not from
    a text walk that has to know a target's line number in advance).
    """
    arg_regs = call_arg_regs(step, prefix, rmap)
    padded = arg_regs + [0] * (CALL_ARITY - len(arg_regs))
    kwargs: dict = {f"arg{i}": r for i, r in enumerate(padded)}
    op = COND if is_cond else STEP
    if not is_cond:
        kwargs["dest"] = dest
    return builder.add(op=op, step_idx=step_idx, **kwargs)


def render_step_thunk(step: Step, local_name: str) -> list[str]:
    """One step's real function, imported and `njit`-wrapped (the same
    pattern `decider2.compile.codegen.emit_kernel_source` uses to call one
    step's compiled dispatcher from another njit'd function — a step
    defined as a closure fails this import, the same pre-existing
    limitation `compile.driver` has for a fused kernel). This is the
    entirety of the ABI-bridging text this build emits per step-use —
    bounded by step COUNT, never by program size, nesting depth or
    iteration count — because `_{local_name}_njit` is called BY NAME from
    `call_step`/`call_cond`'s own switch (`render_switch_branch`), never
    through a pointer.
    """
    return [
        f"from {step.fn.__module__} import {step.fn.__name__} as _{local_name}_src",
        f"_{local_name}_njit = njit(cache=True)(_{local_name}_src)",
    ]


def render_switch_branch(step: Step, local_name: str, step_idx: int, *, is_cond: bool) -> str:
    """One `elif step_idx == N: return ...` line of `call_step`'s or
    `call_cond`'s own switch — casting each of `call_step`'s fixed `a0..
    a{CALL_ARITY-1}` float64 arguments down to the step's own declared
    type on the way in (only as many as the step actually declares — the
    padded, unused trailing ones are never referenced), and the real
    function's own return value back to float64 (or, for a COND, boolean —
    already the right type, never cast) on the way out — the same
    float64-in/float64-out convention `decider2.trees.interpreter` gives
    every register.
    """
    try:
        sig = inspect.signature(step.fn, eval_str=True)
    except (NameError, TypeError):
        sig = inspect.signature(step.fn)
    param_by_name = {p.name: p for p in step.params}
    input_by_name = {i.name: i for i in step.inputs}

    arg_words: list[str] = []
    for pname in sig.parameters:
        if pname in param_by_name:
            arg_words.append(type_word(param_by_name[pname].annotation))
        else:
            ann = input_by_name[pname].annotation if pname in input_by_name else Any
            arg_words.append(type_word(ann))

    thunk_args = [f"a{i}" for i in range(len(arg_words))]
    cast_in = [
        f"int({nm})" if w == "int" else (f"({nm} != 0.0)" if w == "bool" else nm)
        for nm, w in zip(thunk_args, arg_words)
    ]
    call_expr = f"_{local_name}_njit({', '.join(cast_in)})"

    if is_cond:
        ret_expr = call_expr
    else:
        ret_word = return_type_word(step.fn)
        if ret_word == "bool":
            ret_expr = f"(1.0 if {call_expr} else 0.0)"
        elif ret_word == "int":
            ret_expr = f"types.float64({call_expr})"
        else:
            ret_expr = call_expr

    return f"        return {ret_expr}"
