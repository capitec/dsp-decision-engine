"""Shared machinery for `Branch` (`branch.py`) and `Loop` (`loop.py`) — doc
03 §8.2/§8.3, §8.4 ("There is one type — Module — and three combinators over
it").

**No generated source, not even glue.** A construct used to emit ONE real,
content-addressed `.py` file (by-name imports of its callees, a rendered
`call_step`/`call_cond` switch, one `def` per output with a named parameter
per leaf/param) and `njit` it through `decider2.compile.driver` like any
other step. `types.Step` is a frozen dataclass whose `inputs`/`params` are
already data — nothing forces them to come from `inspect.signature` — so
every construct output is now built here directly as a `Step(packed=True)`
(`build_packed_step`): `fn` a real closure over the output's own program
arrays and its construct's walker (`decider2.graph.control_flow.
interpreter`), `inputs`/`params` built straight from the encoder's own
bookkeeping. Exactly what `decider2.trees.encode` and `decider2.tables.
encode` already do for a tree's/table's own steps, and the same `packed`
calling convention `decider2.compile.driver` already drives them through
in every execution mode.

**Registers are laid out by the wrapper's own signature.** `RegisterMap`
is seeded with the output step's `inputs` (registers `0..N-1`) and
`params` (`N..N+P-1`) BEFORE any node is encoded, so `interpreter._seed_
regs` can scatter `(args, params)` into place with one runtime loop and
no per-name code; every other register (a carry's `loop_idx`, an arm's
result, a non-carry body output) is appended after. `Branch`/`Loop` decide
the wrapper signature first and encode second — the reverse of the
text-rendering build, which discovered its registers during the encode
and only then rendered a signature to match.

**Input annotations across a construct boundary.** `decider2.compile.
driver._packed_args_kind` gathers a packed step's inputs into ONE
homogeneous float64 array when there is more than one of them, and
requires every such input to be float-annotated (exactly one input of any
other type is passed through untouched as a 1-tuple, `"raw1"`). A
construct output therefore declares each of its inputs with the callee's
own annotation when it is the step's ONLY input, and as `float` otherwise
(`boundary_inputs`) — a bool/int leaf is exactly representable in float64
and the register array is float64 regardless, so nothing changes what any
callee actually receives (`interpreter.make_call_adapter` casts each
argument back to the callee's declared type). A `str` leaf must stay
`str` (that is how the boundary knows to dictionary-encode the column and
`runtime.invoke._resolve_str_param_code` knows which categories a `str`
param resolves against), so a construct output reading a `str` leaf
ALONGSIDE another input is the one shape the driver's gather cannot hold
today — rejected here at build time with the exact `compile/` change that
would lift it (see `boundary_inputs`), never silently miscast.

**Scope, stated once here.** A `Branch` arm and a `Loop` `should_continue`
must each be exactly ONE step (matching every example in doc 03 §8.2/§8.3),
so no per-arm identifier qualification scheme is needed; `Loop`'s `body`
may be an arbitrary multi-step `Module`/`Pipeline`. Every callee's `param()`
fields are namespaced by the arm/body's own module instance name, prefixed
onto the field (`arm0__cap`, `prefixed_param_decls`) — doc 03 §4.1's "params
are namespaced by module instance name", applied one level up — and the
original pydantic `FieldInfo` is forwarded verbatim, so `ge=`/`le=` bounds
survive the boundary (the rendered `param({default!r})` used to drop them).
"""
from __future__ import annotations

import dataclasses
import inspect
import keyword
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from decider2.compile.driver import _packed_args_kind, step_return_annotation
from decider2.graph.control_flow.interpreter import (
    COND,
    STEP,
    Program,
    ProgramBuilder,
    make_call_adapter,
    make_packed_adapter,
    make_step_fn,
    make_walker,
)
from decider2.graph.pipeline import Pipeline, flow
from decider2.types import Input, NullPolicy, ParamDecl, Step

__all__ = [
    "normalize",
    "single_step",
    "safe_ident",
    "callee_annotation",
    "output_annotation",
    "merge_inputs",
    "boundary_inputs",
    "prefixed_param_decls",
    "RegisterMap",
    "NodeTable",
    "call_arg_regs",
    "encode_call",
    "build_packed_step",
]

_IDENT_BAD = re.compile(r"[^0-9A-Za-z_]")


def safe_ident(name: str) -> str:
    """A valid identifier for a prefixed `ParamDecl` name — still needed
    even though nothing here writes source any more: a param name is a
    pydantic field name (`build_params_model`), reported by `params_
    schema()`/`GET /params/schema`. Same job as `decider2.trees.encode.
    safe_ident`; a tiny copy rather than an import across layers."""
    ident = _IDENT_BAD.sub("_", name)
    if not ident or ident[0].isdigit():
        ident = f"v_{ident}"
    if keyword.iskeyword(ident):
        ident = f"{ident}_"
    return ident


def normalize(element: Any) -> Pipeline:
    """A Branch/Loop condition, arm or body may be a bare function, a
    `Step`, a `Module` or a `Pipeline` — the same `Element` union `flow()`
    itself accepts (doc 03 §8.4). This is also how nesting works:
    `Branch(...)`/`Loop(...)` return a plain `Module`, itself a valid
    `element` here, so `Loop(Branch(...))` needs nothing special.
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


def _signature(fn: Any) -> inspect.Signature:
    try:
        return inspect.signature(fn, eval_str=True)
    except (NameError, TypeError):
        return inspect.signature(fn)


_TYPES: tuple[Any, ...] = (float, int, bool, str)


def callee_annotation(annotation: Any) -> Any:
    """A declared annotation normalised to the four a kernel boundary
    knows (`decider2.compile.driver._NUMBA_BY_ANNOTATION`); anything else
    (unannotated `typing.Any`, an exotic alias) is float64, the boundary's
    own default for an undeclared value (doc 05 §1.5)."""
    return annotation if annotation in _TYPES else float


def output_annotation(step: Step) -> Any:
    """What `step` produces, as one of the four boundary types — read off
    `Step.output_annotation` for a packed step (a nested construct's own
    output), else its function's return annotation (`decider2.compile.
    driver.step_return_annotation`, the one reader every mode shares)."""
    ann = step_return_annotation(step)
    return callee_annotation(float if ann is inspect.Signature.empty else ann)


def _require_required(inp: Input, *, role: str) -> None:
    if inp.null_policy is not NullPolicy.REQUIRED:
        raise ValueError(
            f"{role}: input '{inp.name}' declares a null policy other than "
            "required (missing_as()/not_applicable_as()/Optional). A "
            "Branch/Loop only supports plain required leaf inputs across "
            "its node boundary — a real, reported scope cut, not a spec "
            "requirement."
        )


def merge_inputs(groups: Sequence[tuple[str, Sequence[Input]]]) -> list[Input]:
    """`groups` is `[(role_name, inputs), ...]`. The union, deduplicated by
    name in first-seen order — same name in two groups must agree on type
    (doc 03 §2's ordinary "same name, same meaning" convention, applied one
    level up). Every input must be a plain required leaf.
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
            if callee_annotation(existing.annotation) != callee_annotation(inp.annotation):
                raise ValueError(
                    f"input '{inp.name}' is declared differently by "
                    f"'{origin[inp.name]}' ({existing.annotation}) and "
                    f"'{role}' ({inp.annotation}). Same name must mean the "
                    "same thing (doc 03 §2) — rename one."
                )
    return list(merged.values())


def boundary_inputs(inputs: Sequence[Input], *, construct: str, output: str) -> tuple[Input, ...]:
    """`inputs` as a construct output step declares them to the boundary —
    see the module docstring: the callee's own annotation for a single
    input, `float` for every non-`str` input otherwise, `str` always kept.
    Checked against `decider2.compile.driver._packed_args_kind`, the exact
    rule the fused row-gather applies, so the one unrepresentable shape
    fails here at build time rather than three frames deep in a kernel
    build."""
    if len(inputs) == 1:
        out = (dataclasses.replace(inputs[0], annotation=callee_annotation(inputs[0].annotation)),)
    else:
        out = tuple(
            dataclasses.replace(inp, annotation=str if inp.annotation is str else float)
            for inp in inputs
        )
    try:
        _packed_args_kind([inp.annotation for inp in out])
    except ValueError as exc:
        str_names = sorted(inp.name for inp in out if inp.annotation is str)
        raise ValueError(
            f"{construct}: output '{output}' reads str-typed leaf(s) "
            f"{str_names} alongside {len(out) - len(str_names)} other "
            "input(s). decider2.compile.driver._packed_args_kind gathers a "
            "multi-input packed step into one float64 array and requires "
            "every input to be float-annotated, while a str leaf must stay "
            "declared `str` for the boundary to dictionary-encode it (doc "
            "05 §1.5) — so this shape has no row-gather today. Lifting it "
            "is a compile/driver.py change (let `_packed_args_kind`/"
            "`_packed_input_arrays`/`_packed_row_args` cast a str column's "
            "int32 codes to float64 alongside the floats), not a Branch/"
            "Loop one. Until then, hoist the str comparison into its own "
            "single-input step before the construct and branch on its "
            f"result. Original error: {exc}"
        ) from exc
    return out


def prefixed_param_decls(step: Step, prefix: str) -> list[ParamDecl]:
    """A step's own `param()` fields, namespaced by `prefix` (the arm/body's
    module instance name) so two arms' same-named-but-different tunables
    never collide in one construct's params model (module docstring).
    `field_info` — the harvested pydantic `FieldInfo`, bounds and all — is
    forwarded verbatim."""
    return [
        ParamDecl(
            name=safe_ident(f"{prefix}__{decl.name}"),
            annotation=decl.annotation if decl.annotation is not Any else type(decl.default),
            default=decl.default,
            field_info=decl.field_info,
        )
        for decl in step.params
    ]


@dataclass
class RegisterMap:
    """One construct output's name -> register-index assignment. Seeded
    from the output step's own `inputs` (registers `0..N-1`, by input
    name) and `params` (`N..N+P-1`, by prefixed param name) — exactly the
    order `interpreter._seed_regs` scatters `(args, params)` in — before
    any node is encoded; `internal()` appends everything else (`loop_idx`,
    an arm's result, a non-carry body output). De-duplicates by name: a
    leaf input and a carry sharing one name (the self-read waterfall idiom,
    doc 03 §3.2, one level up) share one register."""

    inputs: Sequence[Input]
    params: Sequence[ParamDecl]
    names: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for inp in self.inputs:
            self._seed(inp.name, "input")
        for decl in self.params:
            self._seed(decl.name, "param")

    def _seed(self, name: str, kind: str) -> None:
        if name in self.names:
            raise ValueError(
                f"RegisterMap: {kind} '{name}' registered twice — the "
                "encoder built a wrapper signature with a duplicate name."
            )
        self.names[name] = len(self.names)

    def internal(self, name: str) -> int:
        """A register no wrapper argument seeds (starts at 0.0)."""
        if name not in self.names:
            self.names[name] = len(self.names)
        return self.names[name]

    def lookup(self, name: str) -> int:
        try:
            return self.names[name]
        except KeyError:
            raise KeyError(
                f"RegisterMap: '{name}' has no register — the encoder read "
                "a name it never declared as an input, param or internal."
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self.names

    @property
    def n_regs(self) -> int:
        return max(len(self.names), 1)


def _callee_arg_names(step: Step, prefix: str) -> list[str]:
    """`step`'s arguments as REGISTER NAMES, in the order its adapter reads
    them: a plain callee's own declared parameter order (a leaf input or
    `loop_idx` by its plain name, a `param()` field by its `prefix`-
    qualified one, matching `prefixed_param_decls`); a packed callee's
    `inputs` then `params` (`types.Step.packed`)."""
    param_names = {p.name for p in step.params}
    if step.packed:
        return [i.name for i in step.inputs] + [safe_ident(f"{prefix}__{p.name}") for p in step.params]
    out: list[str] = []
    for pname in _signature(step.fn).parameters:
        out.append(safe_ident(f"{prefix}__{pname}") if pname in param_names else pname)
    return out


def _callee_arg_annotations(step: Step) -> list[Any]:
    """A plain callee's argument annotations, in declared parameter order
    — what `interpreter.make_call_adapter` casts each register to."""
    param_by_name = {p.name: p for p in step.params}
    input_by_name = {i.name: i for i in step.inputs}
    out: list[Any] = []
    for pname in _signature(step.fn).parameters:
        if pname in param_by_name:
            decl = param_by_name[pname]
            out.append(callee_annotation(decl.annotation if decl.annotation is not Any else type(decl.default)))
        elif pname in input_by_name:
            out.append(callee_annotation(input_by_name[pname].annotation))
        else:
            out.append(float)
    return out


def _check_callee(step: Step, *, role: str) -> None:
    if step.reads_params or step.reads_shared:
        raise ValueError(
            f"{role}: step '{step.name}' reads a bare `params`/`shared` "
            "bundle (doc 03 §4.2). A Branch/Loop passes each callee its "
            "values as individual registers and has no bundle to hand it "
            "— declare the knobs as `param()` fields instead (a real, "
            "reported scope cut)."
        )
    for inp in step.inputs:
        _require_required(inp, role=f"{role}: step '{step.name}'")


class NodeTable:
    """One construct's `STEP`/`COND` node adapters (`interpreter.
    make_call_adapter`/`make_packed_adapter`), in first-registration order
    — the `step_idx` a node carries. De-duplicated by (callee function,
    registers read): the same arm at the same registers in two programs,
    or `is_true` reused as both `condition` and `should_continue` at one
    layout, gets ONE adapter. A callee is validated once (`_check_callee`),
    however many nodes call it. `walker()` builds the construct's shared
    kernel over the adapters once every program is encoded."""

    def __init__(self) -> None:
        self._index: dict[tuple, int] = {}
        self._checked: set[int] = set()
        self.adapters: list[Callable] = []

    def register(self, step: Step, arg_regs: Sequence[int], *, role: str) -> int:
        key = (id(step.fn), tuple(arg_regs))
        if key in self._index:
            return self._index[key]
        if id(step.fn) not in self._checked:
            _check_callee(step, role=role)
            self._checked.add(id(step.fn))
        if step.packed:
            input_annotations = [i.annotation for i in step.inputs]
            adapter = make_packed_adapter(
                step.fn,
                input_annotations,
                [p.annotation if p.annotation is not Any else type(p.default) for p in step.params],
                _packed_args_kind(input_annotations) if step.inputs else "array",
                arg_regs,
            )
        else:
            adapter = make_call_adapter(step.fn, _callee_arg_annotations(step), arg_regs)
        idx = len(self.adapters)
        self.adapters.append(adapter)
        self._index[key] = idx
        return idx

    def walker(self) -> Callable:
        return make_walker(self.adapters)


def call_arg_regs(step: Step, prefix: str, rmap: RegisterMap) -> list[int]:
    """`step`'s arguments as REGISTER INDICES (`_callee_arg_names`, looked
    up in `rmap`). Every name must already have a register — an input, a
    param, or an internal the construct registered before encoding — so a
    wiring bug surfaces here as a `KeyError` naming it."""
    return [rmap.lookup(n) for n in _callee_arg_names(step, prefix)]


def encode_call(
    builder: ProgramBuilder, rmap: RegisterMap, nodes: NodeTable, step: Step, prefix: str,
    *, role: str, is_cond: bool, dest: "int | None" = None,
) -> int:
    """Append one `STEP`/`COND` node calling `step` at the registers `rmap`
    gives its arguments (registering that node's adapter in `nodes`) and
    — for a `STEP`, never a `COND`, which has no result to store — writing
    to register `dest`. Returns the new node's index; `next_`/`alt` are
    the caller's to wire via `ProgramBuilder.set_next`."""
    idx = nodes.register(step, call_arg_regs(step, prefix, rmap), role=role)
    if is_cond:
        return builder.add(COND, step_idx=idx)
    return builder.add(STEP, step_idx=idx, dest=dest)


def build_packed_step(
    *, name: str, program: Program, walk: Callable, inputs: Sequence[Input],
    params: Sequence[ParamDecl], output: Any, doc: str,
) -> Step:
    """One construct output as a `types.Step` — `fn` the packed closure
    `interpreter.make_step_fn` builds over `program` and this construct's
    `walk`, `inputs`/`params` in exactly the register order `RegisterMap`
    was seeded in."""
    return Step(
        name=name,
        fn=make_step_fn(walk, program, output, len(params)),
        inputs=tuple(inputs),
        params=tuple(params),
        doc=doc,
        packed=True,
        output_annotation=output,
    )
