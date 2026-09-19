"""The pipeline scope (doc 03 §3): an ordered sequence of modules.

`|` is a sequence, not a dependency-resolved graph — doc 03 §8.1's own
correction of an earlier draft that asserted both. Written order is
execution order, full stop; only within one module is order derived (doc 03
§2 — a pure, topologically-sorted DAG). `flow(*elements)` is the general
constructor and `|`/`Module.__or__` is sugar over it (doc 03 §5.3), which is
why `types.Module.__or__` reaches into `decider2.graph.pipeline.compose`.

This is also where doc 03 §2.1's "one error that matters" and §3.2/§3.3's
version chains live: both are properties of a *sequence* of modules, not of
any one module, so neither belongs in `graph/interface.py`.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence, Union

from decider2.graph.interface import effective_interface, topological_steps
from decider2.graph.module import module as _module
from decider2.graph.resolve import suggest_name
from decider2.types import Decision, Emit, Input, Interface, MissingInputPolicy, Module, Step

__all__ = ["Pipeline", "flow", "compose"]

Element = Union[Module, Callable[..., Any], "Pipeline"]


@dataclasses.dataclass(frozen=True)
class Pipeline:
    """An ordered sequence of `Module`s plus the pipeline-level metadata doc
    03 §7 and §1's "null must produce a decision" describe: what to keep
    beyond the default additive frame (`emits`), what to drop from it
    (`dropped`), and how a missing input routes (`missing_input_policy`).

    Not a `types.py` seam: only `Module` needs to be understood by every
    other layer (compile, runtime, observe). A pipeline is graph-layer
    structure over modules, built fresh by `flow`/`compose`/`.emit`/`.drop`
    rather than mutated (doc 03 — "`.emit()` and `.drop()` return a new
    pipeline; they do not mutate").
    """

    elements: tuple[Module, ...]
    emits: tuple[Emit, ...] = ()
    dropped: tuple[str, ...] = ()
    missing_input_policy: MissingInputPolicy = MissingInputPolicy()

    def __or__(self, other: "Element") -> "Pipeline":
        return compose(self, other)

    def __ror__(self, other: "Element") -> "Pipeline":
        return compose(other, self)

    @property
    def interface(self) -> Interface:
        return _pipeline_interface(self)

    def params_schema(self) -> dict[str, dict[str, Any]]:
        """Doc 03 §4.1 — the composed set, namespaced by module instance."""
        return {m.name: m.params_schema() for m in self.elements if m.params_model is not None}

    def step(self, name: str) -> Step:
        for m in self.elements:
            for s in m.steps:
                if s.name == name:
                    return s
        raise KeyError(f"pipeline has no step '{name}'")

    def schema(self) -> dict[str, tuple]:
        """Doc 03 §2.2 — "lists every unbound input at once"."""
        iface = self.interface
        return {"inputs": iface.inputs, "outputs": iface.outputs, "terminals": iface.terminals}

    def versions(self) -> dict[str, tuple[str, ...]]:
        """The version chain for every produced name, in production order —
        the data `name@module` qualification resolves against (doc 03 §3.3,
        §7)."""
        return _walk(self.elements)[1]

    def flatten_for_runtime(self) -> tuple[tuple[Step, ...], tuple[int, ...], tuple[Any, ...]]:
        """`(steps, group_ids, param_spaces)` in the shape
        `decider2.runtime.invoke.apply`/`score` take (doc 02 §3.5).

        Execution order is written order, module by module (doc 03 §8.1) —
        never re-derived across the sequence — with each module's own steps
        topologically sorted first (doc 03 §2). `group_ids` gives every
        step in one module the same id, so the default compiles "one kernel
        per module, nothing fused implicitly" (doc 03 §3.2) rather than
        `runtime.invoke`'s own fallback of one kernel per *step*, which is
        only correct by coincidence for a module that happens to have one
        step.

        `param_spaces` is one `runtime.invoke.ParamSpace` per module
        instance — the namespace doc 03 §10 says a caller's `params={...}`
        is keyed by ("Params are namespaced by module instance name"). It
        carries the module's own name, *all* its step names, its model and
        its `.bind()`-frozen values, because none of those are recoverable
        from the flat step list: a multi-step module's knobs live in one
        flat namespace named after the module (§4.1), not after whichever
        step happens to declare each one, and `bound` (§4.3) exists nowhere
        else. Keying this by step name instead — as an earlier draft of
        this contract did — made `params={"band": {"lo": 20.0}}` a silent
        no-op for every module whose steps are not named after it.
        """
        from decider2.runtime.invoke import ParamSpace

        steps: list[Step] = []
        group_ids: list[int] = []
        param_spaces: list[ParamSpace] = []
        for gid, m in enumerate(self.elements):
            member_names: list[str] = []
            for s in topological_steps(m.steps):
                steps.append(s)
                group_ids.append(gid)
                member_names.append(s.name)
            if m.params_model is not None or any(s.reads_params for s in m.steps):
                param_spaces.append(
                    ParamSpace(
                        module=m.name,
                        step_names=tuple(member_names),
                        model=m.params_model,
                        bound=dict(m.bound),
                    )
                )
        return tuple(steps), tuple(group_ids), tuple(param_spaces)

    def emit(self, *names: str) -> "Pipeline":
        """Doc 03 §7. `name`, `name@module` or `name@*`; returns a NEW
        pipeline. Checked against the version chain immediately — "the
        framework tells you when [a qualifier is] needed" applies just as
        much to one that resolves to nothing (§5.2's spirit, applied here).
        """
        parsed = tuple(_parse_emit(n) for n in names)
        _, versions, _ = _walk(self.elements)
        leaf_names = {i.name for i in self.interface.inputs}
        for e in parsed:
            _check_emittable(e, versions, leaf_names)
        return dataclasses.replace(self, emits=_dedup_emits(self.emits + parsed))

    def drop(self, *names: str) -> "Pipeline":
        """Doc 03 §7 — declared on the pipeline, so "this flow does not
        emit X" is a static property. "Dropping a column a later module
        reads is a build error": guarded below against dropping an internal
        wire, which isn't part of the additive frame to begin with, rather
        than a real column being removed out from under a consumer.
        """
        iface = self.interface
        in_output = {i.name for i in iface.inputs} | set(iface.terminals) | {e.name for e in self.emits}
        wired = _all_wired_names(self.elements)
        for n in names:
            if n not in in_output and n in wired:
                raise ValueError(
                    f"pipeline.drop('{n}'): '{n}' is an internal value some "
                    "module reads — not a leaf input, a terminal or an "
                    "emitted value — so there is nothing in the output frame "
                    "to drop (doc 03 §7)."
                )
        return dataclasses.replace(self, dropped=_dedup_strs(self.dropped + tuple(names)))

    def on_missing_input(
        self, *, default: Decision = Decision.REFER, reason: int = 4101, raise_for: Sequence[str] = ()
    ) -> "Pipeline":
        """Doc 03 §1's routing policy. Matches `types.MissingInputPolicy`'s
        actual fields; see this agent's report for where doc 03's
        `Decision.refer(reason=...)` sketch and the fixed `types.py` seam
        disagree.
        """
        policy = MissingInputPolicy(default=default, reason=reason, raise_for=tuple(raise_for))
        return dataclasses.replace(self, missing_input_policy=policy)

    def apply(
        self, frame, *, mode: str = "fused", params: dict | None = None,
        shared: dict | None = None, origin: str | None = None,
    ):
        """Doc 03 §6. Delegates to `decider2.runtime.invoke` — the three
        execution modes and the boundary/compile machinery are a different
        layer's responsibility (doc 00-BUILD.md Layer 2/3); this call site
        is the contract between them. `.emit()`'s `name@module` qualifier
        is not passed through: `runtime.invoke` takes plain names only as
        of this writing (see this agent's report).
        """
        from decider2.runtime.invoke import apply as _apply

        steps, group_ids, param_spaces = self.flatten_for_runtime()
        return _apply(
            steps, frame, interface=self.interface, group_ids=group_ids,
            params=params, shared=shared, origin=origin, mode=mode,
            emit=tuple(e.name for e in self.emits), drop=self.dropped,
            param_spaces=param_spaces, policy=self.missing_input_policy,
        )

    def score(self, record: dict, *, params: dict | None = None, shared: dict | None = None):
        """Doc 03 §6 — the realtime path. Takes a dict, not kwargs (measured:
        EXPERIMENTS.md §N2)."""
        from decider2.runtime.invoke import score as _score

        steps, group_ids, param_spaces = self.flatten_for_runtime()
        return _score(
            steps, record, interface=self.interface, group_ids=group_ids,
            params=params, shared=shared, mode="fused",
            emit=tuple(e.name for e in self.emits), param_spaces=param_spaces,
            policy=self.missing_input_policy,
        )


# --- construction -----------------------------------------------------------


def _to_module(e: Callable | Module | Step) -> Module:
    if isinstance(e, Module):
        return e
    if callable(e):
        return _module(e)
    raise TypeError(f"{e!r} is not a Module or a callable step (doc 03 §5.3)")


def flow(*elements: "Element") -> Pipeline:
    """Doc 03 §5.3 — the general pipeline constructor. `|` is sugar over it:
    a plain function has no `__or__`, so a pipeline of only bare functions
    (§1.1's one-rule-one-artefact case) cannot be built with `|` alone, and
    `flow(...)` is what makes it expressible. `flow(a, b) | Scoring` mixes
    freely because both paths end here.
    """
    if not elements:
        raise ValueError("flow() needs at least one element")

    modules: list[Module] = []
    emits: tuple[Emit, ...] = ()
    dropped: tuple[str, ...] = ()
    policy = MissingInputPolicy()

    for e in elements:
        if isinstance(e, Pipeline):
            modules.extend(e.elements)
            emits = _dedup_emits(emits + e.emits)
            dropped = _dedup_strs(dropped + e.dropped)
            policy = e.missing_input_policy
        else:
            modules.append(_to_module(e))

    _check_unique_instance_names(modules)
    modules_t = tuple(modules)
    _walk(modules_t)  # raises on a build-time wiring error before returning

    return Pipeline(elements=modules_t, emits=emits, dropped=dropped, missing_input_policy=policy)


def compose(left: "Element", right: "Element") -> Pipeline:
    """`a | b` (doc 03 §8.1) — the same object `flow(a, b)` builds; this is
    what `types.Module.__or__` calls."""
    return flow(left, right)


def _check_unique_instance_names(modules: Sequence[Module]) -> None:
    seen: dict[str, int] = {}
    for i, m in enumerate(modules):
        if m.name in seen:
            raise ValueError(
                f"pipeline uses module instance name '{m.name}' twice "
                f"(position {seen[m.name]} and {i}). Doc 03 §5.3: reusing a "
                "module or bare function twice in one pipeline needs a "
                f"second, independently-named instance — {m.name}(name="
                f"'{m.name}_2') — because params are namespaced per module "
                "instance (§4.1) and an audit record must say which one ran."
            )
        seen[m.name] = i


# --- cross-module wiring: the waterfall, the version chain, §2.1's error --


def _walk(
    elements: tuple[Module, ...],
) -> tuple[dict[str, Input], dict[str, tuple[str, ...]], dict[str, bool]]:
    """One pass over the sequence, doc 03 §2.1 + §3.2/§3.3 + §5.1's terminal
    definition, generalised from "one module" to "the whole pipeline":

    - a module's local leaf that some earlier module already produces
      resolves to that most-recent producer (§2.1's waterfall) — and marks
      that produced value as consumed, so a superseded version is correctly
      excluded from the output frame even though a later step reads it;
    - a module's local leaf that nothing has produced yet is a genuine
      pipeline leaf (§2.2), unless it closely matches something already in
      scope, which is treated as the typo it almost certainly is;
    - each production of a name resets its terminal flag — a fresh version
      is terminal again until *something after it* reads it, which is what
      lets a waterfall's intermediate versions still count as "consumed"
      while its final version can still be a terminal.
    """
    produced_so_far: dict[str, list[str]] = {}
    terminal_flag: dict[str, bool] = {}
    leaves: dict[str, Input] = {}

    for m in elements:
        iface = effective_interface(m)
        for inp in iface.inputs:
            if inp.name in produced_so_far:
                terminal_flag[inp.name] = False
                continue
            if inp.name in leaves:
                continue
            near = suggest_name(inp.name, produced_so_far.keys())
            if near is not None:
                raise ValueError(
                    f"module '{m.name}' input '{inp.name}' is not produced by "
                    "any module earlier in this pipeline and is not a "
                    f"declared input column. Did you mean '{near}' (produced "
                    f"by module '{produced_so_far[near][-1]}')? Qualify or "
                    f"rename with {m.name}.relabel(reads={{'{inp.name}': "
                    f"'{near}'}}) (doc 03 §2.2, §5.2)."
                )
            leaves[inp.name] = inp
        for out in iface.outputs:
            produced_so_far.setdefault(out, []).append(m.name)
            terminal_flag[out] = True

    versions = {name: tuple(chain) for name, chain in produced_so_far.items()}
    return leaves, versions, terminal_flag


def _pipeline_interface(p: Pipeline) -> Interface:
    leaves, versions, terminal_flag = _walk(p.elements)
    terminals = tuple(name for name, is_terminal in terminal_flag.items() if is_terminal)
    return Interface(
        inputs=tuple(leaves.values()),
        outputs=tuple(versions),
        terminals=terminals,
        # A pipeline has no single params model of its own — §4.1's composed
        # view is `.params_schema()`, namespaced per module instance.
        params_model=None,
        shared_fields=(),
    )


def _all_wired_names(elements: tuple[Module, ...]) -> set[str]:
    """Every name any step anywhere reads as an input — used only to guard
    `.drop()` against removing an active internal wire (see `Pipeline.drop`).
    """
    return {inp.name for m in elements for s in m.steps for inp in s.inputs}


# --- .emit() parsing and validation -----------------------------------------


def _parse_emit(spec: str) -> Emit:
    """Doc 03 §7: `"term_cap"`, `"term_cap@sector_cap"` or `"term_cap@*"`."""
    if "@" in spec:
        name, at = spec.split("@", 1)
        return Emit(name=name, at=at)
    return Emit(name=spec, at=None)


def _check_emittable(e: Emit, versions: dict[str, tuple[str, ...]], leaf_names: set[str]) -> None:
    produced = e.name in versions
    leaf = e.name in leaf_names
    if not produced and not leaf:
        near = suggest_name(e.name, set(versions) | leaf_names)
        hint = f" Did you mean '{near}'?" if near else ""
        raise ValueError(
            f"pipeline.emit('{e.name}'): no step produces '{e.name}' and it "
            f"is not a declared input column.{hint}"
        )
    if e.at not in (None, "*"):
        if not produced:
            raise ValueError(
                f"pipeline.emit('{e.name}@{e.at}'): '{e.name}' is never "
                "produced by any module — it is only ever a leaf input "
                "column, so it has no version to qualify (doc 03 §7)."
            )
        if e.at not in versions[e.name]:
            raise ValueError(
                f"pipeline.emit('{e.name}@{e.at}'): '{e.name}' is never "
                f"produced by module '{e.at}'. Producers, in order: "
                f"{list(versions[e.name])} (doc 03 §7 — qualify by producing "
                "module name, not position)."
            )


def _dedup_emits(emits: tuple[Emit, ...]) -> tuple[Emit, ...]:
    seen: set[tuple[str, str | None]] = set()
    out: list[Emit] = []
    for e in emits:
        key = (e.name, e.at)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return tuple(out)


def _dedup_strs(xs: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return tuple(out)
