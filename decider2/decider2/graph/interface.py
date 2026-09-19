"""A module's interface: inferred from its steps, then materialised as data
(doc 03 §5.1). Never declared — 0 of 204 corpus modules pass `outputs=`
(doc 00-BUILD.md §6) — so this is the only place a `Module`'s `Interface` is
built.

Two entry points:

- `raw_interface` — pure inference from a module's own steps: §3.1's
  no-overwrite rule (two steps, one output, is a build error) and §2.2's
  did-you-mean for a name that resolves to neither a sibling step's output
  nor a leaf.
- `effective_interface` — `raw_interface` plus `.relabel()` (doc 03 §5.2):
  the module's interior is untouched, only the boundary names change, which
  is what a pipeline should see when this module is one element of a
  sequence (`graph/pipeline.py`).
"""
from __future__ import annotations

from typing import Any

from decider2.graph.resolve import suggest_name
from decider2.types import Input, Interface, Module, Step


def _duplicate_output_error(module_name: str, first: Step, second: Step) -> ValueError:
    first_fn = getattr(first.fn, "__qualname__", repr(first.fn))
    second_fn = getattr(second.fn, "__qualname__", repr(second.fn))
    return ValueError(
        f"module '{module_name}': both '{first_fn}' and '{second_fn}' declare "
        f"output '{second.name}' (doc 03 §3.1). Give them distinct output "
        "names (e.g. `@step(output=...)`), or split them into separate "
        "modules composed with `|`."
    )


def _unbound_input_error(module_name: str, step_name: str, input_name: str, near: str) -> ValueError:
    return ValueError(
        f"step '{step_name}' input '{input_name}' is not produced by any step "
        "in scope and is not a declared input column. Did you mean "
        f"'{near}' (produced by step '{near}' in module '{module_name}')?"
    )


def raw_interface(module_name: str, steps: tuple[Step, ...], params_model: Any) -> Interface:
    """The interface implied by `steps` alone — no pipeline context.

    Inputs are leaves (needed, not produced within), outputs are everything
    produced, terminals are outputs nothing *else in this module* reads
    (doc 03 §5.1). A module's own terminals are a local notion for
    inspecting it standalone; `graph/pipeline.py` recomputes terminal-ness
    across the whole sequence for the output frame (doc 03 §7), since a
    value this module produces may still be consumed by a *later* module.
    """
    producers: dict[str, Step] = {}
    for s in steps:
        if s.name in producers:
            raise _duplicate_output_error(module_name, producers[s.name], s)
        producers[s.name] = s

    produced = tuple(producers)
    consumed_locally = {name: False for name in produced}
    leaves: dict[str, Input] = {}

    for s in steps:
        for inp in s.inputs:
            if inp.name in consumed_locally:
                consumed_locally[inp.name] = True
                continue
            if inp.name in leaves:
                continue
            near = suggest_name(inp.name, produced)
            if near is not None:
                raise _unbound_input_error(module_name, s.name, inp.name, near)
            leaves[inp.name] = inp

    terminals = tuple(name for name in produced if not consumed_locally[name])

    return Interface(
        inputs=tuple(leaves.values()),
        outputs=produced,
        terminals=terminals,
        params_model=params_model,
        # Doc 03 §4.2: a module using `shared` declares "a required-fields
        # contract (names + types), checked at composition." Nothing in a
        # step's signature says *which* fields of `shared` it reads — only
        # that it reads `shared` at all (`Step.reads_shared`) — so there is
        # no source to infer this contract from yet without either a second
        # declaration or reading step bodies (out of scope here; see report).
        shared_fields=(),
    )


def topological_steps(steps: tuple[Step, ...]) -> tuple[Step, ...]:
    """Doc 03 §2: "Inside a module, ordering is derived by topological sort
    — you never declare execution order." `raw_interface` already proves
    the module has no duplicate outputs; this is the companion pass that
    turns "distinct names, no declared order" into a concrete execution
    order for `compile/driver.py` (doc 02 §6: "topological sort/ordering is
    the graph layer's responsibility").

    A depth-first post-order over declaration order, so two modules with the
    same steps in the same DAG shape always compile to the same order
    (doc 05 §4.2's "stable topological tie-break") regardless of how someone
    happened to list independent steps.
    """
    by_name = {s.name: s for s in steps}
    deps: dict[str, list[str]] = {
        s.name: [inp.name for inp in s.inputs if inp.name in by_name and inp.name != s.name]
        for s in steps
    }

    ordered: list[str] = []
    state: dict[str, int] = {}  # 0 unseen (absent), 1 in progress, 2 done

    def visit(name: str) -> None:
        st = state.get(name, 0)
        if st == 2:
            return
        if st == 1:
            raise ValueError(
                f"step '{name}' is part of a dependency cycle — a step "
                "cannot (even transitively) need its own output as an input."
            )
        state[name] = 1
        for dep in deps[name]:
            visit(dep)
        state[name] = 2
        ordered.append(name)

    for s in steps:
        visit(s.name)

    return tuple(by_name[n] for n in ordered)


def effective_interface(module: Module) -> Interface:
    """`raw_interface` plus `.relabel()` (doc 03 §5.2): the boundary names a
    pipeline should wire against, not the module's interior names.

    `.relabel()` is "declared data on the instance... applied at the scope
    boundary, so the module's interior is untouched" — so only the *names*
    reported here change; the steps themselves keep referring to their
    original names, and rewriting the actual calls is a compile-time
    concern (doc 05), not a graph one.

    Deliberately does NOT narrow `params_model` for `.bind()` (§4.3) — the
    field doc 03 §4.3 says a bound value "leaves the caller-facing params
    interface" is `Module.params_schema()`, a flat name->default dict,
    which already excludes bound fields. Rebuilding a second pydantic model
    with fields removed, just so `Interface.params_model` agreed, seemed
    like the wrong place to spend that complexity for no test that needs
    it — see this agent's report.
    """
    raw = raw_interface(module.name, module.steps, module.params_model)

    reads = module.relabel_reads
    writes = module.relabel_writes

    inputs = tuple(
        Input(
            name=reads.get(i.name, i.name),
            annotation=i.annotation,
            null_policy=i.null_policy,
            fill=i.fill,
        )
        for i in raw.inputs
    )
    outputs = tuple(writes.get(o, o) for o in raw.outputs)
    terminals = tuple(writes.get(t, t) for t in raw.terminals)

    return Interface(
        inputs=inputs,
        outputs=outputs,
        terminals=terminals,
        params_model=raw.params_model,
        shared_fields=raw.shared_fields,
    )
