"""`module(...)` — assembling one or more steps into a Module (doc 03 §5).

`types.Module` is the fixed data shape (a frozen dataclass, not the pydantic
`BaseModel` doc 02 §2 describes it as — see this agent's report for that
discrepancy). Everything about *building* one from steps, materialising its
interface, and the instance operations doc 03 declares (`.bind()` §4.3,
`.relabel()` §5.2, `.params_schema()`, re-instantiating under a new name for
reuse §5.3) is attached here from the outside, so `types.py` stays free of
the heavy stack (doc 02 §2's "the graph is data" — no pydantic model-building
machinery needs to live where every layer imports from).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable

from decider2.graph.interface import effective_interface
from decider2.graph.step import make_step
from decider2.params import build_params_model
from decider2.types import Interface, Module, Step

__all__ = ["module"]


def _derived_name(steps: tuple) -> str | None:
    """The name a single-step module takes from its function (doc 03 §5.3).

    `Step.name` is the OUTPUT name, which `@step(output=...)` detaches from
    the function name. The module is named after the function.
    """
    if len(steps) != 1:
        return None
    fn = steps[0].fn
    return getattr(fn, "__name__", None) or steps[0].name


def module(
    *elements: Callable | Step,
    name: str | None = None,
    params: Any = None,
    contract: bool | str | None = None,
) -> Module:
    """Doc 03 §5. `elements` are bare functions and/or already-harvested
    `Step`s (mixing is fine — `make_step` is idempotent on a `Step`).

    `name=` is required only when it cannot be derived: a single-step module
    takes its name from the step (§5 — "which is what makes §5.3's bare
    function work"). `params=` is the explicit-model escape hatch for
    several steps sharing knobs or a cross-field validator (§1.1); it is
    mutually exclusive with any step declaring a `param()` default in the
    same module (§4.4's "two ways to declare a param... a lint forbids both
    in one module", doc 07 §6).
    """
    if not elements:
        raise ValueError("module() needs at least one step")

    steps = tuple(make_step(e) for e in elements)

    # Doc 03 §5.3: a bare function used in a pipeline "becomes a single-step
    # module, NAME FROM THE FUNCTION". The function's name, not the step's
    # output name — those differ whenever @step(output=...) renames the
    # output, which is exactly the waterfall idiom (§3.2). Deriving from the
    # output made every `@step(output="term_cap")` rule a module called
    # `term_cap`, so two rules narrowing the same value collided on their
    # instance name and a legal waterfall could not be expressed at all.
    derived = _derived_name(steps)
    if name is None:
        if derived is None:
            raise ValueError(
                "module(...) with more than one step needs name= — there is "
                "no single function to derive one from (doc 03 §5)."
            )
        name = derived
    elif derived is not None and name == derived:
        raise ValueError(
            f"module({derived}, name={name!r}) restates the derived "
            "name — drop name= (doc 07 §6 lint: 'writing module(fn, "
            "name=\"fn\") is the same name twice')."
        )

    harvested_params = tuple(p for s in steps for p in s.params)
    if params is not None:
        if harvested_params:
            raise ValueError(
                f"module '{name}' passes params= and also has step(s) "
                f"declaring param() defaults ({', '.join(p.name for p in harvested_params)}). "
                "Doc 03 §4.4 allows one way to declare a module's params per "
                "module, not both (doc 07 §6 lint)."
            )
        params_model = params
    else:
        _check_no_duplicate_param_names(name, steps)
        params_model = build_params_model(name, harvested_params)

    built = Module(name=name, steps=steps, params_model=params_model)

    # Validate eagerly (doc 03 §3.1, §2.2 both say "build-time error") rather
    # than waiting for `.interface` to be read.
    effective_interface(built)

    if contract:
        _check_contract(built, contract)

    return built


def _check_no_duplicate_param_names(module_name: str, steps: tuple[Step, ...]) -> None:
    owner: dict[str, str] = {}
    for s in steps:
        for p in s.params:
            if p.name in owner:
                raise ValueError(
                    f"module '{module_name}': steps '{owner[p.name]}' and "
                    f"'{s.name}' both declare param '{p.name}'. Params share "
                    "one flat namespace per module instance (doc 03 §4.1) — "
                    "rename one, or split into modules composed with `|`."
                )
            owner[p.name] = s.name


# --- contract= : freeze the interface, check it never drifts silently -----
# doc 03 §5.1. Opt-in per module; `True` means the derivable path so nobody
# spells out `contracts/{name}.json` by hand (doc 01 §5.3's law again).


def _contract_path(mod: Module, contract: bool | str) -> Path:
    return Path(contract) if isinstance(contract, str) else Path(f"contracts/{mod.name}.json")


def _check_contract(mod: Module, contract: bool | str) -> None:
    path = _contract_path(mod, contract)
    iface = effective_interface(mod)
    snapshot = {
        "inputs": sorted(
            [{"name": i.name, "null_policy": i.null_policy.value} for i in iface.inputs],
            key=lambda d: d["name"],
        ),
        "outputs": sorted(iface.outputs),
        "terminals": sorted(iface.terminals),
        "params": sorted(iface.params_model.model_fields) if iface.params_model else [],
    }
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
        return
    on_disk = json.loads(path.read_text())
    if on_disk != snapshot:
        raise ValueError(
            f"module '{mod.name}' interface no longer matches {path} — "
            f"on disk: {on_disk}, now: {snapshot}. If this change is "
            f"intended, update or delete {path} (doc 03 §5.1)."
        )


# --- behaviour attached to types.Module, kept out of types.py (doc 02 §2) --


def _interface_property(self: Module) -> Interface:
    return effective_interface(self)


def _params_schema(self: Module) -> dict[str, Any]:
    if self.params_model is None:
        return {}
    return {
        name: field.default
        for name, field in self.params_model.model_fields.items()
        if name not in self.bound
    }


def _step(self: Module, name: str) -> Step:
    for s in self.steps:
        if s.name == name:
            return s
    raise KeyError(f"module '{self.name}' has no step '{name}'")


def _bind(self: Module, **values: Any) -> Module:
    """Doc 03 §4.3 — freeze a knob at composition. A bound value "stays a
    runtime value with a fixed default, not a compile-time constant", so
    this only ever edits `Module.bound`; nothing here touches a step's
    signature or triggers a recompile.
    """
    known = set(self.params_model.model_fields) if self.params_model is not None else set()
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"module '{self.name}'.bind(): unknown param(s) {sorted(unknown)} "
            f"— declared params are {sorted(known)}."
        )
    return dataclasses.replace(self, bound={**self.bound, **values})


def _relabel(
    self: Module, *, reads: dict[str, str] | None = None, writes: dict[str, str] | None = None
) -> Module:
    """Doc 03 §5.2 — the instance-local escape hatch. `reads=`/`writes=` are
    symmetric diffs against this module's *current* boundary names, not a
    replacement of the whole mapping.
    """
    return dataclasses.replace(
        self,
        relabel_reads={**self.relabel_reads, **(reads or {})},
        relabel_writes={**self.relabel_writes, **(writes or {})},
    )


def _rename(self: Module, *, name: str) -> Module:
    """`ApplyCap(name="cap_primary")` — doc 03 §5.3's remedy for reusing one
    graph fragment twice in a pipeline: a second, independently-named,
    independently-tunable instance of the same steps.
    """
    return dataclasses.replace(self, name=name)


def _apply(self: Module, frame, *, mode: str = "fused", params=None, shared=None, origin=None):
    """Doc 03 §6 shows a bare module scored/applied directly. Delegates to
    the same runtime path a pipeline uses, treating a standalone module as
    the one-element pipeline it is.
    """
    from decider2.graph.pipeline import flow

    return flow(self).apply(frame, mode=mode, params=params, shared=shared, origin=origin)


def _score(self: Module, record: dict, *, params=None, shared=None):
    from decider2.graph.pipeline import flow

    return flow(self).score(record, params=params, shared=shared)


Module.interface = property(_interface_property)
Module.params_schema = _params_schema
Module.step = _step
Module.bind = _bind
Module.relabel = _relabel
Module.__call__ = _rename
Module.apply = _apply
Module.score = _score
