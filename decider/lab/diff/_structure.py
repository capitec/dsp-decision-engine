from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from decider.lab._refs import EdgeRef, StepRef
from decider.lab.diff._hash import Snapshot, snapshot

_PARAM_FIELDS = ("annotation", "default", "bounds", "shared_key", "on_invalid")


class StepMove(BaseModel):
    """A step matched across versions whose name or parent changed."""

    model_config = ConfigDict(frozen=True)

    old: StepRef
    new: StepRef


class StepChange(BaseModel):
    """A matched step whose `code`, `consts`, `inputs`, `outputs` or `params` declarations differ."""

    model_config = ConfigDict(frozen=True)

    ref: StepRef
    code: bool = False
    consts: bool = False
    inputs: bool = False
    outputs: bool = False
    params: bool = False


class ParamChange(BaseModel):
    """One param declaration that differs; `old` and `new` are short displays (None when absent)."""

    model_config = ConfigDict(frozen=True)

    ref: StepRef
    name: str
    field: Literal["added", "removed", "annotation", "default", "bounds", "shared_key", "on_invalid"]
    old: str | None = None
    new: str | None = None


class EdgeChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    change: Literal["added", "removed"]
    edge: EdgeRef


class StructureDiff(BaseModel):
    """What changed between two versions of a pipeline, without running either.

    Steps match by `(flow_id, step_id)` durable id first, then by path.
    `identical` is True only when the IR content hashes are equal; a rename
    alone leaves it True, an unhashable constant makes it False.
    """

    model_config = ConfigDict(frozen=True)

    added: tuple[StepRef, ...] = ()
    removed: tuple[StepRef, ...] = ()
    renamed: tuple[StepMove, ...] = ()
    moved: tuple[StepMove, ...] = ()
    changed_steps: tuple[StepChange, ...] = ()
    params: tuple[ParamChange, ...] = ()
    edges: tuple[EdgeChange, ...] = ()
    identical: bool


def structure(old: Any, new: Any) -> StructureDiff:
    """Compare two pipelines' structure, params and wiring.

    Example::

        s = diff.structure(old, new)
        s.changed_steps[0].consts   # True: a constant of that step changed
    """
    return compare(_snap(old), _snap(new))


def _snap(step: Any) -> Snapshot:
    return step if isinstance(step, Snapshot) else snapshot(step)


def compare(a: Snapshot, b: Snapshot) -> StructureDiff:
    pairs = _match(a.description.nodes, b.description.nodes)
    matched_old, matched_new = {i for i, _ in pairs}, {j for _, j in pairs}
    old_nodes, new_nodes = a.description.nodes, b.description.nodes
    renamed, moved, changed, params = [], [], [], []
    for i, j in pairs:
        o, n = old_nodes[i], new_nodes[j]
        if o.name != n.name:
            renamed.append(StepMove(old=o, new=n))
        if _parent(o) != _parent(n):
            moved.append(StepMove(old=o, new=n))
        flags = {k: a.parts[i].get(k) != b.parts[j].get(k) for k in ("code", "consts", "inputs", "outputs", "params")}
        if "other" in a.parts[i] or "other" in b.parts[j]:
            flags["code"] = a.parts[i].get("other") != b.parts[j].get("other")
        if any(flags.values()):
            changed.append(StepChange(ref=n, **flags))
        params += _param_changes(n, a.decls[i], b.decls[j])
    where = {old_nodes[i].path: new_nodes[j].path for i, j in pairs}
    return StructureDiff(
        added=tuple(n for j, n in enumerate(new_nodes) if j not in matched_new),
        removed=tuple(n for i, n in enumerate(old_nodes) if i not in matched_old),
        renamed=tuple(renamed), moved=tuple(moved), changed_steps=tuple(changed), params=tuple(params),
        edges=tuple(_edge_changes(a, b, where)), identical=a.hash == b.hash)


def _parent(ref: StepRef) -> str:
    return ref.path.rpartition("/")[0]


def _match(old: tuple[StepRef, ...], new: tuple[StepRef, ...]) -> list[tuple[int, int]]:
    by_id = {(n.flow_id, n.step_id): j for j, n in enumerate(new) if n.step_id is not None}
    by_path = {n.path: j for j, n in enumerate(new)}
    pairs: dict[int, int] = {}
    taken: set[int] = set()
    for i, o in enumerate(old):
        j = by_id.get((o.flow_id, o.step_id)) if o.step_id is not None else None
        if j is not None:
            pairs[i] = j
            taken.add(j)
    for i, o in enumerate(old):
        j = by_path.get(o.path)
        if i not in pairs and j is not None and j not in taken:
            pairs[i] = j
            taken.add(j)
    return sorted(pairs.items())


def _param_changes(ref: StepRef, old: dict, new: dict) -> list[ParamChange]:
    out = [ParamChange(ref=ref, name=n, field="removed") for n in old if n not in new]
    out += [ParamChange(ref=ref, name=n, field="added") for n in new if n not in old]
    for name in old.keys() & new.keys():
        out += [ParamChange(ref=ref, name=name, field=f, old=old[name][f][0], new=new[name][f][0])
                for f in _PARAM_FIELDS if old[name][f][1] != new[name][f][1]]
    return out


def _edge_changes(a: Snapshot, b: Snapshot, where: dict[str, str]) -> list[EdgeChange]:
    # Old edges are renamed to the new paths first, so a renamed step doesn't read as a rewire.
    old = {(where.get(e.from_path, e.from_path), where.get(e.to_path, e.to_path), e.values): e
           for e in a.description.edges if e.kind == "data"}
    new = {(e.from_path, e.to_path, e.values): e for e in b.description.edges if e.kind == "data"}
    return ([EdgeChange(change="removed", edge=e) for k, e in old.items() if k not in new]
            + [EdgeChange(change="added", edge=e) for k, e in new.items() if k not in old])
