"""Runtime lineage: which recorded versions a value was computed from, up to now."""
from __future__ import annotations

from decider.engine.wiring.plan import Branch, Call, Loop, Sequence


def _producers(root) -> dict[int, tuple]:
    """Version id -> how it was made: ("call", Call), ("merge", Branch, Merge) or ("carry", Carry)."""
    out: dict[int, tuple] = {}

    def visit(r):
        if isinstance(r, Call):
            for w in r.writes:
                out[w.id] = ("call", r)
        elif isinstance(r, Sequence):
            for c in r.children:
                visit(c)
        elif isinstance(r, Branch):
            visit(r.condition)
            for a in r.arms:
                visit(a)
            for m in r.merges:
                out[m.version.id] = ("merge", r, m)
        elif isinstance(r, Loop):
            visit(r.condition)
            visit(r.body)
            for c in r.carries:
                out[c.version.id] = ("carry", c)

    visit(root)
    return out


def _value(state, v, row):
    values, valid = state.read(v)
    if row is None:
        return None
    return None if valid is not None and not valid[row] else values[row].item() if hasattr(values[row], "item") else values[row]


def _sources(state, how, row):
    if how[0] == "call":
        return list(how[1].reads or ())
    if how[0] == "merge":
        branch, merge = how[1], how[2]
        if row is not None:
            picked, valid = state.read(branch.condition.writes[0])
            if valid is None or valid[row]:
                p = picked[row]
                k = (0 if p else 1) if picked.dtype == bool else int(p)
                return [branch.condition.writes[0], merge.arms[k] or merge.prior]
        return [branch.condition.writes[0], *(a for a in merge.arms if a), *([merge.prior] if merge.prior else [])]
    return [how[1].last, how[1].initial]


def latest(state, versions, row):
    """The version a record sees: the latest one written on its row.

    Inside a branch arm the newest version covers only that arm's rows, so a
    record on another arm still sees the value from before the branch.
    """
    if row is not None:
        for v in reversed(versions):
            valid = state.valid.get(v.id)
            if valid is None or valid[row]:
                return v
    # ponytail: a genuinely null value on the row also falls through to an older version; track written rows if that misleads.
    return versions[-1] if versions else None


def lineage(session, name: str, row: int | None = None, depth: int = 8) -> dict:
    """The latest written version of `name` and, recursively, the versions it was computed from.

    With `row`, each entry carries that record's value, and a branch merge
    follows only the arm the record took.
    """
    state = session.state
    producers = _producers(session.executable.plan.root)
    written = [v for v in state.chains.get(name, ()) if v.id in state.values]
    inputs = [v for v in session.executable.plan.versions if v.producer is None and v.name == name]
    start = latest(state, inputs + written, row)
    if start is None:
        return {"name": name, "producer": None, "value": None, "inputs": [], "missing": True}
    seen: set[int] = set()

    def entry(v, left):
        e = {"name": v.name, "producer": v.producer, "value": _value(state, v, row), "inputs": []}
        how = producers.get(v.id)
        if how is None or left == 0 or v.id in seen:
            return e
        seen.add(v.id)
        if how[0] != "call":
            e["via"] = how[0]
        e["inputs"] = [entry(s, left - 1) for s in _sources(state, how, row)
                       if s.producer is None or s.id in state.values]
        return e

    return entry(start, depth)
