"""Fork a paused session: re-run from the same point with other params or values, many times.

A running generator can't be copied, so a fork is a fresh session replayed to
the same checkpoint, with the original session's overrides re-applied where
they were made. Steps are pure, so the replay reaches the same state.
"""
from __future__ import annotations

from decider.engine import Engine
from decider.engine.params import document_key
from runs import collect


def checkpoint_key(session):
    """`(path, when, n)`: the current checkpoint and how many times the run has reached it."""
    cp = session.current
    if cp is None:
        return None
    kind = "node_started" if cp.when == "before" else "node_finished"
    n = sum(1 for e in session.events if e.kind == kind and e.origin.path == cp.origin.path)
    return cp.origin.path, cp.when, n


def merge(base, extra):
    out = dict(base or {})
    for k, v in (extra or {}).items():
        out[k] = merge(out.get(k), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def fork(step, frame, params, history, target, scenario):
    """One scenario from `target`: replay to it, apply the scenario's values and params, run to the end.

    Args:
        history: `(key, name, value)` overrides the original session made, in order.
        target: the checkpoint key to fork at; `None` forks before anything runs.
        scenario: `{"params": {...}, "overrides": {...}, "row": int | None}`; params
            merge over the run's document and apply only to nodes that run after `target`.
    """
    s = Engine().bind(step).session(frame, params)
    pending = list(history)

    def apply(key):
        while pending and pending[0][0] == key:
            _, name, value = pending.pop(0)
            s.set(name, value)

    apply(None)
    while target is not None and checkpoint_key(s) != target:
        if s.step_into() is None:
            raise RuntimeError(f"the replay never reached {target}; was the session rewound?")
        apply(checkpoint_key(s))
    for name, value in (scenario.get("overrides") or {}).items():
        row = scenario.get("row")
        if row is None:
            s.set(name, value)
        else:
            values = s.value(name).to_list()
            values[row] = value
            s.set(name, values)
    if scenario.get("params"):
        # ponytail: swaps the private RunParams document in place; ask Session for a params setter if this sticks.
        doc = merge(params, scenario["params"])
        s._params.doc, s._params.key = doc, document_key(doc)
    error = None
    try:
        while not s.finished:
            s.resume()
    except Exception as e:  # the trace up to the failing step is still worth comparing
        error = f"{type(e).__name__}: {e}"
    return collect(s, error)


def sweep(step, frame, params, history, target, scenarios):
    """The original continuation plus one fork per scenario, as traces ready to compare."""
    baseline = fork(step, frame, params, history, target, {})
    return {"baseline": baseline, "results": [{"label": sc.get("label"), **fork(step, frame, params, history, target, sc)}
                                               for sc in scenarios]}
