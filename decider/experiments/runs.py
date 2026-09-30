"""Run a pipeline to the end and keep what each call wrote, for comparisons."""
from __future__ import annotations

from decider.engine import Engine
from decider.engine.debug.controls import Controls


def trace(step, frame, params: dict | None, ir: dict | None = None, forces=()) -> dict:
    """Run to the end and keep what every call wrote, for a step-by-step diff against another run.

    `forces` (branch arms or loop iteration counts, see `Controls.set`) need the described `ir`.
    """
    session = Engine().bind(step).session(frame, params)
    if forces:
        steer(session, ir, forces)
    error = None
    try:
        session.resume()
        while not session.finished:
            session.resume()
    except Exception as e:  # the trace up to the failing step is still worth comparing
        error = f"{type(e).__name__}: {e}"
    return collect(session, error)


def steer(session, ir, forces):
    c = Controls(ir)
    c.set(forces)
    return c.attach(session)


def collect(session, error):
    """What every call of a run wrote, and the output if it finished."""
    state = session.state
    steps = {}
    for call in session.executable.plan.calls:
        written = {v.name: state.column(v.name, v).to_list() for v in call.writes if v.id in state.values}
        if written:
            steps[call.node.origin.path] = written
    output = None if error else {c: s.to_list() for c, s in session.output().to_dict().items()}
    return {"steps": steps, "output": output, "error": error}
