"""Experiment-interface spike: two candidate interfaces over one core runner.

Proves, against the existing machinery:

- candidate A (declarative ``experiment.yaml``) and candidate B (a thin Python
  builder) describe the *same* schema and both drive one ``run_experiment`` core;
- that core reuses ``decider_bridge.forks.fork`` (a fresh session replayed to a
  checkpoint and overridden) for execution, not a new path;
- input fingerprint, symbolic + resolved revision, declared-override
  validation, float-tolerance comparison, nondeterminism detection,
  cancellation, per-scenario failure, resume, and manifest generation.

Run::

    PYTHONPATH=tools/decider-bridge uv run python \
        notes/vscode-redesign/experimentation/03-experiment-interface/prototype.py
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
import threading
from dataclasses import dataclass, field

import polars as pl

from decider import flow, param
from decider.engine.wiring import resolve

# forks.py lives in the bridge; the feedback already flags rehoming it into core.
import decider_bridge.forks as forks


# --- A representative pipeline: arithmetic + a param ----------------------------------------

def ratio(income: float, debt: float) -> float:
    return debt / income if income > 0 else 1.0


def approved(ratio: float, limit: float = param(0.4, ge=0.0, le=1.0)) -> bool:
    return ratio <= limit


def tier(approved: bool, income: float) -> float:
    return 1.0 if approved else (0.5 if income > 0 else 0.0)


def build() -> flow:
    return flow(ratio, approved, tier, name="demo")


def frame(n: int) -> pl.DataFrame:
    return pl.DataFrame({"income": [1000.0] * n, "debt": [i * 200.0 for i in range(n)]})


# --- Experiment schema (candidate A = YAML; candidate B = the same dict, built in Python) ---

@dataclass
class Spec:
    name: str
    entry: str
    revision: dict
    input: dict
    scenarios: list[dict] = field(default_factory=list)
    comparison: dict = field(default_factory=dict)
    summaries: list[dict] = field(default_factory=list)


# --- The one core runner (thin over forks.fork) ----------------------------------------------

def fingerprint(df: pl.DataFrame) -> str:
    buf = io.BytesIO()
    df.write_ipc(buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()[:16]


def resolve_revision(symbolic: str) -> dict:
    sha = subprocess.run(["git", "rev-parse", symbolic], capture_output=True, text=True).stdout.strip()
    return {"symbolic": symbolic, "resolved": sha or None}


def producers(step) -> dict[str, list[str]]:
    plan = resolve(step)
    inputs = {i.name for i in plan.inputs}
    out = {}
    for c in plan.calls:
        for w in c.writes:
            out.setdefault(w.name, []).append(c.node.origin.path)
    for name in inputs:
        out.setdefault(name, []).append(None)  # inputs are settable before anything runs
    return out


def validate_overrides(step, scenarios) -> list[str]:
    prod = producers(step)
    problems = []
    for sc in scenarios:
        at = sc.get("at")
        for name in sc.get("overrides", {}):
            paths = prod.get(name)
            if paths is None:
                problems.append(f"{sc['name']}: override {name!r} is neither an input nor a produced value")
            elif None in paths:
                continue  # an input: settable before anything runs
            elif at is None:
                problems.append(f"{sc['name']}: override {name!r} is produced, not an input; declare 'at: <step path>'")
            elif not any(p and (p == at or p.startswith(at + "/")) for p in paths):
                problems.append(f"{sc['name']}: no step at/under {at!r} produces {name!r}")
    return problems


def _target(sc) -> tuple | None:
    # fork's `target` is a checkpoint key `(path, "after", 1)`: replay to just after
    # the first finish of the step named by `at`, then apply the overrides.
    at = sc.get("at")
    return None if at is None else (at, "after", 1)


def close(a, b, tol) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= tol["atol"] + tol["rtol"] * max(abs(a), abs(b))
    return a == b


def diff(baseline, scenario, tol) -> list[str]:
    bo, so = baseline["output"], scenario["output"]
    if bo is None or so is None:
        return ["a run errored"]
    out = []
    for col in bo:
        for i, (x, y) in enumerate(zip(bo[col], so[col])):
            if not close(x, y, tol):
                out.append(f"{col}[{i}]: {x!r} != {y!r}")
    return out


def run_experiment(step, df, params, spec: Spec, cancel=None, tol=None, resume: dict | None = None) -> dict:
    tol = tol or {"rtol": 1e-6, "atol": 1e-9}
    fp = fingerprint(df)
    problems = validate_overrides(step, spec.scenarios)
    if problems:
        raise ValueError("; ".join(problems))
    # Baseline twice: a differing rerun means the step is not pure.
    base = forks.fork(step, df, params, [], None, {})
    base_again = forks.fork(step, df, params, [], None, {})
    nondeterministic = diff(base, base_again, {"rtol": 0.0, "atol": 0.0}) != []
    manifest = {
        "version": 1,
        "name": spec.name,
        "revision": resolve_revision(spec.revision["symbolic"]),
        "input": {"fingerprint": fp, "row_count": df.height, "schema": {k: str(v) for k, v in df.schema.items()}},
        "nondeterministic": nondeterministic,
        "scenarios": [],
    }
    done = {sc["name"] for sc in (resume or {}).get("scenarios", []) if sc["status"] == "completed"}
    for sc in spec.scenarios:
        if cancel is not None and cancel.is_set():
            manifest["cancelled"] = True
            break
        if sc["name"] in done:
            manifest["scenarios"].append({"name": sc["name"], "status": "resumed"})
            continue
        try:
            trace = forks.fork(step, df, params, [], _target(sc), sc)
        except Exception as e:
            trace = {"steps": {}, "output": None, "error": f"{type(e).__name__}: {e}"}
        divergences = diff(base, trace, tol)
        manifest["scenarios"].append({
            "name": sc["name"], "status": "failed" if trace["error"] else "completed",
            "error": trace["error"], "divergences": divergences,
        })
    return manifest


# --- Candidate B: a thin builder over the same schema ---------------------------------------

class Experiment:
    def __init__(self, name: str, step, frame, params=None, revision="HEAD"):
        self.step, self.frame, self.params = step, frame, params
        self.spec = Spec(name=name, entry="", revision={"symbolic": revision}, input={}, scenarios=[])

    def scenario(self, name, overrides=None, params=None, at=None):
        self.spec.scenarios.append({
            "name": name, "overrides": overrides or {}, "params": params or {}, "at": at,
        })
        return self

    def run(self, tol=None, cancel=None, resume=None):
        return run_experiment(self.step, self.frame, self.params, self.spec, cancel, tol, resume)


# --- The exercise ---------------------------------------------------------------------------

def demo():
    step = build()
    df = frame(4)
    params = step.parameters().defaults()

    # Candidate A: the YAML would parse to exactly this dict.
    yaml_spec = Spec(
        name="demo-drift",
        entry="demo:build",
        revision={"symbolic": "HEAD"},
        input={"path": "data/sample.parquet", "id_column": None},
        scenarios=[
            {"name": "baseline"},
            {"name": "limit_0.3", "params": {"demo": {"approved": {"limit": 0.3}}}},
            {"name": "input_override", "overrides": {"income": 500.0}},
            {"name": "step_output_override", "at": "demo/ratio", "overrides": {"ratio": 0.1}},
            {"name": "undeclared_override", "overrides": {"ratio": 0.1}},
            {"name": "bad_override", "at": "demo/ratio", "overrides": {"not_a_value": 1.0}},
        ],
        comparison={"outputs": ["tier"], "tolerance": {"rtol": 1e-6, "atol": 1e-9}},
    )

    # Candidate B: the same spec assembled through the builder.
    py_spec = (Experiment("demo-drift", step, df, params)
               .scenario("baseline")
               .scenario("limit_0.3", params={"demo": {"approved": {"limit": 0.3}}})
               .scenario("input_override", overrides={"income": 500.0})
               .scenario("step_output_override", overrides={"ratio": 0.1}, at="demo/ratio"))

    print("== producers ==")
    print(producers(step))

    print("\n== declared override validation (candidate A, pre-flight) ==")
    print(validate_overrides(step, yaml_spec.scenarios) or "ok")

    print("\n== fingerprint + revision ==")
    print("fingerprint:", fingerprint(df))
    print("revision:", resolve_revision("HEAD"))

    print("\n== run (candidate B, clean spec) ==")
    out = py_spec.run()
    print(json.dumps(out, indent=1, default=str)[:1400])

    print("\n== per-scenario failure (one bad cast, rest still complete) ==")
    py_spec.scenario("bad_cast", overrides={"ratio": "not-a-float"}, at="demo/ratio")
    out = py_spec.run()
    print([f"{s['name']}:{s['status']}" + (f" error={s['error']}" if s['error'] else "")
           for s in out["scenarios"]])

    print("\n== cancellation (candidate B, clean spec) ==")
    cancel = threading.Event()
    cancel.set()
    out = py_spec.run(cancel=cancel)
    print("cancelled:", out.get("cancelled"), "| scenarios ran:", len(out["scenarios"]))

    print("\n== resume (skip a completed scenario) ==")
    resume = {"scenarios": [{"name": "baseline", "status": "completed"}]}
    out = py_spec.run(resume=resume)
    print([f"{s['name']}:{s['status']}" for s in out["scenarios"]])


def demo_nondeterminism():
    df = frame(4)
    counter = {"n": 0}

    def drift(income: float, debt: float) -> float:
        counter["n"] += 1
        return (debt / income if income > 0 else 1.0) + counter["n"] * 1e-12

    impure = flow(drift, approved, tier, name="demo")
    params = impure.parameters().defaults()
    spec = Spec(name="impure", entry="", revision={"symbolic": "HEAD"}, input={},
                scenarios=[{"name": "baseline"}])
    out = run_experiment(impure, df, params, spec)
    print("nondeterministic detected:", out["nondeterministic"])


if __name__ == "__main__":
    demo()
    print("\n--- nondeterminism ---")
    demo_nondeterminism()
