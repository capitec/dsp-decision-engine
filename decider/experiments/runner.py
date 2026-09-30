"""Run an experiment: validate scenarios, fork each one, compare, and write a manifest.

`run_experiment` is the only place execution happens. It reuses the rehomed
fork/sweep primitives (a fresh session replayed to a checkpoint and overridden)
for execution, `describe` for the static flow contract, `preflight` for
compatibility and cost, and the lifecycle `RunManifest`/`JobHandle` for
provenance and job semantics. It adds no new execution path.
"""
from __future__ import annotations

import importlib
import math
import platform
import subprocess
from importlib import metadata
from typing import Any

import polars as pl

from decider.contract import describe
from decider.data import LoadedData, preflight
from decider.experiments.forks import fork
from decider.experiments.model import ComparisonPolicy, ExperimentDef, Tolerance
from decider.experiments.result import (ExperimentResult, Finding, RunStatus, ScenarioResult, ScenarioStatus)
from decider.lifecycle import Environment, JobHandle, ResultRef, Revision, RunManifest, new_manifest_id


def run_experiment(def_: ExperimentDef, step, frame, params=None, *, job: JobHandle | None = None,
                   resume: ExperimentResult | None = None) -> ExperimentResult:
    """Run every scenario of `def_` next to a baseline and compare, returning an `ExperimentResult`.

    Each scenario runs through `fork` (a fresh session replayed to a checkpoint,
    with the scenario's params and overrides applied), so one failing scenario
    never aborts the sweep. `job` reports progress and can cancel or time out the
    run between scenarios; `resume` (a prior `ExperimentResult`) skips a completed
    scenario only when the input fingerprint and resolved revision still match.

    Example::

        result = run_experiment(def_, pipeline, df, pipeline.parameters().defaults())
        result.status  # RunStatus.COMPLETED, or NON_REPRODUCIBLE / PARTIAL
    """
    if params is None:
        params = step.parameters().defaults()
    problems = validate_scenarios(def_, step)
    if problems:
        raise ValueError("; ".join(problems))
    report = preflight(step, frame, params=params)
    if job is not None:
        job.start()
        job.progress(0, total=len(def_.scenarios))

    baseline = fork(step, frame, params, [], None, {})
    nondeterministic = False
    if def_.comparison.determinism:
        nondeterministic = _diff(baseline["output"], fork(step, frame, params, [], None, {})["output"],
                                 def_.comparison, exact=True) != ()

    data = LoadedData.from_frame(frame, dataset=def_.input.path)
    resolved = resolve_revision(def_.revision)
    done = _resumable(resume, data, resolved)

    results: list[ScenarioResult] = []
    for i, sc in enumerate(def_.scenarios):
        if job is not None:
            if job.timed_out:
                job.timeout()
            if job.cancelled or job.timed_out:
                break
            job.progress(i, total=len(def_.scenarios), message=sc.name)
        if sc.name in done:
            results.append(ScenarioResult(name=sc.name, status=ScenarioStatus.RESUMED))
            continue
        target = None if sc.at is None else (sc.at, "after", 1)
        scenario = {"params": sc.params, "overrides": sc.overrides, "row": sc.row}
        try:
            trace = fork(step, frame, params, [], target, scenario)
        except Exception as e:
            trace = {"output": None, "error": f"{type(e).__name__}: {e}"}
        if trace["error"]:
            results.append(ScenarioResult(
                name=sc.name, status=ScenarioStatus.FAILED, error=trace["error"],
                divergences=(Finding(kind="error", scenario=sc.name, message=trace["error"]),),
            ))
        else:
            results.append(ScenarioResult(
                name=sc.name, status=ScenarioStatus.COMPLETED,
                divergences=_diff(baseline["output"], trace["output"], def_.comparison),
                output_ref=ResultRef(name=sc.name, location=f"results/{sc.name}.parquet"),
                row_count=frame.height,
            ))

    status = _run_status(results, nondeterministic, job)
    manifest = _manifest(def_, step, data, resolved, results)
    if job is not None and not job.terminal:
        job.succeed(result={"status": status.value, "scenarios": len(results)})
    return ExperimentResult(manifest=manifest, status=status, nondeterministic=nondeterministic,
                            scenarios=tuple(results), preflight=report,
                            job=job.snapshot() if job is not None else None)


def validate_scenarios(def_: ExperimentDef, step) -> list[str]:
    """Problems in `def_.scenarios`: overrides at an undeclared point, or no point for a produced value.

    A value the flow produces must name the step output it overrides (`at`); an
    input column is settable before anything runs and needs no point. Returned,
    not raised, so a caller surfaces every problem at once.
    """
    slots = _override_targets(step)
    problems = []
    for sc in def_.scenarios:
        for name in sc.overrides:
            paths = slots.get(name)
            if paths is None:
                problems.append(f"{sc.name}: override {name!r} is neither an input nor a produced value")
            elif any(p is None for p in paths):
                continue
            elif sc.at is None:
                problems.append(f"{sc.name}: override {name!r} is produced, not an input; declare 'at: <step path>'")
            elif not any(p and (p == sc.at or p.startswith(sc.at + "/")) for p in paths):
                problems.append(f"{sc.name}: no step at/under {sc.at!r} produces {name!r}")
    return problems


def resolve_revision(authored: str) -> str | None:
    """Resolve an authored revision (`"HEAD"`, `"HEAD^"`) to a git SHA, or `None` when unresolvable."""
    try:
        out = subprocess.run(["git", "rev-parse", authored], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        return None
    return out or None


def load_flow(entry: str):
    """Import `entry` (`"module:qualname"`) and call it if it is a zero-argument factory."""
    module, _, qualname = entry.partition(":")
    if not module or not qualname:
        raise ValueError(f"entry {entry!r} must be 'module:qualname'")
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj() if callable(obj) else obj


def _override_targets(step) -> dict[str, list[str | None]]:
    slots: dict[str, list[str | None]] = {}
    for v in describe(step).value_slots:
        slots.setdefault(v.name, []).append(v.path)
    return slots


def _diff(base, other, comparison: ComparisonPolicy, exact: bool = False) -> tuple[Finding, ...]:
    if base is None or other is None:
        return (Finding(kind="error", message="a run errored"),)
    tol = Tolerance(rtol=0.0, atol=0.0) if exact else comparison.tolerance
    cols = comparison.outputs or tuple(base)
    out = []
    for col in cols:
        left, right = base.get(col), other.get(col)
        if left is None or right is None:
            continue
        for i, (a, b) in enumerate(zip(left, right)):
            if not _close(a, b, tol):
                out.append(Finding(kind="divergence", location=f"{col}[{i}]", expected=a, actual=b))
    return tuple(out)


def _close(a: Any, b: Any, tol: Tolerance) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= tol.atol + tol.rtol * max(abs(a), abs(b))
    return a == b


def _run_status(results: list[ScenarioResult], nondeterministic: bool, job: JobHandle | None) -> RunStatus:
    if job is not None and job.cancelled:
        return RunStatus.CANCELLED
    if job is not None and job.timed_out:
        return RunStatus.TIMED_OUT
    if nondeterministic:
        return RunStatus.NON_REPRODUCIBLE
    if all(r.status is ScenarioStatus.COMPLETED for r in results):
        return RunStatus.COMPLETED
    return RunStatus.PARTIAL


def _resumable(resume: ExperimentResult | None, data, resolved: str | None) -> set[str]:
    if resume is None:
        return set()
    if resume.manifest.input is not None and resume.manifest.input.fingerprint != data.fingerprint:
        return set()
    if resolved is not None and resume.manifest.revision is not None \
            and resume.manifest.revision.resolved != resolved:
        return set()
    return {s.name for s in resume.scenarios if s.status is ScenarioStatus.COMPLETED}


def _manifest(def_: ExperimentDef, step, data: LoadedData, resolved: str | None,
              results: list[ScenarioResult]) -> RunManifest:
    return RunManifest(
        manifest_id=new_manifest_id(),
        kind="experiment_run",
        flow=describe(step).flow,
        revision=Revision(authored=def_.revision, resolved=resolved),
        environment=Environment(python=platform.python_version(), decider=_decider_version()),
        input=data.summary,
        outputs=tuple(r.output_ref for r in results if r.output_ref is not None),
    )


def _decider_version() -> str:
    try:
        return metadata.version("decider")
    except metadata.PackageNotFoundError:
        return ""
