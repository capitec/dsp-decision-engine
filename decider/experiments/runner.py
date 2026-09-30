"""Run an experiment: validate scenarios, run baseline and variants, compare, and write a manifest.

`run_experiment` is the only place execution happens. It reuses the rehomed
fork/sweep primitives (a fresh session replayed to a checkpoint and overridden)
for execution, `describe` for the static flow contract, `preflight` for
compatibility and cost, and the lifecycle `RunManifest`/`JobHandle` for
provenance and job semantics. Param-only scenarios route to the fused
`step.run(frame, params=...)` path; a scenario that also overrides a value or a
single row runs through `fork`. `run` is the headless entry point: it resolves
the flow and data from the definition, so a plain Python caller runs an
experiment end to end without VS Code.
"""
from __future__ import annotations

import importlib
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import polars as pl

from decider.contract import describe
from decider.data import LoadedData, load, preflight
from decider.exceptions import DeciderError
from decider.experiments.compare import close, diff_outputs
from decider.experiments.forks import fork, merge
from decider.experiments.graphs import path_sankey
from decider.experiments.model import ComparisonPolicy, ExperimentDef, Tolerance, check_experiment_version
from decider.experiments.result import (ExperimentResult, Finding, RunStatus, ScenarioResult, ScenarioStatus)
from decider.experiments.revision import compatible, materialise, remove, resolve
from decider.experiments.save import write as write_results
from decider.experiments.summaries import summarise
from decider.lifecycle import Environment, JobHandle, ResultRef, Revision, RunManifest, new_manifest_id


def run(def_: ExperimentDef, *, frame: pl.DataFrame | None = None, out_dir: str | None = None,
        job: JobHandle | None = None, resume: ExperimentResult | None = None) -> ExperimentResult:
    """Run `def_` headlessly: load the flow by `flow.entry` and, without `frame`, the input by `input.path`.

    A plain Python caller can define, run and compare an experiment end to end
    with no VS Code::

        def_ = ExperimentDef.model_validate(load_yaml(Path("experiments/drift/experiment.yaml").read_text()))
        result = run(def_, out_dir="results/drift")
    """
    check_experiment_version(def_.version)
    step = load_flow(def_.flow.entry)
    if frame is None:
        if not def_.input.path:
            raise DeciderError("no input data: set input.path in the definition, or pass frame=")
        frame = load(def_.input.path).frame
    return run_experiment(def_, step, frame, job=job, resume=resume, out_dir=out_dir)


def run_experiment(def_: ExperimentDef, step, frame, params=None, *, job: JobHandle | None = None,
                   resume: ExperimentResult | None = None, out_dir: str | None = None) -> ExperimentResult:
    """Run every scenario of `def_` next to a baseline and compare, returning an `ExperimentResult`.

    Each scenario runs through `fork` (a fresh session replayed to a checkpoint,
    with the scenario's params and overrides applied), so one failing scenario
    never aborts the sweep; a scenario that only changes params runs on the fused
    `step.run(frame, params=...)` path instead. `job` reports progress and can
    cancel or time out the run between scenarios; `resume` (a prior
    `ExperimentResult`) skips a completed scenario only when the input
    fingerprint and resolved revision still match. `out_dir` writes each
    scenario's output plus the manifest, summary and Sankey.

    Example::

        result = run_experiment(def_, pipeline, df, pipeline.parameters().defaults())
        result.status  # RunStatus.COMPLETED, or NON_REPRODUCIBLE / PARTIAL
    """
    if params is None:
        params = step.parameters().defaults()
    if def_.flow.decider and not compatible(def_.flow.decider, _decider_version()):
        raise DeciderError(f"flow needs decider {def_.flow.decider}, installed {_decider_version()}")
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

    data = LoadedData.from_frame(frame, dataset=Path(def_.input.path).name if def_.input.path else "")
    resolved = resolve(def_.revision)
    done = _resumable(resume, data, resolved)

    results: list[ScenarioResult] = []
    scenario_outputs: dict[str, dict | None] = {}
    scenario_traces: list[dict] = []
    trees: list[Path] = []
    try:
        for i, sc in enumerate(def_.scenarios):
            if job is not None:
                if job.timed_out:
                    job.timeout()
                if job.cancelled or job.timed_out:
                    break
                job.progress(i, total=len(def_.scenarios), message=sc.name)
            if sc.name in done:
                results.append(ScenarioResult(name=sc.name, status=ScenarioStatus.RESUMED,
                                              params=sc.params, overrides=sc.overrides, row=sc.row))
                continue
            scenario_revision, step_for_sc = _scenario_revision(def_, sc, step, resolved, trees)
            if isinstance(step_for_sc, str):
                results.append(ScenarioResult(name=sc.name, status=ScenarioStatus.FAILED,
                                              error=step_for_sc, revision=scenario_revision,
                                              params=sc.params, overrides=sc.overrides, row=sc.row))
                continue
            if _param_only(sc):
                trace = _fused_run(step_for_sc, frame, params, sc)
            else:
                trace = _fork_run(step_for_sc, frame, params, sc)
            scenario_outputs[sc.name] = trace["output"]
            scenario_traces.append({"name": sc.name, **trace})
            if trace["error"]:
                results.append(ScenarioResult(
                    name=sc.name, status=ScenarioStatus.FAILED, error=trace["error"], revision=scenario_revision,
                    params=sc.params, overrides=sc.overrides, row=sc.row,
                    divergences=(Finding(kind="error", scenario=sc.name, message=trace["error"]),),
                ))
            else:
                results.append(ScenarioResult(
                    name=sc.name, status=ScenarioStatus.COMPLETED,
                    divergences=_divergences(baseline["output"], trace["output"], def_.comparison),
                    output_ref=ResultRef(name=sc.name, location=f"results/{sc.name}.parquet"),
                    row_count=frame.height, revision=scenario_revision,
                    params=sc.params, overrides=sc.overrides, row=sc.row,
                ))
    finally:
        for tree in trees:
            remove(tree)

    status = _run_status(results, nondeterministic, job)
    manifest = _manifest(def_, step, data, resolved, results)
    summary = summarise(def_, baseline, scenario_traces)
    sankey = path_sankey(step, frame, params, baseline, def_.comparison.outputs) if baseline["output"] else None
    if out_dir is not None:
        written = write_results(out_dir, ExperimentResult(manifest=manifest, status=status), summary, sankey,
                                {"baseline": baseline["output"], **scenario_outputs})
        for r in results:
            if r.output_ref is not None and r.name in written:
                r.output_ref = r.output_ref.model_copy(update={"location": written[r.name]})
    if job is not None and not job.terminal:
        job.succeed(result={"status": status.value, "scenarios": len(results)})
    return ExperimentResult(manifest=manifest, status=status, name=def_.name, nondeterministic=nondeterministic,
                            scenarios=tuple(results), preflight=report,
                            job=job.snapshot() if job is not None else None,
                            summary=summary, sankey=sankey)


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
    return resolve(authored)


def load_flow(entry: str):
    """Import `entry` (`"module:qualname"`) and call it if it is a zero-argument factory."""
    module, _, qualname = entry.partition(":")
    if not module or not qualname:
        raise ValueError(f"entry {entry!r} must be 'module:qualname'")
    return _load_flow_in(entry)


def _load_flow_in(entry: str, tree: Path | None = None):
    module, _, qualname = entry.partition(":")
    if tree is not None:
        sys.path.insert(0, str(tree))
    try:
        obj = importlib.import_module(module)
    finally:
        if tree is not None:
            sys.path.pop(0)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj() if callable(obj) else obj


def _param_only(sc) -> bool:
    return sc.at is None and not sc.overrides and sc.revision is None and sc.row is None


def _scenario_revision(def_: ExperimentDef, sc, step, resolved: str | None, trees: list[Path]):
    """The scenario's resolved revision and the step to run it on; a string return is the failure reason."""
    if sc.revision is None:
        return None, step
    sha = resolve(sc.revision)
    revision = Revision(authored=sc.revision, resolved=sha)
    if sha is None:
        return revision, f"revision {sc.revision!r} did not resolve"
    if not compatible(def_.flow.decider, _decider_version()):
        return revision, f"revision {sha} is not engine-compatible with decider {_decider_version()}"
    if sha == resolved or resolved is None:
        return revision, step
    tree = materialise(sha)
    trees.append(tree)
    return revision, _load_flow_in(def_.flow.entry, tree)


def _fused_run(step, frame, params, sc) -> dict:
    doc = merge(params, sc.params)
    try:
        out = step.run(frame, params=doc)
    except Exception as e:
        return {"output": None, "steps": {}, "paths": {}, "error": f"{type(e).__name__}: {e}"}
    return {"output": {c: out[c].to_list() for c in out.columns}, "steps": {}, "paths": {}, "error": None}


def _fork_run(step, frame, params, sc) -> dict:
    target = None if sc.at is None else (sc.at, "after", 1)
    scenario = {"params": sc.params, "overrides": sc.overrides, "row": sc.row}
    try:
        return fork(step, frame, params, [], target, scenario)
    except Exception as e:
        return {"output": None, "steps": {}, "paths": {}, "error": f"{type(e).__name__}: {e}"}


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
            if not close(a, b, tol):
                out.append(Finding(kind="divergence", location=f"{col}[{i}]", expected=a, actual=b))
    return tuple(out)


def _divergences(base, other, comparison: ComparisonPolicy) -> tuple[Finding, ...]:
    changed = diff_outputs(base, other, comparison)
    out = []
    for col, rows in changed.items():
        for i in rows:
            out.append(Finding(kind="divergence", location=f"{col}[{i}]",
                               expected=base[col][i], actual=other[col][i]))
    return tuple(out)


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
