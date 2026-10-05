"""Run an experiment: the baseline and every scenario over the same batches of records, feeding callbacks.

The runner only executes. For each batch it runs the baseline, then each
scenario, and hands every output (and, to callbacks that select nodes, what
each matching node wrote) to the shared callbacks and to that scenario's copy
of each metric. A flow runs fused unless something selects nodes or a scenario
overrides a produced value; then it runs one compiled kernel per node. A
scenario on another git revision runs whole in a fresh subprocess
(`decider.experiments._worker`) and is read back batch by batch, in step with
the rest.
"""
from __future__ import annotations

import importlib
import json
import pickle
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import replace
from fnmatch import fnmatchcase
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator, Sequence

import polars as pl

from decider.contract import describe
from decider.data import LoadedData, preflight, scan
from decider import Engine
from decider._engine.debug.session import _cast
from decider._engine.ir.decls import base_annotation
from decider._engine.params import merge
from decider._engine.run.state import State, dtype_of
from decider._engine.wiring.plan import Branch, Loop, ScatterGather
from decider._engine.wiring.plan import Sequence as Sequence_
from decider.exceptions import DeciderError
from decider.experiments.callbacks import (Batch, Callback, JobProgress, NodeBatch, Run, WriteOutputs, overrides,
                                           selects)
from decider.experiments._experiments import (BASELINE, ExperimentDef, ExperimentResult, RunStatus, Scenario,
                                             ScenarioResult, ScenarioStatus, check_experiment_version, input_frame, read_definition,
                                             validate)
from decider.experiments.metrics._base import Metric
from decider.experiments.plots._base import Figure
from decider.diff._revisions import compatible, materialise, remove, resolve
from decider.contract import Environment, InputFingerprint, JobHandle, ResultRef, Revision, RunManifest
from decider.contract._manifest import new_manifest_id

Frame = pl.DataFrame | pl.LazyFrame


def run(def_: ExperimentDef | str | Path, *, frame: Frame | None = None, out_dir: str | None = None,
        job: JobHandle | None = None, callbacks: Sequence[Callback] = ()) -> ExperimentResult:
    """Run an experiment headlessly: load the flow by `flow.entry` and, without `frame`, scan `input.path` or `input.dataset`.

    Give the path of an `experiment.json` and `input.path` is relative to that file and
    `input.dataset` is looked up in its project. Given an `ExperimentDef`, `input.path`
    is relative to the working directory.

    Example::

        result = run("experiments/drift/experiment.json", out_dir="results/drift")
    """
    file = None
    if not isinstance(def_, ExperimentDef):
        file = Path(def_)
        def_ = read_definition(file)
    check_experiment_version(def_.version)
    step = load_flow(def_.flow.entry)
    if frame is None:
        frame = input_frame(def_, file)
    return run_experiment(def_, step, frame, callbacks=callbacks, job=job, out_dir=out_dir)


def run_experiment(def_: ExperimentDef, step, frame: Frame, params=None, *, callbacks: Sequence[Callback] = (),
                   job: JobHandle | None = None, out_dir: str | None = None) -> ExperimentResult:
    """Run the baseline and every scenario of `def_` over `frame`, batch by batch, collecting its metrics.

    `params` is the baseline's params document (the flow's defaults when
    omitted). `callbacks` see every scenario; `job` reports progress and can
    cancel or time out the run between batches; `out_dir` gets each scenario's
    output as parquet plus `result.json`. A failing scenario never stops the
    others; a failing baseline stops the run.

    Example::

        result = run_experiment(def_, pipeline, df)
        result.status   # RunStatus.COMPLETED, or NON_REPRODUCIBLE / PARTIAL / FAILED
    """
    if def_.flow.decider and not compatible(def_.flow.decider, _decider_version()):
        raise DeciderError(f"flow needs decider {def_.flow.decider}, installed {_decider_version()}")
    problems = validate(def_, step)
    if problems:
        raise ValueError("; ".join(problems))
    base_params = params if params is not None else step.parameters().defaults()
    total = _height(frame)
    report = preflight(step, _collect(frame.head(def_.batch_size)), params=base_params)
    scenarios = [Scenario(name=BASELINE), *def_.scenarios]
    shared = [*callbacks, *([JobProgress(job)] if job else []), *([WriteOutputs(out_dir)] if out_dir else [])]
    metrics = {sc.name: [_fresh(m) for m in def_.metrics] for sc in scenarios}
    everyone = [*shared, *(m for ms in metrics.values() for m in ms)]
    listening = [cb for cb in [*shared, *def_.metrics] if overrides(cb, "on_node_end")]
    selectors = {"nodes": sorted({cb.nodes for cb in listening if getattr(cb, "nodes", None)}),
                 "ids": sorted({cb.node_id for cb in listening if getattr(cb, "node_id", None)})}
    run_ = Run(def_.name, step, tuple(sc.name for sc in scenarios), total, def_.batch_size)
    errors: dict[str, str] = {}
    records: Counter[str] = Counter()
    trees: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="decider-experiment-") as tmp:
        try:
            executors, revisions = _executors(def_, step, frame, params, base_params, scenarios, selectors,
                                              Path(tmp), trees, errors)
            for cb in everyone:
                cb.on_run_start(run_)
            nondeterministic = _drive(def_, frame, scenarios, executors, shared, metrics, everyone, run_,
                                      errors, records)
        finally:
            for tree in trees.values():
                remove(tree)

    status = _status(run_.stop, errors, nondeterministic)
    run_.status = status.value
    for cb in everyone:
        cb.on_run_end(run_)
    summaries: dict[str, dict[str, Any]] = {m.key: {} for m in def_.metrics}
    for sc in scenarios:
        if sc.name not in errors:
            for m in metrics[sc.name]:
                summaries[m.key][sc.name] = m.summarise()
    results = tuple(ScenarioResult(
        name=sc.name, status=ScenarioStatus.FAILED if sc.name in errors else ScenarioStatus.COMPLETED,
        error=errors.get(sc.name), revision=revisions.get(sc.name), params=sc.params, overrides=sc.overrides,
        at=sc.at, row=sc.row, records=records[sc.name],
        output_ref=ResultRef(name=sc.name, location=str(Path(out_dir) / sc.name)) if out_dir else None,
    ) for sc in scenarios)
    result = ExperimentResult(
        manifest=_manifest(def_, step, frame, total, resolve(def_.revision), results), status=status,
        name=def_.name, nondeterministic=nondeterministic, scenarios=results, preflight=report,
        job=job.snapshot() if job is not None else None, summaries=summaries,
        plots={p.key: Figure.of(p.render(summaries[p.metric])) for p in def_.plots}, definition=def_,
    )
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "result.json").write_text(result.model_dump_json(indent=2))
    return result


def _drive(def_, frame, scenarios, executors, shared, metrics, everyone, run_, errors, records) -> bool:
    nondeterministic = False
    for index, (offset, batch) in enumerate(_batches(frame, def_.batch_size)):
        baseline: pl.DataFrame | None = None
        baseline_nodes: dict[tuple[str, int | None], NodeBatch] = {}
        for sc in scenarios:
            if sc.name in errors:
                continue
            try:
                out, nodes = executors[sc.name].run(index, offset, batch)
                if sc.name == BASELINE and def_.determinism and not nondeterministic:
                    nondeterministic = not out.equals(executors[BASELINE].run(index, offset, batch)[0])
            except Exception as e:  # noqa: BLE001  one failing scenario never stops the others
                errors[sc.name] = f"{type(e).__name__}: {e}"
                continue
            if sc.name == BASELINE:
                baseline, baseline_nodes = out, {(n.path, n.iteration): n for n in nodes}
            listeners = [*shared, *metrics[sc.name]]
            for n in nodes:
                node = replace(n, scenario=sc.name, baseline=baseline_nodes.get((n.path, n.iteration)))
                for cb in listeners:
                    if selects(cb, node.path, node.id):
                        cb.on_node_end(node)
            done = Batch(sc.name, index, offset, out, baseline)
            for cb in listeners:
                cb.on_batch_end(done)
            records[sc.name] += out.height
        if BASELINE in errors:
            break
        for cb in everyone:
            cb.on_progress(run_, offset + batch.height)
        if run_.stop:
            break
    return nondeterministic


def _executors(def_, step, frame, params, base_params, scenarios, selectors, tmp: Path, trees, errors):
    executors: dict[str, Any] = {}
    revisions: dict[str, Revision] = {}
    resolved = resolve(def_.revision)
    data = None
    for sc in scenarios:
        if sc.revision is None:
            executors[sc.name] = _Local(step, base_params, sc, selectors)
            continue
        sha = resolve(sc.revision)
        revisions[sc.name] = Revision(authored=sc.revision, resolved=sha)
        if sha is None:
            errors[sc.name] = f"revision {sc.revision!r} did not resolve"
        elif sha == resolved or resolved is None:
            executors[sc.name] = _Local(step, base_params, sc, selectors)
        else:
            try:
                tree = trees[sha] = trees.get(sha) or materialise(sha)
            except DeciderError as e:
                errors[sc.name] = str(e)
                continue
            if data is None:
                data = tmp / "input.parquet"
                frame.sink_parquet(data) if isinstance(frame, pl.LazyFrame) else frame.write_parquet(data)
            executors[sc.name] = _Revision({
                "entry": def_.flow.entry, "tree": str(tree), "input": str(data), "params": params,
                "scenario": sc.model_dump(exclude={"name", "revision"}), "selectors": selectors,
                "batch_size": def_.batch_size, "out": str(tmp / f"revision-{len(executors)}"),
            })
    return executors, revisions


class _Local:
    """Runs one scenario's batches in this process."""

    def __init__(self, step, params, sc: Scenario, selectors: dict[str, list[str]]):
        self.sc = sc
        self.patterns, self.ids = selectors["nodes"], set(selectors["ids"])
        self.params = merge(params, sc.params)
        stepped = self.patterns or self.ids or sc.at is not None
        self.exe = Engine().bind(step, mode="stepped" if stepped else "fused")
        # What each node hands a node metric: a call's outputs, a branch's `modifies` names.
        self.writes = {c.node.origin.path: c.writes for c in self.exe.plan.calls}
        self.writes.update((b.node.origin.path, tuple(m.version for m in b.merges))
                           for b in _branches(self.exe.plan.root))

    def run(self, index: int, offset: int, batch: pl.DataFrame) -> tuple[pl.DataFrame, list[NodeBatch]]:
        sc = self.sc
        if sc.overrides and sc.at is None:
            batch = _set_inputs(batch, sc, offset)
        state, params = self.exe.prepare(batch, self.params)
        pending = bool(sc.overrides) and sc.at is not None
        nodes = []
        for cp in self.exe.runner.iterate(self.exe.plan, state, params):
            if cp.when != "after":
                continue
            path = cp.origin.path
            if pending and path == sc.at:
                _set_values(self.exe.plan, state, sc, offset)
                pending = False
            if cp.origin.id in self.ids or any(fnmatchcase(path, p) for p in self.patterns):
                nodes.append(NodeBatch("", path, cp.origin.id, index, offset, self._values(state, path, cp.rows),
                                       cp.rows, cp.arm, cp.iteration))
        return self.exe.output(state), nodes

    def _values(self, state: State, path: str, rows) -> pl.DataFrame:
        frame = pl.DataFrame([state.column(v.name, v) for v in self.writes.get(path, ()) if v.id in state.values])
        return frame if rows is None else frame[rows]


class _Revision:
    """Runs one scenario whole in a subprocess on first use, then reads its batches back."""

    def __init__(self, request: dict[str, Any]):
        self.request = request
        self.ran = False

    def run(self, index: int, offset: int, batch: pl.DataFrame) -> tuple[pl.DataFrame, list[NodeBatch]]:
        if not self.ran:
            self.ran = True
            proc = subprocess.run([sys.executable, "-m", "decider.experiments._worker"],
                                  input=json.dumps(self.request), capture_output=True, text=True)
            reply = _reply(proc.stdout, proc.stderr, proc.returncode)
            if not reply["ok"]:
                raise DeciderError(reply["error"])
        return pickle.loads((Path(self.request["out"]) / f"{index}.pkl").read_bytes())


def _branches(r) -> Iterator[Branch]:
    children: tuple = ()
    if isinstance(r, Branch):
        yield r
        children = (r.condition, *r.arms)
    elif isinstance(r, Sequence_):
        children = r.children
    elif isinstance(r, Loop):
        children = (r.condition, r.body)
    elif isinstance(r, ScatterGather):
        children = (r.body,) if r.accumulate is None else (r.body, r.accumulate)
    for child in children:
        yield from _branches(child)


def _reply(stdout: str, stderr: str, returncode: int) -> dict:
    try:
        return json.loads(stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"ok": False, "error": f"revision subprocess exited {returncode}: {(stderr or stdout)[-500:]}"}


def _set_inputs(batch: pl.DataFrame, sc: Scenario, offset: int) -> pl.DataFrame:
    if sc.row is not None and not offset <= sc.row < offset + batch.height:
        return batch
    for name, value in sc.overrides.items():
        new = pl.lit(value).cast(batch.schema[name]) if name in batch.columns else pl.lit(value)
        if sc.row is not None:
            new = pl.when(pl.int_range(pl.len()) == sc.row - offset).then(new).otherwise(pl.col(name))
        batch = batch.with_columns(new.alias(name))
    return batch


def _set_values(plan, state: State, sc: Scenario, offset: int) -> None:
    # Every version of the name written so far, as a debug session's `set` does: later nodes may read any of them.
    if sc.row is not None and not offset <= sc.row < offset + state.n:
        return
    for name, value in sc.overrides.items():
        inputs = [v for v in plan.versions if v.producer is None and v.name == name]
        targets = inputs + [v for v in state.chains.get(name, ()) if v.id in state.values and v not in inputs]
        if not targets:
            raise KeyError(f"{name!r} is neither an input column nor a value produced by {sc.at!r}")
        if sc.row is not None:
            current = state.column(name, targets[-1]).to_list()
            current[sc.row - offset] = value
            value = current
        values, valid = _cast(name, value, dtype_of(base_annotation(targets[-1].annotation)), state.n)
        for v in targets:
            state.write(v, values.copy(), valid=None if valid is None else valid.copy())


def load_flow(entry: str, tree: Path | None = None):
    """Import `entry` (`"module:qualname"`), from `tree` when given, and call it if it is a zero-argument factory."""
    module, _, qualname = entry.partition(":")
    if not module or not qualname:
        raise ValueError(f"entry {entry!r} must be 'module:qualname'")
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


def _batches(frame: Frame, size: int) -> Iterator[tuple[int, pl.DataFrame]]:
    # ponytail: a lazy CSV rescans from the start for each slice; batch with collect_batches if that bites.
    for offset in range(0, _height(frame), size):
        yield offset, _collect(frame.slice(offset, size))


def _height(frame: Frame) -> int:
    return frame.height if isinstance(frame, pl.DataFrame) else frame.select(pl.len()).collect().item()


def _collect(frame: Frame) -> pl.DataFrame:
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def _fresh(metric: Metric) -> Metric:
    copy = metric.model_copy(deep=True)
    copy.reset()
    return copy


def _status(stop: str | None, errors: dict[str, str], nondeterministic: bool) -> RunStatus:
    if stop:
        return RunStatus(stop)
    if BASELINE in errors:
        return RunStatus.FAILED
    if nondeterministic:
        return RunStatus.NON_REPRODUCIBLE
    return RunStatus.PARTIAL if errors else RunStatus.COMPLETED


def _manifest(def_: ExperimentDef, step, frame: Frame, total: int, resolved: str | None,
              results: tuple[ScenarioResult, ...]) -> RunManifest:
    dataset = def_.input.dataset or Path(def_.input.path).name
    if isinstance(frame, pl.DataFrame):
        data = LoadedData.from_frame(frame, dataset=dataset).summary
    else:
        data = InputFingerprint(dataset=dataset, row_count=total,
                                columns=tuple((c, str(t)) for c, t in frame.collect_schema().items()))
    return RunManifest(
        manifest_id=new_manifest_id(),
        kind="experiment_run",
        flow=describe(step).flow,
        revision=Revision(authored=def_.revision, resolved=resolved),
        environment=Environment(python=platform.python_version(), decider=_decider_version()),
        input=data,
        outputs=tuple(r.output_ref for r in results if r.output_ref is not None),
    )


def _decider_version() -> str:
    try:
        return metadata.version("decider")
    except metadata.PackageNotFoundError:
        return ""
