"""An experiment as data, its result, and the `Experiment` builder notebooks use.

`ExperimentDef` is the whole asset, saved as `experiment.json`
(`model_dump_json` / `model_validate_json`): a flow entry point, the input,
scenarios to run next to the unchanged baseline, metrics to collect and plots
to draw from them. Execution lives in `decider.experiments._runner`.
"""
from __future__ import annotations

from collections import Counter
from fnmatch import fnmatchcase
from enum import Enum
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, SerializeAsAny, model_validator

from decider.contract import describe
from decider.checks import Report as CheckReport
from decider.data import dataset, scan
from decider.experiments._cost import CostEstimate
from decider.ir import iter_nodes, to_ir
from decider.exceptions import DeciderError
from decider.experiments.callbacks import Callback
from decider.experiments.metrics._base import Metric, MetricRef, Summary
from decider.experiments.report import _blocks as report
from decider.experiments.plots._base import Figure, Plot, PlotRef, ipython_display
from decider.experiments.report import Block, Report
from decider.contract import Job, ResultRef, Revision, RunManifest
from decider.experiments._entry import import_flow, infer_entry
from decider.experiments._handle import RunHandle

EXPERIMENT_VERSION = "2"
BASELINE = "baseline"


class FlowSpec(BaseModel):
    """The flow an experiment runs: an import path and the engine window it needs.

    `entry` is `"module:qualname"`, a zero-argument factory or a module-level
    pipeline; `decider` is a version specifier (`">=1.0,<2.0"`) the installed
    decider must satisfy.
    """

    model_config = ConfigDict(extra="forbid")

    entry: str
    decider: str = ""


class InputSpec(BaseModel):
    """The input data: a `.parquet`, `.csv`, `.json` or `.jsonl` `path`, or the `dataset` name of one of the project's datasets, not both."""

    model_config = ConfigDict(extra="forbid")

    path: str = ""
    dataset: str = ""

    @model_validator(mode="after")
    def _one_source(self):
        if self.path and self.dataset:
            raise ValueError("set input.path or input.dataset, not both")
        return self


class Scenario(BaseModel):
    """One controlled change to run next to the baseline.

    `params` merge over the run's params document; `overrides` write values at a
    declared point — `at` names the step whose output to override, or is `None`
    for an input column before anything runs. `revision` runs the scenario on
    another git revision (e.g. `"HEAD^"`); `row` overrides one record only.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    revision: str | None = None
    params: dict[str, Any] = {}
    overrides: dict[str, Any] = {}
    at: str | None = None
    row: int | None = None


class ExperimentDef(BaseModel):
    """The `experiment.json` asset: everything one experiment runs and measures.

    The baseline is the unchanged flow, always run first, under the name
    `"baseline"`. `batch_size` records run per batch, so progress shows and no
    metric needs the whole input; `determinism` reruns each baseline batch and
    flags a differing rerun as non-reproducible. `report` lays out what
    `result.show()` and `result.save_html()` show.
    """

    model_config = ConfigDict(extra="forbid")

    version: str = EXPERIMENT_VERSION
    name: str
    description: str = ""
    flow: FlowSpec
    revision: str = "HEAD"
    input: InputSpec = InputSpec()
    scenarios: list[Scenario] = []
    metrics: list[MetricRef] = []
    plots: list[PlotRef] = []
    batch_size: int = 100_000
    determinism: bool = True
    report: Report | None = None


class RunStatus(str, Enum):
    """The terminal quality of an experiment run.

    `NON_REPRODUCIBLE` means a baseline rerun differed; `PARTIAL` means some
    scenario failed; `FAILED` means the baseline did. None of them reads as
    `COMPLETED`.
    """

    COMPLETED = "completed"
    NON_REPRODUCIBLE = "non_reproducible"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class ScenarioStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"


class ScenarioResult(BaseModel):
    """One scenario's outcome: its status, error, resolved revision and how many records it ran."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: ScenarioStatus
    error: str | None = None
    revision: Revision | None = None
    params: dict[str, Any] = {}
    overrides: dict[str, Any] = {}
    at: str | None = None
    row: int | None = None
    records: int = 0
    output_ref: ResultRef | None = None


class ExperimentResult(BaseModel):
    """A finished (or stopped) run: the manifest, statuses, and what the metrics and plots produced.

    `summaries` is `{metric: {scenario: Summary}}`, for scenarios that
    completed; `plots` is `{plot: Figure}`. In a notebook, `show()` draws the
    report and `plots[name].show()` one chart; `save_html` writes the report.
    """

    model_config = ConfigDict(extra="forbid")

    manifest: RunManifest
    status: RunStatus
    name: str = ""
    nondeterministic: bool = False
    scenarios: tuple[ScenarioResult, ...] = ()
    preflight: CheckReport | None = None
    cost: CostEstimate | None = None
    job: Job | None = None
    summaries: dict[str, dict[str, SerializeAsAny[Summary]]] = {}
    plots: dict[str, Figure] = {}
    definition: ExperimentDef | None = None

    def show(self) -> None:
        """Show the report in a notebook: the experiment's own, or an overview and a section per metric."""
        ipy = ipython_display()
        ipy.display(ipy.HTML(report.fragment(self)))

    def html(self) -> str:
        """The report as a standalone HTML page."""
        return report.page(self)

    def save_html(self, path: str | Path) -> Path:
        """Write the report to `path` as one HTML page; its charts load Vega from the CDN when opened."""
        path = Path(path)
        path.write_text(self.html())
        return path


def check_experiment_version(version: str | None) -> None:
    """Raise if `version` is newer than this decider supports."""
    if version is None:
        return
    try:
        given = int(version.partition(".")[0])
    except ValueError:
        raise DeciderError(f"unrecognised experiment version {version!r}") from None
    if given > int(EXPERIMENT_VERSION):
        raise DeciderError(f"experiment version {version} is newer than {EXPERIMENT_VERSION}; upgrade decider")


def read_definition(path: Path) -> ExperimentDef:
    def_ = ExperimentDef.model_validate_json(path.read_text())
    check_experiment_version(def_.version)
    return def_


def input_frame(def_: ExperimentDef, file: Path | None = None) -> pl.LazyFrame:
    """The lazy frame `def_.input` names; `file` is the definition's own path, which paths and the project are found from."""
    spec = def_.input
    if spec.dataset:
        return dataset(spec.dataset, start=file).scan()
    if not spec.path:
        raise DeciderError("no input data: set input.path or input.dataset in the definition, or pass frame=")
    return scan(str(file.parent / spec.path if file else spec.path))


def validate(def_: ExperimentDef, step) -> list[str]:
    """Every problem in `def_` against `step`, returned rather than raised so a caller shows them all.

    Checks scenario names, that a scenario on another revision has a flow
    `entry` to import (or one that can be inferred from the pipeline), that each override targets an input or a produced
    value (a produced one needs `at`), that metric and plot names are unique,
    and that each plot's metric exists and gives the summary the plot requires.
    """
    problems = []
    names = Counter(sc.name for sc in def_.scenarios)
    if BASELINE in names:
        problems.append(f"{BASELINE!r} is the unchanged run; give the scenario another name")
    problems += [f"two scenarios are named {n!r}" for n, k in names.items() if k > 1]
    if not def_.flow.entry and any(sc.revision for sc in def_.scenarios) and not infer_entry(step):
        problems += [f"{sc.name}: revision {sc.revision!r} runs in a subprocess that imports flow.entry, and the "
                     "pipeline's module has no variable or zero-argument function that rebuilds it; set flow.entry"
                     for sc in def_.scenarios if sc.revision]
    problems += _override_problems(def_, step)
    problems += _node_problems(def_, step)
    metrics: dict[str, Metric] = {}
    for m in def_.metrics:
        if m.key in metrics:
            problems.append(f"two metrics are named {m.key!r}; set name= on one")
        metrics[m.key] = m
    plots = Counter(p.key for p in def_.plots)
    problems += [f"two plots are named {n!r} (an unnamed plot takes its metric's name); set name= on one"
                 for n, k in plots.items() if k > 1]
    if def_.report is not None:
        problems += report.problems(def_.report, set(metrics), set(plots))
    for p in def_.plots:
        m = metrics.get(p.metric)
        if m is None:
            problems.append(f"plot {p.key!r} draws metric {p.metric!r}, which the experiment doesn't collect")
        elif not issubclass(m.summary, p.requires):
            problems.append(f"plot {p.key!r} needs a {p.requires.__name__} summary; "
                            f"metric {p.metric!r} gives {m.summary.__name__}")
    return problems


def _node_problems(def_: ExperimentDef, step) -> list[str]:
    nodes = list(iter_nodes(to_ir(step)))
    paths, ids = [n.origin.path for n in nodes], {n.origin.id for n in nodes}
    problems = []
    for m in def_.metrics:
        if m.nodes is not None and m.node_id is not None:
            problems.append(f"metric {m.key!r} sets both nodes and node_id; pick one")
        elif m.node_id is not None and m.node_id not in ids:
            problems.append(f"metric {m.key!r}: no step has id {m.node_id!r}")
        elif m.nodes is not None and not any(fnmatchcase(p, m.nodes) for p in paths):
            problems.append(f"metric {m.key!r}: nodes={m.nodes!r} matches no step")
    return problems


def _override_problems(def_: ExperimentDef, step) -> list[str]:
    slots: dict[str, list[str | None]] = {}
    for v in describe(step).value_slots:
        slots.setdefault(v.name, []).append(v.path)
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


class Experiment:
    """A builder for one experiment in a notebook: scenarios, metrics, plots, callbacks, then `run`.

    `params` is the baseline's params document; without one, each revision
    runs on its own code's defaults.

    Example::

        result = (Experiment("drift", pipeline, df)
                  .scenario("limit_0.3", params={"demo": {"approved": {"limit": 0.3}}})
                  .metric(Divergence(column="approved"))
                  .plot(Changes(metric="divergence"))
                  .run())
        result.summaries["divergence"]["limit_0.3"].changed
    """

    def __init__(self, name: str, step, frame: pl.DataFrame | pl.LazyFrame, *, params=None, revision: str = "HEAD",
                 batch_size: int = 100_000):
        self.step = step
        self.frame = frame
        self.params = params
        self.callbacks: list[Callback] = []
        self.def_ = ExperimentDef(name=name, flow=FlowSpec(entry=""), revision=revision, batch_size=batch_size)

    @classmethod
    def from_def(cls, def_: ExperimentDef, step=None, frame: pl.DataFrame | pl.LazyFrame | None = None, *,
                 params=None) -> Experiment:
        """An experiment from a whole `ExperimentDef`, including its protocol-only fields (`flow`, `input`, `determinism`).

        Without `step` the flow is imported from `flow.entry`; without `frame`
        the input is scanned from `input.path` or `input.dataset`.

        Example::

            exp = Experiment.from_def(ExperimentDef(name="x", flow={"entry": "credit:build"}, input={"path": "loans.parquet"}))
        """
        check_experiment_version(def_.version)
        exp = cls(def_.name, step if step is not None else import_flow(def_.flow.entry),
                  frame if frame is not None else input_frame(def_), params=params)
        exp.def_ = def_
        return exp

    def scenario(self, name: str, *, params: dict[str, Any] | None = None,
                 overrides: dict[str, Any] | None = None, at: str | None = None,
                 revision: str | None = None, row: int | None = None) -> Experiment:
        self.def_.scenarios.append(Scenario(name=name, params=params or {}, overrides=overrides or {},
                                            at=at, revision=revision, row=row))
        return self

    def metric(self, *metrics: Metric) -> Experiment:
        self.def_.metrics.extend(metrics)
        return self

    def plot(self, *plots: Plot) -> Experiment:
        self.def_.plots.extend(plots)
        return self

    def report(self, *blocks: Block, title: str = "") -> Experiment:
        """Lay out the report `result.show()` draws, top to bottom; see `decider.experiments.report`."""
        self.def_.report = Report(title=title, blocks=list(blocks))
        return self

    def callback(self, *callbacks: Callback) -> Experiment:
        """Add callbacks that see every scenario (progress, writers); they are not saved with the definition."""
        self.callbacks.extend(callbacks)
        return self

    def save(self, path: str | Path) -> Path:
        """Write the definition to `path` as JSON; `Experiment.load` reads it back.

        An empty `flow.entry` is filled from the pipeline when it can be inferred, so the file runs elsewhere.
        """
        path = Path(path)
        def_ = self.def_
        if not def_.flow.entry:
            def_ = def_.model_copy(update={"flow": def_.flow.model_copy(update={"entry": infer_entry(self.step)})})
        path.write_text(def_.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path, *, step=None, frame: pl.DataFrame | pl.LazyFrame | None = None,
             params=None) -> Experiment:
        """An experiment from an `experiment.json`, ready to change further or `run`.

        Without `step` the flow is imported from `flow.entry`; without `frame`
        the input is scanned from `input.path` (relative to the JSON file) or
        `input.dataset` (of the project holding the JSON file).

        Example::

            result = Experiment.load("experiments/premium_sweep/experiment.json").run()
        """
        path = Path(path)
        def_ = read_definition(path)
        return cls.from_def(def_, step, frame if frame is not None else input_frame(def_, path), params=params)

    def run(self, *, out_dir: str | None = None) -> ExperimentResult:
        """Run the baseline and every scenario to the end; `out_dir` gets each scenario's output and `result.json`."""
        from decider.experiments._runner import _run

        return _run(self.def_, self.step, self.frame, self.params, callbacks=self.callbacks, out_dir=out_dir)

    def start(self, *, out_dir: str | None = None) -> RunHandle:
        """Run in a background thread and return a handle with `.status`, `.progress`, `.cancel()` and `.result()`.

        Example::

            handle = experiment.start()
            handle.cancel()          # stops at the next batch
            handle.result().status   # RunStatus.CANCELLED
        """
        from decider.experiments._runner import _run

        return RunHandle(lambda job: _run(self.def_, self.step, self.frame, self.params, callbacks=self.callbacks,
                                          job=job, out_dir=out_dir))
