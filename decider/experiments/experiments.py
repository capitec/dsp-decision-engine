"""An experiment as data, its result, and the `Experiment` builder notebooks use.

`ExperimentDef` is the whole asset, saved as `experiment.json`
(`model_dump_json` / `model_validate_json`): a flow entry point, the input,
scenarios to run next to the unchanged baseline, metrics to collect and plots
to draw from them. Execution lives in `decider.experiments.runner`.
"""
from __future__ import annotations

from collections import Counter
from fnmatch import fnmatchcase
from enum import Enum
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, SerializeAsAny

from decider.contract import describe
from decider.data import PreflightReport
from decider.engine.ir.context import to_ir
from decider.engine.ir.nodes import iter_nodes
from decider.exceptions import DeciderError
from decider.experiments.callbacks import Callback
from decider.experiments.metrics.base import Metric, MetricRef, Summary
from decider.experiments import report
from decider.experiments.plots.base import Figure, Plot, PlotRef, ipython_display
from decider.experiments.report import Block, Report
from decider.lifecycle import Job, JobHandle, ResultRef, Revision, RunManifest

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
    """The input data: a `.parquet`, `.csv`, `.json` or `.jsonl` path."""

    model_config = ConfigDict(extra="forbid")

    path: str = ""


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
    preflight: PreflightReport | None = None
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


def validate(def_: ExperimentDef, step) -> list[str]:
    """Every problem in `def_` against `step`, returned rather than raised so a caller shows them all.

    Checks scenario names, that each override targets an input or a produced
    value (a produced one needs `at`), that metric and plot names are unique,
    and that each plot's metric exists and gives the summary the plot requires.
    """
    problems = []
    names = Counter(sc.name for sc in def_.scenarios)
    if BASELINE in names:
        problems.append(f"{BASELINE!r} is the unchanged run; give the scenario another name")
    problems += [f"two scenarios are named {n!r}" for n, k in names.items() if k > 1]
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
                 entry: str = "", decider: str = "", input_path: str = "", batch_size: int = 100_000,
                 determinism: bool = True):
        self.step = step
        self.frame = frame
        self.params = params
        self.callbacks: list[Callback] = []
        self.def_ = ExperimentDef(name=name, flow=FlowSpec(entry=entry, decider=decider), revision=revision,
                                  input=InputSpec(path=input_path), batch_size=batch_size, determinism=determinism)

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
        """Write the definition to `path` as JSON; `Experiment.load` reads it back."""
        path = Path(path)
        path.write_text(self.def_.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path, *, step=None, frame: pl.DataFrame | pl.LazyFrame | None = None,
             params=None) -> Experiment:
        """An experiment from an `experiment.json`, ready to change further or `run`.

        Without `step` the flow is imported from `flow.entry`; without `frame`
        the input is scanned from `input.path`, relative to the JSON file.

        Example::

            result = Experiment.load("experiments/premium_sweep/experiment.json").run()
        """
        from decider.experiments.runner import load_flow, scan

        path = Path(path)
        def_ = ExperimentDef.model_validate_json(path.read_text())
        check_experiment_version(def_.version)
        if frame is None:
            if not def_.input.path:
                raise DeciderError(f"{path} has no input.path; pass frame=")
            frame = scan(path.parent / def_.input.path)
        exp = cls(def_.name, step if step is not None else load_flow(def_.flow.entry), frame, params=params)
        exp.def_ = def_
        return exp

    def run(self, *, job: JobHandle | None = None, out_dir: str | None = None) -> ExperimentResult:
        from decider.experiments.runner import run_experiment

        return run_experiment(self.def_, self.step, self.frame, self.params, callbacks=self.callbacks, job=job,
                              out_dir=out_dir)
