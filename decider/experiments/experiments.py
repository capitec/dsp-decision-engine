"""An experiment as data, its result, and the `Experiment` builder notebooks use.

`ExperimentDef` is the whole asset, saved as `experiment.json`
(`model_dump_json` / `model_validate_json`): a flow entry point, the input,
scenarios to run next to the unchanged baseline, metrics to collect and plots
to draw from them. Execution lives in `decider.experiments.runner`.
"""
from __future__ import annotations

import json
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
from decider.experiments.plots.base import Figure, Plot, PlotRef, ipython_display
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
    flags a differing rerun as non-reproducible.
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
    completed; `plots` is `{plot: Figure}`. In a notebook, `show()` lays the
    whole run out, and `plots[name].show()` one chart.
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
        """Show the run in a notebook: its status and scenarios, then each metric with its table and charts.

        A metric no plot draws gets the built-in plot for its summary type, so every summary is drawn.
        """
        ipy = ipython_display()
        ipy.display(ipy.Markdown(self._heading()), pl.DataFrame([{
            "scenario": s.name, "status": s.status.value, "records": s.records,
            "revision": s.revision.resolved[:12] if s.revision and s.revision.resolved else None,
            "changes": _changes(s), "error": s.error,
        } for s in self.scenarios]))
        metrics = {m.key: m for m in self.definition.metrics} if self.definition else {}
        plots = self.definition.plots if self.definition else []
        for key, by_scenario in self.summaries.items():
            metric = metrics.get(key)
            rows = [{"scenario": name, **_flat(s.model_dump())} for name, s in by_scenario.items()]
            figures = [self.plots[p.key] for p in plots if p.metric == key and p.key in self.plots]
            if not figures and by_scenario:
                figures = [Figure.of(p.render(by_scenario)) for p in _default_plots(key, by_scenario)]
            ipy.display(ipy.Markdown(f"#### {key}\n\n{_where(metric)}"),
                        pl.DataFrame(rows, infer_schema_length=None), *figures)

    def _heading(self) -> str:
        revision = self.manifest.revision
        records = self.manifest.input.row_count if self.manifest.input else None
        parts = [f"{records:,} records" if records is not None else "",
                 f"revision `{(revision.resolved or revision.authored)[:12]}`" if revision else "",
                 f"{len(self.scenarios) - 1} scenario(s) next to the baseline"]
        lines = [f"### {self.name}: {self.status.value}", " · ".join(p for p in parts if p)]
        if self.nondeterministic:
            lines.append("**A rerun of the baseline differed: this flow is not reproducible.**")
        return "\n\n".join(lines)


def _changes(s: ScenarioResult) -> str:
    bits = []
    if s.params:
        bits.append(f"params {json.dumps(s.params)}")
    if s.overrides:
        where = (f" after {s.at}" if s.at else "") + (f" on record {s.row}" if s.row is not None else "")
        bits.append(f"set {json.dumps(s.overrides)}{where}")
    if s.revision:
        bits.append(f"revision {s.revision.authored}")
    return "; ".join(bits) or "unchanged"


def _where(metric: Metric | None) -> str:
    if metric is None:
        return ""
    column = getattr(metric, "column", None) or getattr(metric, "outcome", None)
    what = f"`{metric.type}`" + (f" of `{column}`" if column else "")
    if metric.node_id:
        return f"{what} at the step with id `{metric.node_id}`"
    return f"{what} at steps matching `{metric.nodes}`" if metric.nodes else f"{what} in the flow's output"


def _default_plots(key: str, by_scenario: dict[str, Summary]) -> list[Plot]:
    summary = type(next(iter(by_scenario.values())))
    found = [cls for cls in _subclasses(Plot) if cls.requires is summary and cls.__module__.startswith("decider.")]
    return [found[0](metric=key)] if found else []


def _subclasses(cls: type) -> list[type]:
    return [c for sub in cls.__subclasses__() for c in (sub, *_subclasses(sub))]


def _flat(fields: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, dict):
            out.update(_flat(value, f"{prefix}{key}."))
        elif isinstance(value, list):
            out[prefix + key] = _short(value)
        else:
            out[prefix + key] = value
    return out


def _short(values: list) -> str:
    if values and all(isinstance(v, (int, float)) for v in values):
        shown = ", ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in values[:6])
        return f"[{shown}{', …' if len(values) > 6 else ''}] ({len(values)})"
    text = json.dumps(values, default=str)
    return text if len(text) <= 80 else text[:77] + "..."


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
