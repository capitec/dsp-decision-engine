"""Experiments: run a flow's baseline and scenarios over the same records, and measure them with metrics.

An `ExperimentDef` (`experiment.json`) or the `Experiment` builder names the
flow, the input, the scenarios (params, overrides, another git revision), the
metrics to collect and the plots to draw. The runner only executes: it feeds
every scenario the same batches and calls the metrics' and callbacks' hooks,
in the style of a training loop's callbacks. Each metric summarises into a
typed `Summary`; each plot requires one; a `Report` lays them out. Metrics,
plots, report blocks and callbacks live in the subnamespaces
`decikit.lab.experiments.metrics`, `.plots`, `.report` and `.callbacks`.
"""
from decikit.lab.experiments import callbacks, metrics, plots, report  # noqa: F401  (registers the built-in types)
from decikit.lab.experiments._experiments import (BASELINE, Experiment, ExperimentDef, ExperimentResult, RunStatus,
                                              Scenario, ScenarioResult, validate)
from decikit.lab.experiments._handle import RunHandle
from decikit.lab.experiments._jobs import Job, JobHandle, JobStatus, PartialResult
from decikit.lab.experiments._manifest import (Environment, Override, ResultRef, Revision, RunManifest, Sample, Selection,
                                           SourceState, manifest_schema)
from decikit.lab.experiments._runner import run, start

__all__ = ["BASELINE", "Environment", "Experiment", "ExperimentDef", "ExperimentResult", "Job", "JobHandle", "JobStatus",
           "Override", "PartialResult", "ResultRef", "Revision", "RunHandle", "RunManifest", "RunStatus", "Sample",
           "Scenario", "ScenarioResult", "Selection", "SourceState", "manifest_schema", "run", "start", "validate"]
