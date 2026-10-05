"""Experiments: run a flow's baseline and scenarios over the same records, and measure them with metrics.

An `ExperimentDef` (`experiment.json`) or the `Experiment` builder names the
flow, the input, the scenarios (params, overrides, another git revision), the
metrics to collect and the plots to draw. The runner only executes: it feeds
every scenario the same batches and calls the metrics' and callbacks' hooks,
in the style of a training loop's callbacks. Each metric summarises into a
typed `Summary`; each plot requires one; a `Report` lays them out. Metrics,
plots, report blocks and callbacks live in the subnamespaces
`decider.experiments.metrics`, `.plots`, `.report` and `.callbacks`.
"""
from decider.experiments import callbacks, metrics, plots, report  # noqa: F401  (registers the built-in types)
from decider.experiments._experiments import (BASELINE, Experiment, ExperimentDef, ExperimentResult, RunStatus,
                                              Scenario, ScenarioResult, validate)
from decider.experiments._handle import RunHandle
from decider.experiments._runner import run, start

__all__ = ["BASELINE", "Experiment", "ExperimentDef", "ExperimentResult", "RunHandle", "RunStatus", "Scenario", "ScenarioResult",
           "run", "start", "validate"]
