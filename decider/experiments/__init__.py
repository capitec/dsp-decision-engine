"""Experiments: run a flow's baseline and scenarios over the same records, and measure them with metrics.

An `ExperimentDef` (`experiment.json`) or the `Experiment` builder names the
flow, the input, the scenarios (params, overrides, another git revision), the
metrics to collect and the plots to draw. The runner only executes: it feeds
every scenario the same batches and calls the metrics' and callbacks' hooks,
in the style of a training loop's callbacks. Each metric summarises into a
typed `Summary`; each plot requires one; a `Report` lays them out.
"""
from decider.experiments.callbacks import Batch, Callback, JobProgress, NodeBatch, Progress, Run, WriteOutputs
from decider.experiments.experiments import (BASELINE, EXPERIMENT_VERSION, Experiment, ExperimentDef,
                                             ExperimentResult, FlowSpec, InputSpec, RunStatus, Scenario,
                                             ScenarioResult, ScenarioStatus, check_experiment_version, validate)
from decider.experiments.metrics import (Bins, BranchFlows, Counts, Describe, Divergence, Divergences, Flows,
                                         Histogram, Metric, MetricRef, Stats, Summary, ValueCounts)
from decider.experiments.report import (Block, BlockRef, Chart, MetricSection, Overview, Report, Row, Section,
                                        Table, Text)
from decider.experiments.plots import Bars, Changes, Distribution, Figure, Plot, PlotRef, Ranges, Sankey
from decider.experiments.revision import compatible, materialise, remove, resolve
from decider.experiments.runner import load_flow, run, run_experiment

__all__ = [
    "BASELINE", "Bars", "Batch", "Bins", "Block", "BlockRef", "BranchFlows", "Callback", "Changes", "Chart",
    "Counts", "Describe", "Distribution", "Divergence", "Divergences", "EXPERIMENT_VERSION", "Experiment",
    "ExperimentDef", "ExperimentResult", "Figure", "FlowSpec", "Flows", "Histogram", "InputSpec", "JobProgress",
    "Metric", "MetricRef", "MetricSection", "NodeBatch", "Overview", "Plot", "PlotRef", "Progress", "Ranges",
    "Report", "Row", "Run", "RunStatus", "Sankey", "Scenario", "ScenarioResult", "ScenarioStatus", "Section",
    "Stats", "Summary", "Table", "Text", "ValueCounts", "WriteOutputs", "check_experiment_version", "compatible",
    "load_flow", "materialise", "remove", "resolve", "run", "run_experiment", "validate",
]
