"""Metrics: pluggable collectors the runner feeds batch by batch, each summarising into a typed `Summary`.

A metric is global (it reads the flow's output in `on_batch_end`) or, with a
`nodes` glob, per node (it reads what matching nodes wrote in `on_node_end`).
Define your own by subclassing `Metric` with a `type` alias; an experiment
names it by that alias, so it loads from JSON.
"""
from decider.lab.experiments.callbacks import Batch
from decider.lab.experiments.metrics._base import Metric, MetricRef, Summary
from decider.lab.experiments.metrics._compare import Divergence, Divergences
from decider.lab.experiments.metrics._outputs import Bins, Counts, Describe, Histogram, Stats, ValueCounts
from decider.lab.experiments.metrics._paths import BranchFlows, Flows

__all__ = ["Batch", "Bins", "BranchFlows", "Counts", "Describe", "Divergence", "Divergences", "Flows", "Histogram", "Metric",
           "MetricRef", "Stats", "Summary", "ValueCounts"]
