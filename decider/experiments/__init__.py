"""Experiment primitives: fork and sweep a debug session, and run whole pipelines to collect traces.

The full experiment interface — scenario assets, run manifests, comparison and
aggregation policy — builds on these primitives; it is not defined here.
"""
from decider.experiments.forks import checkpoint_key, fork, merge, sweep
from decider.experiments.runs import collect, steer, trace

__all__ = ["checkpoint_key", "collect", "fork", "merge", "steer", "sweep", "trace"]
