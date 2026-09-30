"""The experiment interface: one versioned schema, two thin facades, one runner.

The experiment asset is a `ExperimentDef` (the `experiment.yaml` schema); a YAML
file and the `Experiment` builder both produce it, and `run_experiment` is the
only place execution happens. Execution reuses the fork/sweep primitives,
comparison reads produced outputs, and results layer the experiment states
(nondeterminism, per-scenario status, findings) over the lifecycle `RunManifest`.
"""
from decider.experiments.builder import Experiment
from decider.experiments.forks import checkpoint_key, fork, merge, sweep
from decider.experiments.model import (EXPERIMENT_VERSION, ComparisonPolicy, ExperimentDef, FlowSpec, InputSpec,
                                       Scenario, SummarySpec, Tolerance, check_experiment_version)
from decider.experiments.result import (ExperimentResult, Finding, RunStatus, ScenarioResult, ScenarioStatus)
from decider.experiments.runner import load_flow, resolve_revision, run_experiment, validate_scenarios
from decider.experiments.runs import collect, steer, trace
from decider.experiments.yaml import dump as dump_yaml
from decider.experiments.yaml import load as load_yaml

__all__ = [
    "EXPERIMENT_VERSION", "ComparisonPolicy", "Experiment", "ExperimentDef", "ExperimentResult", "Finding", "FlowSpec",
    "InputSpec", "RunStatus", "Scenario", "ScenarioResult", "ScenarioStatus", "SummarySpec", "Tolerance",
    "check_experiment_version", "checkpoint_key", "collect", "dump_yaml", "fork", "load_flow", "load_yaml", "merge",
    "resolve_revision", "run_experiment", "steer", "sweep", "trace", "validate_scenarios",
]
