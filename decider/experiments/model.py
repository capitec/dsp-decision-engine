"""The experiment asset: a versioned `experiment.yaml` schema, in Python.

An experiment is data, never code: a flow entry point, an authored revision, an
input reference, scenarios, a comparison policy and requested summaries.
Optional Python `tests/` and `graphs/` are references from the YAML, not code
embedded in it.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from decider.exceptions import DeciderError

EXPERIMENT_VERSION = "1"


class FlowSpec(BaseModel):
    """The flow an experiment runs: an import path and the engine window it needs.

    `entry` is `"module:qualname"`, a zero-argument factory or a module-level
    pipeline; `decider` pins the supported engine window (e.g. `">=1.0,<2.0"`).
    """

    model_config = ConfigDict(extra="forbid")

    entry: str
    decider: str = ""


class InputSpec(BaseModel):
    """The input data reference: a path and the column that identifies records."""

    model_config = ConfigDict(extra="forbid")

    path: str = ""
    id_column: str | None = None


class Scenario(BaseModel):
    """One controlled change to run next to the baseline.

    `params` merge over the run's params document; `overrides` write values at a
    declared point — `at` names the step output to override, or is `None` for an
    input column before anything runs. `revision` runs the scenario on another
    source revision (e.g. `"HEAD^"`); `row` overrides one record only.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    revision: str | None = None
    params: dict[str, Any] = {}
    overrides: dict[str, Any] = {}
    at: str | None = None
    row: int | None = None


class Tolerance(BaseModel):
    """Float comparison tolerance: `abs(a-b) <= atol + rtol * max(abs(a), abs(b))`."""

    model_config = ConfigDict(extra="forbid")

    rtol: float = 1e-6
    atol: float = 1e-9


class ComparisonPolicy(BaseModel):
    """How scenarios compare to a baseline.

    `outputs` names the produced columns to compare (input overrides are not
    reflected in the output frame, so comparison reads produced columns).
    `determinism` reruns the baseline and flags a differing rerun as
    non-reproducible.
    """

    model_config = ConfigDict(extra="forbid")

    baseline: str = "baseline"
    outputs: tuple[str, ...] = ()
    tolerance: Tolerance = Tolerance()
    determinism: bool = True


class SummarySpec(BaseModel):
    """A requested built-in summary: counts by output, or first divergence."""

    model_config = ConfigDict(extra="forbid")

    type: str
    output: str = ""
    against: str | None = None


class ExperimentDef(BaseModel):
    """The `experiment.yaml` asset, the canonical serialization of one experiment.

    Both a YAML file and the `Experiment` builder produce this schema, and both
    feed one `run_experiment`.
    """

    model_config = ConfigDict(extra="forbid")

    version: str = EXPERIMENT_VERSION
    name: str
    description: str = ""
    flow: FlowSpec
    revision: str = "HEAD"
    input: InputSpec = InputSpec()
    scenarios: list[Scenario] = []
    comparison: ComparisonPolicy = ComparisonPolicy()
    summaries: list[SummarySpec] = []
    tests: str = ""
    graphs: str = ""


def check_experiment_version(version: str | None) -> None:
    """Raise if `version` is newer than this decider supports; mirrors `check_version`."""
    if version is None:
        return
    try:
        given = int(version.partition(".")[0])
        supported = int(EXPERIMENT_VERSION.partition(".")[0])
    except ValueError:
        raise DeciderError(f"unrecognised experiment version {version!r}") from None
    if given > supported:
        raise DeciderError(
            f"experiment version {version} is newer than {EXPERIMENT_VERSION}; upgrade decider to read it"
        )
