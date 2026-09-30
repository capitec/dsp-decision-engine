"""The experiment result: a run manifest layered with comparison, scenario and nondeterminism state.

The generic `RunManifest` (from the lifecycle contract) records provenance:
flow, resolved revision, environment, input fingerprint, output references.
This module adds the experiment-specific states the spec requires to be
unambiguous: a run status that never lets a non-reproducible or partial run
look like a deterministic completed one, per-scenario statuses, and portable
finding descriptors.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from decider.data import PreflightReport
from decider.lifecycle import Job, ResultRef, RunManifest


class RunStatus(str, Enum):
    """The terminal quality of an experiment run, distinct from the job's lifecycle.

    `NON_REPRODUCIBLE` means every scenario completed but a baseline rerun
    diverged; `PARTIAL` means some scenario did not complete. Neither is
    `COMPLETED`, so a non-reproducible or partial run never reads as a
    deterministic finished one.
    """

    COMPLETED = "completed"
    NON_REPRODUCIBLE = "non_reproducible"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class ScenarioStatus(str, Enum):
    """One scenario's outcome; `RESUMED` means a prior run already completed it."""

    COMPLETED = "completed"
    FAILED = "failed"
    RESUMED = "resumed"
    SKIPPED = "skipped"


class Finding(BaseModel):
    """A portable finding: one divergence or error, with its location and values."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    scenario: str = ""
    location: str = ""
    expected: Any = None
    actual: Any = None
    message: str = ""


class ScenarioResult(BaseModel):
    """One scenario's outcome: status, error, divergences and an output reference."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: ScenarioStatus
    error: str | None = None
    divergences: tuple[Finding, ...] = ()
    output_ref: ResultRef | None = None
    row_count: int | None = None


class ExperimentResult(BaseModel):
    """A completed (or stopped) experiment run: the manifest plus comparison state.

    `manifest` is the generic `RunManifest`; `nondeterministic` and `status`
    add the experiment's first-class result states; `preflight` is the
    compatibility and cost report captured before anything ran.
    """

    model_config = ConfigDict(extra="forbid")

    manifest: RunManifest
    status: RunStatus
    nondeterministic: bool = False
    scenarios: tuple[ScenarioResult, ...] = ()
    preflight: PreflightReport | None = None
    job: Job | None = None
