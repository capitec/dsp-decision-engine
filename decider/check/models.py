from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from decider.contract.refs import FlowRef, StepRef


class Severity(Enum):
    """How serious a finding is; the client maps this to warning vs release-blocking.

    `decider` only reports, it never fails a build: the client owns the policy
    (CI can treat `error` — or any stricter threshold — as a release gate,
    notebooks and VS Code can render `warning`/`info` without failing).

    Example::

        Severity.ERROR in {f.severity for f in report.findings}
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Finding(BaseModel):
    """One check result: what was found, where, and how serious.

    `step` is the capture-time step reference (`path`, `name`, `source`) plus
    its durable `flow_id`/`step_id`; `line` is the resolved source line when
    known; `detail` holds machine-readable specifics the check chooses. `check`
    names the check that produced it; when empty, `run` fills it from the check
    function's name.

    Example::

        Finding(check="wall_clock", severity=Severity.WARNING,
                message="reads the wall clock", step=StepRef(path="term/cap"),
                line=12, detail={"call": "time.time()"})
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    check: str = ""
    severity: Severity
    message: str
    step: StepRef | None = None
    line: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class Report(BaseModel):
    """The result of one `check.run`: the flow checked and its findings.

    Serialisable as-is for CI, notebooks and editors
    (`report.model_dump_json()`); `ok` is True when no finding is `error`
    severity, which a client may use as its release gate.

    Example::

        report = check.run(pipeline)
        report.ok
        [f.message for f in report.findings if f.severity is Severity.ERROR]
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    flow: FlowRef
    findings: tuple[Finding, ...] = ()

    @property
    def ok(self) -> bool:
        return not any(f.severity is Severity.ERROR for f in self.findings)
