from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from decider.contract import FlowRef, StepRef


class Severity(Enum):
    """How serious a finding is; the client maps this to advice or a release gate.

    `decider` only reports. `Report.fail_on` and `decider check --fail-on` are the gate.

    Example::

        Severity.ERROR in {f.severity for f in report.findings}
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

    @property
    def rank(self) -> int:
        return list(Severity).index(self)


class Suppression(BaseModel):
    """Why a finding is silenced: the step `path` whose `suppress_checks` named `name`.

    Example::

        Suppression(path="legacy", name="int64_overflow")
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    name: str


class Finding(BaseModel):
    """One check result: what was found, where, and how serious.

    `code` is what CI pins and users suppress (`"DCD101"`), `check` the alias
    (`"int64_overflow"`). `step` is the contract reference, so it joins with
    `describe()` nodes; `fix` is one line saying what to do instead and
    `detail` holds machine-readable specifics. A suppressed finding stays in
    the report and doesn't count against `ok`.

    Example::

        Finding(code="ACME001", check="no_float_money", severity=Severity.WARNING,
                message="fee is held as a float", step=StepRef(path="term/cap"), line=12)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = ""
    check: str = ""
    severity: Severity
    message: str
    step: StepRef | None = None
    file: str | None = None
    line: int | None = None
    fix: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    suppressed: Suppression | None = None


class SkippedCheck(BaseModel):
    """A check that didn't run, and why (it needs data it wasn't given).

    Example::

        SkippedCheck(check="missing_required", code="DCD504", reason="needs data=")
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    check: str
    code: str
    reason: str


class Report(BaseModel):
    """The result of one `checks.run`: the flow checked, its findings and the checks that were skipped.

    `ok` is True when no unsuppressed finding is an error. `gate` is the
    severity `fail_on()` uses by default (the project's `fail_on`).
    `model_dump_json()` is the JSON form.

    Example::

        report = checks.run(pipeline)
        print(report.to_text())
        report.ok, report.counts, report.fail_on("warning")
        sarif = report.to_sarif()
    """

    # "ignore" so a dumped report, with its computed `ok` and `counts`, loads back.
    model_config = ConfigDict(frozen=True, extra="ignore")

    flow: FlowRef
    findings: tuple[Finding, ...] = ()
    skipped: tuple[SkippedCheck, ...] = ()
    gate: Severity = Severity.ERROR

    @computed_field
    @property
    def ok(self) -> bool:
        return not self.fail_on(Severity.ERROR)

    @computed_field
    @property
    def counts(self) -> dict[Severity, int]:
        live = [f.severity for f in self.findings if f.suppressed is None]
        return {s: live.count(s) for s in Severity}

    def fail_on(self, severity: Severity | str | None = None) -> bool:
        """True when an unsuppressed finding is at least `severity` (default: the report's `gate`)."""
        floor = self.gate if severity is None else Severity(severity)
        return any(f.suppressed is None and f.severity.rank >= floor.rank for f in self.findings)

    def to_text(self) -> str:
        """One line per finding (`file:line: severity code check: message`), then skipped checks and a summary."""
        lines = []
        for f in self.findings:
            where = f"{f.file}:{f.line}: " if f.file and f.line else ""
            note = f" (suppressed by {f.suppressed.name} on {f.suppressed.path})" if f.suppressed else ""
            lines.append(f"{where}{f.severity.value} {f.code} {f.check}: {f.message}{note}")
            if f.fix:
                lines.append(f"    fix: {f.fix}")
        lines += [f"skipped {s.code} {s.check}: {s.reason}" for s in self.skipped]
        counts = ", ".join(f"{n} {s.value}" for s, n in self.counts.items())
        suppressed = sum(f.suppressed is not None for f in self.findings)
        lines.append(f"{self.flow.name or 'flow'}: {counts}" + (f", {suppressed} suppressed" if suppressed else ""))
        return "\n".join(lines)

    def to_sarif(self) -> dict[str, Any]:
        """SARIF 2.1.0, readable by GitHub code scanning; `partialFingerprints` is `flow_id/step_id/code`."""
        level = {Severity.INFO: "note", Severity.WARNING: "warning", Severity.ERROR: "error"}
        rules = {f.code: {"id": f.code, "name": f.check} for f in self.findings}
        return {
            "version": "2.1.0",
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "runs": [{
                "tool": {"driver": {"name": "decider", "rules": list(rules.values())}},
                "results": [_result(self, f, level[f.severity]) for f in self.findings],
            }],
        }


def _result(report: Report, f: Finding, level: str) -> dict[str, Any]:
    flow_id = (f.step.flow_id if f.step else None) or report.flow.flow_id or report.flow.name
    step_id = (f.step.step_id or f.step.path) if f.step else ""
    result: dict[str, Any] = {
        "ruleId": f.code, "level": level, "message": {"text": f.message},
        "partialFingerprints": {"decider/v1": f"{flow_id}/{step_id}/{f.code}"},
    }
    if f.file:
        region = {"startLine": f.line} if f.line else {}
        result["locations"] = [{"physicalLocation": {"artifactLocation": {"uri": f.file}, "region": region}}]
    if f.suppressed:
        result["suppressions"] = [{"kind": "inSource", "justification": f"{f.suppressed.name} on {f.suppressed.path}"}]
    return result
