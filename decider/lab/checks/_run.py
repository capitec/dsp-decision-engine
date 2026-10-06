from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from decider.lab.checks._base import STANDARD, CheckContext, registered
from decider.lab.checks._config import ChecksConfig, build, find_config, load_config
from decider.lab.checks._report import Finding, Report, Severity, SkippedCheck
from decider.lab._refs import FlowRef
from decider.exceptions import DeciderError

_FAMILIES = {"0": "determinism", "1": "numeric", "2": "structure", "3": "params", "4": "ids", "5": "runtime"}


def run(pipeline: Any, checks: Iterable[Any] | None = None, *, config: str | Path | None = None,
        data: Any = None, params: Any = None) -> Report:
    """Run checks over `pipeline` and collect a `Report`.

    `checks` is a flat list of `Check` instances, aliases, config dicts or check
    sets such as `checks.standard`. With none, the project's `checks.yaml` is
    used if there is one, else `standard`. `config` names another file; it sets
    the check list when `checks` is omitted, and its `disable` and `fail_on` apply
    either way. A check that needs a frame (`data=`) and wasn't given one is
    listed in `report.skipped`. A step's `suppress_checks` marks matching
    findings suppressed instead of dropping them.

    Example::

        report = checks.run(pipeline)
        report = checks.run(pipeline, [checks.standard, NoFloatMoney(allow=["fee"])])
        report = checks.run(pipeline, config="ci/checks.yaml", data=frame, params=doc)
        report.ok, report.to_text()
    """
    cfg = load_config(config) if config is not None else _project_config()
    if checks is None:
        instances = cfg.instances() if cfg else build(STANDARD)
    else:
        instances = [c for c in build(list(checks)) if not cfg or not {c.type, c.code} & set(cfg.disable)]
    ctx = CheckContext(pipeline, params, data)
    findings: list[Finding] = []
    skipped: list[SkippedCheck] = []
    for c in instances:
        if c.needs_data and data is None:
            skipped.append(SkippedCheck(check=c.type, code=c.code, reason="needs data=; none was given"))
            continue
        ctx._current = c
        for finding in c.run(ctx):
            if not isinstance(finding, Finding):
                raise DeciderError(f"check {c.type!r} yielded {type(finding).__name__}, not a Finding")
            findings.append(finding)
    ctx._current = None
    findings += _step_findings(ctx)
    if cfg:
        findings = [f for f in findings if not {f.code, f.check} & set(cfg.disable)]
    ir = ctx.ir
    flow = FlowRef(name=ir.origin.path, source=ir.origin.source, flow_id=ir.origin.id)
    return Report(flow=flow, findings=tuple(findings), skipped=tuple(skipped + ctx.skipped),
                  gate=cfg.fail_on if cfg else Severity.ERROR)


def _project_config() -> ChecksConfig | None:
    path = find_config()
    return load_config(path) if path else None


def _step_findings(ctx: CheckContext) -> list[Finding]:
    from decider.steps import ConfigurableStep

    by_path: dict[str, Any] = {}
    for n in ctx.nodes:
        by_path.setdefault(n.origin.path, n)
    out: list[Finding] = []
    for path, step in ctx.steps.items():
        if isinstance(step, ConfigurableStep) and path in by_path:
            ctx.node = by_path[path]
            out += step.check(ctx)
    ctx.node = None
    return out


def catalogue() -> list[dict[str, Any]]:
    """Every registered check: alias, code, family, needs_data, executes, severity and config schema.

    Example::

        [(c["code"], c["alias"]) for c in checks.catalogue()]
    """
    rows = []
    for tag, cls in sorted(registered().items(), key=lambda kv: (getattr(kv[1], "code", ""), kv[0])):
        code = getattr(cls, "code", "")
        family = _FAMILIES.get(code[3:4], "custom") if code.startswith("DCD") else "custom"
        rows.append({"alias": tag, "code": code, "family": family, "needs_data": cls.needs_data,
                     "executes": cls.executes, "severity": cls.model_fields["severity"].default.value,
                     "schema": cls.model_json_schema()})
    return rows
