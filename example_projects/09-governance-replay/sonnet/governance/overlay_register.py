"""The overlay register, ageing and stack-off run (spec 09 §5.14).

Every flow already owns a `credit_core.adjustments.AdjustmentRegister` (00's
own mechanism: append-only, `due_for_review`, `apply_stack(stack_enabled=...)`
through one implementation). This module does not reimplement any of that --
it is the one estate-wide *view* over the registers 01, 03 and 05 already
built, which is exactly what §5.14.1 asks for ("one view across all eight
flows") and what none of those three projects builds on its own, since each
only needs its own register, not the estate's.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date

from governance import diff, flows, paths
from governance.evidence_store import EvidenceRecord, json_safe


def _flow_registers(flow_code: str) -> list[tuple[str, object]]:
    """Every `(adjustment_set_id, AdjustmentRegister)` this flow owns -- reached the same
    way 05's own NOTES.md documents reaching another project's module-level singleton
    (`_ENTITY_OVERLAYS`, `_BUSINESS_OVERLAYS`): a plain attribute import, not a declared
    "list your registers" API, because none of 01/03/05 publish one (each only needed its
    own register, never an estate-wide enumeration of them)."""
    adapter = flows.get(flow_code)
    project_dir = adapter.project_dir()
    for extra in adapter.extra_projects:
        paths.ensure_on_path(paths.sibling_project(extra))
    paths.ensure_on_path(project_dir)

    if flow_code == "01":
        from fraud_interdiction.overlays import ADJUSTMENT_SET_ID, OVERLAY_REGISTER
        return [(ADJUSTMENT_SET_ID, OVERLAY_REGISTER)]
    if flow_code == "03":
        from loan_granting.scoring import ADJUSTMENT_SET_ID, SCORE_ADJUSTMENTS
        return [(ADJUSTMENT_SET_ID, SCORE_ADJUSTMENTS)]
    if flow_code == "05":
        from business_nested.grade import _BUSINESS_ADJUSTMENT_SET_ID, _BUSINESS_OVERLAYS
        from business_nested.overlays import EVENT_ADJUSTMENT_SET_ID, EVENT_THRESHOLD_OVERLAYS
        from business_nested.scoring import _ENTITY_ADJUSTMENT_SET_ID, _ENTITY_OVERLAYS
        return [
            (EVENT_ADJUSTMENT_SET_ID, EVENT_THRESHOLD_OVERLAYS),
            (_ENTITY_ADJUSTMENT_SET_ID, _ENTITY_OVERLAYS),
            (_BUSINESS_ADJUSTMENT_SET_ID, _BUSINESS_OVERLAYS),
        ]
    raise KeyError(f"no overlay registers known for flow {flow_code!r}")


@dataclass(frozen=True)
class OverlayRow:
    flow_code: str
    adjustment_set_id: str
    adjustment_id: str
    kind: str
    target: str
    scope: dict
    stack_position: int
    owner: str
    approval_reference: str
    rationale: str
    effective_from: str
    effective_to: str | None
    review_date: str
    tighten_only: bool
    enabled: bool


def combined_register(flow_codes: tuple[str, ...] = ("01", "03", "05")) -> list[OverlayRow]:
    """§5.14.1's "one view across all eight flows" -- here, across the three this harness
    was built against. Answers the two aggregate questions unmodified: "how much of the
    estate is overlaid" is `len(rows)`/by-flow counts; "where do overlays concentrate" is
    `by target`/`by kind` counts over these rows -- both computed by the caller from this
    flat table, not baked into it, since which aggregate matters changes by audience."""
    rows = []
    for flow_code in flow_codes:
        for set_id, register in _flow_registers(flow_code):
            for a in register._all:  # `AdjustmentRegister` has no public iterator; `._all` is its
                # one internal list, reached the same "past the frozen entry point" way 02/05's own
                # NOTES.md document for `core.obligations._process`/`_ALWAYS_DISQUALIFYING`.
                rows.append(OverlayRow(
                    flow_code=flow_code, adjustment_set_id=set_id, adjustment_id=a.adjustment_id,
                    kind=a.kind, target=a.target, scope=dict(a.scope), stack_position=a.stack_position,
                    owner=a.owner, approval_reference=a.approval_reference, rationale=a.rationale,
                    effective_from=a.effective_from.isoformat(),
                    effective_to=a.effective_to.isoformat() if a.effective_to else None,
                    review_date=a.review_date.isoformat(), tighten_only=a.tighten_only, enabled=a.enabled,
                ))
    return rows


def ageing_report(flow_codes: tuple[str, ...], as_of: date) -> dict:
    """§5.14.2's monthly report: overlays past review, by age; and (the row "with teeth")
    a per-overlay unwind estimate, produced by `unwind_estimate` below rather than left as
    an unanswerable "we don't know"."""
    due = []
    for flow_code in flow_codes:
        for set_id, register in _flow_registers(flow_code):
            for a in register.due_for_review(as_of):
                age_days = (as_of - a.review_date).days
                due.append({
                    "flow_code": flow_code, "adjustment_set_id": set_id, "adjustment_id": a.adjustment_id,
                    "review_date": a.review_date.isoformat(), "age_days": age_days,
                    "owner": a.owner, "kind": a.kind, "target": a.target,
                })
    due.sort(key=lambda r: -r["age_days"])
    return {"as_of": as_of.isoformat(), "overlays_due_for_review": due, "count": len(due)}


def _set_every(params: dict, name: str, value) -> dict:
    """A deep copy of `params` with every key literally named `name` set to `value`,
    wherever it sits (`diff.find_param` finds the paths; this applies the change) --
    the same "disable the whole stack with one flag" mechanism §5.14.3 requires, and the
    same recursive-walk pattern 03's own `tests/test_pipeline.py` already uses for it."""
    out = copy.deepcopy(params)
    for path in diff.find_param(params, name):
        node = out
        parts = path.split("/")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return out


def stack_off_run(evidence: EvidenceRecord, *, mode: str = "interpreted") -> dict:
    """§5.14.3: the same decision, the same implementation, the overlay stack disabled.
    §5.14.4's "estimated effect of unwinding" for one decision is the outcome delta this
    produces; `unwind_estimate` below runs it over a population for the monthly report."""
    adapter = flows.get(evidence.flow_code)
    built = adapter.build(evidence.config_version, mode=mode)
    off_params = _set_every(evidence.params, "adjustment_stack_enabled", False)
    typed_request = adapter.type_record(evidence.request)
    counterfactual = json_safe(built.executable.score(typed_request, off_params))
    outcome_field = adapter.outcome_field
    return {
        "decision_id": evidence.decision_id, "flow_code": evidence.flow_code,
        "adjusted_outcome": evidence.record.get(outcome_field),
        "unadjusted_outcome": counterfactual.get(outcome_field),
        "outcome_changed": evidence.record.get(outcome_field) != counterfactual.get(outcome_field),
        "counterfactual_record": counterfactual,
    }


def unwind_estimate(flow_code: str, evidences: list[EvidenceRecord], *, mode: str = "interpreted") -> dict:
    """§5.14.2's "estimated effect of unwinding each expired overlay", over a population of
    already-captured decisions (a monthly run would use the flow's full production volume;
    this harness's demo runs it over whatever this flow's evidence store holds)."""
    changed = 0
    moved_from: dict = {}
    for evidence in evidences:
        result = stack_off_run(evidence, mode=mode)
        if result["outcome_changed"]:
            changed += 1
            moved_from[evidence.decision_id] = (result["adjusted_outcome"], result["unadjusted_outcome"])
    n = len(evidences)
    return {
        "flow_code": flow_code, "population_size": n, "decisions_that_would_change": changed,
        "rate": (changed / n) if n else 0.0, "moves": moved_from,
    }
