"""Single-decision explanation, three audiences (spec 09 §5.2).

Built entirely from a captured `EvidenceRecord` -- no re-scoring, no live
call. The three renderings differ in *what they may contain*, not just in
length (§5.2's own table): the consultant rendering states the dominant
reason in plain language and nothing internal; the analyst rendering is
unbounded and includes every mechanism-specific detail this flow recorded
(03's cap chain, 01's fired/evaluated rule sets, 05's entity/event
attribution); the adjudicator rendering narrates the same facts for a lay
reader, with every identifier glossed, thresholds included (§5.2's second
tension: the analyst and adjudicator renderings state both reasons and
thresholds, the consultant rendering states reasons only).
"""
from __future__ import annotations

from dataclasses import dataclass

from governance.evidence_store import EvidenceRecord

# Each flow's own reason registry (imported directly, the same "reach the one
# stable governance object" pattern 00/01/03/05 already use for theirs) --
# used only for the reason code -> description text, never re-derived.
_REGISTRIES: dict[str, object] = {}


def _registry(flow_code: str):
    if flow_code in _REGISTRIES:
        return _REGISTRIES[flow_code]
    from governance import flows, paths

    adapter = flows.get(flow_code)
    project_dir = adapter.project_dir()
    for extra in adapter.extra_projects:
        paths.ensure_on_path(paths.sibling_project(extra))
    paths.ensure_on_path(project_dir)
    if flow_code == "01":
        from fraud_interdiction.reasons import REASON_REGISTRY as registry
    elif flow_code == "03":
        from loan_granting.reasons import REGISTRY as registry
    elif flow_code == "05":
        from business_nested.outcome import REASON_REGISTRY as registry
    else:
        registry = None
    _REGISTRIES[flow_code] = registry
    return registry


def _describe(flow_code: str, code: int | None) -> str | None:
    if code is None:
        return None
    registry = _registry(flow_code)
    if registry is None:
        return str(code)
    entry = registry._by_code.get(code)  # the registry's own public surface is `.rank()`/`__contains__`;
    return entry.description if entry else str(code)  # reaching `_by_code` for wording is the same
    # "reach past the one stable entry point" trade-off 02/05's own NOTES.md document.


@dataclass(frozen=True)
class Explanation:
    decision_id: str
    flow_code: str
    audience: str
    text: str
    detail: dict


def _mechanism_detail(evidence: EvidenceRecord) -> dict:
    """Flow-specific "what happened" detail -- 09 §5.2's gates/rules/nodes/cap-chain/
    score-contribution list, read from whatever this flow's own evidence carries (each
    flow's `pipeline.py` `.emit()` list is what decides what is even available here)."""
    r = evidence.record
    if evidence.flow_code == "01":
        return {
            "fired_rule_ids": r.get("fired_rule_ids"),
            "unevaluable_rule_ids": r.get("unevaluable_rule_ids"),
            "fired_on_overlay_ids": r.get("fired_on_overlay_ids"),
            "counterfactual_fired_rule_ids": r.get("counterfactual_fired_rule_ids"),
            "counterfactual_action_code": r.get("counterfactual_action_code"),
            "action_code": r.get("action_code"), "action_source_rule_id": r.get("action_source_rule_id"),
            "hard_block_code": r.get("hard_block_code"), "degraded_mode_code": r.get("degraded_mode_code"),
        }
    if evidence.flow_code == "03":
        return {
            "eligibility_gate_ids": r.get("eligibility_gate_ids"),
            "eligibility_gate_verdicts": r.get("eligibility_gate_verdicts"),
            "amount_cap_chain_rule_ids": r.get("amount_cap_chain_rule_ids"),
            "amount_cap_chain_values": r.get("amount_cap_chain_values"),
            "amount_cap_binding_rule_id": r.get("amount_cap_binding_rule_id"),
            "score": r.get("score"), "score_unadjusted": r.get("score_unadjusted"),
            "risk_grade": r.get("risk_grade"),
            "recommended_amount": r.get("recommended_amount"), "recommended_term": r.get("recommended_term"),
            "offer_binding_constraint_codes": r.get("offer_binding_constraint_codes"),
        }
    if evidence.flow_code == "05":
        return {
            "attributing_entity_id": r.get("attributing_entity_id"),
            "attributing_event_ids": r.get("attributing_event_ids"),
            "business_decline_binding_rule_actual": r.get("business_decline_binding_rule_actual"),
            "entity_verdict_code": r.get("entity_verdict_code"),
            "risk_grade": r.get("risk_grade"),
        }
    return {}


def _unadjusted_pairs(evidence: EvidenceRecord) -> dict:
    """Every `<x>_unadjusted` field paired with its adjusted sibling (09 §5.14.5:
    "explanation shows the adjusted and unadjusted values side by side")."""
    r = evidence.record
    pairs = {}
    for key in r:
        if key.endswith("_unadjusted"):
            base = key[: -len("_unadjusted")]
            if base in r:
                pairs[base] = {"adjusted": r[base], "unadjusted": r[key]}
    return pairs


def explain(evidence: EvidenceRecord, audience: str) -> Explanation:
    if audience not in ("consultant", "analyst", "adjudicator"):
        raise ValueError(f"unknown audience {audience!r}")
    r = evidence.record
    primary = r.get(evidence_primary_field(evidence.flow_code))
    reasons = r.get(evidence_reason_field(evidence.flow_code)) or []
    dominant_text = _describe(evidence.flow_code, primary)

    if audience == "consultant":
        if dominant_text:
            text = dominant_text
        else:
            from governance import flows

            text = f"No decline or referral reason applies (outcome {r.get(flows.get(evidence.flow_code).outcome_field)})."
        detail = {"primary_reason": dominant_text}
    elif audience == "analyst":
        text = (
            f"Decision {evidence.decision_id} (flow {evidence.flow_code}), "
            f"decided {evidence.decision_date}, config {evidence.config_version}.\n"
            f"Primary reason: {dominant_text}.\n"
            f"All reasons: {[_describe(evidence.flow_code, c) for c in reasons]}."
        )
        detail = {
            "primary_reason": dominant_text,
            "all_reasons": [{"code": c, "description": _describe(evidence.flow_code, c)} for c in reasons],
            "reason_registry_version": r.get(evidence_registry_version_field(evidence.flow_code)),
            "mechanism": _mechanism_detail(evidence),
            "adjustments_applied": r.get("adjustments_applied"),
            "adjustment_set_id": r.get("adjustment_set_id"),
            "unadjusted_vs_adjusted": _unadjusted_pairs(evidence),
            "full_record": r,
        }
    else:  # adjudicator
        text = (
            f"This decision was made on {evidence.decision_date} under policy version "
            f"{evidence.config_version} of the {evidence.flow_code} flow.\n"
            f"The outcome and its dominant, published reason: {dominant_text}.\n"
            f"Every reason considered, in order of severity: "
            f"{'; '.join(str(_describe(evidence.flow_code, c)) for c in reasons)}.\n"
            f"The reason registry in force at the decision date was version "
            f"{r.get(evidence_registry_version_field(evidence.flow_code))}."
        )
        detail = {
            "primary_reason": dominant_text,
            "all_reasons": [{"code": c, "description": _describe(evidence.flow_code, c)} for c in reasons],
            "reason_registry_version": r.get(evidence_registry_version_field(evidence.flow_code)),
            "policy_version": evidence.config_version,
            "unadjusted_vs_adjusted": _unadjusted_pairs(evidence),
        }
    return Explanation(decision_id=evidence.decision_id, flow_code=evidence.flow_code, audience=audience,
                        text=text, detail=detail)


def evidence_primary_field(flow_code: str) -> str:
    from governance import flows

    return flows.get(flow_code).primary_reason_field


def evidence_reason_field(flow_code: str) -> str:
    from governance import flows

    return flows.get(flow_code).reason_field


def evidence_registry_version_field(flow_code: str) -> str:
    from governance import flows

    return flows.get(flow_code).registry_version_field
