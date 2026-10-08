"""The adverse verdict of an entity: what its classified events add up to, and which events say so.

It is not the worst event and not an average. Three minor events in a year are material, and so are six
unsatisfied judgments of R30 000 that are each minor alone. Every verdict names the rule that decided it and
the whole set of events that satisfied the rule, so a decline can say which entity and which events.
"""
from __future__ import annotations

from decider import allow_fallback, param, step

from credit_core.adverse_events import ADMINISTRATION_ORDER, FRAUD_CONVICTION, IMMATERIAL, SEQUESTRATION

from business_nested import vocab

AE_R_01 = "AE-R-01"  # any disqualifying event
AE_R_02 = "AE-R-02"  # several minor events in the last year
AE_R_06 = "AE-R-06"  # unsatisfied amount adds up to material
AE_R_07 = "AE-R-07"  # unsatisfied amount adds up to disqualifying
AE_R_11 = "AE-R-11"  # nothing but a couple of immaterial events
AE_R_12 = "AE-R-12"  # a peripheral entity is capped at material
WORST_EVENT = "worst-event"  # no rule fired: the verdict is the worst single event

# A peripheral entity's disqualifying verdict is capped at material unless one of these drove it.
NEVER_CAPPED_EVENT_TYPES = (SEQUESTRATION, FRAUD_CONVICTION, ADMINISTRATION_ORDER)


def rollup(
    events: list[dict], severity_key: str, criticality: int, requested_amount: float, *,
    recent_months: int, minor_count: int, material_floor: float, material_pct: float,
    disqualifying_floor: float, disqualifying_pct: float, immaterial_count: int,
) -> tuple[int, str, list[int]]:
    """(verdict_code, binding rule, attributing event ids) from events carrying `severity_key`.

    Example: three minor events in the last year give `(MATERIAL, "AE-R-02", [the three ids])`.
    """
    if not events:
        return vocab.CLEAR, AE_R_11, []
    # Rules are listed in rule order, so a tie in severity goes to the lower-numbered rule.
    fired: list[tuple[int, str, list[int]]] = []

    def ids(chosen):
        return [e["event_id"] for e in chosen]

    if disqualifying := [e for e in events if e[severity_key] == vocab.DISQUALIFYING]:
        fired.append((vocab.DISQUALIFYING, AE_R_01, ids(disqualifying)))
    recent_minor = [e for e in events if e[severity_key] == vocab.MINOR and e["age_months"] <= recent_months]
    if len(recent_minor) >= minor_count:
        fired.append((vocab.MATERIAL, AE_R_02, ids(recent_minor)))
    unsatisfied = [e for e in events if not e["is_satisfied"] and (e["amount"] or 0.0) > 0]
    total = sum(e["amount"] for e in unsatisfied)
    if total > max(disqualifying_floor, disqualifying_pct * requested_amount):
        fired.append((vocab.DISQUALIFYING, AE_R_07, ids(unsatisfied)))
    elif total > max(material_floor, material_pct * requested_amount):
        fired.append((vocab.MATERIAL, AE_R_06, ids(unsatisfied)))
    if all(e[severity_key] == IMMATERIAL for e in events) and len(events) <= immaterial_count:
        fired.append((vocab.CLEAR, AE_R_11, ids(events)))
    if not fired:
        worst = max(events, key=lambda e: e[severity_key])
        fired.append((worst[severity_key], WORST_EVENT, ids([worst]) if worst[severity_key] > vocab.CLEAR else []))
    verdict, rule, chosen = max(fired, key=lambda f: f[0])
    if criticality == vocab.PERIPHERAL and verdict == vocab.DISQUALIFYING and not any(
            e["event_type_code"] in NEVER_CAPPED_EVENT_TYPES for e in events if e["event_id"] in chosen):
        verdict, rule = vocab.MATERIAL, AE_R_12
    return verdict, rule, sorted(chosen)


@allow_fallback
@step(outputs=("adverse_verdict_code", "adverse_binding_rule", "adverse_event_ids", "adverse_verdict_provisional",
               "adverse_verdict_code_unadjusted", "adverse_verdict_overlay_sensitive",
               "adverse_verdict_code_undisputed"))
def adverse_verdict(
    adverse_events: list[dict], criticality_class: float, requested_amount: float,
    recent_months: int = param(12, ge=1), minor_count: int = param(3, ge=1),
    aggregate_material_floor: float = param(150_000.0, ge=0.0), aggregate_material_pct: float = param(0.15, ge=0.0),
    aggregate_disqualifying_floor: float = param(500_000.0, ge=0.0),
    aggregate_disqualifying_pct: float = param(0.50, ge=0.0), immaterial_count: int = param(2, ge=0),
) -> tuple[int, str, list[int], bool, int, bool, int]:
    """The verdict, the rule that bound it and its events, and three variants of the same rules.

    `adverse_verdict_provisional`: a disputed event is among those that decide it.
    `adverse_verdict_code_unadjusted`: the verdict against the base thresholds, without any overlay.
    `adverse_verdict_code_undisputed`: the verdict if every disputed event were left out.
    """
    policy = dict(recent_months=recent_months, minor_count=minor_count, material_floor=aggregate_material_floor,
                  material_pct=aggregate_material_pct, disqualifying_floor=aggregate_disqualifying_floor,
                  disqualifying_pct=aggregate_disqualifying_pct, immaterial_count=immaterial_count)
    verdict, rule, chosen = rollup(adverse_events, "event_severity_code", criticality_class, requested_amount, **policy)
    unadjusted = rollup(adverse_events, "event_severity_code_unadjusted", criticality_class, requested_amount, **policy)[0]
    undisputed = rollup([e for e in adverse_events if not e["is_disputed"]], "event_severity_code",
                        criticality_class, requested_amount, **policy)[0]
    provisional = any(e["is_disputed"] for e in adverse_events if e["event_id"] in chosen)
    return verdict, rule, chosen, provisional, unadjusted, verdict != unadjusted, undisputed
