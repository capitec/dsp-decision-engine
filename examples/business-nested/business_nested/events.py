"""Adverse event classification: one severity per event, on its own merits.

`credit_core.adverse_events` classifies an event against two amount thresholds. Here the thresholds depend on
how critical the entity the event sits on is, and a scoped, approved overlay may tighten them before any event is
classified. `event_unit()` is the flow `each` runs on every event of every entity.
"""
from __future__ import annotations

import math
from datetime import date

from decider import DecisionTableConfig, allow_fallback, flow, missing_as, param, step

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister
from credit_core.adverse_events import (
    DEFAULT_LISTING, EVENT_TYPES, GARNISHEE_ORDER, JUDGMENT, LITIGATION, MINOR, TAX_NON_COMPLIANCE,
    event_age_months, event_severity_code,
)
from credit_core.evidence import cell_id

from business_nested import vocab

THRESHOLD_VERSION = "ev-thresh-2026.09"
THRESHOLD_ADJUSTMENT_SET_ID = "AS-05-EVT-2026.09"

# Event type -> (material, disqualifying) amounts for a critical, significant and peripheral entity, in rand.
# Garnishee orders stand in for the municipal and rental bucket; any other type uses the default row.
_THRESHOLDS = {
    JUDGMENT: ((10_000, 50_000), (25_000, 100_000), (50_000, 250_000)),
    DEFAULT_LISTING: ((5_000, 40_000), (7_500, 75_000), (15_000, 150_000)),
    TAX_NON_COMPLIANCE: ((25_000, 150_000), (50_000, 300_000), (100_000, 500_000)),
    LITIGATION: ((250_000, 1_000_000), (500_000, 2_000_000), (750_000, 3_000_000)),
    GARNISHEE_ORDER: ((10_000, 60_000), (20_000, 120_000), (40_000, 200_000)),
}
_DEFAULT_THRESHOLDS = ((5_000, 50_000), (10_000, 100_000), (20_000, 200_000))

# Halve the judgment materiality threshold for construction until the end of 2026: the policy lever the
# thresholds exist for. A materiality threshold is a ceiling that only shrinks, hence `cap_adjustment`.
THRESHOLD_OVERLAYS = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-05-014", kind="cap_adjustment", target="material_threshold",
        effect=AdjustmentEffect("multiply", 0.5), scope={"sector_code": 412, "event_type_code": JUDGMENT},
        stack_position=1,
        owner="Business Credit Risk Policy", approval_reference="CRC-2026-071",
        rationale="Halve judgment materiality thresholds for construction, two quarters",
        effective_from=date(2026, 1, 1), effective_to=date(2026, 12, 31), review_date=date(2026, 11, 1),
        tighten_only=True),
])


def build_threshold_table() -> DecisionTableConfig:
    """Writes `material_threshold_base`, `disqualifying_threshold_base` and `threshold_cell_id` for an event's
    `event_type_code` and `criticality_class`."""
    rows = []
    for event_type in EVENT_TYPES:
        for klass, (material, disqualifying) in zip((vocab.CRITICAL, vocab.SIGNIFICANT, vocab.PERIPHERAL),
                                                    _THRESHOLDS.get(event_type, _DEFAULT_THRESHOLDS)):
            rows.append({"event_type": event_type, "criticality": klass, "material": float(material),
                         "disqualifying": float(disqualifying),
                         "cell_id": cell_id("event_thresholds", THRESHOLD_VERSION, event_type, klass)})
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "event_thresholds",
        "columns": {"event_type": "Int64", "criticality": "Int64", "material": "Float64",
                    "disqualifying": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "event_type_code", "value_column": "event_type"},
            {"type": "eq", "variable": "criticality_class", "value_column": "criticality"}]},
        "outputs": ["material", "disqualifying", "cell_id"],
        "default": [*map(float, _DEFAULT_THRESHOLDS[0]), None],
    }).relabel(writes={"material": "material_threshold_base", "disqualifying": "disqualifying_threshold_base",
                       "cell_id": "threshold_cell_id"})


@allow_fallback
@step(outputs=("material_threshold", "disqualifying_threshold", "threshold_overlay_ids"))
def overlaid_thresholds(
    material_threshold_base: float, disqualifying_threshold_base: float, sector_code: int, criticality_class: float,
    event_type_code: float, decision_date: date, event_overlays_enabled: bool = param(True),
) -> tuple[float, float, list[str]]:
    """The thresholds after the overlays in force on the decision date, and the ids of those that applied.

    Switching `event_overlays_enabled` off runs the same code and returns the base thresholds.
    """
    provenance = {"sector_code": sector_code, "criticality_class": criticality_class, "event_type_code": event_type_code}
    results = [THRESHOLD_OVERLAYS.apply_stack(target, base, provenance, decision_date, THRESHOLD_ADJUSTMENT_SET_ID,
                                              stack_enabled=event_overlays_enabled)
               for target, base in (("material_threshold", material_threshold_base),
                                    ("disqualifying_threshold", disqualifying_threshold_base))]
    ids = sorted({a for r in results for a in r.adjustments_applied})
    return results[0].adjusted_value, results[1].adjusted_value, ids


@allow_fallback
def event_age(event_date: date, decision_date: date) -> int:
    return event_age_months(event_date, decision_date)


def amount_unknown(amount: float = missing_as(math.nan)) -> bool:
    return math.isnan(amount)


@allow_fallback
def event_severity(
    event_type_code: float, amount_unknown: bool, status_is_active: bool, is_disputed: bool, is_satisfied: bool,
    material_threshold: float, disqualifying_threshold: float, amount: float = missing_as(0.0),
) -> int:
    """`credit_core`'s severity against this entity's thresholds. A disputed event is classified one class lower,
    never below minor; an event of unknown amount is taken to be exactly material."""
    severity = event_severity_code(
        event_type_code, material_threshold if amount_unknown else amount, status_is_active, is_disputed,
        is_satisfied, material_threshold, disqualifying_threshold)
    return severity - 1 if is_disputed and severity > MINOR else severity


# The same rule against the base thresholds, to show what the overlay changed.
event_severity_unadjusted = step(event_severity, output="event_severity_code_unadjusted").named("unadjusted").relabel(
    reads={"material_threshold": "material_threshold_base", "disqualifying_threshold": "disqualifying_threshold_base"})


def event_unit():
    """The flow for one event. Reads the event's own fields; adds its age, thresholds, overlays and severities."""
    return flow(
        step(event_age, output="age_months"), step(amount_unknown),
        build_threshold_table(), overlaid_thresholds,
        step(event_severity, output="event_severity_code"), event_severity_unadjusted,
        name="event",
    ).emit("age_months", "amount_unknown", "material_threshold_base", "disqualifying_threshold_base", "threshold_cell_id",
           "material_threshold", "disqualifying_threshold", "threshold_overlay_ids", "event_severity_code")
