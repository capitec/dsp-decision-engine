"""Adverse event classification: one severity per event, on its own merits.

`credit_core.adverse_events` classifies an event against two amount thresholds. Here the thresholds depend on
how critical the entity the event sits on is, and a scoped, approved overlay may tighten them before any event is
classified. `event_unit()` is the flow `each` runs on every event of every entity.
"""
from __future__ import annotations

import math
from datetime import date
from types import SimpleNamespace
from typing import Any, Literal

from decider import (
    ConfigurableStep, DecisionTableConfig, ParamRef, Value, allow_fallback, flow, missing_as, step,
)
from pydantic import model_validator

from credit_core.adjustments import AdjustmentStack, Overlay, evaluate
from credit_core.adverse_events import (
    DEFAULT_LISTING, EVENT_TYPES, GARNISHEE_ORDER, JUDGMENT, LITIGATION, MINOR, TAX_NON_COMPLIANCE,
    event_age_months, event_severity_code,
)
from credit_core.evidence import cell_id

from business_nested import vocab
from business_nested.documents import shipped

THRESHOLD_VERSION = "ev-thresh-2026.09"

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


TARGETS = ("material_threshold", "disqualifying_threshold")


@allow_fallback
def overlay_thresholds(
    material_threshold_base: float, disqualifying_threshold_base: float, sector_code: int, criticality_class: float,
    event_type_code: float, decision_date: date, material_overlays: tuple, disqualifying_overlays: tuple,
    event_overlays_enabled: bool,
) -> tuple[float, float, list[str]]:
    provenance = {"sector_code": sector_code, "criticality_class": criticality_class, "event_type_code": event_type_code}
    on = decision_date.toordinal()
    material, applied_material, _ = evaluate(material_overlays, material_threshold_base, provenance, on,
                                             event_overlays_enabled)
    disqualifying, applied_disqualifying, _ = evaluate(
        disqualifying_overlays, disqualifying_threshold_base, provenance, on, event_overlays_enabled)
    return material, disqualifying, sorted({*applied_material, *applied_disqualifying})


class OverlaidThresholds(ConfigurableStep):
    """The event thresholds after the overlays in force on the decision date, and the ids of those that applied.

    The overlays are fields, so a release check sees them. Each targets `material_threshold` or
    `disqualifying_threshold` and may be scoped on `sector_code`, `criticality_class` and `event_type_code`, which an
    `AdjustmentStack` step cannot match. The param `event_overlays_enabled` switches them all off.

    Example: `ConfigurableStep.load("configs/0.1.0/event_threshold_overlays.json")`.
    """

    type: Literal["event_threshold_overlays"] = "event_threshold_overlays"
    name: str = "overlaid_thresholds"
    set_id: str
    adjustments: tuple[Overlay, ...]
    enabled: Value[bool] = ParamRef(param="event_overlays_enabled", default=True)

    @model_validator(mode="after")
    def _overlays_are_for_a_threshold(self) -> OverlaidThresholds:
        stray = [a.adjustment_id for a in self.adjustments if a.target not in TARGETS]
        if stray:
            raise ValueError(f"overlays {stray} are not for {list(TARGETS)}")
        self._stack(TARGETS[0]), self._stack(TARGETS[1])  # unique ids, tighten-only
        return self

    def _stack(self, target: str) -> AdjustmentStack:
        return AdjustmentStack(target=target, set_id=self.set_id,
                               adjustments=[a for a in self.adjustments if a.target == target])

    def to_ir(self, ctx: Any) -> Any:
        # `_ordered` packs a stack in stack order for `evaluate`; credit_core has no public name for it.
        return ctx.call(
            self, overlay_thresholds,
            values={"material_overlays": self._stack(TARGETS[0])._ordered(),
                    "disqualifying_overlays": self._stack(TARGETS[1])._ordered(),
                    "event_overlays_enabled": self.enabled},
            outputs=("material_threshold", "disqualifying_threshold", "threshold_overlay_ids"))


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


def event_unit(overlays: OverlaidThresholds | None = None):
    """The flow for one event. Reads the event's own fields; adds its age, thresholds, overlays and severities.

    `overlays` is the `event_threshold_overlays` document; by default the one in `configs/0.1.0`.
    """
    return flow(
        step(event_age, output="age_months"), step(amount_unknown),
        build_threshold_table(), overlays or shipped("event_threshold_overlays"),
        step(event_severity, output="event_severity_code"), event_severity_unadjusted,
        name="event",
    ).emit("age_months", "amount_unknown", "material_threshold_base", "disqualifying_threshold_base", "threshold_cell_id",
           "material_threshold", "disqualifying_threshold", "threshold_overlay_ids", "event_severity_code")


# governance-replay lists the overlays through this name; no step reads it.
THRESHOLD_OVERLAYS = SimpleNamespace(_all=shipped("event_threshold_overlays").adjustments)
