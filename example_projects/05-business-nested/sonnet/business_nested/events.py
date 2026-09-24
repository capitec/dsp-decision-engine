"""Stage 3 -- adverse event classification (spec 05 §5.5).

Reuses `core.adverse_events.event_severity_code` (project 00) **unchanged**
-- it already takes `material_threshold`/`disqualifying_threshold` as
caller-supplied params, which is exactly spec 05 §13 Q3's answer for "where
does the parameterisation of a shared capability by its context live": the
threshold table lives here (project 05, keyed by event type x this
project's own criticality class), the classification arithmetic stays in
00, called directly as a plain function -- a *wrapped* capability, not a
fork and not a second copy of the satisfied/amount/status logic.

Project 05's own 14 event types (spec 05 §4.4) are not re-declared: the
classification mechanism (satisfied/amount/status-only/always-disqualifying)
is identical to 00's `core.adverse_events` 14-type enum, so this module
imports 00's type constants directly as project 05's event-type vocabulary
too, rather than inventing a second numbering for the same four behaviours.
See NOTES.md "Reuse".

The threshold table itself is a `DecisionTableConfig` (spec 05 §5.5's "the
commonest policy intervention... halve the judgment materiality threshold"
-- exactly the table-as-config-document pattern 00 §9 and this project's
09-C checklist require), read through a **nested `Engine`** bound once at
import time (see `scoring.py`'s docstring for why: no `decider` construct
runs one step once per element of a ragged collection nested inside a row,
so a second `Engine.run()` call over a small per-application "events"
frame is how a table lookup composes with per-event Python classification
logic in the same stage).
"""
from __future__ import annotations

import polars as pl

from decider import Engine, frame_step
from decider.steps.tables import DecisionTableConfig

from credit_core.adjustments import AdjustmentRegister
from credit_core.adverse_events import (
    DEFAULT_LISTING, GARNISHEE_ORDER, IMMATERIAL, JUDGMENT, LITIGATION, MINOR, TAX_NON_COMPLIANCE,
    DISQUALIFYING as EVENT_DISQUALIFYING, event_severity_code,
)
from credit_core.evidence import cell_id as _cell_id

from business_nested import vocab

EVENT_THRESHOLD_VERSION = "ev-thresh-2026.09"

# (event_type_code, criticality_class) -> (material, disqualifying), rand.
# Four rows straight from spec 05 §5.5's table; GARNISHEE_ORDER stands in for the
# spec's fifth "municipal / rental" bucket (project 05 has no separate code for it --
# see NOTES.md "Reuse"). Any other event type falls through to the DEFAULT row.
_THRESHOLDS = {
    (JUDGMENT, vocab.CRITICAL): (10_000.0, 50_000.0),
    (JUDGMENT, vocab.SIGNIFICANT): (25_000.0, 100_000.0),
    (JUDGMENT, vocab.PERIPHERAL): (50_000.0, 250_000.0),
    (DEFAULT_LISTING, vocab.CRITICAL): (5_000.0, 40_000.0),
    (DEFAULT_LISTING, vocab.SIGNIFICANT): (7_500.0, 75_000.0),
    (DEFAULT_LISTING, vocab.PERIPHERAL): (15_000.0, 150_000.0),
    (TAX_NON_COMPLIANCE, vocab.CRITICAL): (25_000.0, 150_000.0),
    (TAX_NON_COMPLIANCE, vocab.SIGNIFICANT): (50_000.0, 300_000.0),
    (TAX_NON_COMPLIANCE, vocab.PERIPHERAL): (100_000.0, 500_000.0),
    (LITIGATION, vocab.CRITICAL): (250_000.0, 1_000_000.0),
    (LITIGATION, vocab.SIGNIFICANT): (500_000.0, 2_000_000.0),
    (LITIGATION, vocab.PERIPHERAL): (750_000.0, 3_000_000.0),
    (GARNISHEE_ORDER, vocab.CRITICAL): (10_000.0, 60_000.0),
    (GARNISHEE_ORDER, vocab.SIGNIFICANT): (20_000.0, 120_000.0),
    (GARNISHEE_ORDER, vocab.PERIPHERAL): (40_000.0, 200_000.0),
}
_DEFAULT_BY_CLASS = {vocab.CRITICAL: (5_000.0, 50_000.0), vocab.SIGNIFICANT: (10_000.0, 100_000.0),
                      vocab.PERIPHERAL: (20_000.0, 200_000.0)}


def build_event_threshold_table() -> DecisionTableConfig:
    rows = []
    for (event_type, cls), (material, disqualifying) in _THRESHOLDS.items():
        rows.append({
            "event_type": event_type, "criticality": cls, "material_threshold": material,
            "disqualifying_threshold": disqualifying,
            "cell_id": _cell_id("event_thresholds", EVENT_THRESHOLD_VERSION, event_type, cls),
        })
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "event_amount_thresholds",
        "columns": {"event_type": "Int64", "criticality": "Int64", "material_threshold": "Float64",
                    "disqualifying_threshold": "Float64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "and", "expressions": [
            {"type": "eq", "variable": "event_type_code", "value_column": "event_type"},
            {"type": "eq", "variable": "criticality_class", "value_column": "criticality"},
        ]},
        "outputs": ["material_threshold", "disqualifying_threshold", "cell_id"],
        "default": [None, None, None],  # filled from _DEFAULT_BY_CLASS below when unmatched
    })


_THRESHOLD_ENGINE = Engine().bind(build_event_threshold_table())


def _lookup_thresholds(event_types: list[int], criticalities: list[int]) -> tuple[list[float], list[float], list[str]]:
    if not event_types:
        return [], [], []
    frame = pl.DataFrame({"event_type_code": event_types, "criticality_class": criticalities})
    result = _THRESHOLD_ENGINE.run(frame, {})
    material = result["material_threshold"].to_list()
    disqualifying = result["disqualifying_threshold"].to_list()
    cells = result["cell_id"].to_list()
    for i, (m, cls) in enumerate(zip(material, criticalities)):
        if m is None:
            material[i], disqualifying[i] = _DEFAULT_BY_CLASS[cls]
            cells[i] = _cell_id("event_thresholds", EVENT_THRESHOLD_VERSION, "default", cls)
    return material, disqualifying, cells


def classify_events(
    ev_entity_id: list[int], ev_event_id: list[int], ev_event_type_code: list[int], ev_amount: list,
    ev_status_is_active: list[bool], ev_is_disputed: list[bool],
    ev_is_satisfied: list[bool], entity_id: list[int], entity_criticality_class: list[int], decision_date,
    threshold_overlays: AdjustmentRegister, adjustment_set_id: str, stack_enabled: bool = True,
    business_provenance: dict | None = None,
) -> dict:
    """Pure function (no `decider` machinery) -- unit-testable directly, and reused by
    `counterfactual.py`'s re-derivation with no pipeline round-trip.

    Overlay position 1 (spec 05 §5.5): the threshold overlay is applied to
    `material_threshold`/`disqualifying_threshold` **before** `event_severity_code`
    ever runs, so it changes the classification itself, not a value derived from it
    (spec 05 §5.5 requirement 3). Both the base and the overlaid threshold are
    recorded on every event, with the overlay id where one applied (requirements 1-2).
    """
    criticality_by_entity = dict(zip(entity_id, entity_criticality_class))
    criticalities = [criticality_by_entity.get(eid, vocab.PERIPHERAL) for eid in ev_entity_id]
    material_base, disqualifying_base, cell_ids = _lookup_thresholds(ev_event_type_code, criticalities)

    severity, threshold_used, threshold_base_out, overlay_ids, provisional = [], [], [], [], []
    for i in range(len(ev_event_id)):
        cls = criticalities[i]
        provenance = dict(business_provenance or {})
        provenance["criticality_class"] = cls
        mat_result = threshold_overlays.apply_stack(
            "material_threshold", material_base[i], provenance, decision_date, adjustment_set_id, stack_enabled,
        )
        dis_result = threshold_overlays.apply_stack(
            "disqualifying_threshold", disqualifying_base[i], provenance, decision_date, adjustment_set_id,
            stack_enabled,
        )
        sev = event_severity_code(
            ev_event_type_code[i], ev_amount[i] or 0.0, ev_status_is_active[i], ev_is_disputed[i],
            ev_is_satisfied[i], mat_result.adjusted_value, dis_result.adjusted_value,
        )
        if ev_is_disputed[i]:
            # AE-C-22 (spec 05 §5.5): one class lower, never below minor, flagged provisional.
            sev = max(MINOR, sev - 1) if sev > IMMATERIAL else MINOR
            provisional.append(True)
        else:
            provisional.append(False)
        severity.append(sev)
        # Record only the overlay(s) that touched the threshold *actually used* to
        # classify this event -- an overlay considered but not decisive (e.g. one
        # scoped to `material_threshold` on an event that classified off the
        # disqualifying threshold instead) must not be attributed as if it moved the
        # answer (spec 05 §5.5 requirement 2: "which threshold value was used").
        if sev == EVENT_DISQUALIFYING:
            threshold_used.append(dis_result.adjusted_value)
            threshold_base_out.append(disqualifying_base[i])
            overlay_ids.append("|".join(dis_result.adjustments_applied))
        else:
            threshold_used.append(mat_result.adjusted_value)
            threshold_base_out.append(material_base[i])
            overlay_ids.append("|".join(mat_result.adjustments_applied))

    return {
        "ev_severity_code": severity, "ev_threshold_used": threshold_used, "ev_threshold_base": threshold_base_out,
        "ev_overlay_ids": overlay_ids, "ev_classification_provisional": provisional,
        "ev_threshold_cell_id": cell_ids,
    }



def make_classify_step(threshold_overlays: AdjustmentRegister, adjustment_set_id: str):
    """Builds the `frame_step` bound to one project's overlay register (spec 00 §7.6's
    "same implementation, stack-off through a flag" -- `adjustment_stack_enabled` is a
    request-level field here, not a `param()`, because §5.14.3 requires the counterfactual
    run to be selectable per call, not only per deployment)."""

    @frame_step(
        reads=["ev_entity_id", "ev_event_id", "ev_event_type_code", "ev_amount",
               "ev_status_is_active", "ev_is_disputed", "ev_is_satisfied", "entity_id",
               "entity_criticality_class", "decision_date", "adjustment_stack_enabled", "sector_code"],
        writes=["ev_severity_code", "ev_threshold_used", "ev_threshold_base", "ev_overlay_ids",
                "ev_classification_provisional", "ev_threshold_cell_id", "ev_severity_code_unadjusted"],
    )
    def classify_events_step(df: pl.DataFrame) -> pl.DataFrame:
        cols = ["ev_entity_id", "ev_event_id", "ev_event_type_code", "ev_amount",
                "ev_status_is_active", "ev_is_disputed", "ev_is_satisfied", "entity_id",
                "entity_criticality_class", "decision_date", "adjustment_stack_enabled", "sector_code"]
        rows = df.select(cols).to_dicts()
        results = []
        for row in rows:
            args = (
                row["ev_entity_id"], row["ev_event_id"], row["ev_event_type_code"], row["ev_amount"],
                row["ev_status_is_active"], row["ev_is_disputed"], row["ev_is_satisfied"],
                row["entity_id"], row["entity_criticality_class"], row["decision_date"],
            )
            adjusted = classify_events(*args, threshold_overlays, adjustment_set_id,
                                        bool(row["adjustment_stack_enabled"]),
                                        business_provenance={"sector_code": row["sector_code"]})
            # §5.6/§7.6: the same implementation, stack disabled, run alongside the adjusted
            # pass so `rollup.py` can tell an overlay-driven verdict change from a real one.
            unadjusted = classify_events(*args, threshold_overlays, adjustment_set_id, False,
                                          business_provenance={"sector_code": row["sector_code"]})
            adjusted["ev_severity_code_unadjusted"] = unadjusted["ev_severity_code"]
            results.append(adjusted)
        return df.with_columns(pl.DataFrame(results))

    return classify_events_step
