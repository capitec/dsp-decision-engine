"""Stage 4 -- entity adverse verdict roll-up (spec 05 §5.6).

Six of the twelve roll-up rules (SCOPE.md), chosen to keep one of each
required shape: a straight override (AE-R-01), a **count** rule that must
attribute to the whole satisfying set rather than one event (AE-R-02,
acceptance §10 item 2), two **aggregate** rules over unsatisfied amount
(AE-R-06/07), a floor (AE-R-11) and a criticality-dependent cap (AE-R-12).

**Attribution (spec 05 §13 Q4, Q6).** Every rule computes, alongside the
verdict, the flat list of `event_id`s that satisfied it -- one for AE-R-01,
three-or-more for AE-R-02, the whole contributing set for AE-R-06/07. The
shape is uniform (always a list of ids) regardless of rule kind; only the
list's length varies. `rollup_entities` returns this as one more pair of
flat, parallel long-form lists (`attr_entity_id`, `attr_event_id`,
`attr_rule_id`) rather than a per-entity nested list, for the same
materialisation reason `structure.py`'s docstring explains -- one row per
(entity, attributing event), joined back to the entity table by
`entity_id`.

Run the same function twice, once over `ev_severity_code` and once over
`ev_severity_code_unadjusted` (spec 00 §7.6's "same implementation, stack
off"), to get `entity_verdict_unadjusted` and `verdict_overlay_sensitive`
(spec 05 §5.6's requirement that a verdict changed by an overlay must say
so) -- see `pipeline.py` for where both calls happen.
"""
from __future__ import annotations

import polars as pl

from decider import frame_step

from business_nested import vocab
# `_ALWAYS_DISQUALIFYING` is 00's underscore-prefixed, non-public set (sequestration,
# fraud conviction, administration order) -- reused directly here rather than forked,
# the same "reach past the one stable entry point" trade-off 02's NOTES.md documents
# for `core.obligations._process` (see NOTES.md "Gaps in what I consumed").
from credit_core.adverse_events import (
    DISQUALIFYING as EV_DISQUALIFYING, IMMATERIAL as EV_IMMATERIAL, MATERIAL as EV_MATERIAL,
    _ALWAYS_DISQUALIFYING,
)

RECENT_MONTHS = 12
MINOR_COUNT_THRESHOLD = 3
AGGREGATE_MATERIAL_FLOOR = 150_000.0
AGGREGATE_MATERIAL_PCT = 0.15
AGGREGATE_DISQUALIFYING_FLOOR = 500_000.0
AGGREGATE_DISQUALIFYING_PCT = 0.50
IMMATERIAL_COUNT_CEILING = 2

AE_R_01 = "AE-R-01"
AE_R_02 = "AE-R-02"
AE_R_06 = "AE-R-06"
AE_R_07 = "AE-R-07"
AE_R_11 = "AE-R-11"
AE_R_12 = "AE-R-12"


def _events_by_entity(entity_ids: list[int], ev_entity_id: list[int], **ev_cols) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {eid: [] for eid in entity_ids}
    n = len(ev_entity_id)
    for i in range(n):
        eid = ev_entity_id[i]
        grouped.setdefault(eid, []).append({k: v[i] for k, v in ev_cols.items()})
    return grouped


def rollup_one_entity(events: list[dict], criticality_class: int, requested_amount: float) -> tuple[int, str, list[int]]:
    """One entity's events (each a dict with `event_id`, `severity`, `amount`,
    `is_satisfied`, `event_date_months_ago`) -> (verdict_code, binding_rule, attributing event ids)."""
    if not events:
        return vocab.CLEAR, AE_R_11, []

    fired: list[tuple[int, str, list[int]]] = []  # (verdict, rule, event_ids), most severe wins

    disqualifying_events = [e["event_id"] for e in events if e["severity"] == EV_DISQUALIFYING]
    if disqualifying_events:
        fired.append((vocab.DISQUALIFYING, AE_R_01, disqualifying_events))

    recent_minor = [e["event_id"] for e in events
                     if e["severity"] == vocab.MINOR and e["months_ago"] <= RECENT_MONTHS]
    if len(recent_minor) >= MINOR_COUNT_THRESHOLD:
        fired.append((vocab.MATERIAL, AE_R_02, recent_minor))

    unsatisfied = [e for e in events if not e["is_satisfied"] and (e["amount"] or 0.0) > 0]
    aggregate = sum(e["amount"] for e in unsatisfied)
    if aggregate > max(AGGREGATE_DISQUALIFYING_FLOOR, AGGREGATE_DISQUALIFYING_PCT * requested_amount):
        fired.append((vocab.DISQUALIFYING, AE_R_07, [e["event_id"] for e in unsatisfied]))
    elif aggregate > max(AGGREGATE_MATERIAL_FLOOR, AGGREGATE_MATERIAL_PCT * requested_amount):
        fired.append((vocab.MATERIAL, AE_R_06, [e["event_id"] for e in unsatisfied]))

    if all(e["severity"] == EV_IMMATERIAL for e in events) and len(events) <= IMMATERIAL_COUNT_CEILING:
        fired.append((vocab.CLEAR, AE_R_11, [e["event_id"] for e in events]))

    if not fired:
        # No roll-up rule fired: the verdict is the worst individual event severity,
        # bound to a synthetic "no roll-up rule" marker rather than a false rule id.
        worst = max(events, key=lambda e: e["severity"])
        base_verdict = {EV_IMMATERIAL: vocab.CLEAR, vocab.MINOR: vocab.MINOR, EV_MATERIAL: vocab.MATERIAL,
                         EV_DISQUALIFYING: vocab.DISQUALIFYING}[worst["severity"]]
        fired.append((base_verdict, "worst-event", [worst["event_id"]] if base_verdict != vocab.CLEAR else []))

    verdict, rule, event_ids = max(fired, key=lambda f: f[0])

    if criticality_class == vocab.PERIPHERAL and verdict == vocab.DISQUALIFYING:
        # AE-R-12: peripheral is capped at material, except a write-off/fraud-marker
        # event drove it (spec 00's `_ALWAYS_DISQUALIFYING` set -- sequestration, fraud
        # conviction, administration order -- stands in for spec 05's "AE-C-18 or AE-C-20").
        bypass = any(e["event_type_code"] in _ALWAYS_DISQUALIFYING for e in events if e["event_id"] in event_ids)
        if not bypass:
            verdict, rule = vocab.MATERIAL, AE_R_12

    return verdict, rule, sorted(set(event_ids))


def rollup_entities(
    entity_id: list[int], entity_criticality_class: list[int], requested_amount: float,
    ev_entity_id: list[int], ev_event_id: list[int], ev_severity: list[int], ev_amount: list,
    ev_is_satisfied: list[bool], ev_age_months: list[int], ev_event_type_code: list[int],
) -> dict:
    grouped = _events_by_entity(
        entity_id, ev_entity_id, event_id=ev_event_id, severity=ev_severity, amount=ev_amount,
        is_satisfied=ev_is_satisfied, months_ago=ev_age_months, event_type_code=ev_event_type_code,
    )
    criticality_by_id = dict(zip(entity_id, entity_criticality_class))

    verdicts, rules = [], []
    attr_entity, attr_event, attr_rule = [], [], []
    for eid in entity_id:  # entity_id is already sorted by structure.py -- order independence
        verdict, rule, events_attributed = rollup_one_entity(
            grouped.get(eid, []), criticality_by_id[eid], requested_amount)
        verdicts.append(verdict)
        rules.append(rule)
        for ev_id in events_attributed:
            attr_entity.append(eid)
            attr_event.append(ev_id)
            attr_rule.append(rule)

    return {
        "entity_verdict_code": verdicts, "entity_verdict_binding_rule": rules,
        "attr_entity_id": attr_entity, "attr_event_id": attr_event, "attr_rule_id": attr_rule,
    }


@frame_step(
    reads=["entity_id", "entity_criticality_class", "requested_amount", "ev_entity_id", "ev_event_id",
           "ev_severity_code", "ev_amount", "ev_is_satisfied", "ev_age_months", "ev_event_type_code",
           "ev_severity_code_unadjusted"],
    writes=["entity_verdict_code", "entity_verdict_binding_rule", "attr_entity_id", "attr_event_id",
            "attr_rule_id", "entity_verdict_code_unadjusted", "entity_verdict_overlay_sensitive"],
)
def rollup_step(df: pl.DataFrame) -> pl.DataFrame:
    cols = ["entity_id", "entity_criticality_class", "requested_amount", "ev_entity_id", "ev_event_id",
            "ev_severity_code", "ev_amount", "ev_is_satisfied", "ev_age_months", "ev_event_type_code",
            "ev_severity_code_unadjusted"]
    results = []
    for row in df.select(cols).to_dicts():
        adjusted = rollup_entities(
            row["entity_id"], row["entity_criticality_class"], row["requested_amount"], row["ev_entity_id"],
            row["ev_event_id"], row["ev_severity_code"], row["ev_amount"], row["ev_is_satisfied"],
            row["ev_age_months"], row["ev_event_type_code"],
        )
        unadjusted = rollup_entities(
            row["entity_id"], row["entity_criticality_class"], row["requested_amount"], row["ev_entity_id"],
            row["ev_event_id"], row["ev_severity_code_unadjusted"], row["ev_amount"], row["ev_is_satisfied"],
            row["ev_age_months"], row["ev_event_type_code"],
        )
        results.append({
            "entity_verdict_code": adjusted["entity_verdict_code"],
            "entity_verdict_binding_rule": adjusted["entity_verdict_binding_rule"],
            "attr_entity_id": adjusted["attr_entity_id"], "attr_event_id": adjusted["attr_event_id"],
            "attr_rule_id": adjusted["attr_rule_id"],
            "entity_verdict_code_unadjusted": unadjusted["entity_verdict_code"],
            "entity_verdict_overlay_sensitive": [
                a != u for a, u in zip(adjusted["entity_verdict_code"], unadjusted["entity_verdict_code"])
            ],
        })
    return df.with_columns(pl.DataFrame(results))
