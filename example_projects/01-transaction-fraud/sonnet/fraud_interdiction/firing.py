"""Applicability, the firing set and the counterfactual (spec 01 §5.8, §5.10-§5.11).

Joins the tree evaluation's raw `<rule_id>.fired` boolean columns (the
predicate "shape", from `rules.py`) with the `RuleCatalog`'s governance
metadata (the "answer": which of those firings actually count) to produce:

- `fired_rule_ids`: applicable live rules whose predicate held and are not
  suppressed as unevaluable (§5.10).
- `unevaluable_rule_ids`: applicable live rules suppressed because a
  referenced velocity aggregate was stale/absent and the rule's declared
  behaviour is "suppress" (§5.4).
- `fired_on_overlay_ids`: the subset of `fired_rule_ids` that fired only
  because an overlay moved their threshold (§5.10) -- computed by
  comparing against the *base* evaluation (`rules.build_overlay_base_document`).
- `counterfactual_fired_rule_ids`: what the base rule set alone would have
  produced, ignoring the overlay stack (§5.12 "Emits").
- `shadow_fired_rule_ids`: kept in a separate output the action-resolution
  step never reads (see its docstring) -- shadow isolation "by construction"
  (§5.11, §10 item 4), not by convention.

This is a `frame_step`, not 635 named function parameters: `decider`'s
`step()` binds by parameter name, so a step reading the whole applicable
population would need one argument per rule -- the same "column-naming
bookkeeping" 00 NOTES.md flags for composing table capabilities, at 10x
the scale. `frame_step` reads the columns it needs by name from the
whole `DataFrame` instead, which is the actual escape hatch for a
variable, data-defined *width* rather than a variable *length* list (00's
usual `frame_step` case) -- worth its own line in NOTES.md.
"""
from __future__ import annotations

from decider import frame_step
import polars as pl

from fraud_interdiction.features import ABSENT, FRESH, STALE
from fraud_interdiction.rules import RuleCatalog, RuleDefinition

# The three velocity aggregates `features.py` tracks freshness for.
TRACKED_VELOCITY_STATE_COLUMNS = {
    "velocity_count_1min": "velocity_count_1min_state",
    "velocity_count_10min": "velocity_count_10min_state",
    "velocity_sum_amount_1h": "velocity_sum_amount_1h_state",
}


def _applicable(rule: RuleDefinition, event_type_code: int, decision_date, client_segments: list[str],
                 degraded_mode_code: str) -> bool:
    if event_type_code not in rule.event_types:
        return False
    if not (rule.effective_from <= decision_date and (rule.effective_to is None or decision_date < rule.effective_to)):
        return False
    if rule.segments is not None and not (set(rule.segments) & set(client_segments or [])):
        return False
    # §5.6: restricted mode suspends the declared-suppressible subset.
    if degraded_mode_code == "restricted" and rule.suppressible:
        return False
    return True


def _unevaluable(rule: RuleDefinition, row: dict) -> bool:
    if rule.velocity_feature not in TRACKED_VELOCITY_STATE_COLUMNS:
        return False
    state = row.get(TRACKED_VELOCITY_STATE_COLUMNS[rule.velocity_feature])
    degraded = state in (STALE, ABSENT)
    if not degraded:
        return False
    if rule.stale_absent_behavior == "suppress":
        return True
    return False  # evaluate_false is handled by treating the rule as not-fired below; last_known_good: no change


def _forced_false(rule: RuleDefinition, row: dict) -> bool:
    if rule.velocity_feature not in TRACKED_VELOCITY_STATE_COLUMNS:
        return False
    state = row.get(TRACKED_VELOCITY_STATE_COLUMNS[rule.velocity_feature])
    return rule.stale_absent_behavior == "evaluate_false" and state in (STALE, ABSENT)


def _resolve_row(row: dict, live: tuple[RuleDefinition, ...], shadow: tuple[RuleDefinition, ...],
                  overlay_eligible_ids: frozenset[str]) -> dict:
    event_type_code = row["event_type_code"]
    decision_date = row["decision_date"]
    segments = row.get("client_segments") or []
    degraded_mode_code = row["degraded_mode_code"]

    fired, unevaluable, on_overlay, counterfactual = [], [], [], []
    for r in live:
        if not _applicable(r, event_type_code, decision_date, segments, degraded_mode_code):
            continue
        if _unevaluable(r, row):
            unevaluable.append(r.rule_id)
            continue
        raw_fired = bool(row.get(f"{r.rule_id}.fired")) and not _forced_false(r, row)
        base_fired = raw_fired
        if r.rule_id in overlay_eligible_ids:
            base_fired = bool(row.get(f"{r.rule_id}.fired_base")) and not _forced_false(r, row)
            if raw_fired and not base_fired:
                on_overlay.append(r.rule_id)
        if raw_fired:
            fired.append(r.rule_id)
        if base_fired:
            counterfactual.append(r.rule_id)

    shadow_fired = [
        r.rule_id for r in shadow
        if _applicable(r, event_type_code, decision_date, segments, degraded_mode_code)
        and bool(row.get(f"{r.rule_id}.fired")) and not _unevaluable(r, row) and not _forced_false(r, row)
    ]
    return {
        "fired_rule_ids": fired, "unevaluable_rule_ids": unevaluable,
        "fired_on_overlay_ids": on_overlay, "counterfactual_fired_rule_ids": counterfactual,
        "shadow_fired_rule_ids": shadow_fired,
    }


def build_resolve_firing_step(catalog: RuleCatalog):
    """A `frame_step` closed over `catalog` -- the rule population is build-time data, not a param."""
    live = catalog.by_status("live")
    shadow = catalog.by_status("shadow")
    overlay_eligible_ids = frozenset(r.rule_id for r in catalog.overlay_eligible())
    reads = (
        [f"{r.rule_id}.fired" for r in live] + [f"{r.rule_id}.fired" for r in shadow]
        + [f"{r.rule_id}.fired_base" for r in live if r.rule_id in overlay_eligible_ids]
        + ["event_type_code", "decision_date", "client_segments", "degraded_mode_code"]
        + list(TRACKED_VELOCITY_STATE_COLUMNS.values())
    )
    writes = ["fired_rule_ids", "unevaluable_rule_ids", "fired_on_overlay_ids",
              "counterfactual_fired_rule_ids", "shadow_fired_rule_ids"]

    def resolve_firing_set(df: pl.DataFrame) -> pl.DataFrame:
        rows = df.select(reads).to_dicts()
        results = [_resolve_row(row, live, shadow, overlay_eligible_ids) for row in rows]
        out = pl.DataFrame(results, schema={w: pl.List(pl.Utf8) for w in writes})
        return df.with_columns(out)

    resolve_firing_set.__name__ = "resolve_firing_set"
    return frame_step(resolve_firing_set, reads=reads, writes=writes)
