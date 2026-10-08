"""Which rules fired: applicability, stale velocity, and the effect of the overlays on each firing."""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from decider import frame_step

from fraud_interdiction import vocab

if TYPE_CHECKING:
    from fraud_interdiction.rules import Rule

STATE_COLUMN = {
    "velocity_count_1min": "velocity_count_1min_state",
    "velocity_count_10min": "velocity_count_10min_state",
    "velocity_sum_amount_1h": "velocity_sum_amount_1h_state",
}
WRITES = ["fired_rule_ids", "unevaluable_rule_ids", "fired_on_overlay_ids", "counterfactual_fired_rule_ids",
          "shadow_fired_rule_ids"]


def _applies(rule: Rule, row: dict) -> bool:
    on = row["decision_date"]
    return (row["event_type_code"] in rule.event_types
            and rule.effective_from <= on and (rule.effective_to is None or on < rule.effective_to)
            and (rule.segments is None or bool(set(rule.segments) & set(row["client_segments"] or ())))
            # Restricted mode suspends the rules that read an aggregate it cannot trust.
            and not (row["degraded_mode_code"] == vocab.Mode.RESTRICTED and rule.velocity_feature))


def _fired(rule: Rule, row: dict, column: str) -> bool | None:
    """Whether the rule's predicate held; None when it cannot be evaluated."""
    if rule.velocity_feature and row[STATE_COLUMN[rule.velocity_feature]] != vocab.FRESH:
        if rule.stale_absent == "suppress":
            return None
        if rule.stale_absent == "evaluate_false":
            return False
    return bool(row[f"{rule.rule_id}.{column}"])


def _row(row: dict, live: list[Rule], shadow: list[Rule]) -> dict[str, list[str]]:
    fired, unevaluable, on_overlay, counterfactual = [], [], [], []
    for rule in filter(lambda r: _applies(r, row), live):
        now = _fired(rule, row, "fired")
        if now is None:
            unevaluable.append(rule.rule_id)
            continue
        base = _fired(rule, row, "fired_base") if rule.base_condition is not None else now
        if now:
            fired.append(rule.rule_id)
            if not base:
                on_overlay.append(rule.rule_id)
        if base:
            counterfactual.append(rule.rule_id)
    shadow_fired = [r.rule_id for r in shadow if _applies(r, row) and _fired(r, row, "fired")]
    return dict(zip(WRITES, (fired, unevaluable, on_overlay, counterfactual, shadow_fired)))


def firing_step(rules: tuple[Rule, ...]):
    """A frame step: the rule population is as wide as the rule set, so it reads columns by name from the frame."""
    live = [r for r in rules if r.status == "live"]
    shadow = [r for r in rules if r.status == "shadow"]
    reads = ([f"{r.rule_id}.fired" for r in rules] + [f"{r.rule_id}.fired_base" for r in live if r.base_condition]
             + ["event_type_code", "decision_date", "client_segments", "degraded_mode_code", *STATE_COLUMN.values()])

    @frame_step(reads=reads, writes=WRITES)
    def resolve_firing(df: pl.DataFrame) -> pl.DataFrame:
        rows = [_row(row, live, shadow) for row in df.select(reads).to_dicts()]
        return df.with_columns(pl.DataFrame(rows, schema={w: pl.List(pl.Utf8) for w in WRITES}))

    return resolve_firing
