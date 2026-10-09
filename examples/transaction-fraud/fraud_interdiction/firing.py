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


# A rule as the frame step sees it: plain values only, so the step's key holds the rule data and the diff compares it.
RuleRow = tuple[str, tuple[int, ...], int, int, tuple[str, ...] | None, str | None, str, bool]


def pack(rule: Rule) -> RuleRow:
    """(id, event types, effective from and to as ordinals with 0 for open, segments, velocity feature, stale
    behaviour, whether it has a base condition)."""
    return (rule.rule_id, rule.event_types, rule.effective_from.toordinal(),
            rule.effective_to.toordinal() if rule.effective_to else 0, rule.segments, rule.velocity_feature,
            rule.stale_absent, rule.base_condition is not None)


def _applies(rule: RuleRow, row: dict, on: int) -> bool:
    _, event_types, start, end, segments, velocity_feature, _, _ = rule
    return (row["event_type_code"] in event_types and start <= on and (end == 0 or on < end)
            and (segments is None or bool(set(segments) & set(row["client_segments"] or ())))
            # Restricted mode suspends the rules that read an aggregate it cannot trust.
            and not (row["degraded_mode_code"] == vocab.Mode.RESTRICTED and velocity_feature))


def _fired(rule: RuleRow, row: dict, column: str) -> bool | None:
    """Whether the rule's predicate held; None when it cannot be evaluated."""
    rule_id, _, _, _, _, velocity_feature, stale_absent, _ = rule
    if velocity_feature and row[STATE_COLUMN[velocity_feature]] != vocab.FRESH:
        if stale_absent == "suppress":
            return None
        if stale_absent == "evaluate_false":
            return False
    return bool(row[f"{rule_id}.{column}"])


def _row(row: dict, live: tuple[RuleRow, ...], shadow: tuple[RuleRow, ...]) -> dict[str, list[str]]:
    on = row["decision_date"].toordinal()
    fired, unevaluable, on_overlay, counterfactual = [], [], [], []
    for rule in filter(lambda r: _applies(r, row, on), live):
        now = _fired(rule, row, "fired")
        if now is None:
            unevaluable.append(rule[0])
            continue
        base = _fired(rule, row, "fired_base") if rule[7] else now
        if now:
            fired.append(rule[0])
            if not base:
                on_overlay.append(rule[0])
        if base:
            counterfactual.append(rule[0])
    shadow_fired = [r[0] for r in shadow if _applies(r, row, on) and _fired(r, row, "fired")]
    return dict(zip(WRITES, (fired, unevaluable, on_overlay, counterfactual, shadow_fired)))


def firing_step(rules: tuple[Rule, ...]):
    """A frame step: the rule population is as wide as the rule set, so it reads columns by name from the frame.

    It closes over tuples of plain values only, which is what lets `decider.lab.diff` see a rule's dates, segments
    or stale-data behaviour change.
    """
    live = tuple(pack(r) for r in rules if r.status == "live")
    shadow = tuple(pack(r) for r in rules if r.status == "shadow")
    reads = tuple([f"{r.rule_id}.fired" for r in rules]
                  + [f"{r.rule_id}.fired_base" for r in rules if r.status == "live" and r.base_condition]
                  + ["event_type_code", "decision_date", "client_segments", "degraded_mode_code",
                     *STATE_COLUMN.values()])

    @frame_step(reads=list(reads), writes=WRITES)
    def resolve_firing(df: pl.DataFrame) -> pl.DataFrame:
        rows = [_row(row, live, shadow) for row in df.select(list(reads)).to_dicts()]
        return df.with_columns(pl.DataFrame(rows, schema={w: pl.List(pl.Utf8) for w in WRITES}))

    return resolve_firing
