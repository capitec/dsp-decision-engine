"""Action resolution (spec 01 §5.12): one action, one source rule, a complete reason set.

Reads `fired_rule_ids` and `counterfactual_fired_rule_ids` from `firing.py`
-- **never** `shadow_fired_rule_ids`, which is spec 01 §5.11's structural
shadow-isolation guarantee: a step that is never wired to read a column
cannot be influenced by it, so "no shadow rule has ever changed an
`action_code`" (§10 item 4) is true by construction of the dag, not by
review discipline.
"""
from __future__ import annotations

from datetime import date

from decider import frame_step
import polars as pl

from fraud_interdiction import overlays, vocab
from fraud_interdiction.rules import RuleCatalog, RuleDefinition


def _effective_action(rule: RuleDefinition, channel_code: int, decision_date: date,
                       adjustment_stack_enabled: bool) -> tuple[int, bool]:
    """`(action_code, escalated)` -- the action-escalation overlay only ever raises severity."""
    if rule.action_code != vocab.ACTION_MONITOR:
        return rule.action_code, False
    escalated, _adj_id = overlays.at_web_action_escalation(
        rule.family, channel_code, decision_date, adjustment_stack_enabled)
    return (vocab.ACTION_STEP_UP, True) if escalated else (rule.action_code, False)


def _tie_key(c: dict) -> tuple:
    return (-c["severity"], c["priority"], vocab.FAMILY_PRECEDENCE.index(c["family"]), c["rule_id"])


def _resolve(fired_ids: list[str], catalog: RuleCatalog, channel_code: int, decision_date: date,
             adjustment_stack_enabled: bool, apply_escalation: bool) -> tuple[int, str, list[int], bool]:
    """One (action_code, action_source_rule_id, reason_codes, governance_exception) for one firing set."""
    candidates = []
    for rid in fired_ids:
        rule = catalog[rid]
        action, escalated = (
            _effective_action(rule, channel_code, decision_date, adjustment_stack_enabled)
            if apply_escalation else (rule.action_code, False)
        )
        candidates.append({"rule_id": rid, "family": rule.family, "severity": rule.severity,
                            "priority": rule.priority, "critical": rule.critical, "action_code": action})

    reason_codes = sorted({catalog[rid].reason_code for rid in fired_ids})
    if not candidates:
        return vocab.ACTION_ALLOW, "", reason_codes, False

    critical = [c for c in candidates if c["critical"]]
    governance_exception = False
    if critical:
        allow_families = {c["family"] for c in critical if c["action_code"] == vocab.ACTION_ALLOW}
        if allow_families:
            candidates = [c for c in candidates if c["critical"] or c["family"] not in allow_families]
            critical = [c for c in candidates if c["critical"]]
        if len({c["action_code"] for c in critical}) > 1:
            governance_exception = True  # §5.12 item 3: two critical rules disagree
        pool = critical
    else:
        pool = candidates

    max_action = max(c["action_code"] for c in pool)
    tied = sorted((c for c in pool if c["action_code"] == max_action), key=_tie_key)
    winner = tied[0]
    return winner["action_code"], winner["rule_id"], reason_codes, governance_exception


def build_resolve_action_step(catalog: RuleCatalog):
    """A `frame_step` closed over the catalog (governance data, not a per-row column)."""

    def resolve_action(df: pl.DataFrame) -> pl.DataFrame:
        rows = df.select(["fired_rule_ids", "counterfactual_fired_rule_ids", "hard_block_code",
                           "hard_block_forced_action", "channel_code", "decision_date",
                           "adjustment_stack_enabled"]).to_dicts()
        results = []
        for row in rows:
            action, source, reasons, exception = _resolve(
                row["fired_rule_ids"] or [], catalog, row["channel_code"], row["decision_date"],
                row["adjustment_stack_enabled"], apply_escalation=True)
            counterfactual_action, _, _, _ = _resolve(
                row["counterfactual_fired_rule_ids"] or [], catalog, row["channel_code"], row["decision_date"],
                row["adjustment_stack_enabled"], apply_escalation=False)
            forced = row["hard_block_forced_action"] or 0
            if forced > action:
                action, source = forced, f"HARD_BLOCK:{row['hard_block_code']}"
            if forced > counterfactual_action:
                counterfactual_action = forced
            results.append({
                "action_code": action, "action_source_rule_id": source, "decline_reason_codes": reasons,
                "governance_exception": exception, "counterfactual_action_code": counterfactual_action,
            })
        out = pl.DataFrame(results, schema={
            "action_code": pl.Int64, "action_source_rule_id": pl.Utf8, "decline_reason_codes": pl.List(pl.Int64),
            "governance_exception": pl.Boolean, "counterfactual_action_code": pl.Int64,
        })
        return df.with_columns(out)

    resolve_action.__name__ = "resolve_action"
    reads = ["fired_rule_ids", "counterfactual_fired_rule_ids", "hard_block_code", "hard_block_forced_action",
             "channel_code", "decision_date", "adjustment_stack_enabled"]
    writes = ["action_code", "action_source_rule_id", "decline_reason_codes", "governance_exception",
              "counterfactual_action_code"]
    return frame_step(resolve_action, reads=reads, writes=writes)
