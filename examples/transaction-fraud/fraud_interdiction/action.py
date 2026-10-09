"""Action resolution: the one action, the rule it came from, every reason, and what the overlays changed."""
from __future__ import annotations

from bisect import bisect_left
from datetime import date
from typing import Any, Literal

from decider import ConfigurableStep, allow_fallback

from credit_core.adjustments import AdjustmentStack
from credit_core.reason_codes import ReasonCodes, rank_codes

from fraud_interdiction import vocab
from fraud_interdiction.overlays import escalated_action
from fraud_interdiction.rules import Rule
from fraud_interdiction.vocab import Action

OUTPUTS = ("action_code", "action_source_rule_id", "decline_reason_codes", "primary_reason_code",
           "reason_registry_version", "governance_exception", "counterfactual_action_code")
# A rule as the resolver sees it, sorted by id so a firing is found by bisection: plain values only, so the
# step's key holds them and the diff compares them.
ID, FAMILY, ACTION, SEVERITY, PRIORITY, CRITICAL, REASON = range(7)
RuleRow = tuple[str, str, int, int, int, bool, int]


def _tie_break(row: RuleRow, action: int) -> tuple:
    return (-action, -row[SEVERITY], row[PRIORITY], vocab.FAMILIES.index(row[FAMILY]), row[ID])


def _winner(fired: list[tuple[RuleRow, int]]) -> tuple[tuple[RuleRow, int] | None, bool]:
    """The firing that decides, and whether two critical rules asked for different actions.

    A critical rule is binding: when any fires, only critical rules compete. Among those, and otherwise
    among all firings, the most severe action wins, then severity, priority, family seniority and rule id.
    """
    critical = [f for f in fired if f[0][CRITICAL]]
    pool = critical or fired
    exception = len({action for _, action in critical}) > 1
    return (min(pool, key=lambda f: _tie_break(*f)) if pool else None), exception


def _rule(rules: tuple[RuleRow, ...], rule_id: str) -> RuleRow:
    return rules[bisect_left(rules, (rule_id,))]


@allow_fallback
def resolve_action(
    fired_rule_ids: list[str], counterfactual_fired_rule_ids: list[str], hard_block_code: int,
    hard_block_forced_action: int, channel_code: int, decision_date: date, adjustment_stack_enabled: bool,
    rules: tuple[RuleRow, ...], escalation: tuple, severity: tuple[tuple[int, int], ...], version: str,
    gates: tuple[tuple[int, str, int, int], ...],
) -> tuple[int, str, list[int], int | None, str, bool, int]:
    on = decision_date.toordinal()
    fired = []
    for rule_id in fired_rule_ids:
        row = _rule(rules, rule_id)
        fired.append((row, escalated_action(row[ACTION], escalation, row[FAMILY], channel_code, on,
                                            adjustment_stack_enabled)))
    win, exception = _winner(fired)
    held = sorted((g for g in gates if hard_block_code & g[0]), key=lambda g: -g[2])
    action, source = (win[1], win[0][ID]) if win else (Action.ALLOW, "")
    primary = win[0][REASON] if win else None
    if hard_block_forced_action > action:
        action, primary = hard_block_forced_action, held[0][3]
        source = "HARD_BLOCK:" + ",".join(g[1] for g in held)
    reasons, _ = rank_codes([row[REASON] for row, _ in fired] + [g[3] for g in held], severity, version)
    # The base rule set alone: the overlays' thresholds and escalations are left out, the hard blocks are not.
    base, _ = _winner([(row, row[ACTION]) for row in (_rule(rules, i) for i in counterfactual_fired_rule_ids)])
    counterfactual = max(base[1] if base else Action.ALLOW, hard_block_forced_action)
    return action, source, reasons, primary, version, exception, counterfactual


class ActionResolver(ConfigurableStep):
    """Turns the firing sets and the hard blocks into the decision.

    It holds everything it decides from: the `rules` (their action, severity, priority and reason code), the
    `escalation` overlays that raise a monitor to a step-up, and the `reason_codes` that rank the reasons.

    Example::

        resolver = ActionResolver(name="resolve_action", rules=rule_set.rules, escalation=escalation, reason_codes=codes)
    """

    type: Literal["action_resolver"] = "action_resolver"
    name: str = "resolve_action"
    rules: tuple[Rule, ...]
    escalation: AdjustmentStack
    reason_codes: ReasonCodes

    def to_ir(self, ctx: Any) -> Any:
        rows = tuple(sorted((r.rule_id, r.family, r.action_code, r.severity, r.priority, r.critical, r.reason_code)
                            for r in self.rules))
        gates = tuple((bit, name, int(action), code) for bit, (name, action, code) in vocab.GATES.items())
        return ctx.call(
            self, resolve_action, outputs=OUTPUTS,
            values={"rules": rows, "escalation": self.escalation.packed(),
                    "severity": tuple((c.code, c.severity_rank) for c in self.reason_codes.codes),
                    "version": self.reason_codes.version, "gates": gates})
