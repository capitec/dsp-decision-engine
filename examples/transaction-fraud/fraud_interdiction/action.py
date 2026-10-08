"""Action resolution: the one action, the rule it came from, every reason, and what the overlays changed."""
from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from decider import allow_fallback, step

from fraud_interdiction import vocab
from fraud_interdiction.overlays import escalated_action
from fraud_interdiction.reasons import REGISTRY
from fraud_interdiction.vocab import Action

if TYPE_CHECKING:
    from fraud_interdiction.rules import Rule

OUTPUTS = ("action_code", "action_source_rule_id", "decline_reason_codes", "primary_reason_code",
           "reason_registry_version", "governance_exception", "counterfactual_action_code")


def _tie_break(rule: Rule, action: int) -> tuple:
    return (-action, -rule.severity, rule.priority, vocab.FAMILIES.index(rule.family), rule.rule_id)


def _winner(fired: list[tuple[Rule, int]]) -> tuple[tuple[Rule, int] | None, bool]:
    """The firing that decides, and whether two critical rules asked for different actions.

    A critical rule is binding: when any fires, only critical rules compete. Among those, and otherwise
    among all firings, the most severe action wins, then severity, priority, family seniority and rule id.
    """
    critical = [f for f in fired if f[0].critical]
    pool = critical or fired
    exception = len({action for _, action in critical}) > 1
    return (min(pool, key=lambda f: _tie_break(*f)) if pool else None), exception


def resolve_action_step(rules: tuple[Rule, ...]):
    """The step that turns the firing sets and the hard blocks into the decision; `rules` is the governance data."""
    by_id = {r.rule_id: r for r in rules}

    @allow_fallback
    def resolve_action(
        fired_rule_ids: list[str], counterfactual_fired_rule_ids: list[str], hard_block_code: int,
        hard_block_forced_action: int, channel_code: int, decision_date: date, adjustment_stack_enabled: bool,
    ) -> tuple[int, str, list[int], int | None, str, bool, int]:
        fired = [(by_id[i], escalated_action(by_id[i].action_code, by_id[i].family, channel_code, decision_date,
                                             adjustment_stack_enabled)) for i in fired_rule_ids]
        win, exception = _winner(fired)
        gates = sorted((g for bit, g in vocab.GATES.items() if hard_block_code & bit), key=lambda g: -g[1])
        action, source = (win[1], win[0].rule_id) if win else (Action.ALLOW, "")
        primary = win[0].reason_code if win else None
        if hard_block_forced_action > action:
            action, source, primary = hard_block_forced_action, "HARD_BLOCK:" + ",".join(g[0] for g in gates), gates[0][2]
        reasons, _ = REGISTRY.rank([r.reason_code for r, _ in fired] + [g[2] for g in gates])
        # The base rule set alone: the overlays' thresholds and escalations are left out, the hard blocks are not.
        base, _ = _winner([(by_id[i], by_id[i].action_code) for i in counterfactual_fired_rule_ids])
        counterfactual = max(base[1] if base else Action.ALLOW, hard_block_forced_action)
        return action, source, reasons, primary, REGISTRY.version, exception, counterfactual

    return step(resolve_action, outputs=OUTPUTS)
