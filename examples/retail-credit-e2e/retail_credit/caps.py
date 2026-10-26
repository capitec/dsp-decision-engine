"""Policy caps: one small step per rule narrows the amount, and says why; the term and the worst acceptable grade follow.

Order is part of the policy: a rule that takes a percentage of the current value gives a different answer after
another rule has run, so the rules are steps of a flow in the order the policy lists them. The amount keeps its whole
chain, so "which rule set my cap" and "which rules looked at it and did nothing" are both answerable, and they are
different facts. Thresholds are literals in each rule's function, so a changed threshold shows in a diff as a changed
step; the appetite by grade (CAP-0104) is the `appetite_by_grade` document.
"""
from __future__ import annotations

import math
from typing import TypedDict

from decikit import allow_fallback, flow, missing_as, param, step

from retail_credit.documents import recorded

SEED, BOUND, NOT_BOUND, NOT_APPLICABLE, RAISED, DECLINED = (
    "seed", "bound", "evaluated_not_bound", "not_applicable", "raised", "declined")

R_CHANNEL_CLOSED = 2011
R_APPETITE_DECLINE = 2012
R_ENQUIRY_VELOCITY_DECLINE = 2013

AMOUNT = ("amount_value", "amount_chain", "amount_declined")


class ChainLink(TypedDict):
    """One rule's entry in the amount chain."""

    rule_id: str
    before: float
    after: float
    status: str


Fold = tuple[float, list[ChainLink], int]


def narrow(value: float, chain: list[ChainLink], declined: int, rule_id: str, applies: bool, cap: float,
           decline_reason: int = 0) -> Fold:
    """One rule's turn. `applies` says whether the application is in the rule's scope, not whether its threshold is
    crossed: a rule in scope whose threshold is not crossed was evaluated and did not bind. A rule with a
    `decline_reason` declines when it narrows the value to zero or below, and a decline ends the fold."""
    if declined:
        return value, chain, declined
    if not applies:
        return value, [*chain, {"rule_id": rule_id, "before": value, "after": value, "status": NOT_APPLICABLE}], 0
    after = min(value, cap)
    stop = bool(decline_reason) and after <= 0.0
    status = DECLINED if stop else BOUND if after != value else NOT_BOUND
    return after, [*chain, {"rule_id": rule_id, "before": value, "after": after, "status": status}], decline_reason if stop else 0


@allow_fallback
def seed(product_max_amount: float) -> Fold:
    return product_max_amount, [{"rule_id": "SEED", "before": product_max_amount, "after": product_max_amount,
                                 "status": SEED}], 0


@allow_fallback
def cap_0029(amount_value: float, amount_chain: list[ChainLink], amount_declined: int,
             channel_open: bool = missing_as(True)) -> Fold:
    return narrow(amount_value, amount_chain, amount_declined, "CAP-0029", not channel_open, 0.0, R_CHANNEL_CLOSED)


@allow_fallback
def cap_0104(amount_value: float, amount_chain: list[ChainLink], amount_declined: int, appetite_amount_cap: float) -> Fold:
    return narrow(amount_value, amount_chain, amount_declined, "CAP-0104", True, appetite_amount_cap, R_APPETITE_DECLINE)


@allow_fallback
def cap_0131(amount_value: float, amount_chain: list[ChainLink], amount_declined: int,
             months_employed: float = missing_as(0.0)) -> Fold:
    return narrow(amount_value, amount_chain, amount_declined, "CAP-0131", True,
                  30_000.0 if months_employed < 9 else 85_000.0 if months_employed < 24 else math.inf)


@allow_fallback
def cap_0158(amount_value: float, amount_chain: list[ChainLink], amount_declined: int,
             is_existing_client: bool = missing_as(True)) -> Fold:
    return narrow(amount_value, amount_chain, amount_declined, "CAP-0158", not is_existing_client, 70_000.0)


@allow_fallback
def cap_0176(amount_value: float, amount_chain: list[ChainLink], amount_declined: int, worst_arrears_months: int) -> Fold:
    return narrow(amount_value, amount_chain, amount_declined, "CAP-0176", True,
                  35_000.0 if worst_arrears_months >= 3 else 140_000.0 if worst_arrears_months >= 2 else math.inf)


@allow_fallback
def cap_0212(amount_value: float, amount_chain: list[ChainLink], amount_declined: int,
             enquiry_velocity_90d: int = missing_as(0)) -> Fold:
    return narrow(amount_value, amount_chain, amount_declined, "CAP-0212", True,
                  40_000.0 if enquiry_velocity_90d >= 7 else math.inf)


@allow_fallback
def cap_0212_decline(amount_value: float, amount_chain: list[ChainLink], amount_declined: int,
                     enquiry_velocity_90d: int = missing_as(0)) -> Fold:
    return narrow(amount_value, amount_chain, amount_declined, "CAP-0212-decline", enquiry_velocity_90d >= 12, 0.0,
                  R_ENQUIRY_VELOCITY_DECLINE)


@allow_fallback
def cap_0248(amount_value: float, amount_chain: list[ChainLink], amount_declined: int, channel_code: int) -> Fold:
    return narrow(amount_value, amount_chain, amount_declined, "CAP-0248", channel_code == 5, 180_000.0)


@allow_fallback
def cap_0266(amount_value: float, amount_chain: list[ChainLink], amount_declined: int,
             on_concentration_watchlist: bool = missing_as(False)) -> Fold:
    return narrow(amount_value, amount_chain, amount_declined, "CAP-0266", on_concentration_watchlist, 60_000.0)


@allow_fallback
def cap_0301(amount_value: float, amount_chain: list[ChainLink], amount_declined: int,
             group_limit: float = missing_as(math.inf), group_exposure: float = missing_as(0.0)) -> Fold:
    return narrow(amount_value, amount_chain, amount_declined, "CAP-0301", True, max(0.0, group_limit - group_exposure))


@allow_fallback
def cap_0310(amount_value: float, amount_chain: list[ChainLink], amount_declined: int, income_evidence_tier: int) -> Fold:
    return narrow(amount_value, amount_chain, amount_declined, "CAP-0310", income_evidence_tier >= 6, 80_000.0)


RULES = (cap_0029, cap_0104, cap_0131, cap_0158, cap_0176, cap_0212, cap_0212_decline, cap_0248, cap_0266, cap_0301,
         cap_0310)


def amount_rules_unit(appetite_by_grade):
    """The amount rules in policy order; writes `amount_value`, `amount_chain` and `amount_declined` (the reason, or 0)."""
    return flow(step(seed, outputs=AMOUNT), appetite_by_grade, *(step(rule, outputs=AMOUNT) for rule in RULES),
                name="amount_rules").emit(*AMOUNT)


def term_cap(risk_grade: int, product_max_term: int) -> float:
    return float(min(product_max_term, (84, 84, 84, 72, 72, 72, 60, 60, 48, 48, 24, 24)[min(max(risk_grade, 1), 12) - 1]))


def worst_acceptable_grade(worst_arrears_months: int) -> int:
    return 9 if worst_arrears_months >= 2 else 12


def binding_rule(chain: list[ChainLink]) -> str:
    """The rule that set the final value: the last to bind, or raise, or "" if nothing narrowed the seed."""
    bound = [c["rule_id"] for c in chain if c["status"] in (BOUND, RAISED)]
    return bound[-1] if bound else ""


@allow_fallback
def policy_caps(
    amount_value: float, amount_chain: list[ChainLink], amount_declined: int, campaign_id: str = missing_as(""),
    uplift_pct: float = param(0.20, ge=0.0, le=1.0), uplift_ceiling: float = param(200_000.0, ge=0.0),
) -> tuple[float, str, list[ChainLink], bool, int]:
    """(amount_cap_before_overlay, amount_cap_binding_rule, amount_cap_chain, p09_declined, p09_decline_reason).

    A campaign may raise the amount cap, the only rule that can, by at most `uplift_pct` and never to more than
    `uplift_ceiling`.
    """
    if campaign_id and not amount_declined:
        raised = max(amount_value, min(amount_value * (1.0 + uplift_pct), uplift_ceiling))
        chain = [*amount_chain, {"rule_id": "CAP-0118", "before": amount_value, "after": raised,
                                 "status": RAISED if raised > amount_value else NOT_BOUND}]
        amount_value = raised
    else:
        chain = [*amount_chain, {"rule_id": "CAP-0118", "before": amount_value, "after": amount_value,
                                 "status": NOT_APPLICABLE}]
    return amount_value, binding_rule(chain), chain, bool(amount_declined), amount_declined


policy_caps_step = step(policy_caps, outputs=(
    "amount_cap_before_overlay", "amount_cap_binding_rule", "amount_cap_chain", "p09_declined", "p09_decline_reason"))
term_cap_step = step(term_cap)
worst_acceptable_grade_step = step(worst_acceptable_grade)


def round_down_to_grid(amount_cap: float, amount_grid: float) -> float:
    return math.floor(amount_cap / amount_grid) * amount_grid


def amount_cap_unit(amount_cap_overlays):
    """Overlays enter after the rules and may only reduce, so they can never do what the campaign uplift does.
    `amount_cap` is written twice, by the overlay and then rounded down to the grid, so the two steps are a flow."""
    return flow(recorded(amount_cap_overlays, "amount_cap"), step(round_down_to_grid, output="amount_cap"),
                name="amount_cap_overlay",
                ).emit("amount_cap", "amount_cap_unadjusted", "amount_cap_adjustment_set_id", "amount_cap_adjustments_applied")
