"""Policy caps: an ordered register narrows the amount, the term and the worst acceptable grade, and says why.

Order is part of the policy: an entry that takes a percentage of the current value gives a different answer after
another entry has run. Each ceiling keeps its whole chain, so "which rule set my cap" and "which rules looked at it and
did nothing" are both answerable, and they are different facts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from decider import allow_fallback, flow, missing_as, param, step

from retail_credit.documents import recorded

AMOUNT, TERM, GRADE = "amount", "term", "grade"

SEED, BOUND, NOT_BOUND, NOT_APPLICABLE, RAISED, DECLINED = (
    "seed", "bound", "evaluated_not_bound", "not_applicable", "raised", "declined")

R_CHANNEL_CLOSED = 2011
R_APPETITE_DECLINE = 2012
R_ENQUIRY_VELOCITY_DECLINE = 2013


@dataclass(frozen=True)
class Entry:
    """One rule. `applies` says whether the application is in the rule's scope, not whether its threshold is crossed:
    a rule in scope whose threshold is not crossed was evaluated and did not bind."""

    rule_id: str
    ceiling: str
    applies: Callable[[dict], bool]
    narrow: Callable[[float, dict], float]
    decline_reason: int = 0  # non-zero: the entry declines when it narrows the value to zero or below


_APPETITE_AMOUNT = {1: 500_000.0, 2: 460_000.0, 3: 420_000.0, 4: 360_000.0, 5: 300_000.0, 6: 220_000.0,
                    7: 180_000.0, 8: 140_000.0, 9: 70_000.0, 10: 50_000.0, 11: 28_000.0}  # grade 12: none
_TERM_BY_GRADE = [84, 84, 84, 72, 72, 72, 60, 60, 48, 48, 24, 24]

REGISTER: tuple[Entry, ...] = (
    Entry("CAP-0029", AMOUNT, lambda c: not c["channel_open"], lambda v, c: 0.0, R_CHANNEL_CLOSED),
    Entry("CAP-0104", AMOUNT, lambda c: True, lambda v, c: min(v, _APPETITE_AMOUNT.get(c["risk_grade"], 0.0)),
          R_APPETITE_DECLINE),
    Entry("CAP-0131", AMOUNT, lambda c: True,
          lambda v, c: min(v, 30_000.0) if c["months_employed"] < 9 else min(v, 85_000.0) if c["months_employed"] < 24 else v),
    Entry("CAP-0158", AMOUNT, lambda c: not c["is_existing_client"], lambda v, c: min(v, 70_000.0)),
    Entry("CAP-0176", AMOUNT, lambda c: True,
          lambda v, c: min(v, 35_000.0) if c["worst_arrears_months"] >= 3 else
          min(v, 140_000.0) if c["worst_arrears_months"] >= 2 else v),
    Entry("CAP-0212", AMOUNT, lambda c: True, lambda v, c: min(v, 40_000.0) if c["enquiry_velocity_90d"] >= 7 else v),
    Entry("CAP-0212-decline", AMOUNT, lambda c: c["enquiry_velocity_90d"] >= 12, lambda v, c: 0.0,
          R_ENQUIRY_VELOCITY_DECLINE),
    Entry("CAP-0248", AMOUNT, lambda c: c["channel_code"] == 5, lambda v, c: min(v, 180_000.0)),
    Entry("CAP-0266", AMOUNT, lambda c: c["on_concentration_watchlist"], lambda v, c: min(v, 60_000.0)),
    Entry("CAP-0301", AMOUNT, lambda c: True, lambda v, c: min(v, max(0.0, c["group_limit"] - c["group_exposure"]))),
    Entry("CAP-0310", AMOUNT, lambda c: c["income_evidence_tier"] >= 6, lambda v, c: min(v, 80_000.0)),
    Entry("CAP-0122", TERM, lambda c: True, lambda v, c: min(v, _TERM_BY_GRADE[min(max(c["risk_grade"], 1), 12) - 1])),
    Entry("CAP-0176-grade", GRADE, lambda c: True, lambda v, c: min(v, 9) if c["worst_arrears_months"] >= 2 else v),
)


def fold(ceiling: str, seed: float, context: dict) -> tuple[float, list[dict], int]:
    """(final value, chain, decline reason): the register's entries for `ceiling` in order. The chain starts with the
    seed; every entry then appears with its before and after values and its status. A declining entry ends the fold."""
    value, chain = seed, [{"rule_id": "SEED", "before": seed, "after": seed, "status": SEED}]
    for entry in (e for e in REGISTER if e.ceiling == ceiling):
        if not entry.applies(context):
            chain.append({"rule_id": entry.rule_id, "before": value, "after": value, "status": NOT_APPLICABLE})
            continue
        after = entry.narrow(value, context)
        declined = entry.decline_reason and after <= 0.0
        chain.append({"rule_id": entry.rule_id, "before": value, "after": after,
                      "status": DECLINED if declined else BOUND if after != value else NOT_BOUND})
        if declined:
            return after, chain, entry.decline_reason
        value = after
    return value, chain, 0


def binding_rule(chain: list[dict]) -> str:
    """The entry that set the final value: the last to bind, or raise, or "" if nothing narrowed the seed."""
    bound = [c["rule_id"] for c in chain if c["status"] in (BOUND, RAISED)]
    return bound[-1] if bound else ""


@allow_fallback
def policy_caps(
    risk_grade: int, worst_arrears_months: int, channel_code: int, income_evidence_tier: int, product_max_amount: float,
    product_max_term: int, months_employed: float = missing_as(0.0), is_existing_client: bool = missing_as(True),
    enquiry_velocity_90d: int = missing_as(0), channel_open: bool = missing_as(True),
    on_concentration_watchlist: bool = missing_as(False), group_limit: float = missing_as(math.inf),
    group_exposure: float = missing_as(0.0), campaign_id: str = missing_as(""),
    uplift_pct: float = param(0.20, ge=0.0, le=1.0), uplift_ceiling: float = param(200_000.0, ge=0.0),
) -> tuple[float, str, list[dict], bool, int, float, int]:
    """(amount_cap_before_overlay, amount_cap_binding_rule, amount_cap_chain, p09_declined, p09_decline_reason,
    term_cap, worst_acceptable_grade).

    A campaign may raise the amount cap, the only entry that can, by at most `uplift_pct` and never to more than
    `uplift_ceiling`.
    """
    context = dict(locals())
    amount, chain, decline = fold(AMOUNT, product_max_amount, context)
    if campaign_id and not decline:
        raised = max(amount, min(amount * (1.0 + uplift_pct), uplift_ceiling))
        chain.append({"rule_id": "CAP-0118", "before": amount, "after": raised,
                      "status": RAISED if raised > amount else NOT_BOUND})
        amount = raised
    else:
        chain.append({"rule_id": "CAP-0118", "before": amount, "after": amount, "status": NOT_APPLICABLE})
    term, _, _ = fold(TERM, product_max_term, context)
    grade, _, _ = fold(GRADE, 12, context)
    return amount, binding_rule(chain), chain, bool(decline), decline, term, int(grade)


policy_caps_step = step(policy_caps, outputs=(
    "amount_cap_before_overlay", "amount_cap_binding_rule", "amount_cap_chain", "p09_declined", "p09_decline_reason",
    "term_cap", "worst_acceptable_grade"))

def round_down_to_grid(amount_cap: float, amount_grid: float) -> float:
    return math.floor(amount_cap / amount_grid) * amount_grid


def amount_cap_unit(amount_cap_overlays):
    """Overlays enter after the register and may only reduce, so they can never do what the campaign uplift does.
    `amount_cap` is written twice, by the overlay and then rounded down to the grid, so the two steps are a flow."""
    return flow(recorded(amount_cap_overlays, "amount_cap"), step(round_down_to_grid, output="amount_cap"),
                name="amount_cap_overlay",
                ).emit("amount_cap", "amount_cap_unadjusted", "amount_cap_adjustment_set_id", "amount_cap_adjustments_applied")
