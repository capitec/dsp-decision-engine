"""P09 -- Policy gates and the cap waterfall (spec 10 §5.10): the hard part.

The spec's own worked example (10 §5.10, Client V's `amount_cap` chain) is
reproduced here nearly verbatim as this project's register -- eleven
entries plus the uplift and one cap overlay -- because that worked example
*is* the attribution requirement, stated as data: for every ceiling, after
the register runs, the final value, which entry bound it, the full
ordered chain, and which entries were evaluated-and-did-not-bind versus
not-applicable must all be answerable (10 §5.10 "The attribution
requirement").

This is a representative register (~13 entries), not the full 118 (47
applicable to product 10) -- cap-register *volume* is project 03's
dominant difficulty (SCOPE.md: 03 builds "the cap waterfall at real
volume (all 52 rules...)"); this project's difficulty is composition and
scale of phases and entry points, and the waterfall mechanism -- ordered
fold, full chain, not-applicable vs. evaluated-and-did-not-bind, the
uplift's authority bound, the cap overlay entering as its own row -- is
identical at 13 entries or at 47. Extending `AMOUNT_CAP_REGISTER` is
mechanical.

Not built on `decider.steps.tables.DecisionTableConfig`: an ordered fold
that must record every intermediate value (10 §5.10 item 3, "the full
chain") is a *reduction* over a sequence, not a keyed lookup -- the table
primitive answers "which one row matches", not "what did every row in
order do to a running value". `credit_core.adjustments.AdjustmentRegister`
is this project's closest precedent for "an ordered, scoped, attributed
fold" and this module follows its shape deliberately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from decider import missing_as, param, step

from retail_credit.overlays import ADJUSTMENT_SET_ID, OVERLAY_REGISTER

AMOUNT_CAP = "amount_cap"
TERM_CAP = "term_cap"
GRADE_CAP = "worst_acceptable_grade"
DECLINE = "decline"

BOUND = "bound"
EVALUATED_NOT_BOUND = "evaluated_not_bound"
NOT_APPLICABLE = "not_applicable"
RAISED = "raised"

R_CHANNEL_CLOSED = 2011
R_APPETITE_DECLINE = 2012
R_ENQUIRY_VELOCITY_DECLINE = 2013


@dataclass(frozen=True)
class CapEntry:
    rule_id: str
    owner_team_code: int
    entry_class: str  # "product" | "policy" | "appetite" | "exposure" | "regulatory" | "campaign"
    ceiling: str       # AMOUNT_CAP | TERM_CAP | GRADE_CAP | DECLINE
    applicable: Callable[[dict], bool]
    narrow: Callable[[float, dict], float] | None = None  # None on a DECLINE-class entry
    decline_reason_code: int | None = None
    raise_allowed: bool = False  # only ever True for CAP-0118


def _grade_appetite_amount(grade: int) -> float:
    table = {1: 500_000.0, 2: 460_000.0, 3: 420_000.0, 4: 360_000.0, 5: 300_000.0, 6: 220_000.0,
             7: 180_000.0, 8: 140_000.0, 9: 70_000.0, 10: 50_000.0, 11: 28_000.0, 12: 0.0}
    return table.get(grade, 0.0)


def _grade_term_cap(grade: int) -> int:
    if grade <= 3:
        return 84
    if grade <= 6:
        return 72
    if grade <= 8:
        return 60
    if grade <= 10:
        return 48
    return 24


AMOUNT_CAP_REGISTER: tuple[CapEntry, ...] = (
    CapEntry("CAP-0011", 7, "product", AMOUNT_CAP, lambda c: True, lambda v, c: min(v, 500_000.0)),
    CapEntry("CAP-0029", 8, "product", DECLINE, lambda c: not c.get("channel_open", True),
             decline_reason_code=R_CHANNEL_CLOSED),
    CapEntry("CAP-0104", 4, "appetite", AMOUNT_CAP, lambda c: True,
             lambda v, c: min(v, _grade_appetite_amount(c["risk_grade"])) if c["risk_grade"] < 12
             else 0.0, decline_reason_code=R_APPETITE_DECLINE),
    # CAP-0131, CAP-0176 (both ceilings) and CAP-0212 are always applicable (10 §5.10's own
    # worked example: CAP-0131 is "applicable: yes (67 months)" even though 67 months is
    # *above* its 24-month threshold and the entry does not bind) -- applicability is scope
    # ("this application has an employer tenure"), not "the threshold happens to trigger".
    # A CapEntry whose predicate encoded the threshold instead would mark every
    # non-triggering application NOT_APPLICABLE, collapsing exactly the distinction 10
    # §5.10 item 4 requires ("evaluated and did not bind" vs. "not applicable" are
    # different facts) -- confirmed as a real bug here (`tests/test_cap_waterfall.py`),
    # not a hypothetical one: an earlier version of this register did exactly that.
    CapEntry("CAP-0131", 7, "policy", AMOUNT_CAP, lambda c: True,
             lambda v, c: min(v, 30_000.0 if c["months_employed"] < 9.0 else
                              85_000.0 if c["months_employed"] < 24.0 else v)),
    CapEntry("CAP-0158", 7, "policy", AMOUNT_CAP, lambda c: not c.get("is_existing_client", True),
             lambda v, c: min(v, 70_000.0)),
    CapEntry("CAP-0176", 4, "policy", AMOUNT_CAP, lambda c: True,
             lambda v, c: min(v, 35_000.0 if c["worst_arrears_months"] >= 3.0 else
                              140_000.0 if c["worst_arrears_months"] >= 2.0 else v)),
    CapEntry("CAP-0176-grade", 4, "policy", GRADE_CAP, lambda c: True,
             lambda v, c: min(v, 9) if c["worst_arrears_months"] >= 2.0 else v),
    CapEntry("CAP-0212", 4, "policy", AMOUNT_CAP, lambda c: True,
             lambda v, c: min(v, 40_000.0) if c["enquiry_velocity_90d"] >= 7 else v),
    CapEntry("CAP-0212-decline", 4, "policy", DECLINE, lambda c: c["enquiry_velocity_90d"] >= 12,
             decline_reason_code=R_ENQUIRY_VELOCITY_DECLINE),
    CapEntry("CAP-0248", 7, "policy", AMOUNT_CAP, lambda c: c["channel_code"] == 5,
             lambda v, c: min(v, 180_000.0)),
    CapEntry("CAP-0266", 10, "exposure", AMOUNT_CAP, lambda c: True,
             lambda v, c: min(v, 60_000.0) if c.get("on_concentration_watchlist", False) else v),
    CapEntry("CAP-0301", 1, "exposure", AMOUNT_CAP, lambda c: True,
             lambda v, c: min(v, max(0.0, c["group_limit"] - c["group_exposure"]))),
    CapEntry("CAP-0122", 4, "appetite", TERM_CAP, lambda c: True,
             lambda v, c: min(v, _grade_term_cap(c["risk_grade"]))),
    CapEntry(
        "CAP-0118", 0, "campaign", AMOUNT_CAP,   # owner 0 = Credit Committee (governs, owns no phase; phases.py)
        lambda c: bool(c.get("campaign_id")),
        narrow=None,  # raise_allowed=True routes through `_raise_amount`, never through `narrow`
        raise_allowed=True,
    ),
)
# ponytail: a DECLINE-class entry (CAP-0029, CAP-0212-decline) is re-evaluated once per
# ceiling fold (amount, grade, term) since `run_cap_waterfall` doesn't know in advance
# which folds share it -- correct (idempotent predicates) but wasteful. Upgrade path: one
# fold over all three ceilings at once if this shows up in a batch-scale profile.


def _raise_amount(v: float, uplift_pct: float, absolute_authority: float, regulatory_ceiling: float) -> float:
    raised = min(v * (1.0 + uplift_pct), v + absolute_authority, regulatory_ceiling)
    return max(raised, v)  # never lower than the seed it raises


def run_cap_waterfall(
    ceiling: str, seed: float, ctx: dict, *, uplift_pct: float = 0.20,
    absolute_authority: float = 200_000.0, regulatory_ceiling: float = float("inf"),
) -> tuple[float, list[dict], bool, int | None]:
    """Fold `AMOUNT_CAP_REGISTER`'s entries for one ceiling over `seed`. Returns
    (final_value, chain, declined, decline_reason_code). `chain` is the full ordered
    attribution (10 §5.10 item 3): every entry, applicable or not, evaluated or not,
    with its before/after and status.
    """
    value = seed
    chain: list[dict] = []
    for entry in AMOUNT_CAP_REGISTER:
        if entry.ceiling not in (ceiling, DECLINE):
            continue
        applies = entry.applicable(ctx)
        if not applies:
            chain.append({"rule_id": entry.rule_id, "applicable": False, "before": value, "after": value,
                          "status": NOT_APPLICABLE})
            continue
        if entry.ceiling == DECLINE:
            chain.append({"rule_id": entry.rule_id, "applicable": True, "before": value, "after": value,
                          "status": "declined"})
            return value, chain, True, entry.decline_reason_code
        before = value
        if entry.raise_allowed:
            after = _raise_amount(before, uplift_pct, absolute_authority, regulatory_ceiling)
            status = RAISED if after > before else EVALUATED_NOT_BOUND
        else:
            after = entry.narrow(before, ctx)
            status = BOUND if after != before else EVALUATED_NOT_BOUND
            if entry.decline_reason_code is not None and after <= 0.0:
                chain.append({"rule_id": entry.rule_id, "applicable": True, "before": before, "after": after,
                              "status": "declined"})
                return after, chain, True, entry.decline_reason_code
        chain.append({"rule_id": entry.rule_id, "applicable": True, "before": before, "after": after,
                      "status": status})
        value = after
    return value, chain, False, None


def binding_entry(chain: list[dict]) -> str | None:
    """Which entry set the final value (10 §5.10 item 2) -- the last one to bind, or None
    if the seed itself was never narrowed. Two entries reducing to the same value: the
    earlier is the binder, the later is *coincident* (10 §5.10) -- a strictly-later BOUND
    entry always overwrites, matching that rule.
    """
    bound = [c for c in chain if c["status"] in (BOUND, RAISED)]
    return bound[-1]["rule_id"] if bound else None


def evaluate_amount_and_grade_caps(
    risk_grade: int, worst_arrears_months: float, channel_code: int,
    months_employed: float = missing_as(0.0), is_existing_client: bool = missing_as(True),
    enquiry_velocity_90d: int = missing_as(0), channel_open: bool = missing_as(True),
    on_concentration_watchlist: bool = missing_as(False), group_limit: float = missing_as(1e12),
    group_exposure: float = missing_as(0.0), campaign_id: str = missing_as(""),
    product_max_amount: float = param(500_000.0, ge=0.0),
) -> tuple[float, str, bool, int, int, float]:
    """P09's "amount, term and grade entries, before P10" pass (O-10): everything except
    `instalment_cap`, which needs `max_affordable_instalment` and therefore runs after P10
    (see `evaluate_instalment_cap`).

    Returns only scalars (amount_cap, the binding rule id, declined, decline reason,
    term_cap, worst_acceptable_grade) -- **not** the full attribution chain. A `list[dict]`
    terminal output from a decider step hits the same result-materialisation crash project
    00 documents for `frame_step` (NOTES.md "Framework friction" §4.2: "cannot parse numpy
    data type dtype('O') into Polars data type", because the generic output path round-trips
    every unconsumed output through `np.where(...).tolist()` regardless of whether it came
    from a `frame_step` or a plain `step`) -- confirmed by reproducing it here with a
    `list[dict]` output on a plain `step`, not only on a `frame_step`. The full chain
    (10 §5.10 item 3) is real and tested (`tests/test_cap_waterfall.py`); it is computed by
    calling `run_cap_waterfall` directly, outside decider, from the same context a served
    request already carries -- an evidence-assembly concern, not a per-request pipeline
    output, exactly as `decision_record.build_decision_record` is (see its own docstring).
    """
    ctx = dict(
        risk_grade=risk_grade, months_employed=months_employed, is_existing_client=is_existing_client,
        worst_arrears_months=worst_arrears_months, enquiry_velocity_90d=enquiry_velocity_90d,
        channel_code=channel_code, channel_open=channel_open, on_concentration_watchlist=on_concentration_watchlist,
        group_limit=group_limit, group_exposure=group_exposure, campaign_id=campaign_id,
    )
    amount_cap, amount_chain, declined, decline_reason = run_cap_waterfall(AMOUNT_CAP, product_max_amount, ctx)
    grade_cap, grade_chain, grade_declined, grade_decline_reason = run_cap_waterfall(GRADE_CAP, 12, ctx)
    term_cap, _term_chain, _, _ = run_cap_waterfall(TERM_CAP, 84, ctx)
    any_decline = declined or grade_declined
    reason = (decline_reason if declined else grade_decline_reason) or 0
    binder = binding_entry(amount_chain) or ""
    return amount_cap, binder, any_decline, reason, term_cap, grade_cap


evaluate_amount_and_grade_caps_step = step(
    evaluate_amount_and_grade_caps,
    outputs=("amount_cap_before_overlay", "amount_cap_binding_rule", "p09_declined", "p09_decline_reason",
              "term_cap", "worst_acceptable_grade"),
)


def amount_cap_overlay_step():
    """The cap overlay (10 §5.10's worked example, ADJ-0087): enters the chain as its own
    row, attributed to the adjustment set rather than to a register entry -- "your cap was
    R134 400, reduced to R120 750 by a policy overlay ... expiring 2027-12-31" is a
    different answer than "a rule bound it" (10 §5.10). May only reduce (`cap_adjustment`
    is tighten-only, `credit_core.adjustments._TIGHTEN_RULES`), so it can never do what
    CAP-0118's authority-bounded uplift does. Same register, resolved once (O-05/O-21).
    """
    return OVERLAY_REGISTER.apply_stack_step(
        "amount_cap", ADJUSTMENT_SET_ID, base_field="amount_cap_before_overlay",
        adjusted_output="amount_cap", unadjusted_output="amount_cap_unadjusted",
    )


def instalment_cap_chain(
    max_affordable_instalment: float, self_employed_haircut: bool = False, is_joint_application: bool = False,
    self_employed_pct: float = 0.90, joint_household_pct: float = 0.95,
) -> tuple[float, list[dict]]:
    """P09's second pass (O-10): `instalment_cap` is seeded from affordability, then
    narrowed by the two entries that act on the instalment -- run *after* P10, because
    they cannot exist before it. Plain-Python defaults here are fine: this function is
    only ever called directly, never wired as a decider step (same `list[dict]`-output
    reason as `evaluate_amount_and_grade_caps`'s docstring) -- see `evaluate_instalment_cap`
    below for the wired, scalar-only version.
    """
    value = max_affordable_instalment
    chain = [{"rule_id": "SEED-AFFORDABILITY", "applicable": True, "before": value, "after": value, "status": BOUND}]
    if self_employed_haircut:
        after = value * self_employed_pct
        chain.append({"rule_id": "CAP-INST-SE", "applicable": True, "before": value, "after": after, "status": BOUND})
        value = after
    else:
        chain.append({"rule_id": "CAP-INST-SE", "applicable": False, "before": value, "after": value,
                      "status": NOT_APPLICABLE})
    if is_joint_application:
        after = value * joint_household_pct
        chain.append({"rule_id": "CAP-INST-JOINT", "applicable": True, "before": value, "after": after,
                      "status": BOUND})
        value = after
    else:
        chain.append({"rule_id": "CAP-INST-JOINT", "applicable": False, "before": value, "after": value,
                      "status": NOT_APPLICABLE})
    return value, chain


def evaluate_instalment_cap(
    max_affordable_instalment: float, self_employed_haircut: bool = missing_as(False),
    is_joint_application: bool = missing_as(False),
) -> float:
    value, _chain = instalment_cap_chain(max_affordable_instalment, self_employed_haircut, is_joint_application)
    return value


evaluate_instalment_cap_step = step(evaluate_instalment_cap, output="instalment_cap")
