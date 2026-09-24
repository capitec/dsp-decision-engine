"""P09's attribution requirement (spec 10 §5.10 item 1-4): final value, binder, full
chain, evaluated-and-did-not-bind vs. not-applicable.
"""
from retail_credit.cap_waterfall import (
    AMOUNT_CAP, BOUND, EVALUATED_NOT_BOUND, NOT_APPLICABLE, RAISED, binding_entry, run_cap_waterfall,
)

_BASE_CTX = dict(
    risk_grade=6, internal_tenure_months=67.0, months_employed=67.0, is_existing_client=True,
    worst_arrears_months=0.0, enquiry_velocity_90d=4, channel_code=3, channel_open=True,
    on_concentration_watchlist=False, group_limit=260_000.0, group_exposure=148_000.0, campaign_id="",
)


def test_final_value_and_binder_match_the_spec_worked_example():
    """Reproduces 10 §5.10's own worked `amount_cap` chain for Client V (minus the overlay,
    which this project applies as a separate step, see NOTES.md)."""
    value, chain, declined, reason = run_cap_waterfall(AMOUNT_CAP, 500_000.0, _BASE_CTX)
    assert declined is False
    assert value == 112_000.0  # CAP-0104 (220k) then CAP-0301 (112k) bind, matching the spec
    assert binding_entry(chain) == "CAP-0301"


def test_every_entry_is_evaluated_and_recorded_even_when_not_applicable():
    _, chain, _, _ = run_cap_waterfall(AMOUNT_CAP, 500_000.0, _BASE_CTX)
    statuses = {c["rule_id"]: c["status"] for c in chain}
    assert statuses["CAP-0158"] == NOT_APPLICABLE  # existing client: not applicable
    assert statuses["CAP-0011"] == EVALUATED_NOT_BOUND  # seeds the product maximum; nothing to narrow from here
    assert statuses["CAP-0212"] == EVALUATED_NOT_BOUND  # 4 enquiries: evaluated, below the 7 threshold
    assert statuses["CAP-0301"] == BOUND  # group exposure headroom actually narrows it


def test_uplift_never_lowers_the_ceiling_and_is_bounded():
    ctx = {**_BASE_CTX, "campaign_id": "7712"}
    value, chain, _, _ = run_cap_waterfall(AMOUNT_CAP, 500_000.0, ctx)
    uplift = next(c for c in chain if c["rule_id"] == "CAP-0118")
    assert uplift["status"] == RAISED
    assert uplift["after"] >= uplift["before"]
    assert uplift["after"] <= uplift["before"] * 1.20 + 1e-6


def test_arrears_tightens_both_amount_and_grade_ceilings():
    ctx = {**_BASE_CTX, "worst_arrears_months": 3.0}
    amount, _, _, _ = run_cap_waterfall(AMOUNT_CAP, 500_000.0, ctx)
    from retail_credit.cap_waterfall import GRADE_CAP
    grade_cap, _, _, _ = run_cap_waterfall(GRADE_CAP, 12, ctx)
    assert amount <= 35_000.0
    assert grade_cap == 9


def test_channel_closed_declines_outright():
    ctx = {**_BASE_CTX, "channel_open": False}
    value, chain, declined, reason = run_cap_waterfall(AMOUNT_CAP, 500_000.0, ctx)
    assert declined is True
    assert reason == 2011  # R_CHANNEL_CLOSED
