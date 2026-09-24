"""§9.2: an account under debt review, outside contact hours, with a channel consent
withdrawal, must show every one of those suspensions -- not the first one found."""
from datetime import date

from decider import Engine

from collections_treatment import suspensions


def test_every_suspension_individually_attributable_no_short_circuit():
    step = suspensions.evaluate_suspensions_step
    record = {
        "decision_date": date(2026, 9, 24),
        "debt_review_stage_code": 1, "debt_review_case_number": "DR-9001",
        "debt_review_default_date": date(2020, 1, 1),
        "administration_order_active": False, "insolvency_active": False, "deceased": False,
        "complaint_open": False, "complaint_logged_date": date(2020, 1, 1),
        "ombud_referral_open": False, "dispute_raised": False,
        "hardship_arrangement_performing": False, "notice_delivered_date": date(2020, 1, 1),
        "litigation_in_progress": False, "prescription_date": date(2030, 1, 1),
        "outside_permitted_hours": True,
        "consent_withdrawn_channels": [1],
        "frequency_cap_reached": False, "frequency_cap_window_reset_date": date(2020, 1, 1),
        "written_communication_only": False, "promise_date": date(2020, 1, 1),
        "no_valid_contact_point": False,
    }
    out = Engine().bind(step).score(record)

    codes = out["suspension_codes"]
    assert set(codes) == {101, 115, 116}, codes
    # A client under debt review (101), outside contact hours (115) AND with a channel
    # consent withdrawal (116) all present at once -- the exact §9.2 scenario ("a client
    # under debt review who was also outside contact hours and had withdrawn SMS
    # consent", spec 08 §2).
    assert len(out["suspension_scopes"]) == len(codes) == len(out["suspension_sources"])


def test_hard_block_suspends_every_treatment():
    permitted = suspensions.permitted_treatment_codes(
        suspension_codes=[106], suspension_blocks_all=[True],
    )
    assert permitted == []


def test_soft_suspension_does_not_block_all_treatments():
    permitted = suspensions.permitted_treatment_codes(
        suspension_codes=[117], suspension_blocks_all=[False],
    )
    assert permitted != []


def test_prescription_is_a_hard_block_even_though_not_flagged_blocks_all_in_the_table():
    codes, _scopes, blocks, _sources, _expiries = suspensions.evaluate_suspensions(
        date(2026, 9, 24), prescription_date=date(2026, 9, 1),
    )
    assert 114 in codes
    assert suspensions.permitted_treatment_codes(codes, blocks) == []


def test_pre_prescription_flag_fires_within_60_days_only():
    assert suspensions.pre_prescription_flag(date(2026, 9, 24), date(2026, 11, 20)) is True
    assert suspensions.pre_prescription_flag(date(2026, 9, 24), date(2027, 6, 1)) is False
    assert suspensions.pre_prescription_flag(date(2026, 9, 24), None) is False
