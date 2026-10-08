from datetime import date

import pytest

from credit_core.bureau import normalise_bureau

DECISION = date(2026, 9, 24)
RESPONSE = {
    "accounts": [{"status_code": 0}, {"status_code": 2}],
    "enquiries": [{"enquiry_date": date(2026, 8, 1)}, {"enquiry_date": date(2026, 2, 1)}, {"enquiry_date": date(2025, 8, 1)}],
    "public_records": [{}],
    "as_of_date": date(2026, 9, 1),
}


def bureau(score, response, **params):
    return score(normalise_bureau, {"bureau_response": response, "decision_date": DECISION},
                 {"normalise_bureau_response": params} if params else None)


def test_the_response_is_summarised(score):
    out = bureau(score, RESPONSE)
    assert out["bureau_account_count"] == 2
    assert out["bureau_worst_status"] == 2
    assert out["bureau_public_record_count"] == 1
    assert out["bureau_as_of_date"] == date(2026, 9, 1)
    assert (out["bureau_is_stale"], out["bureau_data_quality"]) == (False, "ok")


def test_enquiries_are_counted_by_calendar_month_windows(score):
    out = bureau(score, RESPONSE)
    # August is 1 month back, February 7 months back, and August last year 13 months back.
    assert (out["bureau_enquiry_count_3m"], out["bureau_enquiry_count_12m"]) == (1, 2)


def test_a_view_older_than_the_tolerance_is_stale(score):
    old = {**RESPONSE, "as_of_date": date(2026, 6, 1)}  # 115 days
    assert bureau(score, old)["bureau_data_quality"] == "stale"
    assert bureau(score, old)["bureau_is_stale"] is True
    assert bureau(score, old, staleness_days=120)["bureau_is_stale"] is False


def test_an_empty_response_is_its_own_quality_verdict(score):
    out = bureau(score, {})
    assert (out["bureau_data_quality"], out["bureau_is_stale"], out["bureau_account_count"]) == ("no_record", True, 0)
    assert out["bureau_worst_status"] == 0


def test_a_response_without_a_date_is_stale(score):
    out = bureau(score, {"accounts": [{"status_code": 1}]})
    assert (out["bureau_is_stale"], out["bureau_data_quality"], out["bureau_as_of_date"]) == (True, "stale", None)
