import pytest

from collections_treatment import state


@pytest.mark.parametrize("days, bucket", [(0, 0), (1, 1), (14, 1), (15, 2), (29, 2), (30, 3), (59, 3), (60, 4),
                                          (89, 4), (90, 5), (119, 5), (120, 6), (179, 6), (180, 7), (364, 7), (365, 8)])
def test_days_past_due_fall_into_the_documented_bucket(days, bucket):
    assert state.arrears_bucket_code(days) == bucket


def test_an_account_under_a_performing_arrangement_stays_in_the_bucket_it_entered_in():
    held = state.arrears_bucket_code(200, under_performing_arrangement=True, bucket_at_arrangement_entry=3)
    assert held == 3
    assert state.arrears_bucket_code(200, under_performing_arrangement=False, bucket_at_arrangement_entry=3) == 7


def test_an_account_cured_twice_is_floored_at_bucket_three(decide):
    assert state.arrears_bucket_code(10, times_cured_12m=2) == 3
    assert state.arrears_bucket_code(10, times_cured_12m=1) == 1
    assert decide(days_past_due=10, times_cured_12m=2)["arrears_bucket_code"] == 3


def test_the_arrangement_hold_comes_from_the_request_not_from_a_param(decide):
    out = decide(days_past_due=200, under_performing_arrangement=True, bucket_at_arrangement_entry=3)
    assert out["arrears_bucket_code"] == 3
    assert decide(days_past_due=200)["arrears_bucket_code"] == 7


def test_balance_bands_follow_the_documented_edges(decide):
    bands = {balance: decide(outstanding_balance=balance, arrears_amount=100.0)["balance_band_code"]
             for balance in (2_499.0, 2_500.0, 9_999.0, 10_000.0, 49_999.0, 50_000.0, 249_999.0, 250_000.0)}
    assert list(bands.values()) == [1, 2, 2, 3, 4, 5, 6, 7]


def test_the_balance_band_names_the_cell_it_read(decide):
    assert decide()["balance_band_cell_id"] == "balance_band@collections-bands-2026.09#3"


@pytest.mark.parametrize("rate, band", [(0.8, 1), (0.5, 1), (0.35, 2), (0.2, 2), (0.1, 3), (0.05, 3), (0.0, 4)])
def test_a_higher_contact_rate_is_a_more_responsive_contact_band(decide, rate, band):
    assert decide(right_party_contact_rate_90d=rate)["contact_band_code"] == band


def test_an_account_never_reached_is_unreachable(decide):
    assert decide(right_party_contact_rate_90d=None)["contact_band_code"] == 4
