from datetime import date

from limit_mgmt.exclusions import exclusion_codes, is_excluded
from limit_mgmt.vocab import Exclusion

CLEAN = dict(decision_date=date(2026, 9, 24), consent_automatic_increase=True, months_on_book=24.0)


def test_a_clean_account_has_no_exclusions():
    assert exclusion_codes(**CLEAN) == [] and is_excluded([]) is False


def test_every_exclusion_that_applies_is_recorded_not_just_the_first():
    codes = exclusion_codes(**CLEAN | {"worst_arrears_months_now": 1, "debt_review_active": True, "consent_withdrawn": True})
    assert codes == [Exclusion.X01_ARREARS_NOW, Exclusion.X03_DEBT_REVIEW, Exclusion.X12_NO_CONSENT]
    assert is_excluded(codes) is True


def test_no_consent_record_excludes_the_account_as_much_as_a_withdrawn_one():
    assert exclusion_codes(**CLEAN | {"consent_automatic_increase": False}) == [Exclusion.X12_NO_CONSENT]


def test_the_cooling_off_window_runs_from_the_last_limit_change():
    inside = exclusion_codes(**CLEAN | {"last_change_date": date(2026, 4, 1)})  # 176 days
    edge = exclusion_codes(**CLEAN | {"last_change_date": date(2026, 3, 28)})  # 180 days
    assert inside == [Exclusion.X09_COOLING_OFF] and edge == []


def test_an_account_under_six_months_old_is_too_young():
    assert exclusion_codes(**CLEAN | {"months_on_book": 5.0}) == [Exclusion.X14_TOO_YOUNG]


def test_the_collections_feed_excludes_an_account_in_treatment():
    assert exclusion_codes(**CLEAN | {"treatment_suspension_active": True}) == [Exclusion.X16_TREATMENT_SUSPENSION]
