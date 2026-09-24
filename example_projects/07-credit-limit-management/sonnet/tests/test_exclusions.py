import datetime

from limit_mgmt.exclusions import exclusion_codes
from limit_mgmt.vocab import Exclusion

# `missing_as(False)`/`missing_as(True)` cannot fall back to a plain Python default
# when a step function is called directly (outside `decider`'s own engine): `bool` can't
# be subclassed, so `decider` has no `bool_MissingAs` the way it has `int_MissingAs`/
# `float_MissingAs`/`list_MissingAs` (which *do* behave as their real value in plain
# Python -- confirmed with `type(missing_as(0))` -> `int_MissingAs`, a real `int`
# subclass). For `bool` it falls back to a generic sentinel object that is truthy
# regardless of the declared default:
#
#   >>> from decider import missing_as
#   >>> bool(missing_as(False))
#   True
#
# so every boolean input is passed explicitly below rather than omitted -- see
# NOTES.md "Framework friction" for the full writeup; this is the #1 finding.
_CLEAN = dict(
    decision_date=datetime.date(2026, 9, 24),
    worst_arrears_months_now=0, worst_arrears_months_6=0,
    debt_review_active=False, deceased_marker=False, fraud_marker=False,
    consent_automatic_increase=True, consent_withdrawn=False,
    months_on_book=24, treatment_suspension_active=False,
)


def test_complete_attribution_not_first_match_only():
    """§5.2: "Every exclusion that applied, not merely the first one found." An account
    in arrears now, under debt review, and with no consent must show all three."""
    codes = exclusion_codes(**{
        **_CLEAN, "worst_arrears_months_now": 1, "debt_review_active": True,
        "consent_automatic_increase": False,
    })
    assert set(codes) == {Exclusion.X01_ARREARS_NOW, Exclusion.X03_DEBT_REVIEW, Exclusion.X12_NO_CONSENT}


def test_clean_account_has_no_exclusions():
    assert exclusion_codes(**_CLEAN) == []


def test_x12_no_consent_is_not_the_same_as_declined_offer():
    """§5.2: "12.4% of the book has no such agreement on file... not offers-that-were-
    declined." No consent record at all still fires X12."""
    codes = exclusion_codes(**{**_CLEAN, "consent_automatic_increase": False})
    assert Exclusion.X12_NO_CONSENT in codes


def test_cooling_off_window():
    codes_within = exclusion_codes(**{
        **_CLEAN, "last_change_date": datetime.date(2026, 8, 1),  # 54 days ago, inside 180
    })
    assert Exclusion.X09_COOLING_OFF in codes_within

    codes_outside = exclusion_codes(**{**_CLEAN, "last_change_date": datetime.date(2025, 1, 1)})
    assert Exclusion.X09_COOLING_OFF not in codes_outside


def test_x16_treatment_suspension_from_the_08_stub():
    """DEPS.md: project 08's treatment-state feed is stubbed as `treatment_suspension_active`."""
    codes = exclusion_codes(**{**_CLEAN, "treatment_suspension_active": True})
    assert Exclusion.X16_TREATMENT_SUSPENSION in codes
