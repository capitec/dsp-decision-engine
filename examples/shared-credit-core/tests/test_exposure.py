from datetime import date

from credit_core.exposure import exposure_headroom, group_exposure_limit_step, group_total_exposure, group_total_exposure_step
from decikit import dag

FACILITIES = [
    {"balance": 100_000.0, "opened_date": date(2020, 1, 1)},
    {"balance": 50_000.0, "opened_date": date(2020, 1, 1), "closed_date": date(2024, 1, 1)},
    {"balance": 80_000.0, "opened_date": date(2027, 1, 1)},
    {"balance": 200_000.0, "opened_date": date(2020, 1, 1), "is_contingent": True, "conversion_factor": 0.5},
]


def test_only_facilities_open_on_the_date_count_and_contingent_ones_at_their_conversion_factor():
    assert group_total_exposure(FACILITIES, date(2026, 1, 1)) == 200_000.0  # 100 000 + 50% of 200 000
    assert group_total_exposure(FACILITIES, date(2023, 1, 1)) == 250_000.0  # the closed one still counts then
    assert group_total_exposure(FACILITIES, date(2027, 6, 1)) == 280_000.0  # and the new one now does


def test_a_facility_counts_on_its_opening_date_and_not_on_its_closing_date():
    facility = [{"balance": 10.0, "opened_date": date(2026, 1, 1), "closed_date": date(2026, 2, 1)}]
    assert group_total_exposure(facility, date(2026, 1, 1)) == 10.0
    assert group_total_exposure(facility, date(2026, 2, 1)) == 0.0


def test_headroom_is_the_limit_less_the_exposure_and_never_negative():
    assert exposure_headroom(200_000.0, 5_000_000.0) == 4_800_000.0
    assert exposure_headroom(6_000_000.0, 5_000_000.0) == 0.0


def test_the_steps_publish_total_limit_and_headroom(score):
    unit = dag(group_total_exposure_step, group_exposure_limit_step, exposure_headroom, name="exposure").emit(
        "group_total_exposure", "group_exposure_limit")
    out = score(unit, {"related_facilities": FACILITIES, "exposure_as_of_date": date(2026, 1, 1)},
                {"exposure": {"group_exposure_limit": {"limit": 1_000_000.0}}})
    assert (out["group_total_exposure"], out["group_exposure_limit"], out["exposure_headroom"]) == (200_000.0, 1_000_000.0, 800_000.0)
