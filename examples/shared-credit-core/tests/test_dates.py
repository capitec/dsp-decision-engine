from datetime import date

import pytest

from credit_core.dates import EffectiveDatedSet, EffectiveVersion

V1 = EffectiveVersion("v1", date(2025, 1, 1), date(2026, 1, 1))
V2 = EffectiveVersion("v2", date(2026, 1, 1))


def test_the_version_in_force_on_a_date_is_resolved():
    versions = EffectiveDatedSet("x", [V1, V2])
    assert versions.resolve(date(2025, 6, 1)).version_id == "v1"
    assert versions.resolve(date(2025, 12, 31)).version_id == "v1"
    assert versions.resolve(date(2026, 1, 1)).version_id == "v2"  # a version starts on its first day
    assert versions.resolve(date(2030, 1, 1)).version_id == "v2"


def test_versions_may_be_given_in_any_order():
    assert EffectiveDatedSet("x", [V2, V1]).resolve(date(2025, 6, 1)).version_id == "v1"


def test_a_date_before_the_first_version_cannot_be_resolved():
    with pytest.raises(LookupError, match="no version of 'x'"):
        EffectiveDatedSet("x", [V2]).resolve(date(2025, 1, 1))


def test_a_date_after_the_last_version_ended_cannot_be_resolved():
    with pytest.raises(LookupError):
        EffectiveDatedSet("x", [V1]).resolve(date(2026, 1, 1))


@pytest.mark.parametrize("second", [
    EffectiveVersion("gap", date(2026, 2, 1)),
    EffectiveVersion("overlap", date(2025, 12, 1)),
])
def test_gaps_and_overlaps_are_rejected(second):
    with pytest.raises(ValueError, match="gap or overlap"):
        EffectiveDatedSet("x", [V1, second])


def test_an_open_ended_version_cannot_be_followed_by_another():
    with pytest.raises(ValueError, match="gap or overlap"):
        EffectiveDatedSet("x", [EffectiveVersion("open", date(2025, 1, 1)), V2])


def test_the_resolver_step_writes_the_version_of_each_decision_date(score):
    step = EffectiveDatedSet("norms", [V1, V2]).resolver_step()
    assert score(step, {"decision_date": date(2025, 6, 1)})["norms_version"] == "v1"
    assert score(step, {"decision_date": date(2026, 6, 1)})["norms_version"] == "v2"


def test_the_resolver_step_can_read_another_date_and_write_another_name(score):
    step = EffectiveDatedSet("norms", [V1, V2]).resolver_step(date_field="knowledge_date", output="agreed_version")
    assert score(step, {"knowledge_date": date(2025, 6, 1)})["agreed_version"] == "v1"


def test_a_dotted_family_name_gives_a_usable_default_column(score):
    step = EffectiveDatedSet("expense_norms.statutory", [V2]).resolver_step()
    assert score(step, {"decision_date": date(2026, 6, 1)})["expense_norms_statutory_version"] == "v2"
