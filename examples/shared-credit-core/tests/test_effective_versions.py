from datetime import date

import pytest
from decider import ConfigurableStep, flow
from decider.lab import diff

from credit_core.dates import EffectiveVersions, Version

V1 = Version(version_id="v1", effective_from=date(2025, 1, 1), effective_to=date(2026, 1, 1))
V2 = Version(version_id="v2", effective_from=date(2026, 1, 1))


def versions(*vs, **fields):
    return EffectiveVersions(family="x", versions=vs, **fields)


def same(a, b):
    return diff.structure(flow(a, name="t"), flow(b, name="t")).identical


def test_the_version_in_force_on_a_date_is_resolved():
    resolved = versions(V1, V2).resolve
    assert resolved(date(2025, 6, 1)).version_id == "v1"
    assert resolved(date(2025, 12, 31)).version_id == "v1"
    assert resolved(date(2026, 1, 1)).version_id == "v2"  # a version starts on its first day
    assert resolved(date(2030, 1, 1)).version_id == "v2"


def test_versions_may_be_given_in_any_order():
    assert versions(V2, V1).resolve(date(2025, 6, 1)).version_id == "v1"


def test_a_date_before_the_first_version_cannot_be_resolved():
    with pytest.raises(LookupError, match="no version of 'x'"):
        versions(V2).resolve(date(2025, 1, 1))


def test_a_date_after_the_last_version_ended_cannot_be_resolved():
    with pytest.raises(LookupError):
        versions(V1).resolve(date(2026, 1, 1))


@pytest.mark.parametrize("second", [
    Version(version_id="gap", effective_from=date(2026, 2, 1)),
    Version(version_id="overlap", effective_from=date(2025, 12, 1)),
])
def test_gaps_and_overlaps_are_rejected_when_the_document_loads(second):
    with pytest.raises(ValueError, match="gap or overlap"):
        versions(V1, second)


def test_an_open_ended_version_cannot_be_followed_by_another():
    with pytest.raises(ValueError, match="gap or overlap"):
        versions(Version(version_id="open", effective_from=date(2025, 1, 1)), V2)


def test_the_step_writes_the_version_of_each_decision_date(score):
    step = EffectiveVersions(family="norms", versions=[V1, V2])
    assert (step.name, step.output) == ("resolve_norms_version", "norms_version")
    assert score(step, {"decision_date": date(2025, 6, 1)})["norms_version"] == "v1"
    assert score(step, {"decision_date": date(2026, 6, 1)})["norms_version"] == "v2"


def test_the_step_can_read_another_date_and_write_another_name(score):
    step = EffectiveVersions(family="norms", versions=[V1, V2], date_field="knowledge_date", output="agreed_version")
    assert score(step, {"knowledge_date": date(2025, 6, 1)})["agreed_version"] == "v1"


def test_a_dotted_family_name_gives_a_usable_default_column(score):
    step = EffectiveVersions(family="expense_norms.statutory", versions=[V2])
    assert score(step, {"decision_date": date(2026, 6, 1)})["expense_norms_statutory_version"] == "v2"


def test_the_document_loads_with_dates_as_json_strings():
    doc = {"type": "effective_versions", "family": "x", "versions": [
        {"version_id": "v1", "effective_from": "2025-01-01", "effective_to": "2026-01-01"},
        {"version_id": "v2", "effective_from": "2026-01-01"}]}
    loaded = ConfigurableStep.load(doc)
    assert loaded == versions(V1, V2) and ConfigurableStep.load(loaded.model_dump_json()) == loaded


def test_the_same_versions_are_unchanged_in_the_diff():
    assert same(versions(V1, V2), ConfigurableStep.load(versions(V1, V2).model_dump_json()))


def test_moving_the_cut_over_a_month_shows_in_the_diff():
    moved = versions(V1.model_copy(update={"effective_to": date(2026, 2, 1)}),
                     V2.model_copy(update={"effective_from": date(2026, 2, 1)}))
    assert not same(versions(V1, V2), moved)


def test_renaming_a_version_shows_in_the_diff():
    assert not same(versions(V1, V2), versions(V1, V2.model_copy(update={"version_id": "v2b"})))


def test_the_step_compiles_with_its_python_boundary_declared():
    fused = flow(versions(V1, V2), name="t").compile("fused")
    assert all(reason.startswith("@allow_fallback") for reason in fused.fallbacks().values())
