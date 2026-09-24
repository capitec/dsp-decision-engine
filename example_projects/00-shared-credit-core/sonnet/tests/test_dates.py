"""`core.dates`: effective-dated resolution, standalone (spec 00 §7.5, §6.19)."""
from datetime import date

import pytest
from decider import Engine

from credit_core.dates import EffectiveDatedSet, EffectiveVersion


def test_resolves_the_version_in_force_on_a_date():
    norms = EffectiveDatedSet("x", [
        EffectiveVersion("v1", date(2025, 1, 1), date(2026, 1, 1)),
        EffectiveVersion("v2", date(2026, 1, 1), None),
    ])
    assert norms.resolve(date(2025, 6, 1)).version_id == "v1"
    assert norms.resolve(date(2026, 1, 1)).version_id == "v2"
    assert norms.resolve(date(2030, 1, 1)).version_id == "v2"


def test_a_decision_before_any_version_starts_is_unresolvable():
    norms = EffectiveDatedSet("x", [EffectiveVersion("v1", date(2026, 1, 1), None)])
    with pytest.raises(LookupError):
        norms.resolve(date(2025, 1, 1))


def test_gaps_and_overlaps_are_rejected_at_construction():
    with pytest.raises(ValueError, match="gap or overlap"):
        EffectiveDatedSet("x", [
            EffectiveVersion("v1", date(2025, 1, 1), date(2026, 1, 1)),
            EffectiveVersion("v2", date(2026, 2, 1), None),  # gap
        ])


def test_change_scenario_1_old_decisions_keep_the_old_table_forever():
    """SCOPE.md change scenario 1: a gazetted change must not touch decisions before it."""
    norms = EffectiveDatedSet("expense_norms.statutory", [
        EffectiveVersion("old", date(2025, 1, 1), date(2026, 10, 1)),
        EffectiveVersion("new", date(2026, 10, 1), None),
    ])
    before, on, after = date(2026, 9, 30), date(2026, 10, 1), date(2027, 1, 1)
    assert norms.resolve(before).version_id == "old"
    assert norms.resolve(on).version_id == "new"
    assert norms.resolve(after).version_id == "new"
    # A 2029 replay of the 2026-09-30 decision must still land on "old" (09 §5.15 item 4).
    assert norms.resolve(before).version_id == "old"


def test_resolver_step_writes_the_version_id_per_record():
    fam = EffectiveDatedSet("fam", [
        EffectiveVersion("a", date(2020, 1, 1), date(2025, 1, 1)),
        EffectiveVersion("b", date(2025, 1, 1), None),
    ])
    resolver = fam.resolver_step()
    exe = Engine().bind(resolver, mode="interpreted")
    assert exe.score({"decision_date": date(2024, 6, 1)})["fam_version"] == "a"
    assert exe.score({"decision_date": date(2025, 6, 1)})["fam_version"] == "b"


def test_resolver_step_can_read_a_different_date_field():
    fam = EffectiveDatedSet("fam", [EffectiveVersion("only", date(2020, 1, 1), None)])
    resolver = fam.resolver_step(date_field="knowledge_date", output="fam_version")
    exe = Engine().bind(resolver, mode="interpreted")
    assert exe.score({"knowledge_date": date(2024, 1, 1)})["fam_version"] == "only"
