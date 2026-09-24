"""suppressions.py: Stage 2 (spec 04 §5.2) -- multiple suppressions recorded, absolute vs
measurement-relevant classification (§5.2 requirement 3).

Bound through `Engine`, not called as plain Python: `suppression_evaluation`'s defaults are
`missing_as(...)` sentinels (decider's own declaration objects), which only resolve to a
real value when the framework fills them in -- calling the bare function directly would
compare against the sentinel object itself, not `False`. 00's own `tests/test_thin_capabilities.py`
follows the identical pattern for every capability with a `missing_as`/`param` default.
"""
from __future__ import annotations

from decider import Engine

from campaign_trees import suppressions


def _run(**overrides):
    exe = Engine().bind(suppressions.suppression_evaluation_step, mode="interpreted")
    return exe.score(overrides)


def test_no_suppressions_when_nothing_applies():
    out = _run()
    assert out["suppression_codes"] == []
    assert out["suppressed_absolute"] is False


def test_a_client_suppressed_by_four_rules_shows_four():
    """§5.2 requirement 1: "A client suppressed by four different rules must show four.\""""
    out = _run(marketing_opt_out=True, consent_permitted=False, in_cooling_off=True, contacts_30d=5)
    assert set(out["suppression_codes"]) == {"S07", "S08", "S21", "S30"}
    assert out["suppressed_absolute"] is False  # all four are measurement-relevant


def test_absolute_suppression_sets_the_flag():
    out = _run(is_deceased=True)
    assert out["suppression_codes"] == ["S01"]
    assert out["suppressed_absolute"] is True


def test_absolute_and_measurement_relevant_together_is_still_absolute():
    out = _run(is_deceased=True, marketing_opt_out=True)
    assert set(out["suppression_codes"]) == {"S01", "S07"}
    assert out["suppressed_absolute"] is True


def test_registry_classifies_every_code_used():
    for code in ["S01", "S02", "S07", "S08", "S11", "S14", "S21", "S23", "S30"]:
        assert code in suppressions.REGISTRY
        _desc, klass = suppressions.REGISTRY[code]
        assert klass in ("absolute", "measurement_relevant")
