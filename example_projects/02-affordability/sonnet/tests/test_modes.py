"""The four assessment modes (spec 02 §5.8): one arithmetic path, four evidence selections."""
from decider import Engine

from assessment import modes


def _tier(mode_code: int) -> int:
    exe = Engine().bind(modes.minimum_income_tier, mode="interpreted")
    return exe.score({"assessment_mode_code": mode_code})["minimum_income_tier"]


def test_new_application_requires_the_strongest_evidence_of_the_four_modes():
    """§5.8: new application (tier 4 typical) is stricter than a limit increase (tier 3,
    internal deposits dominant) and much stricter than an arrangement (tier 6, "often
    unverifiable")."""
    assert _tier(modes.NEW_APPLICATION) < _tier(modes.ARRANGEMENT)


def test_limit_increase_accepts_weaker_evidence_than_a_new_application():
    assert _tier(modes.LIMIT_INCREASE) <= _tier(modes.NEW_APPLICATION)


def test_arrangement_accepts_the_weakest_evidence_of_the_four():
    for mode in (modes.NEW_APPLICATION, modes.LIMIT_INCREASE, modes.SCENARIO):
        assert _tier(modes.ARRANGEMENT) >= _tier(mode)


def test_every_mode_is_covered():
    for mode in (modes.NEW_APPLICATION, modes.LIMIT_INCREASE, modes.ARRANGEMENT, modes.SCENARIO):
        assert _tier(mode) in range(1, 7)
