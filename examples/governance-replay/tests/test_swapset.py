from governance.flows import BUSINESS, FRAUD
from governance.overlays import switched_off
from governance.swapset import attribute, swap_set

from conftest import requests_of


def refer_from_grade(grade):
    def edit(config):
        config["params"]["business_credit"]["decision"]["outcome"]["business_outcome"]["refer_cutoff_grade"] = grade
    return edit


def test_the_same_version_twice_moves_no_one(evidence):
    population = requests_of(evidence, BUSINESS)
    now = BUSINESS.replayer()
    s = swap_set(BUSINESS, population, now, now)
    assert (s.population, s.moved, s.reasons_only, s.uncomparable) == (5, (), (), {})


def test_a_tighter_refer_cutoff_moves_the_approvals_it_catches_and_nobody_else(evidence, tmp_path):
    population = requests_of(evidence, BUSINESS)
    s = swap_set(BUSINESS, population, BUSINESS.replayer(), BUSINESS.candidate(refer_from_grade(7), tmp_path))
    assert s.moved == (("biz-2026-0900123-v1", 1, 3), ("biz-2026-0900123-v2", 1, 3))
    assert s.reasons_appeared == {5603: 4} and s.reasons_disappeared == {} and s.uncomparable == {}


def test_a_record_that_cannot_be_scored_is_reported_not_fatal(evidence):
    population = requests_of(evidence, BUSINESS) + [{"decision_id": "broken", "decision_date": "not a date"}]
    now = BUSINESS.replayer()
    s = swap_set(BUSINESS, population, now, now)
    assert s.population == 6 and list(s.uncomparable) == ["broken"]


def test_n_changes_give_n_swap_sets_each_measured_against_the_run_before_it(evidence, tmp_path):
    population = requests_of(evidence, BUSINESS)
    base = BUSINESS.replayer()
    params = base.module_fn().params
    stack_off = BUSINESS.replayer(params=switched_off(params))
    tighter = BUSINESS.candidate(refer_from_grade(6), tmp_path)
    tighter.params = switched_off(tighter.module_fn().params)  # the second change on top of the first
    first, second = attribute(BUSINESS, population, [("base", base), ("overlays off", stack_off), ("refer from 6", tighter)])
    assert (first.label, second.label) == ("overlays off", "refer from 6")
    assert first.moved == ()
    assert [m[0] for m in second.moved] == ["biz-2026-0900123-v1", "biz-2026-0900123-v2"]
