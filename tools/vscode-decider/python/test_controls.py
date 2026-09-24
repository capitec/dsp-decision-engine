import os
import sys

import pytest

HERE = os.path.dirname(__file__)
LOAN = os.path.join(HERE, "..", "examples", "loan.py")
sys.path.insert(0, HERE)
from bridge import Bridge  # noqa: E402

UNCAPPED = {"term": {"cap_by_income": {"cap": 100.0}}}  # so the sector caps decide term_cap


def started(**kw):
    b = Bridge()
    b.start(LOAN, **kw)
    return b


def test_a_forced_arm_routes_every_record_down_it():
    r = Bridge().trace(LOAN, params=UNCAPPED, forces=[{"path": "term/by_sector", "arm": 1}])
    assert r["output"]["term_cap"] == [60.0, 36.0]  # record 1 is private but gets the public cap
    assert "term/by_sector/cap_private" not in r["steps"]


def test_a_forced_arm_can_target_one_record():
    r = Bridge().trace(LOAN, params=UNCAPPED, forces=[{"path": "term/by_sector", "arm": 0, "row": 1}])
    assert r["output"]["term_cap"] == [54.0, 36.0]
    assert r["steps"]["term/by_sector/cap_private"]["term_cap"] == [54.0, 36.0]
    assert "term/by_sector/cap_public" not in r["steps"]


def test_a_loop_forced_to_n_iterations_runs_exactly_n():
    two = Bridge().trace(LOAN, forces=[{"path": "sizing/shrink_offer", "iterations": 2, "row": 0}])
    assert two["output"]["offer"] == [96000.0, 90000.0]  # record 2 loops as it would: not at all
    ten = Bridge().trace(LOAN, forces=[{"path": "sizing/shrink_offer", "iterations": 10}])
    assert ten["output"]["offer"] == pytest.approx([150000 * 0.8 ** 10, 90000 * 0.8 ** 10])


def test_forces_that_do_not_fit_are_refused():
    with pytest.raises(ValueError):
        started(forces=[{"path": "term/by_sector", "arm": 2}])
    with pytest.raises(KeyError):
        started(forces=[{"path": "term", "arm": 0}])


def test_a_sweep_compares_iteration_counts_from_the_start():
    r = Bridge().sweep([{"label": str(n), "forces": [{"path": "sizing/shrink_offer", "iterations": n}]} for n in (5, 10)],
                       from_here=False, file=LOAN)
    five, ten = (x["output"]["offer"][0] for x in r["results"])
    assert (five, ten) == pytest.approx((150000 * 0.8 ** 5, 150000 * 0.8 ** 10))


def test_a_sweep_from_the_pause_compares_arms():
    b = started(breakpoints=["term"], params=UNCAPPED)
    b.handle({"cmd": "resume"})
    r = b.handle({"cmd": "sweep", "scenarios": [{"label": f"arm {a}", "forces": [{"path": "term/by_sector", "arm": a}]}
                                                 for a in (0, 1)]})
    assert r["baseline"]["output"]["term_cap"] == [54.0, 36.0]
    assert [x["output"]["term_cap"] for x in r["results"]] == [[54.0, 36.0], [60.0, 36.0]]  # record 2 asks for 36, under both caps


def test_a_forced_arm_set_mid_run_applies_from_there():
    b = started(breakpoints=["term"], params=UNCAPPED)
    b.handle({"cmd": "resume"})
    b.handle({"cmd": "set_controls", "forces": [{"path": "term/by_sector", "arm": 1}]})
    assert b.handle({"cmd": "resume"})["finished"]
    assert b.session.output()["term_cap"].to_list() == [60.0, 36.0]


def test_a_rerun_goes_back_forces_and_returns_to_the_pause():
    b = started(breakpoints=["sizing"], params=UNCAPPED)
    b.handle({"cmd": "resume"})
    b.handle({"cmd": "set_controls", "forces": [{"path": "term/by_sector", "arm": 1}]})
    r = b.handle({"cmd": "rerun", "path": "term/by_sector/is_private", "back": "sizing"})
    assert r["current"] == {"path": "sizing", "when": "before"}
    assert b.session.value("term_cap").to_list() == [60.0, 36.0]


def test_a_force_set_while_paused_just_after_the_condition_still_routes():
    b = started(breakpoints=["term/by_sector/is_private"], params=UNCAPPED)
    b.handle({"cmd": "resume"})
    assert b.handle({"cmd": "step"})["current"] == {"path": "term/by_sector/is_private", "when": "after"}
    b.handle({"cmd": "set_controls", "forces": [{"path": "term/by_sector", "arm": 1}]})
    assert b.handle({"cmd": "resume"})["finished"]
    assert b.session.output()["term_cap"].to_list() == [60.0, 36.0]


def test_a_breakpoint_on_an_iteration_pauses_before_it():
    b = started(watches=[{"path": "sizing/shrink_offer", "iteration": 3}])
    r = b.handle({"cmd": "resume"})
    assert r["current"] == {"path": "sizing/shrink_offer/too_big", "when": "before", "iteration": 3}
    assert b.session.current.iteration == 3 and r["hit"]["text"] == "iteration 3 of shrink_offer"
    assert b.session.value("offer").to_list()[0] == 96000.0


def test_a_value_breakpoint_pauses_where_a_record_first_meets_it():
    b = started(watches=[{"name": "offer", "op": "<", "value": 70000}])
    r = b.handle({"cmd": "resume"})
    assert r["current"]["path"] == "sizing/shrink_offer/shrink" and r["hit"]["rows"] == [0]
    assert r["hit"]["values"] == [61440.0]
    assert b.session.current.iteration == 4  # the shrink that wrote 61440, not the one after
    assert b.handle({"cmd": "resume"})["finished"]  # still under 70000 after the next shrink: no second pause


def test_a_value_breakpoint_can_be_scoped_to_steps_and_one_record():
    b = started(watches=[{"name": "term_cap", "op": "==", "value": 36.0, "scope": ["term/by_sector"]}])
    r = b.handle({"cmd": "resume"})
    assert r["current"]["path"] == "term/by_sector/cap_public" and r["hit"]["rows"] == [1]
    b = started(watches=[{"name": "offer", "op": "<=", "value": 90000, "row": 0}])
    assert b.handle({"cmd": "resume"})["hit"]["values"] == [76800.0]


def test_a_value_breakpoint_that_never_matches_lets_the_run_finish():
    b = started()
    b.handle({"cmd": "set_controls", "watches": [{"name": "offer", "op": ">", "value": 10 ** 6}]})
    assert b.handle({"cmd": "resume"})["finished"]
