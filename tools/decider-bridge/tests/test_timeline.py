import os
import sys

HERE = os.path.dirname(__file__)
EXAMPLES = os.path.join(HERE, "..", "..", "vscode-decider", "examples")
LOAN = os.path.join(EXAMPLES, "loan.py")
sys.path.insert(0, os.path.join(HERE, ".."))
from decider_bridge.bridge import Bridge  # noqa: E402


def finished(**kw):
    b = Bridge()
    b.start(LOAN, **kw)
    b.handle({"cmd": "resume"})
    return b


def test_a_records_history_lists_each_change_with_the_step_and_iteration():
    h = finished().handle({"cmd": "changes", "name": "offer", "row": 0})
    assert [(c["path"].split("/")[-1], c["iteration"], c["before"], c["value"]) for c in h["changes"]] == [
        ("offer", None, None, 150000.0),
        ("shrink", 1, 150000.0, 120000.0),
        ("shrink", 2, 120000.0, 96000.0),
        ("shrink", 3, 96000.0, 76800.0),
        ("shrink", 4, 76800.0, 61440.0),
        ("shrink", 5, 61440.0, 49152.0),
    ]
    assert not h["input"]


def test_a_step_that_writes_the_same_value_is_kept_not_changed():
    h = finished().handle({"cmd": "changes", "name": "term_cap", "row": 1})
    assert [(c["path"], c.get("kept", False)) for c in h["changes"]] == [
        ("term/term_cap", False), ("term/cap_by_income", True), ("term/by_sector/cap_public", True)]  # the caps leave 36
    batch = finished().handle({"cmd": "changes", "name": "term_cap"})
    assert [c["path"] for c in batch["changes"]] == ["term/term_cap", "term/cap_by_income"]  # 60 → 48 for record 1


def test_a_forced_branch_shows_as_its_own_change():
    b = Bridge()
    b.start(LOAN, forces=[{"path": "term/by_sector", "arm": 1, "row": 0}])
    b.handle({"cmd": "resume"})
    h = b.handle({"cmd": "changes", "name": "is_private", "row": 0})
    assert [(c["path"], c["value"]) for c in h["changes"]] == [("term/by_sector/is_private", True), ("force@term/by_sector", False)]


def test_the_status_says_which_steps_ran_since_a_rewind():
    b = finished()
    assert "sizing/shrink_offer/shrink" in b.status()["ran"]
    r = b.handle({"cmd": "rewind", "path": "sizing"})
    assert "term/cap_by_income" in r["ran"] and "sizing/shrink_offer/shrink" not in r["ran"]


def test_a_step_that_passes_an_input_on_names_it_and_where_it_came_from():
    b = finished()
    first = b.handle({"cmd": "changes", "name": "offer", "row": 0})["changes"][0]
    assert (first["via"], first["viaPath"]) == ("requested_amount", None)  # offer returns the input as it is
    shrunk = b.handle({"cmd": "changes", "name": "offer", "row": 0})["changes"][1]
    assert "via" not in shrunk


def test_a_record_that_left_a_loop_is_not_said_to_be_kept_by_later_iterations():
    h = finished().handle({"cmd": "changes", "name": "offer", "row": 1})
    assert [c["path"] for c in h["changes"]] == ["sizing/offer"]  # record 2 never loops


def test_the_batch_history_counts_the_records_each_change_touched():
    h = finished().handle({"cmd": "changes", "name": "offer"})
    assert h["changes"][0]["rows"] == 2 and h["changes"][1]["rows"] == 1


def test_an_input_starts_from_its_value_and_an_override_is_a_change():
    b = Bridge()
    b.start(LOAN, breakpoints=["term/cap_by_income"])
    b.handle({"cmd": "resume"})
    b.handle({"cmd": "set", "name": "requested_amount", "value": 1000.0})
    h = b.handle({"cmd": "changes", "name": "requested_amount", "row": 0})
    assert h["input"] and h["initial"] == 150000.0
    assert h["changes"][0]["path"] == "override@term/cap_by_income" and h["changes"][0]["value"] == 1000.0


def test_going_back_to_a_step_by_path_pauses_after_its_last_write_for_the_record():
    b = finished()
    r = b.handle({"cmd": "go_to", "path": "affordability/disposable_income", "row": 0})
    assert r["current"] == {"path": "affordability/disposable_income", "when": "after"}


def test_a_value_not_set_yet_has_no_breakdown_rather_than_an_error():
    b = finished()
    b.handle({"cmd": "go_to", "path": "affordability/disposable_income", "row": 0})
    assert b.handle({"cmd": "lineage", "name": "offer", "row": 0})["unset"]


def test_the_state_mid_iteration_shows_what_the_loop_body_just_wrote():
    b = Bridge()
    b.start(LOAN, breakpoints=["sizing/shrink_offer/shrink"])
    b.handle({"cmd": "resume"})
    b.handle({"cmd": "step"})  # just after the first shrink; the carry still holds 150000
    offer = next(c for c in b.state(row=0)["columns"] if c["name"] == "offer")
    assert offer["value"] == 120000.0 and offer["preview"] == [120000.0, 90000.0]


def test_going_back_to_a_change_pauses_just_after_the_step_that_made_it():
    b = finished()
    third = b.handle({"cmd": "changes", "name": "offer", "row": 0})["changes"][3]
    r = b.handle({"cmd": "go_to", "change": third["change"]})
    assert r["current"] == {"path": "sizing/shrink_offer/shrink", "when": "after", "iteration": 3}
    assert b.session.current.iteration == 3
    assert b.session.value("offer@sizing/shrink_offer/shrink").to_list()[0] == 76800.0
    after = b.handle({"cmd": "changes", "name": "offer", "row": 0})["changes"]
    assert [c["value"] for c in after if not c.get("pending")] == [150000.0, 120000.0, 96000.0, 76800.0]
    assert [c["value"] for c in after if c.get("pending")] == [61440.0, 49152.0]  # undone, to happen again
    assert b.handle({"cmd": "resume"})["finished"]
    assert len(b.handle({"cmd": "changes", "name": "offer", "row": 0})["changes"]) == 6
