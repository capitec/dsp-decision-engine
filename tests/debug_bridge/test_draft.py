import os

from decider.debug_bridge.bridge import Bridge

HERE = os.path.dirname(__file__)
LOAN = os.path.join(HERE, "..", "..", "tools", "vscode-decider", "examples", "loan.py")


def started(**kw):
    b = Bridge()
    b.start(LOAN, **kw)
    return b


def test_start_applies_input_overrides_for_one_record():
    b = started(overrides={"requested_amount": 10000.0}, row=0)
    b.handle({"cmd": "resume"})
    assert b.session.frame["requested_amount"].to_list() == [10000.0, 90000.0]
    assert b.session.output()["offer"].to_list() == [10000.0, 90000.0]


def test_draft_lists_converted_overrides_and_dropped_edits():
    b = started(breakpoints=["term/cap_by_income"])
    b.handle({"cmd": "resume"})
    b.handle({"cmd": "set", "name": "term_cap", "value": 12.0})
    b.handle({"cmd": "skip", "path": "term/cap_by_income"})
    d = b.handle({"cmd": "draft"})
    assert {"target": "term_cap@term/cap_by_income", "value": 12.0, "source": "set"} in d["converted"]
    assert {"kind": "code_edit", "path": "term/cap_by_income", "detail": "step skipped"} in d["dropped"]
    assert any(x["kind"] == "console_mutation" for x in d["dropped"])


def test_draft_reports_forces_as_declared_overrides():
    b = started()
    b.handle({"cmd": "set_controls", "forces": [{"path": "term/by_sector", "arm": 1}], "watches": []})
    d = b.handle({"cmd": "draft"})
    assert {"target": "force@term/by_sector", "value": 1, "source": "force", "row": None} in d["converted"]


def test_status_reports_trace_as_off():
    b = started()
    assert b.status()["trace"] == {"available": False, "reason": "trace capture is off; showing live state"}
