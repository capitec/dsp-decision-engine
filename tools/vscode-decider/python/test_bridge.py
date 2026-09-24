import glob
import json
import os
import socket
import subprocess
import sys

import pytest

HERE = os.path.dirname(__file__)
LOAN = os.path.join(HERE, "..", "examples", "loan.py")
sys.path.insert(0, HERE)
from bridge import Bridge  # noqa: E402


def started(**kw):
    b = Bridge()
    b.start(LOAN, **kw)
    return b


def find(node, path):
    if node["path"] == path:
        return node
    for c in node.get("children", ()):
        if (hit := find(c, path)) is not None:
            return hit


def test_describe_reports_every_node_kind_with_its_source():
    d = Bridge().describe(LOAN)
    assert d["pipeline"] == "pipeline"
    ir = d["ir"]
    assert [c["path"] for c in ir["children"]] == ["join_bureau", "affordability", "banding", "term", "sizing", "risk_tree"]
    assert find(ir, "join_bureau")["callKind"] == "frame"
    loop = find(ir, "sizing/shrink_offer")
    assert loop["kind"] == "loop" and loop["carries"] == ["offer"]
    assert [c["path"] for c in loop["children"]] == ["sizing/shrink_offer/too_big", "sizing/shrink_offer/shrink"]
    tree = find(ir, "risk_tree")
    assert tree["callKind"] == "row" and tree["inputs"] == ["bureau_score", "ratio"]
    assert tree["line"] and tree["python"]["bodyLine"]  # a row node debugs into its Python reference
    assert find(ir, "term")["line"] and find(ir, "term/cap_by_income")["file"].endswith("loan.py")


def test_break_set_resume_changes_the_output_and_the_trail():
    b = started(breakpoints=["term/cap_by_income"])
    r = b.handle({"cmd": "resume"})
    assert r["current"] == {"path": "term/cap_by_income", "when": "before"}
    assert r["events"][-1]["kind"] == "paused"
    b.handle({"cmd": "set", "name": "term_cap", "value": 12.0})
    assert b.handle({"cmd": "resume"})["finished"]
    assert b.session.output()["term_cap"].to_list() == [12.0, 12.0]
    producers = [v["producer"] for v in b.column("term_cap")["versions"]]
    assert "override@term/cap_by_income" in producers


def test_step_out_stops_after_the_enclosing_node():
    b = started(breakpoints=["term/by_sector/is_private"])
    b.handle({"cmd": "resume"})
    r = b.handle({"cmd": "step_out"})
    assert r["current"] == {"path": "term/by_sector", "when": "after"}


def test_a_loop_is_stepped_iteration_by_iteration():
    b = started(breakpoints=["sizing/shrink_offer/shrink"])
    hits = 0
    while not b.handle({"cmd": "resume"})["finished"]:
        hits += 1
    assert hits == 5  # 150000 shrinks five times for client 1; client 2 never loops
    producers = {v["producer"] for v in b.column("offer")["versions"]}
    assert producers == {"sizing/offer", "sizing/shrink_offer", "sizing/shrink_offer/shrink"}


def test_a_row_node_reports_tree_visits_and_breaks_on_a_locator():
    b = started(breakpoints=["risk_tree#bureau_score < 680"])
    r = b.handle({"cmd": "resume"})
    assert r["current"] == {"path": "risk_tree", "when": "after"}
    visits = {e["origin"]["locator"]: e["rows"] for e in r["events"] if e["kind"] == "node_visited"}
    assert visits == {"root": 2, "bureau_score < 680": 1, "ratio >= 3": 1, "bureau_score >= 680": 1}


def test_state_can_focus_one_record():
    b = started()
    b.handle({"cmd": "resume"})
    cols = {c["name"]: c for c in b.state(row=1)["columns"]}
    assert cols["term_cap"]["value"] == 36.0
    assert cols["min_net_salary"]["value"] is None and cols["min_net_salary"]["nulls"] == 1
    assert cols["bureau_score"]["producer"] == "join_bureau"
    assert [v["values"] for v in b.column("offer", row=0)["versions"]][-1] == [49152.0]


def test_runtime_lineage_follows_the_arm_a_record_took():
    b = started()
    b.handle({"cmd": "resume"})
    private = b.handle({"cmd": "lineage", "name": "term_cap", "row": 0})
    assert private["producer"] == "term/by_sector" and private["via"] == "merge" and private["value"] == 48.0
    assert [i["producer"] for i in private["inputs"]] == ["term/by_sector/is_private", "term/by_sector/cap_private"]
    public = b.handle({"cmd": "lineage", "name": "term_cap", "row": 1})
    assert public["inputs"][1]["producer"] == "term/by_sector/cap_public"
    assert {"name": "requested_term", "producer": None, "value": 36.0, "inputs": []} in flatten(public)


def flatten(entry):
    yield entry
    for i in entry["inputs"]:
        yield from flatten(i)


def test_lineage_only_sees_what_has_run_so_far():
    b = started(breakpoints=["term/by_sector"])
    b.handle({"cmd": "resume"})
    l = b.handle({"cmd": "lineage", "name": "term_cap", "row": 0})
    assert l["producer"] == "term/cap_by_income"


def test_a_failing_step_is_reported_not_fatal():
    rows = [{**r, "sector_code": None} for r in __import__("loan").SAMPLE]
    b = started(data=rows)
    r = b.handle({"cmd": "resume"})
    assert "sector_code" in r["error"]
    assert r["events"][-1]["kind"] == "error"


def run_bridge(*extra, env=None):
    return subprocess.Popen([sys.executable, os.path.join(HERE, "bridge.py"), *extra], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, env=env)


def test_the_stdio_protocol_round_trips():
    p = run_bridge()

    def call(**req):
        p.stdin.write(json.dumps(req) + "\n")
        p.stdin.flush()
        return json.loads(p.stdout.readline())

    assert call(id=1, cmd="start", file=LOAN, breakpoints=["banding"])["ok"]
    assert call(id=2, cmd="resume")["result"]["current"]["path"] == "banding"
    assert call(id=3, cmd="bogus")["ok"] is False
    call(id=4, cmd="exit")
    assert p.wait(10) == 0


DEBUGPY_LIBS = sorted(glob.glob(os.path.expanduser("~/.vscode*/extensions/ms-python.debugpy-*/bundled/libs")))


@pytest.mark.skipif(not DEBUGPY_LIBS, reason="ms-python.debugpy extension not installed")
def test_debugpy_flag_opens_an_attach_port():
    p = run_bridge("--debugpy", "5689", env={**os.environ, "PYTHONPATH": DEBUGPY_LIBS[-1]})
    try:
        p.stdin.write('{"id": 1, "cmd": "describe", "file": %s}\n' % json.dumps(LOAN))
        p.stdin.flush()
        assert json.loads(p.stdout.readline())["ok"]  # listen() returned before serving
        socket.create_connection(("127.0.0.1", 5689), 2).close()
    finally:
        p.kill()


def test_describe_carries_the_params_schema_and_a_code_fingerprint():
    d = Bridge().describe(LOAN)
    assert d["params"]["term/cap_by_income"]["cap"]["default"] == 48.0
    assert len(find(d["ir"], "term/cap_by_income")["code"]) == 12


def test_trace_records_every_call_and_applies_params_and_overrides():
    base = Bridge().trace(LOAN)
    assert base["error"] is None
    assert base["steps"]["term/cap_by_income"]["term_cap"] == [48.0, 36.0]
    tuned = Bridge().trace(LOAN, params={"term": {"cap_by_income": {"cap": 24.0}}})
    assert tuned["steps"]["term/cap_by_income"]["term_cap"] == [24.0, 24.0]  # a null salary is filled with 0
    what_if = Bridge().trace(LOAN, overrides={"requested_amount": 10000.0}, row=0)
    assert [r["requested_amount"] for r in what_if["data"]] == [10000.0, 90000.0]
    assert what_if["output"]["offer"] == [10000.0, 90000.0]


def test_tree_path_rewalks_one_record():
    b = started()
    b.handle({"cmd": "resume"})
    assert b.handle({"cmd": "tree_path", "path": "risk_tree", "row": 0}) == {
        "path": "risk_tree", "row": 0, "visited": ["root", "bureau_score < 680", "ratio >= 3"], "result": [1]}
    assert b.handle({"cmd": "tree_path", "path": "risk_tree", "row": 1})["visited"] == ["root", "bureau_score >= 680"]


def test_scenarios_fork_from_the_pause_and_keep_earlier_overrides():
    b = started(breakpoints=["term/cap_by_income"])
    b.handle({"cmd": "resume"})
    b.handle({"cmd": "set", "name": "requested_amount", "value": 120000.0})
    r = b.handle({"cmd": "sweep", "scenarios": [
        {"label": "ceiling 10", "params": {"term": {"term_cap": {"ceiling": 10.0}}}},
        {"label": "cap 6", "params": {"term": {"cap_by_income": {"cap": 6.0}}}},
        {"label": "record 1 term 12", "overrides": {"term_cap": 12.0}, "row": 1},
    ]})
    assert r["at"] == {"path": "term/cap_by_income", "when": "before", "n": 1}
    base = r["baseline"]["output"]
    assert base["term_cap"] == [48.0, 36.0] and base["offer"][1] == 120000.0  # the earlier override replays
    ceiling, cap, record = (x["output"] for x in r["results"])
    assert ceiling["term_cap"] == [48.0, 36.0]  # term_cap already ran: a param upstream of the pause changes nothing
    assert cap["term_cap"] == [6.0, 6.0]
    assert record["term_cap"] == [48.0, 12.0]
    assert b.handle({"cmd": "resume"})["finished"]  # the original session is untouched


def test_scenarios_from_the_start_without_a_session():
    r = Bridge().sweep([{"label": "cap 24", "params": {"term": {"cap_by_income": {"cap": 24.0}}}}],
                       from_here=False, file=LOAN)
    assert r["at"] is None
    assert r["results"][0]["output"]["term_cap"] == [24.0, 24.0]


def test_debug_condition_matches_only_the_focused_record():
    b = started(breakpoints=["term/cap_by_income"])
    b.handle({"cmd": "resume"})
    cond = b.handle({"cmd": "debug_condition", "path": "term/cap_by_income", "row": 1})["condition"]
    assert cond == "term_cap == 36.0 and min_net_salary == 0.0"  # the null arrives filled
    assert eval(cond, {}, {"term_cap": 36.0, "min_net_salary": 0.0})
    assert not eval(cond, {}, {"term_cap": 60.0, "min_net_salary": 4000.0})


def test_skipping_a_step_mid_run_reruns_without_it():
    b = started(breakpoints=["term/cap_by_income"])
    b.handle({"cmd": "resume"})
    r = b.handle({"cmd": "skip", "path": "term/cap_by_income"})
    assert any(e["kind"] == "edited" and e["action"] == "delete" for e in r["events"])
    assert b.handle({"cmd": "resume"})["finished"]
    assert b.session.output()["term_cap"].to_list() == [54.0, 36.0]


def test_an_edited_step_is_swapped_in_mid_run(tmp_path):
    loan = tmp_path / "loan.py"
    loan.write_text(open(LOAN).read())
    b = Bridge()
    b.start(str(loan), breakpoints=["term/cap_by_income"])
    b.handle({"cmd": "resume"})
    loan.write_text(loan.read_text().replace("return min(term_cap, cap) if", "return min(term_cap, cap) - 1 if"))
    r = b.handle({"cmd": "reload_step", "path": "term/cap_by_income"})
    assert r["current"] == {"path": "term/cap_by_income", "when": "before"}
    assert r["formula"] == "min(term_cap, cap) - 1 if min_net_salary < 5000 else term_cap"
    assert [line[0] + line[1:].strip() for line in r["diff"]] == ["-return min(term_cap, cap) if min_net_salary < 5000 else term_cap",
                                                    "+return min(term_cap, cap) - 1 if min_net_salary < 5000 else term_cap"]
    assert b.handle({"cmd": "resume"})["finished"]
    assert b.session.output()["term_cap"].to_list() == [47.0, 35.0]  # record 1's missing salary fills as 0, so the edit hits it too


def test_a_swapped_step_can_be_put_back(tmp_path):
    loan = tmp_path / "loan.py"
    loan.write_text(open(LOAN).read())
    b = Bridge()
    b.start(str(loan), breakpoints=["term/cap_by_income"])
    b.handle({"cmd": "resume"})
    loan.write_text(loan.read_text().replace("return min(term_cap, cap) if", "return min(term_cap, cap) - 1 if"))
    b.handle({"cmd": "reload_step", "path": "term/cap_by_income"})
    b.handle({"cmd": "restore", "path": "term/cap_by_income"})
    assert b.handle({"cmd": "resume"})["finished"]
    assert b.session.output()["term_cap"].to_list() == [48.0, 36.0]
    assert b.edits == []


def test_a_skipped_step_can_be_put_back():
    b = started(breakpoints=["term/cap_by_income"])
    b.handle({"cmd": "resume"})
    b.handle({"cmd": "skip", "path": "term/cap_by_income"})
    b.handle({"cmd": "restore", "path": "term/cap_by_income"})
    # The flow re-runs with the step back in, so its breakpoint is met again on the way.
    assert b.handle({"cmd": "resume"})["current"] == {"path": "term/cap_by_income", "when": "before"}
    assert b.handle({"cmd": "resume"})["finished"]
    assert b.session.output()["term_cap"].to_list() == [48.0, 36.0]
    assert b.edits == []


def test_skipping_the_only_producer_is_refused_and_changes_nothing():
    b = started(breakpoints=["term/cap_by_income"])
    b.handle({"cmd": "resume"})
    with pytest.raises(Exception, match="nothing produces"):
        b.handle({"cmd": "skip", "path": "affordability/ratio"})
    assert b.handle({"cmd": "resume"})["finished"]
    assert b.session.output()["term_cap"].to_list() == [48.0, 36.0]


def test_edits_compare_against_the_flow_as_started():
    b = started(breakpoints=["term/cap_by_income"])
    b.handle({"cmd": "resume"})
    b.handle({"cmd": "skip", "path": "term/cap_by_income"})
    r = b.handle({"cmd": "compare_edits"})
    assert r["a"]["output"]["term_cap"] == [48.0, 36.0]
    assert r["b"]["output"]["term_cap"] == [54.0, 36.0]
    assert "term/cap_by_income" in r["a"]["steps"] and "term/cap_by_income" not in r["b"]["steps"]


def test_each_edit_compares_on_its_own(tmp_path):
    loan = tmp_path / "loan.py"
    loan.write_text(open(LOAN).read())
    b = Bridge()
    b.start(str(loan), breakpoints=["term/cap_by_income"])
    b.handle({"cmd": "resume"})
    b.handle({"cmd": "skip", "path": "term/cap_by_income"})
    loan.write_text(loan.read_text().replace("cap: float = param(54.0)", "cap: float = param(50.0)"))
    b.handle({"cmd": "reload_step", "path": "term/by_sector/cap_private"})
    only_skip = b.handle({"cmd": "compare_edits", "path": "term/cap_by_income"})["b"]["output"]["term_cap"]
    only_edit = b.handle({"cmd": "compare_edits", "path": "term/by_sector/cap_private"})["b"]["output"]["term_cap"]
    both = b.handle({"cmd": "compare_edits"})["b"]["output"]["term_cap"]
    assert (only_skip, only_edit, both) == ([54.0, 36.0], [48.0, 36.0], [50.0, 36.0])


def test_a_one_line_step_reports_its_formula():
    assert find(Bridge().describe(LOAN)["ir"], "term/term_cap")["formula"] == "min(requested_term, ceiling)"
