import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(__file__)
LOAN = os.path.join(HERE, "..", "examples", "loan.py")
sys.path.insert(0, HERE)
import decider_stub as dc  # noqa: E402
from bridge import Bridge  # noqa: E402


def load_loan():
    return Bridge().describe(LOAN)


def test_describe_finds_the_pipeline_and_its_ir():
    d = load_loan()
    assert [p["name"] for p in d["pipelines"]] == ["pipeline"]
    assert d["pipelines"][0]["line"] > 0
    paths = [c["path"] for c in d["ir"]["children"]]
    assert paths == ["affordability", "banding", "term"]
    by_sector = d["ir"]["children"][2]["children"][2]
    assert by_sector["kind"] == "branch"
    assert [c["path"] for c in by_sector["children"]] == [
        "term/by_sector/is_private", "term/by_sector/cap_private", "term/by_sector/cap_public"]
    assert "term_cap" in d["columns"]


def test_dag_orders_by_dependencies():
    b = Bridge()
    b.describe(LOAN)
    aff = b.ir.children_[0]
    assert [n.origin.path for n in aff.children_] == [
        "affordability/disposable_income", "affordability/ratio", "affordability/affordable"]


def test_break_set_resume_changes_the_output():
    b = Bridge()
    r = b.start(LOAN, breakpoints=["term/cap_by_income"])
    r = b.resume()
    assert r["current"]["path"] == "term/cap_by_income"
    assert r["events"][-1] == {"event": "Paused", "path": "term/cap_by_income", "reason": "breakpoint"}
    b.session.set("term_cap", 12.0)
    r = b.resume()
    assert r["finished"]
    assert b.session.state.columns["term_cap"] == [12.0, 12.0]
    producers = [p for p, _ in b.session.state.versions["term_cap"]]
    assert producers == ["term/term_cap", "override@term/cap_by_income", "term/cap_by_income",
                         "term/by_sector/cap_private", "term/by_sector/cap_public"]


def test_step_steps_over_a_branch_and_step_into_enters_it():
    b = Bridge()
    b.start(LOAN, breakpoints=["term/by_sector"])
    b.resume()
    b.session.step()
    assert b.session.finished
    b.start(LOAN, breakpoints=["term/by_sector"])
    b.resume()
    b.session.step_into()
    assert b.session.current.path == "term/by_sector/is_private"
    b.session.step()
    assert b.session.current.path == "term/by_sector/cap_private"


def test_rewind_reruns_from_a_node_with_the_current_state():
    b = Bridge()
    b.start(LOAN)
    b.resume()
    b.session.set("requested_term", 10.0)
    b.session.rewind("term/term_cap")
    assert b.session.current.path == "term/term_cap"
    b.resume()
    assert b.session.state.columns["term_cap"] == [10.0, 10.0]


def test_lineage_walks_back_to_inputs():
    b = Bridge()
    b.describe(LOAN)
    l = b.lineage("term_cap", "term/by_sector/cap_private")
    assert l["producer"] == "term/cap_by_income"
    inner = l["inputs"][0]
    assert inner["producer"] == "term/term_cap"
    assert inner["inputs"][0] == {"name": "requested_term", "producer": None, "inputs": []}


def test_the_stdio_protocol_round_trips():
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "bridge.py")], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True)

    def call(**req):
        p.stdin.write(json.dumps(req) + "\n")
        p.stdin.flush()
        return json.loads(p.stdout.readline())

    assert call(id=1, cmd="describe", file=LOAN)["ok"]
    r = call(id=2, cmd="start", file=LOAN, breakpoints=["banding"])
    assert r["ok"] and r["result"]["current"] is None
    r = call(id=3, cmd="resume")
    assert r["result"]["current"]["path"] == "banding"
    assert call(id=4, cmd="state")["result"]["columns"][0]["name"] == "affordable"
    assert call(id=5, cmd="bogus")["ok"] is False
    call(id=6, cmd="exit")
    assert p.wait(5) == 0


DEBUGPY_LIBS = sorted(__import__("glob").glob(os.path.expanduser("~/.vscode*/extensions/ms-python.debugpy-*/bundled/libs")))


@pytest.mark.skipif(not DEBUGPY_LIBS, reason="ms-python.debugpy extension not installed")
def test_debugpy_flag_opens_an_attach_port():
    import socket
    env = {**os.environ, "PYTHONPATH": DEBUGPY_LIBS[-1]}
    p = subprocess.Popen([sys.executable, os.path.join(HERE, "bridge.py"), "--debugpy", "5689"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, text=True)
    try:
        p.stdin.write('{"id": 1, "cmd": "describe", "file": %s}\n' % json.dumps(LOAN))
        p.stdin.flush()
        assert json.loads(p.stdout.readline())["ok"]  # listen() returned before serving
        socket.create_connection(("127.0.0.1", 5689), 2).close()
    finally:
        p.kill()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
