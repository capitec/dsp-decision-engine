import os

import polars as pl

import decider_jupyter
from decider import flow, param, step
from decider_jupyter.serving import serve

LOAN = os.path.join(os.path.dirname(__file__), "..", "..", "vscode-decider", "examples", "loan.py")


class Comm:
    def __init__(self):
        self.sent = []

    def on_msg(self, fn):
        self.handler = fn

    def send(self, data):
        self.sent.append(data)

    def ask(self, **req):
        self.handler({"content": {"data": {"id": len(self.sent), **req}}})
        reply = self.sent[-1]
        assert reply["id"] == len(self.sent) - 1
        return reply


def ok(reply):
    assert reply["ok"], reply.get("error")
    return reply["result"]


def test_a_flow_file_is_described_and_stepped_over_the_comm():
    c = serve(Comm())
    assert ok(c.ask(cmd="describe", file=LOAN))["pipeline"] == "pipeline"
    ok(c.ask(cmd="start", file=LOAN, breakpoints=["affordability"]))
    assert ok(c.ask(cmd="resume"))["current"] == {"path": "affordability", "when": "before"}
    names = [col["name"] for col in ok(c.ask(cmd="state", row=0))["columns"]]
    assert "bureau_score" in names


def test_a_fresh_request_leaves_the_debug_session_where_it_was():
    c = serve(Comm())
    ok(c.ask(cmd="start", file=LOAN, breakpoints=["affordability"]))
    ok(c.ask(cmd="resume"))
    traced = ok(c.ask(cmd="trace", file=LOAN, params={"term": {"cap_by_income": {"cap": 24.0}}}, fresh=True))
    assert traced["output"]["term_cap"] == [24.0, 24.0]
    assert ok(c.ask(cmd="step"))["current"] == {"path": "affordability/disposable_income", "when": "before"}


def test_errors_come_back_as_the_reply_not_a_dead_comm():
    c = serve(Comm())
    reply = c.ask(cmd="describe", file="no_such_flow.py")
    assert not reply["ok"] and "no_such_flow" in reply["error"]
    assert ok(c.ask(cmd="cwd")) == os.getcwd()


def total(a: float, b: float) -> float:
    return a + b


@step(output="total")
def scale(total: float, k: float = param(2.0)) -> float:
    return total * k


@step(output="total", name="scale")
def add_k_fifties(total: float, k: float = param(2.0)) -> float:
    return total + 50 * k


def test_debug_runs_a_notebook_pipeline_on_its_frame_and_reloads_a_redefined_step(monkeypatch):
    ns = {"flow_a": flow(total, scale)}
    comms = []
    monkeypatch.setattr(decider_jupyter, "_user_ns", lambda: ns)
    monkeypatch.setattr(decider_jupyter, "notebook_comm", lambda name: comms.append((name, Comm())) or comms[-1][1])
    decider_jupyter.debug(ns["flow_a"], pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}), {"scale": {"k": 10.0}})
    name, c = comms[0]
    assert name == "flow_a"
    d = ok(c.ask(cmd="describe", pipeline=name))
    assert [n["path"] for n in d["ir"]["children"]] == ["total", "scale"]
    ok(c.ask(cmd="start", pipeline=name, breakpoints=["scale"]))
    assert ok(c.ask(cmd="resume"))["current"] == {"path": "scale", "when": "before"}

    # The cell that defines the flow ran again, with another `scale`.
    ns["flow_a"] = flow(total, add_k_fifties)
    assert ok(c.ask(cmd="reload_step", path="scale"))["current"] == {"path": "scale", "when": "before"}
    ok(c.ask(cmd="resume"))
    assert ok(c.ask(cmd="state"))["columns"][-1]["preview"] == [504.0, 506.0]
