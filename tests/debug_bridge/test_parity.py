"""The bridge drives a debug session, so the cross-mode parity contract holds for the flows it debugs.

`decider.testing.assert_equivalent` is the parity contract: `run()`, `score()`
and a resumed session give the same frame in every mode, exactly (NaN equals
NaN, nothing else is approximate). A bridge session runs in interpreted mode,
so a flow debugged through the bridge must produce the same numeric results a
compiled run does.
"""
import os

from decider.testing import assert_equivalent

from decider.debug_bridge.bridge import Bridge

FLOW = """from decider import flow, param


def net(a: float, b: float) -> float:
    return a - b


def scaled(net: float, factor: float = param(2.0, ge=0)) -> float:
    return net * factor


pipeline = flow(net, scaled, name="f")
SAMPLE = [{"a": 10.0, "b": 3.0}, {"a": 20.0, "b": 1.0}]
"""


def test_a_flow_debugged_through_the_bridge_matches_compiled_runs_exactly(tmp_path):
    f = tmp_path / "flow.py"
    f.write_text(FLOW)
    b = Bridge()
    b.start(str(f))
    out = assert_equivalent(b.original, b.session.frame, b.doc)
    assert out["scaled"].to_list() == [14.0, 38.0]
