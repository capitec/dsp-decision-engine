from __future__ import annotations

from decikit._tooling.bridge.protocol import Lineage, PROTOCOL_TS, render, Status

def test_protocol_ts_matches_the_reply_models():
    text = PROTOCOL_TS.read_text()
    assert render(text) == text, "protocol.ts is stale: run `uv run python -m decikit._tooling.bridge.protocol` to regenerate it"


def test_a_reply_carries_only_the_fields_it_set():
    assert Status(finished=True, current=None, events=[]).wire() == {"finished": True, "current": None, "events": []}
    assert Lineage(name="x", producer=None, value=None, inputs=[]).wire() == {"name": "x", "producer": None, "value": None, "inputs": []}
