"""Request handling for the governance harness's servable pipeline.

Every top-level input here (`flow_code`, `decision_id`) is a plain `str`, so
this project should be clear of 00/01/03's own documented `decider build`
warm-up bug (`_warm`'s synthesiser only mistypes `date`/`list`/`dict`
inputs -- see 00's NOTES.md "Framework friction" 4.1). It still needs its
own override, for a different reason: the synthetic record's dummy string
is `""` for every `str` input, and `""` is not a known `flow_code` (nor a
`decision_id` this harness has evidence for), so `run_replay("", "", ...)`
raises `KeyError` during warm-up rather than mistyping anything. Warming
with this project's own `sample_request.json` -- a real, previously
captured decision -- avoids that the same way 00/01/03's overrides avoid
their own warm-up failure, even though the underlying defect is not theirs.
"""
from __future__ import annotations

import json
from pathlib import Path

import decider.serving.handler as _handler
from decider.serving.handler import RequestHandler

_SAMPLE_REQUEST_PATH = Path(__file__).parent / "sample_request.json"


def _warm_with_sample_record(exe, params) -> None:
    record = json.loads(_SAMPLE_REQUEST_PATH.read_text())
    exe.score(record, params)
    import polars as pl
    exe.run(pl.DataFrame([record]), params)


_handler._warm = _warm_with_sample_record


class Handler(RequestHandler):
    pass
