"""Request handling for fraud_interdiction.

Same `decider.serving.handler._warm` workaround as 00's `inference.py`
(see its docstring for the full repro): the built-in warm-up synthesises
a dummy record typed only for `bool`/`int`/`str`/`bytes`, defaulting to
the float `1.0` for anything else -- including this pipeline's
`decision_date: date` and `client_segments: list[str]` -- and crashes
before the first real request:

    TypeError: '<=' not supported between instances of 'datetime.date' and 'float'

This replaces `_warm` at import time with one that warms using this
project's own `sample_request.json`, a real record, instead of the
synthetic one.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import decider.serving.handler as _handler
from decider.serving.handler import RequestHandler

_SAMPLE_REQUEST_PATH = Path(__file__).parent / "sample_request.json"


def _typed_sample_record() -> dict:
    record = json.loads(_SAMPLE_REQUEST_PATH.read_text())
    record["decision_date"] = datetime.date.fromisoformat(record["decision_date"])
    return record


def _warm_with_sample_record(exe, params) -> None:
    record = _typed_sample_record()
    exe.score(record, params)
    import polars as pl
    exe.run(pl.DataFrame([record]), params)


_handler._warm = _warm_with_sample_record


class Handler(RequestHandler):
    pass
