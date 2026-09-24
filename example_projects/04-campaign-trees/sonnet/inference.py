"""Request handling for the campaign-23 pipeline.

`decider build`/`decider serve` warm every kernel with a synthetic record before serving
(`decider.serving.handler._warm`). That synthesiser only knows `bool`/`int`/`str`/`bytes`
(`decider/serving/handler.py`'s `_DUMMY` dict) and falls back to the float `1.0` for every
other type -- including `datetime.date`. This pipeline's `cycle_date: date` input (mandatory
per 09 §5.15 item 4, "no reliance on today") means the built-in warm-up crashes before any
request is ever served -- exactly the finding 00 and 03's own NOTES.md already document
(00 NOTES.md "Framework friction" #1; 03's confirms it again), now a third time
independently. This module replaces `_warm` with one that warms from this project's own
`sample_request.json`, the same workaround 00/03 use, for the same reason.
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
    record["cycle_date"] = datetime.date.fromisoformat(record["cycle_date"])
    return record


def _warm_with_sample_record(exe, params) -> None:
    record = _typed_sample_record()
    exe.score(record, params)
    import polars as pl
    exe.run(pl.DataFrame([record]), params)


_handler._warm = _warm_with_sample_record


class Handler(RequestHandler):
    pass
