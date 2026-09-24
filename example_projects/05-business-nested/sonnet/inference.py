"""Request handling for the business-nested-entities pipeline.

`decider build`/`decider serve` warm every kernel with a synthetic record before
serving (`decider.serving.handler._warm`). That synthesiser only knows
`bool`/`int`/`str`/`bytes` (`decider/serving/handler.py`'s `_DUMMY` dict) and
falls back to the float `1.0` for every other type -- including `list`, `dict`
and `datetime.date`. This pipeline's top-level inputs include `decision_date: date`
and the nested `entities: list[struct]` collection (spec 05's own dominant
difficulty), so the built-in warm-up crashes before any request is ever served --
exactly the gap project 00's NOTES.md documents ("any decider project with a
`date`, `list` or `dict` top-level input cannot `decider build` unchanged today").
This module replaces `_warm`, at import time, with one that warms using this
project's own `sample_request.json` instead of the synthetic record -- the same
workaround 00 and 02 both use.
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
    for entity in record.get("entities", []):
        for event in entity.get("adverse_events", []):
            event["event_date"] = datetime.date.fromisoformat(event["event_date"])
    return record


def _warm_with_sample_record(exe, params) -> None:
    record = _typed_sample_record()
    exe.score(record, params)
    import polars as pl
    exe.run(pl.DataFrame([record]), params)


_handler._warm = _warm_with_sample_record


class Handler(RequestHandler):
    pass
