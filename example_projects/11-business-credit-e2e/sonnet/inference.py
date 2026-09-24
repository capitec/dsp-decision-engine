"""Request handling for the business credit end-to-end (EP-1) pipeline.

Same gap 00, 02, 05 and 07's NOTES.md all document: `decider build`/`decider
serve`'s built-in warm-up (`decider.serving.handler._warm`) only knows
`bool`/`int`/`str`/`bytes` and falls back to `1.0` for `date` and `list`
inputs -- and this pipeline's `decision_date`/`knowledge_date: date`,
`existing_accounts: list[dict]` and the nested `entities: list[struct]`
(inherited unmodified from project 05) are exactly that shape. Replaced at
import time with a warm-up from this project's own `sample_request.json`,
exactly as 00/02/05/07 do.
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
    record["knowledge_date"] = datetime.date.fromisoformat(record["knowledge_date"])
    for account in record.get("existing_accounts", []):
        if account.get("opened_date"):
            account["opened_date"] = datetime.date.fromisoformat(account["opened_date"])
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
