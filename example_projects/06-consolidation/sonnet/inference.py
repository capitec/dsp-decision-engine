"""Request handling for consolidation and restructure.

`decider build`/`decider serve` warm every kernel with a synthetic record before
serving (`decider.serving.handler._warm`). That synthesiser only knows
`bool`/`int`/`str`/`bytes` and falls back to the float `1.0` for every other type --
including `list`, `dict` and `datetime.date`. This pipeline's top-level inputs
include `decision_date: date` and several ragged lists (`accounts`,
`client_nominated_settle`, ...), so the built-in warm-up crashes before any request
is ever served -- the same finding projects 00, 02 and 03's NOTES.md all document.
Replaces `_warm` at import time with one that warms using this project's own
`sample_request.json`, exactly like those three projects.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import decider.serving.handler as _handler
from decider.serving.handler import RequestHandler

_SAMPLE_REQUEST_PATH = Path(__file__).parent / "sample_request.json"
_DATE_FIELDS = ("decision_date", "bureau_as_of_date", "last_consolidation_date")


def _typed_sample_record() -> dict:
    record = json.loads(_SAMPLE_REQUEST_PATH.read_text())
    for field in _DATE_FIELDS:
        if record.get(field):
            record[field] = datetime.date.fromisoformat(record[field])
    for account in record.get("accounts", []):
        for field in ("opened_date", "quotation_expiry_date"):
            if account.get(field):
                account[field] = datetime.date.fromisoformat(account[field])
    return record


def _warm_with_sample_record(exe, params) -> None:
    record = _typed_sample_record()
    exe.score(record, params)
    import polars as pl
    exe.run(pl.DataFrame([record]), params)


_handler._warm = _warm_with_sample_record


class Handler(RequestHandler):
    pass
