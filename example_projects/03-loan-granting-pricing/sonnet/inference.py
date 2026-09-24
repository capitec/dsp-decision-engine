"""Request handling for loan granting and pricing.

`decider build`/`decider serve` warm every kernel with a synthetic record
before serving (`decider.serving.handler._warm`). That synthesiser only
knows `bool`/`int`/`str`/`bytes` (`decider/serving/handler.py`'s `_DUMMY`
dict) and falls back to the float `1.0` for every other type -- including
`list`, `dict` and `datetime.date`. This pipeline's top-level inputs
include `decision_date: date` and several ragged lists (`bureau_accounts`,
`exclusion_list_hits`, ...), so the built-in warm-up crashes before any
request is ever served -- the same finding project 00's NOTES.md documents
(`TypeError: 'float' object is not iterable`), hit again here because every
downstream project inherits the same `decision_date`-is-mandatory contract
(00 §7.3). As there, this module replaces `_warm` at import time with one
that warms using this project's own `sample_request.json`.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import decider.serving.handler as _handler
from decider.serving.handler import RequestHandler

_SAMPLE_REQUEST_PATH = Path(__file__).parent / "sample_request.json"
_DATE_FIELDS = ("decision_date", "bureau_as_of_date")
_ACCOUNT_LIST_FIELDS = (
    "bureau_accounts", "internal_accounts", "applicant2_bureau_accounts", "applicant2_internal_accounts",
)


def _typed_sample_record() -> dict:
    record = json.loads(_SAMPLE_REQUEST_PATH.read_text())
    for field in _DATE_FIELDS:
        if record.get(field):
            record[field] = datetime.date.fromisoformat(record[field])
    for field in _ACCOUNT_LIST_FIELDS:
        for account in record.get(field, []):
            if account.get("opened_date"):
                account["opened_date"] = datetime.date.fromisoformat(account["opened_date"])
    return record


def _warm_with_sample_record(exe, params) -> None:
    record = _typed_sample_record()
    exe.score(record, params)
    import polars as pl
    exe.run(pl.DataFrame([record]), params)


_handler._warm = _warm_with_sample_record


class Handler(RequestHandler):
    pass
