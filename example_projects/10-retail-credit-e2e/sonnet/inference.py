"""Request handling for the retail_credit entry-point-1 demo.

`decider build`/`decider serve` warm every kernel with a synthetic record
before serving (`decider.serving.handler._warm`). That synthesiser only
knows `bool`/`int`/`str`/`bytes` (`decider/serving/handler.py`'s `_DUMMY`
dict) and falls back to the float `1.0` for every other type -- including
`list`, `dict` and `datetime.date`. This pipeline's top-level inputs
include `decision_date: date`, `bureau_accounts`/`internal_accounts:
list[dict]` and `variable_pay_history: list[float]`, so the built-in
warm-up crashes before any request is ever served -- exactly the defect
project 00's NOTES.md "Framework friction" §4.1 documents (this project
hits the identical failure, confirming 00's prediction that "every other
project in this set" would).

There is no `Handler.*_fn` override that reaches `_warm` -- `stage()`
calls the module-level function directly, with no extension point -- so
this module replaces it, at import time, with one that warms using this
project's own `sample_request.json` instead of a synthesised record. See
project 00's NOTES.md for the fuller writeup and repro; this file is the
same workaround, reused verbatim because the underlying framework gap is
identical.
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
    for account in record.get("bureau_accounts", []) + record.get("internal_accounts", []):
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
