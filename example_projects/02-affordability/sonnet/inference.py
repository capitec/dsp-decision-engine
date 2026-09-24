"""Request handling for the affordability assessment.

`decider build`/`decider serve` warm every kernel with a synthetic record
before serving (`decider.serving.handler._warm`); that synthesiser only
knows `bool`/`int`/`str`/`bytes` and falls back to the float `1.0` for
everything else -- including this pipeline's `decision_date: date`,
`applicant1_bureau_accounts: list[dict]` and `applicant1_declared_expenses:
dict[str, float]` inputs (spec 02 §4.1 makes several inputs ragged lists and
dicts by nature). 00 NOTES.md "Framework friction" 4.1 documents the same
gap and the same fix: there is no `Handler.*_fn` override for this, so this
module replaces `_warm` at import time with one that warms from this
project's own `sample_request.json` instead of a synthesised record.
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
    if record.get("bureau_as_of_date"):
        record["bureau_as_of_date"] = datetime.date.fromisoformat(record["bureau_as_of_date"])
    for key in ("applicant1_bureau_accounts", "applicant1_internal_accounts",
                "applicant2_bureau_accounts", "applicant2_internal_accounts"):
        for account in record.get(key, []):
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
