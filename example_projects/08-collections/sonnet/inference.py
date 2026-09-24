"""Request handling for collections treatment assignment.

`decider build`/`decider serve` warm every kernel with a synthetic record
before serving (`decider.serving.handler._warm`); that synthesiser only
knows `bool`/`int`/`str`/`bytes` and falls back to the float `1.0` for
everything else -- including this pipeline's `decision_date: date`,
several `date | None` inputs, and `consent_withdrawn_channels: list[int]`.
00 NOTES.md "Framework friction" 4.1 documents the same gap on project 00
and the same fix, reused here unmodified: there is no `Handler.*_fn`
override for this, so this module replaces `_warm` at import time with
one that warms from this project's own `sample_request.json`.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import decider.serving.handler as _handler
from decider.serving.handler import RequestHandler

_SAMPLE_REQUEST_PATH = Path(__file__).parent / "sample_request.json"

_DATE_FIELDS = (
    "decision_date", "debt_review_default_date", "complaint_logged_date",
    "notice_delivered_date", "prescription_date", "frequency_cap_window_reset_date",
    "promise_date", "bureau_as_of_date",
)

# 02's own accounts carry `opened_date` -- reused here unmodified since this project's
# `arrangements.py` feeds 02's evidence/capacity units directly (see that module's
# docstring on why a bare `import pipeline` can't reuse 02's own helper for this).
_ACCOUNT_LIST_FIELDS = (
    "applicant1_bureau_accounts", "applicant1_internal_accounts",
    "applicant2_bureau_accounts", "applicant2_internal_accounts",
)


def _typed_sample_record() -> dict:
    record = json.loads(_SAMPLE_REQUEST_PATH.read_text())
    for key in _DATE_FIELDS:
        if record.get(key):
            record[key] = datetime.date.fromisoformat(record[key])
    for key in _ACCOUNT_LIST_FIELDS:
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
