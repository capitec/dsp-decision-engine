"""Request handling for credit limit management.

Same gap 00 and 02's NOTES.md already document: `decider build`/`decider
serve`'s built-in warm-up (`decider.serving.handler._warm`) only knows
`bool`/`int`/`str`/`bytes` and falls back to `1.0` for `date` and `list`
inputs -- and this pipeline's `decision_date: date`, `cycle_balances:
list[float]`, `applicant1_bureau_accounts: list[dict]` (via project 02)
inputs are exactly that shape. Replaced at import time with a warm-up from
this project's own `sample_request.json`, exactly as 00/02 do.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

import decider.serving.handler as _handler
from decider.serving.handler import RequestHandler

_SAMPLE_REQUEST_PATH = Path(__file__).parent / "sample_request.json"

_DATE_FIELDS = ("decision_date", "bureau_as_of_date", "last_change_date", "salary_deposit_as_of_date")
_ACCOUNT_LIST_FIELDS = (
    "applicant1_bureau_accounts", "applicant1_internal_accounts",
    "applicant2_bureau_accounts", "applicant2_internal_accounts",
)


def _typed_sample_record() -> dict:
    record = json.loads(_SAMPLE_REQUEST_PATH.read_text())
    for field in _DATE_FIELDS:
        if record.get(field):
            record[field] = datetime.date.fromisoformat(record[field])
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
