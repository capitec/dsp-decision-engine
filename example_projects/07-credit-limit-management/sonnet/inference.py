"""Request handling for credit limit management.

Staging warms up with `sample_request.json`.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

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


class Handler(RequestHandler):
    pass
