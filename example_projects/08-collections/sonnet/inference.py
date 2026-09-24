"""Request handling for collections treatment assignment.

Staging warms up with `sample_request.json`.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

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


class Handler(RequestHandler):
    pass
