import datetime
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_DATE_FIELDS = ("decision_date", "bureau_as_of_date", "last_change_date", "salary_deposit_as_of_date")
_ACCOUNT_LIST_FIELDS = (
    "applicant1_bureau_accounts", "applicant1_internal_accounts",
    "applicant2_bureau_accounts", "applicant2_internal_accounts",
)


def _typed(record: dict) -> dict:
    record = dict(record)
    for field in _DATE_FIELDS:
        if record.get(field):
            record[field] = datetime.date.fromisoformat(record[field])
    for key in _ACCOUNT_LIST_FIELDS:
        record[key] = [dict(a) for a in record.get(key, [])]
        for a in record[key]:
            if a.get("opened_date"):
                a["opened_date"] = datetime.date.fromisoformat(a["opened_date"])
    return record


@pytest.fixture(scope="module")
def sample_record() -> dict:
    path = Path(__file__).resolve().parents[1] / "sample_request.json"
    return _typed(json.loads(path.read_text()))


@pytest.fixture
def typed_record():
    return _typed
