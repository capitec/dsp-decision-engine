"""The date parsing the serving handler does, for callers that score in process."""
from __future__ import annotations

from datetime import date

from business_nested.records import parse_dates as parse_entity_dates


def parse_dates(record: dict) -> dict:
    """A copy of a JSON request with its dates parsed, as `Executable.score` takes them.

    Example: `parse_dates(json.load(open("sample_request.json")))["decision_date"]` is `date(2026, 9, 24)`.
    """
    record = parse_entity_dates(record)
    for field in ("decision_date", "knowledge_date"):
        if isinstance(record.get(field), str):
            record[field] = date.fromisoformat(record[field])
    for account in record.get("existing_accounts", []):
        account["opened_date"] = date.fromisoformat(account["opened_date"])
    return record
