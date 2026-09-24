"""Request handling for the business credit end-to-end (EP-1) pipeline.

Staging warms up with `sample_request.json`.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

from decider.serving.handler import RequestHandler

_SAMPLE_REQUEST_PATH = Path(__file__).parent / "sample_request.json"


def _typed_sample_record() -> dict:
    record = json.loads(_SAMPLE_REQUEST_PATH.read_text())
    record["decision_date"] = datetime.date.fromisoformat(record["decision_date"])
    record["knowledge_date"] = datetime.date.fromisoformat(record["knowledge_date"])
    for account in record.get("existing_accounts", []):
        if account.get("opened_date"):
            account["opened_date"] = datetime.date.fromisoformat(account["opened_date"])
    for entity in record.get("entities", []):
        for event in entity.get("adverse_events", []):
            event["event_date"] = datetime.date.fromisoformat(event["event_date"])
    return record


class Handler(RequestHandler):
    pass
