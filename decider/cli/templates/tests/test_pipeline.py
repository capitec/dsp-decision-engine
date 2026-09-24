"""Scores requests through the served handler, raw JSON bytes in, as `decider serve` does."""
import asyncio
import json
from pathlib import Path

from decider.config import JsonFileStore

from {{name}}.inference import Handler

ROOT = Path(__file__).parents[1]
HANDLER = Handler(JsonFileStore(basepath=str(ROOT / "configs")), "{{name}}.pipeline:build")
HANDLER.stage()
HANDLER.activate()


def score(body: bytes) -> dict:
    response = asyncio.run(HANDLER.process_fn(body, "application/json", "application/json"))
    return json.loads(response.content)


def test_the_sample_request_is_approved():
    assert score((ROOT / "sample_request.json").read_bytes())["approved"] is True


def test_month_end_applications_meet_the_lower_limit():
    assert score(b'{"income": 1000, "debt": 350, "applied_on": "2026-01-10"}')["approved"] is True
    assert score(b'{"income": 1000, "debt": 350, "applied_on": "2026-01-28"}')["approved"] is False
