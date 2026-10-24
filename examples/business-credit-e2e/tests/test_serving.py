"""The decision as served: raw request bytes through the handler `decikit serve` uses, in its compiled default mode."""
import json
from pathlib import Path

import polars as pl
import pytest
from decikit import FusedEngine
from decikit.config import JsonFileStore
from decikit.lab.testing import assert_equivalent
from decikit.serving import RequestHandler

from business_credit_e2e.pipeline import OUTPUTS, build

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def handler():
    handler = RequestHandler(JsonFileStore(basepath=str(ROOT / "configs")), "business_credit_e2e.pipeline:build",
                             sample_request=str(ROOT / "sample_request.json"))
    handler.stage()
    handler.activate()
    return handler


def post(handler, request: dict) -> dict:
    response = handler.process_fn(json.dumps(request).encode(), "application/json", "application/json")
    return json.loads(response.content)


def test_the_raw_sample_request_is_approved_with_its_covenant_bound(handler, sample):
    out = post(handler, sample)
    assert (out["outcome_code"], out["risk_grade"], out["offered_amount"], out["instalment"]) == (1, 7, 1_000_000.0, 30392.91)
    assert (out["covenant_instance_id"], out["covenant_definition_version"], out["measured_dscr"]) == (
        "COV-DSCR-500001", "dscr-2026.03", 1.3911)


def test_the_compiled_pipeline_agrees_with_the_interpreted_one_on_every_output(handler, decide, sample):
    served, interpreted = post(handler, sample), decide()
    for name in OUTPUTS[1:]:  # the entities carry dates, which are strings once served
        expected = interpreted[name]
        assert served[name] == (pytest.approx(expected) if isinstance(expected, float) else expected), name


def test_a_disqualifying_judgment_is_declined_and_still_carries_the_covenant(handler, sample):
    sample["entities"][0]["adverse_events"].append({
        "event_id": 41, "event_type_code": 1, "amount": 184000.0, "event_date": "2025-03-14", "status_is_active": True,
        "is_disputed": False, "is_satisfied": False})
    out = post(handler, sample)
    assert (out["outcome_code"], out["attributing_event_ids"], out["attributing_rule_id"]) == (4, [41], "AE-R-01")
    assert out["covenant_definition_version"] == "dscr-2026.03"


def test_a_revolving_facility_is_served_with_its_minimum_payment(handler, sample):
    out = post(handler, sample | {"product_code": 51, "requested_amount": 400000.0})
    assert (out["offered_amount"], out["new_facility_instalment"]) == (400_000.0, 20_000.0)


def test_every_mode_and_a_single_record_give_the_same_answers(request_for):
    assert_equivalent(build(), pl.DataFrame([request_for()]))


def test_every_step_that_runs_in_python_says_so_on_purpose():
    executable = FusedEngine(strict_compile=True).bind(build())  # raises on an unannounced fallback
    assert executable.fallbacks()
    assert all(reason.startswith("@allow_fallback") for reason in executable.fallbacks().values())
