"""The decision as served: raw request bytes through the handler `decikit serve` uses, in its compiled default mode."""
import json
from pathlib import Path

import polars as pl
import pytest
from decikit import FusedEngine
from decikit.config import JsonFileStore
from decikit.lab.testing import assert_equivalent
from decikit.serving import RequestHandler

from business_nested.pipeline import DECISION_OUTPUTS, build
from business_nested.records import parse_dates

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def handler():
    handler = RequestHandler(JsonFileStore(basepath=str(ROOT / "configs")), "business_nested.pipeline:build",
                             sample_request=str(ROOT / "sample_request.json"))
    handler.stage()
    handler.activate()
    return handler


def post(handler, request: dict) -> dict:
    response = handler.process_fn(json.dumps(request).encode(), "application/json", "application/json")
    return json.loads(response.content)


def test_the_raw_sample_request_is_declined_with_its_dates_as_json_strings(handler, sample):
    out = post(handler, sample)
    assert (out["outcome_code"], out["primary_reason_code"]) == (4, 5402)
    assert (out["attributing_entity_id"], out["attributing_event_ids"], out["attributing_rule_id"]) == (1, [41], "AE-R-01")
    assert out["assessed_entities"][0]["adverse_events"][0]["event_date"] == "2025-03-14"


def test_the_compiled_pipeline_agrees_with_the_interpreted_one_on_every_figure(handler, assess, sample):
    served, interpreted = post(handler, sample), assess()
    for name in DECISION_OUTPUTS:
        expected = interpreted[name]
        assert served[name] == (pytest.approx(expected) if isinstance(expected, float) else expected), name
    assert [e["adverse_verdict_code"] for e in served["assessed_entities"]] == [4, 1, 3]


def test_a_request_without_the_judgment_is_approved_and_priced(handler, sample):
    sample["entities"][0]["adverse_events"].pop(0)
    out = post(handler, sample)
    assert (out["outcome_code"], out["offered_amount"], out["instalment"]) == (1, 3_500_000.0, 106_202.67)


def test_a_sole_proprietor_is_served_with_the_statutory_assessment(handler, sample):
    request = sample | {"legal_form_code": 1, "entities": [{
        "entity_id": 1, "entity_key": "PRINCIPAL", "is_natural_person": True, "relationship_type_code": 11,
        "effective_ownership_pct": 100.0, "is_controlling": True, "is_sole_principal": True, "bureau_score": 705.0,
        "months_on_record": 100.0, "worst_delinquency_months": 0.0, "adverse_events": []}],
        "principal_internal_accounts": [{"account_type_code": 1, "balance": 90000.0, "instalment": 2500.0,
                                         "months_in_arrears": 0, "opened_date": "2021-06-15", "closed": False,
                                         "is_internal": True}]}
    out = post(handler, request)
    assert out["sole_proprietor_assessment_ran"] is True and out["sole_proprietor_affordability_verdict_code"] == 3
    assert (out["outcome_code"], out["decline_reason_codes"]) == (4, [5601])


def test_a_request_with_no_entities_is_a_referral_for_its_structure(handler, sample):
    out = post(handler, sample | {"entities": []})
    assert (out["outcome_code"], out["decline_reason_codes"]) == (3, [5001])


def test_every_mode_and_a_single_record_give_the_same_answers(sample):
    assert_equivalent(build(), pl.DataFrame([parse_dates(sample)]))


def test_every_step_that_runs_in_python_says_so_on_purpose():
    executable = FusedEngine(strict_compile=True).bind(build())  # raises on an unannounced fallback
    assert executable.fallbacks()
    assert all(reason.startswith("@allow_fallback") for reason in executable.fallbacks().values())
