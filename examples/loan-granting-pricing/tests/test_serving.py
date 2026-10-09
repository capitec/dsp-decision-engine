"""The decision as served: raw request bytes through the handler `decider serve` uses, in its compiled default mode."""
import json
from pathlib import Path

import pytest
from decider import Engine
from decider.config import JsonFileStore
from decider.serving import RequestHandler

from loan_granting.pipeline import build

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ("outcome_code", "risk_grade", "score", "probability_of_default", "amount_cap", "term_cap",
           "worst_acceptable_grade", "max_affordable_instalment", "affordability_verdict_code", "has_any_offer",
           "solve_total_evaluations", "decline_reason_codes", "adjustments_applied", "final_validation_passed")


@pytest.fixture(scope="module")
def handler():
    handler = RequestHandler(JsonFileStore(basepath=str(ROOT / "configs")), "loan_granting.pipeline:build",
                             sample_request=str(ROOT / "sample_request.json"))
    handler.stage()
    handler.activate()
    return handler


def post(handler, request: dict) -> dict:
    response = handler.process_fn(json.dumps(request).encode(), "application/json", "application/json")
    return json.loads(response.content)


def test_the_raw_sample_request_is_decided_with_its_dates_as_json_strings(handler, sample):
    out = post(handler, sample)
    assert (out["outcome_code"], out["risk_grade"], out["max_affordable_instalment"]) == (1, 9, 5309.05)
    best = out["recommended_offer"]
    assert (best["term_months"], best["offered_amount"], best["instalment"]) == (36, 120_000.0, 4969.47)


def test_the_compiled_pipeline_agrees_with_the_interpreted_one_on_every_figure(handler, decide, sample):
    served, interpreted = post(handler, sample), decide()
    assert {name: served[name] for name in FIGURES} == {name: interpreted[name] for name in FIGURES}
    assert served["offers"] == interpreted["offers"] and served["amount_cap_chain"] == interpreted["amount_cap_chain"]


@pytest.mark.parametrize("change", [
    {"channel_code": 4}, {"fraud_verdict_code": 3}, {"bureau_as_of_date": "2026-07-01"}, {"applicant_age_years": 15.0},
    {"requested_amount": None, "requested_term_months": None}, {"applicant1_payslip_income": 3000.0},
    {"is_joint_application": True, "applicant2_payslip_income": 9000.0, "applicant2_employment_type_code": 2},
])
def test_other_decisions_are_served_the_same_as_interpreted(handler, decide, sample, change):
    served, interpreted = post(handler, {**sample, **change}), decide(**change)
    assert {name: served[name] for name in FIGURES} == {name: interpreted[name] for name in FIGURES}


def test_every_step_that_runs_in_python_says_so_on_purpose(documents):
    executable = Engine(strict_compile=True).bind(build(**documents), mode="fused")  # raises on an unannounced fallback
    assert executable.fallbacks()
    assert all(reason.startswith("@allow_fallback") for reason in executable.fallbacks().values())
