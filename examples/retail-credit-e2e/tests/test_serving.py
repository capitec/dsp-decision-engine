"""The decision as served: raw request bytes through the handler `decikit serve` uses, in its compiled default mode."""
import json
from pathlib import Path

import pytest
from decikit import FusedEngine
from decikit.config import JsonFileStore
from decikit.serving import RequestHandler

from retail_credit.pipeline import build

from conftest import parse_dates

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ("outcome_code", "risk_grade", "segment_code", "amount_cap", "max_affordable_instalment", "instalment_cap",
           "offer_amount", "offer_binding_constraint", "nominal_annual_rate", "instalment", "amount_financed",
           "total_cost_of_credit", "rate_cell_id", "decline_reason_codes", "amount_cap_adjustments_applied",
           "norm_table_version", "fraud_verdict_code")


@pytest.fixture(scope="module")
def handler():
    handler = RequestHandler(JsonFileStore(basepath=str(ROOT / "configs")), "retail_credit.pipeline:build",
                             sample_request=str(ROOT / "sample_request.json"))
    handler.stage()
    handler.activate()
    return handler


def post(handler, request: dict) -> dict:
    response = handler.process_fn(json.dumps(request).encode(), "application/json", "application/json")
    return json.loads(response.content)


def test_the_raw_sample_request_is_approved_with_its_dates_as_json_strings(handler, sample):
    out = post(handler, sample)
    assert (out["outcome_code"], out["offer_amount"], out["instalment"]) == (1, 76250.0, 2578.72)
    assert (out["offer_binding_constraint"], out["rate_cell_id"]) == ("BIND-AFF", "rate_card.flex_loan@rc-2026.09#37|39|7")


def test_the_compiled_pipeline_agrees_with_the_interpreted_one_on_every_figure(handler, decide, sample):
    served, interpreted = post(handler, sample), decide()
    assert {name: served[name] for name in FIGURES} == {name: interpreted[name] for name in FIGURES}


def test_a_declined_request_is_served_with_its_reasons(handler, sample):
    out = post(handler, sample | {"is_sanctioned": True})
    assert (out["outcome_code"], out["decline_reason_codes"], out["primary_reason_code"]) == (4, [1006], 1006)


def test_a_consolidation_offer_is_served(handler, sample):
    from test_consolidation import OVERLOADED

    out = post(handler, sample | OVERLOADED)
    assert (out["outcome_code"], out["offer_basis_code"], out["offer_amount"]) == (2, 2, 30_000.0)


def test_every_step_that_runs_in_python_says_so_on_purpose(sample, params):
    strict = FusedEngine(strict_compile=True).bind(build())
    undeclared = {step for step, reason in strict.fallbacks().items() if not reason.startswith("@allow_fallback")}
    assert strict.fallbacks() and not undeclared
    # A node that compiles on first use only meets strict mode when it is scored.
    assert strict.run(parse_dates(sample), params)["offer_amount"] == 76250.0
