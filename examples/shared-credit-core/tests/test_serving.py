"""The pipeline as served: raw request bytes through the handler `decikit serve` uses, in its default compiled mode."""
import json
from pathlib import Path

import pytest
from decikit import FusedEngine
from decikit.config import JsonFileStore
from decikit.serving import RequestHandler

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "0.1.0"


@pytest.fixture(scope="module")
def handler():
    handler = RequestHandler(JsonFileStore(basepath=str(CONFIGS.parent)), "credit_core.pipeline:build",
                             sample_request=str(ROOT / "sample_request.json"))
    handler.stage()
    handler.activate()
    return handler


def post(handler, body: bytes) -> dict:
    response = handler.process_fn(body, "application/json", "application/json")
    return json.loads(response.content)


def test_the_raw_sample_request_is_approved_with_its_dates_as_json_strings(handler):
    out = post(handler, (ROOT / "sample_request.json").read_bytes())
    assert out["outcome_code"] == 1
    assert (out["instalment"], out["nominal_annual_rate"], out["score"], out["risk_grade"]) == (2355.49, 0.1604, 642.0, 8)
    assert out["adjustments_applied"] == ["ADJ-2026-001", "ADJ-2026-002"]
    assert out["rate_cell_id"] == "rate_card.flex_loan@rc-2026.09#12|21|8"
    assert out["decision_id"] == "d5f5f6b0-3f8a-4f2a-9c4b-2f6e7a1b0001"


def test_the_compiled_pipeline_agrees_with_the_interpreted_one_on_every_figure(handler, quote, params, sample):
    served = post(handler, (ROOT / "sample_request.json").read_bytes())
    interpreted = quote.run(sample, params)
    for name in ("gross_monthly_income", "net_monthly_income", "existing_obligations", "max_affordable_instalment",
                 "score", "probability_of_default", "risk_grade", "nominal_annual_rate", "instalment",
                 "total_cost_of_credit", "outcome_code", "norm_table_version", "tax_table_cell_id"):
        assert served[name] == interpreted[name], name


def test_a_declined_request_is_served_with_its_reasons(handler):
    request = json.loads((ROOT / "sample_request.json").read_text())
    request.update(is_sanctioned=True, requested_amount=500000.0, term_months=6)
    out = post(handler, json.dumps(request).encode())
    assert out["outcome_code"] == 4
    assert out["decline_reason_codes"] == [1006, 4001] and out["primary_reason_code"] == 1006


def test_every_step_that_runs_in_python_says_so_on_purpose(quote_pipeline):
    executable = FusedEngine(strict_compile=True).bind(quote_pipeline)  # raises on an unannounced fallback
    assert executable.fallbacks()
    assert all(reason.startswith("@allow_fallback") for reason in executable.fallbacks().values())
