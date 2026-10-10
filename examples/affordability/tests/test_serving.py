"""The assessment as served: raw request bytes through the handler `decikit serve` uses, in its compiled default mode."""
import json
import math
from pathlib import Path

import pytest
from decikit import Engine
from decikit.config import JsonFileStore
from decikit.serving import RequestHandler

from assessment.pipeline import build

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ("gross_monthly_income", "net_monthly_income", "living_expenses", "existing_obligations",
           "discretionary_income", "max_affordable_instalment", "max_affordable_instalment_unadjusted",
           "affordability_buffer_applied", "affordability_verdict_code", "evidence_sufficiency_code",
           "norm_table_version", "applicant1_tax_table_cell_id", "adjustments_applied")


@pytest.fixture(scope="module")
def handler():
    handler = RequestHandler(JsonFileStore(basepath=str(ROOT / "configs")), "assessment.pipeline:build",
                             sample_request=str(ROOT / "sample_request.json"))
    handler.stage()
    handler.activate()
    return handler


def post(handler, request: dict) -> dict:
    response = handler.process_fn(json.dumps(request).encode(), "application/json", "application/json")
    return json.loads(response.content)


def test_the_raw_sample_request_is_assessed_with_its_dates_as_json_strings(handler, sample):
    out = post(handler, sample)
    assert (out["max_affordable_instalment"], out["affordability_verdict_code"]) == (5980.99, 1)
    assert out["discretionary_income_after"] == 5713.5


def test_the_compiled_pipeline_agrees_with_the_interpreted_one_on_every_figure(handler, assess, sample):
    served, interpreted = post(handler, sample), assess()
    assert {name: served[name] for name in FIGURES} == {name: interpreted[name] for name in FIGURES}


def test_a_joint_application_is_served_and_agrees_with_the_interpreted_pipeline(handler, assess, sample):
    joint = sample | {"is_joint_application": True, "applicant2_dependants_count": 3,
                      "applicant2_employment_type_code": 2, "applicant2_payslip_income": 9000.0}
    served, interpreted = post(handler, joint), assess(**joint)
    assert served["gross_monthly_income"] == 26125.0
    assert {name: served[name] for name in FIGURES} == {name: interpreted[name] for name in FIGURES}


def test_a_request_for_capacity_alone_is_served_without_a_margin(handler, sample):
    out = post(handler, {k: v for k, v in sample.items() if k != "proposed_instalment"})
    assert out["max_affordable_instalment"] == 5980.99 and out["affordability_verdict_code"] == 1
    assert out["discretionary_income_after"] is None or math.isnan(out["discretionary_income_after"])


def test_a_stale_bureau_is_served_as_indeterminate(handler, sample):
    out = post(handler, sample | {"bureau_as_of_date": "2026-08-01"})
    assert (out["affordability_verdict_code"], out["decline_reason_codes"]) == (4, [204])


def test_every_step_that_runs_in_python_says_so_on_purpose():
    executable = Engine(strict_compile=True).bind(build(), mode="fused")  # raises on an unannounced fallback
    assert executable.fallbacks()
    assert all(reason.startswith("@allow_fallback") for reason in executable.fallbacks().values())
