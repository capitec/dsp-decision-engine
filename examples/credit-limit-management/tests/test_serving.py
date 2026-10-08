"""The pipeline as served: raw request bytes through the handler `decider serve` uses, in its compiled default mode."""
import asyncio
import json

import pytest
from decider import Engine
from decider.config import JsonFileStore
from decider.serving import RequestHandler

from limit_mgmt.matrix import build_matrix_table
from limit_mgmt.pipeline import build
from population import ROOT

FIGURES = ("matrix_cell_id", "matrix_multiplier", "matrix_multiplier_unadjusted", "policy_proposed_limit",
           "policy_proposed_limit_unadjusted", "final_proposed_limit", "binding_cap_code", "increase_path_code",
           "change_type_code", "notice_class_code", "behaviour_grade", "probability_of_default", "exclusion_codes",
           "max_affordable_instalment", "affordability_buffer_applied", "evidence_tier_code", "decline_reason_codes")


@pytest.fixture(scope="module")
def handler():
    handler = RequestHandler(JsonFileStore(basepath=str(ROOT / "configs")), "limit_mgmt.pipeline:build",
                             sample_request=str(ROOT / "sample_request.json"))
    handler.stage()
    handler.activate()
    return handler


def post(handler, request: dict) -> dict:
    response = asyncio.run(handler.process_fn(json.dumps(request).encode(), "application/json", "application/json"))
    return json.loads(response.content)


def test_the_raw_sample_request_is_decided_with_its_dates_as_json_strings(handler, sample):
    out = post(handler, sample)
    assert (out["final_proposed_limit"], out["change_type_code"], out["matrix_cell_id"]) == (
        20_000.0, 1, "limit_matrix@limit-matrix-2026.09#20|2|5|5")


@pytest.mark.parametrize("changes", [
    {}, {"worst_arrears_months_now": 1}, {"consent_automatic_increase": False}, {"verified_salary_amount": 11_000.0},
    {"bureau_as_of_date": "2026-08-10"}, {"product_code": 21}, {"salary_deposit_variability_pct": 0.3}])
def test_the_compiled_pipeline_agrees_with_the_interpreted_one(handler, decide, sample, changes):
    served, interpreted = post(handler, sample | changes), decide(**changes)
    assert {name: served[name] for name in FIGURES} == {name: interpreted[name] for name in FIGURES}


def test_the_served_pipeline_reads_the_zero_utilisation_cell(handler, sample):
    assert post(handler, sample | {"cycle_balances": [0.0] * 6})["matrix_cell_id"].endswith("#20|2|1|5")


def test_every_step_that_runs_in_python_says_so_on_purpose():
    executable = Engine(strict_compile=True).bind(build(build_matrix_table()), mode="fused")  # raises on an unannounced fallback
    assert executable.fallbacks()
    assert all(reason.startswith("@allow_fallback") for reason in executable.fallbacks().values())
