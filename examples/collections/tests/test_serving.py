"""The decision as served: raw request bytes through the handler `decikit serve` uses, in its compiled default mode."""
import json
from pathlib import Path

import pytest
from decikit import FusedEngine
from decikit.config import JsonFileStore
from decikit.serving import RequestHandler

from collections_treatment import vocab as v

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ("arrears_bucket_code", "collections_score", "collections_band_code", "roll_probability",
           "matrix_cell_id", "treatment_code", "treatment_intensity", "permitted_treatment_codes", "path_position",
           "path_position_as_at_now", "arrangement_sustainable", "discretionary_income", "pool_code", "allocated",
           "non_selection_reason_code")


@pytest.fixture(scope="module")
def handler():
    handler = RequestHandler(JsonFileStore(basepath=str(ROOT / "configs")), "collections_treatment.pipeline:build",
                             sample_request=str(ROOT / "sample_request.json"))
    handler.stage()
    handler.activate()
    return handler


def post(handler, request: dict) -> dict:
    response = handler.process_fn(json.dumps(request).encode(), "application/json", "application/json")
    return json.loads(response.content)


def test_the_raw_sample_request_gets_a_voice_message_with_its_dates_as_json_strings(handler, sample):
    out = post(handler, sample)
    assert (out["treatment_code"], out["allocated"], out["non_selection_reason_code"]) == (v.AUTOMATED_VOICE_MESSAGE, True, None)
    assert out["suspension_codes"] == [116] and out["decision_date"] == "2026-09-24"


def test_the_compiled_pipeline_agrees_with_the_interpreted_one_on_every_figure(handler, decide, sample):
    served, interpreted = post(handler, sample), decide()
    assert {name: served[name] for name in FIGURES} == {name: interpreted[name] for name in FIGURES}


def test_a_suspended_account_is_served_with_no_action(handler, sample):
    out = post(handler, sample | {"debt_review_stage_code": 1, "debt_review_case_number": "DR-9001"})
    assert (out["treatment_code"], out["allocated"], out["non_selection_reason_code"]) == (v.NO_ACTION, False, v.NS_SUSPENDED)
    assert out["suspension_codes"] == [101, 116]


def test_a_request_without_a_proposed_instalment_is_served_with_no_arrangement_verdict(handler, sample):
    out = post(handler, {k: v for k, v in sample.items() if k != "proposed_instalment"})
    assert (out["arrangement_sustainable"], out["arrangement_failure_detail"]) == (False, "no_proposal")


def test_every_step_that_runs_in_python_says_so_on_purpose(matrix_document, build_pipeline):
    executable = FusedEngine(strict_compile=True).bind(build_pipeline(matrix_document))
    assert executable.fallbacks()
    assert all(reason.startswith("@allow_fallback") for reason in executable.fallbacks().values())
