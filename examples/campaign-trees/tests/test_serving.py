"""The decision as served: raw request bytes through the handler `decider serve` uses, in its compiled default mode."""
import json
from pathlib import Path

import pytest
from decider import Engine
from decider.config import JsonFileStore
from decider.serving import RequestHandler

from campaign_trees.pipeline import build

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ("evaluated", "tree_path", "unadjusted_tree_path", "leaf", "pre_assessed_amount", "suppression_codes",
           "assignment_outcome_code", "advertised_amount", "overlay_stack_id", "is_control")


@pytest.fixture(scope="module")
def handler():
    handler = RequestHandler(JsonFileStore(basepath=str(ROOT / "configs")), "campaign_trees.pipeline:build",
                             sample_request=str(ROOT / "sample_request.json"))
    handler.stage()
    handler.activate()
    return handler


def post(handler, request: dict) -> dict:
    response = handler.process_fn(json.dumps(request).encode(), "application/json", "application/json")
    return json.loads(response.content)


def test_the_raw_sample_request_is_decided_with_its_date_as_a_json_string(handler, sample):
    out = post(handler, sample)
    assert (out["tree_path"], out["assignment_outcome_code"], out["advertised_amount"]) == (
        "n1>n2>n4>n6>n8>n10>leaf_912", 1, 20_800.0)
    assert out["pre_assessment_valid_to"] == "2026-10-29"


@pytest.mark.parametrize("change", [
    {}, {"is_deceased": True}, {"contacts_30d": 5}, {"discretionary_income": 2400.0}, {"estimated_discretionary_income": 1000.0},
    {"cycle_date": "2026-10-05", "settlement_ratio": 0.5, "behaviour_score": 700, "worst_arrears_months_12m": 0,
     "discretionary_income": 5000.0, "income_band_code": 7, "estimated_discretionary_income": 40_000.0, "app_logins_90d": 5},
])
def test_other_clients_are_served_the_same_as_interpreted(handler, decide, sample, change):
    served, interpreted = post(handler, {**sample, **change}), decide(**change)
    assert {name: served[name] for name in FIGURES} == {name: interpreted[name] for name in FIGURES}


def test_every_step_that_runs_in_python_says_so_on_purpose(tree, rate_card):
    executable = Engine(strict_compile=True).bind(build(tree, rate_card), mode="fused")  # raises on an unannounced fallback
    assert executable.fallbacks()
    assert all(reason.startswith("@allow_fallback") for reason in executable.fallbacks().values())
