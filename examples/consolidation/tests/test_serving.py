"""The decision as served: raw request bytes through the handler `decikit serve` uses, in its compiled default mode."""
import json
from pathlib import Path

import pytest
from decikit import FusedEngine
from decikit.config import JsonFileStore
from decikit.serving import RequestHandler

from consolidation.pipeline import build

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ("outcome_code", "decline_reason_codes", "eligibility_route", "short_circuit_applies", "conditions",
           "offer_valid_until", "objective_weights", "objective_overlay_ids", "search_record", "baseline", "invariants",
           "before_and_after")


@pytest.fixture(scope="module")
def handler():
    handler = RequestHandler(JsonFileStore(basepath=str(ROOT / "configs")), "consolidation.pipeline:build",
                             sample_request=str(ROOT / "sample_request.json"))
    handler.stage()
    handler.activate()
    return handler


def post(handler, request: dict) -> dict:
    response = handler.process_fn(json.dumps(request).encode(), "application/json", "application/json")
    return json.loads(response.content)


def test_the_raw_sample_request_is_decided_with_its_dates_as_json_strings(handler, sample):
    out = post(handler, sample)
    offer = out["chosen_offer"]
    assert (out["outcome_code"], offer["scenario_id"], offer["settlement_set"], offer["product_code"], offer["term_months"]) == (
        2, 41, [1, 2, 3, 4, 5, 6, 7], 11, 72)
    assert (offer["offered_amount"], offer["instalment"]) == (211_103.11, 5_117.67)
    assert out["search_record"]["consumption"] == 178 and out["offer_valid_until"] == "2026-10-15"


def test_the_compiled_pipeline_agrees_with_the_interpreted_one_on_every_figure_and_every_scenario(handler, assess, sample):
    served, interpreted = post(handler, sample), assess()
    assert {name: served[name] for name in FIGURES} == {name: interpreted[name] for name in FIGURES}
    assert served["scenarios"] == interpreted["scenarios"] and served["chosen_offer"] == interpreted["chosen_offer"]


@pytest.mark.parametrize("change", [
    {"channel_code": 2}, {"client_under_debt_review": True}, {"client_nominated_settle": [5]}, {"client_excluded_settle": [2]},
    {"requested_amount": None}, {"last_balance_transfer_date": "2026-03-01"}, {"consolidations_last_24_months": 2},
])
def test_other_decisions_are_served_the_same_as_interpreted(handler, assess, sample, change):
    request = {k: v for k, v in (sample | change).items() if v is not None}
    served, interpreted = post(handler, request), assess(**change)
    assert {name: served[name] for name in FIGURES} == {name: interpreted[name] for name in FIGURES}


def present(value):
    """A frame row holds every declared key, null where a single record left it out; compare with those keys read as absent."""
    if isinstance(value, dict):
        return {k: present(v) for k, v in value.items() if v is not None}
    return [present(v) for v in value] if isinstance(value, list) else value


def test_a_json_array_of_requests_is_decided_like_the_same_requests_one_by_one(handler, sample):
    requests = [sample, sample | {"decision_id": "con-2026-0000555002", "channel_code": 2}]
    batch = post(handler, requests)
    assert len(batch) == 2
    for served, request in zip(batch, requests):
        alone = post(handler, request)
        assert present({name: served[name] for name in FIGURES}) == present({name: alone[name] for name in FIGURES})


def test_every_step_that_runs_in_python_says_so_on_purpose(documents):
    executable = FusedEngine(strict_compile=True).bind(build(*documents))  # raises on an unannounced fallback
    assert executable.fallbacks()
    assert all(reason.startswith("@allow_fallback") for reason in executable.fallbacks().values())
