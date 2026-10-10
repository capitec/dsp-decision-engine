import json

import polars as pl
import pytest
from decikit.config import JsonFileStore
from decikit.lab.testing import assert_equivalent
from decikit.serving import RequestHandler

from governance.pipeline import build

from conftest import BASE_IDS, ROOT


@pytest.fixture(scope="module")
def handler():
    h = RequestHandler(JsonFileStore(basepath=str(ROOT / "configs")), "governance.pipeline:build",
                       sample_request=str(ROOT / "sample_request.json"))
    h.stage()
    h.activate()
    return h


def post(handler, request: dict) -> dict:
    response = handler.process_fn(json.dumps(request).encode(), "application/json", "application/json")
    return json.loads(response.content)


def test_the_sample_request_replays_a_loan_decision(handler):
    sample = json.loads((ROOT / "sample_request.json").read_text())
    assert post(handler, sample) == {**sample, "replay_verdict": "reproduced", "divergence_count": 0,
                                     "diverged_fields": [], "first_divergence": ""}


def test_a_fraud_and_a_business_decision_replay_too(handler):
    for flow_code in ("fraud", "business"):
        assert post(handler, {"flow_code": flow_code, "decision_id": BASE_IDS[flow_code]})["replay_verdict"] == "reproduced"


def test_a_decision_with_no_evidence_is_an_error_not_a_verdict(handler):
    with pytest.raises(FileNotFoundError):
        post(handler, {"flow_code": "loan", "decision_id": "no-such-decision"})


def test_every_mode_and_a_single_record_give_the_same_verdicts():
    frame = pl.DataFrame({"flow_code": ["loan", "business"], "decision_id": [BASE_IDS["loan"], BASE_IDS["business"]]})
    out = assert_equivalent(build(), frame)
    assert out["replay_verdict"].to_list() == ["reproduced", "reproduced"]
