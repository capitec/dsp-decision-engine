import json

import pytest

from governance.evidence import EvidenceStore, capture, digest
from governance.flows import BUSINESS


def test_capture_keeps_the_request_as_received_and_the_digest_of_every_config_document(evidence):
    request = json.loads((BUSINESS.project() / "sample_request.json").read_text())
    e = capture(BUSINESS, request)
    assert e.request == request and isinstance(e.request["decision_date"], str)
    assert e.config_version == "0.1.0" and e.mode == "fused"
    assert e.documents == {"params": digest(e.params)}
    assert (e.response["outcome_code"], e.response["attributing_rule_id"]) == (4, "AE-R-01")
    assert e == evidence.load("business", e.decision_id)


def test_saving_the_same_decision_twice_refuses_to_overwrite(evidence, tmp_path):
    store = EvidenceStore(tmp_path)
    e = evidence.all("business")[0]
    store.save(e)
    with pytest.raises(FileExistsError):
        store.save(e)


def test_the_digest_changes_when_a_document_changes_by_one_value():
    assert digest({"a": {"b": 1}}) != digest({"a": {"b": 2}})
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
