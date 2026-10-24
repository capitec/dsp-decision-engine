import dataclasses
import shutil
import json

from decikit.config import JsonFileStore

from governance.evidence import EvidenceStore
from governance.flows import BUSINESS, FLOWS
from governance.replay import grade, replay

from conftest import BASE_IDS, ROOT


def test_a_shipped_decision_replays_bit_identically(decision):
    v = replay(decision)
    assert (v.verdict, v.divergences, v.first) == ("reproduced", (), None)


def test_every_shipped_decision_replays(evidence, flow_code):
    verdicts = {replay(e).verdict for e in evidence.all(flow_code)}
    assert verdicts == {"reproduced"}


def test_replaying_in_another_mode_gives_the_same_answers(evidence):
    assert replay(evidence.load("loan", BASE_IDS["loan"]), mode="interpreted").verdict == "reproduced"


def test_an_edited_outcome_does_not_reproduce_and_the_step_that_wrote_it_is_named(decision):
    flow = FLOWS[decision.flow]
    edited = dataclasses.replace(decision, response={**decision.response, flow.outcome: 999})
    v = replay(edited)
    assert v.verdict == "not_reproduced"
    assert [d.field for d in v.divergences] == [flow.outcome]
    assert v.first.recorded == 999 and v.first.replayed == decision.response[flow.outcome]
    assert v.first.step.endswith("outcome") or v.first.step.endswith("resolve_action") or "/" in v.first.step


def test_the_first_divergence_is_in_execution_order_not_alphabetical(evidence):
    loan = evidence.load("loan", BASE_IDS["loan"])
    v = replay(dataclasses.replace(loan, request={**loan.request, "requested_amount": 1000.0}))
    fields = [d.field for d in v.divergences]
    assert v.verdict == "not_reproduced"
    assert fields[0] == "requested_amount" and fields != sorted(fields)  # the changed input comes first
    assert fields.index("offers") < fields.index("outcome_code")  # the offers are built before the outcome is set


def test_a_config_document_edited_after_capture_cannot_be_replayed(evidence, tmp_path):
    e = evidence.load("business", BASE_IDS["business"])
    version = e.config_version
    shutil.copytree(BUSINESS.project() / "configs" / version, tmp_path / version)
    params = tmp_path / version / "params.json"
    document = json.loads(params.read_text())
    document["business_credit"]["decision"]["outcome"]["business_outcome"]["refer_cutoff_grade"] = 7
    params.write_text(json.dumps(document))
    v = replay(e, store=JsonFileStore(basepath=str(tmp_path)))
    assert (v.verdict, v.changed_documents, v.divergences) == ("not_reproducible", ("params",), ())


def test_floats_inside_the_declared_band_are_within_tolerance_and_outside_it_diverge():
    assert grade("instalment", 4969.47, 4969.470000001) == "tolerance"
    assert grade("instalment", 4969.47, 4969.48) == "diverged"
    assert grade("nominal_annual_rate", 0.1764, 0.17640001) == "tolerance"
    assert grade("score", 640.0, 640.0000001) == "tolerance" and grade("score", 640.0, 640.01) == "diverged"


def test_codes_lists_and_booleans_compare_exactly():
    assert grade("outcome_code", 1, 2) == "diverged"
    assert grade("decline_reason_codes", [1, 2], [2, 1]) == "diverged"
    assert grade("flag", True, False) == "diverged" and grade("flag", True, True) == "exact"
    assert grade("offers", [{"instalment": 1.0}], [{"instalment": 1.01}]) == "diverged"
    assert grade("x", float("nan"), float("nan")) == "exact"
