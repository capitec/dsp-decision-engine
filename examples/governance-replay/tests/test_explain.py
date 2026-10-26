import pytest

from governance.explain import explain

from conftest import BASE_IDS


def test_the_consultant_hears_the_deciding_reason_and_nothing_internal(evidence):
    e = evidence.load("business", BASE_IDS["business"])
    x = explain(e, "consultant")
    assert x.text == "An entity behind the business carries a disqualifying adverse record"
    assert x.detail == {} and "AE-R-01" not in x.text and e.config_version not in x.text


def test_the_consultant_of_a_fraud_decision_learns_nothing_about_the_rule(evidence):
    x = explain(evidence.load("fraud", BASE_IDS["fraud"]), "consultant")
    assert x.text == "The client is shown wording.hold_for_review.v1 and nothing else."


def test_the_analyst_sees_every_reason_the_mechanism_and_adjusted_beside_unadjusted(evidence):
    e = evidence.load("business", BASE_IDS["business"])
    x = explain(e, "analyst")
    assert [r["code"] for r in x.detail["reasons"]] == [5402]
    assert x.detail["how"]["attributing_rule_id"] == "AE-R-01" and x.detail["how"]["attributing_event_ids"] == [41]
    assert x.detail["adjusted_vs_unadjusted"]["risk_grade"] == {"adjusted": 7, "unadjusted": 6}
    assert x.detail["documents"] == e.documents


def test_the_adjudicator_gets_sentences_with_the_policy_version(evidence):
    e = evidence.load("fraud", BASE_IDS["fraud"])
    x = explain(e, "adjudicator")
    assert "2026-01-10" in x.text and "0.1.0" in x.text and "Outcome 40" in x.text
    assert x.detail["registry"] == "fraud-reasons-2026.01" and len(x.detail["reasons"]) == 16


def test_an_approval_has_no_reason_to_give(evidence):
    e = evidence.load("loan", BASE_IDS["loan"])
    assert explain(e, "consultant").text == "No reason applies (outcome 1)."


def test_an_unknown_audience_is_refused(evidence):
    with pytest.raises(ValueError):
        explain(evidence.load("loan", BASE_IDS["loan"]), "regulator")
