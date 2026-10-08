import pytest

from governance.certify import certify
from governance.flows import LOAN

from test_versions import raise_rates, reverse_rows


@pytest.fixture(scope="module")
def golden(evidence):
    return evidence.all("loan")


def test_a_reordered_rate_card_is_a_no_effect_change_and_passes(golden, tmp_path):
    c = certify(golden, LOAN.candidate(reverse_rows, tmp_path, "interpreted"), accountable="J. Moyo")
    assert (c.passed, c.moved, c.unexpected, c.unmoved) == (True, (), (), ())
    assert c.changed_documents == ("rate_card_flex_loan",) and c.accountable == "J. Moyo"


def test_a_rate_rise_declared_as_no_effect_is_blocked_with_the_decisions_it_moved(golden, tmp_path):
    c = certify(golden, LOAN.candidate(raise_rates(1.0), tmp_path, "interpreted"))
    everyone = tuple(sorted(e.decision_id for e in golden))
    assert not c.passed and c.moved == everyone and c.unexpected == everyone and c.unmoved == ()


def test_a_declared_change_passes_only_if_exactly_the_declared_decisions_move(golden, tmp_path):
    candidate = LOAN.candidate(raise_rates(1.0, grade=12), tmp_path, "interpreted")
    worst = {e.decision_id for e in golden if e.response["risk_grade"] == 12}
    other = sorted({e.decision_id for e in golden} - worst)
    assert worst and other
    c = certify(golden, candidate, worst)
    assert c.passed and c.moved == tuple(sorted(worst))
    c = certify(golden, candidate, worst | {other[0]})
    assert not c.passed and c.unmoved == (other[0],) and c.unexpected == ()
    c = certify(golden, candidate, worst - {sorted(worst)[0]})
    assert not c.passed and c.unexpected == (sorted(worst)[0],) and c.unmoved == ()


def test_the_certificate_is_a_plain_document_with_the_verdict(golden, tmp_path):
    c = certify(golden, LOAN.candidate(reverse_rows, tmp_path, "interpreted"), approvals=("CRC-2026-14",))
    doc = c.to_json()
    assert doc["passed"] is True and doc["approvals"] == ["CRC-2026-14"] and doc["flow"] == "loan"
