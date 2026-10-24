from governance.coverage import coverage
from governance.flows import BUSINESS, FRAUD, LOAN


def responses(evidence, flow):
    return [e.response for e in evidence.all(flow.code)]


def test_business_rules_that_never_bound_are_dead_and_ids_outside_the_list_are_reported_apart(evidence):
    c = coverage(BUSINESS, responses(evidence, BUSINESS))
    assert c.decisions == 5 and set(c.hits) == {"AE-R-01", "AE-R-02", "AE-R-06", "AE-R-07", "AE-R-11", "AE-R-12"}
    assert c.hits["AE-R-01"] > 0 and c.dead == ("AE-R-06", "AE-R-12")
    assert c.dominant == ("AE-R-02", "AE-R-11")
    assert c.unknown == ("worst-event",)


def test_loan_waterfall_rules_count_only_when_they_bound(evidence):
    c = coverage(LOAN, responses(evidence, LOAN))
    assert len(c.hits) == 12 and len(c.dead) == 9
    assert c.dominant == ("CAP-0100", "CAP-0210", "CAP-0420")
    assert all(c.hits[r] == 0 for r in c.dead) and c.cells["risk_grade_cell_id"] == 4


def test_fraud_universe_is_the_live_rules_only(evidence):
    c = coverage(FRAUD, responses(evidence, FRAUD))
    assert len(c.hits) == 521 and c.unknown == () and len(c.dead) == 494
    assert "MS-0208" in c.hits and c.hits["MS-0021"] > 0


def test_static_findings_come_from_the_toolkits_own_checks(evidence):
    c = coverage(FRAUD, responses(evidence, FRAUD))
    assert c.static_findings == ("DCD208 fraud_interdiction/wording/client_wording",)
