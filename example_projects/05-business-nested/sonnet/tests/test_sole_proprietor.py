from datetime import date

from business_nested import vocab
from business_nested.sole_proprietor import assess_sole_proprietor


def test_only_sole_proprietors_trigger_the_regulated_affordability_call():
    ran, verdict, max_instalment = assess_sole_proprietor(
        legal_form_code=3, decision_date=date(2026, 9, 24), entity_id=[1],
        entity_relationship_type_code=[vocab.SHAREHOLDER], entity_key=["CO-B"],
        declared_annual_turnover=800_000.0, requested_amount=300_000.0,
    )
    assert ran is False and verdict == 0


def test_sole_proprietor_calls_project_02_and_gets_a_real_verdict():
    """Spec 05 §5.2/§5.12 item 7: a statutory affordability assessment through project 02."""
    ran, verdict, max_instalment = assess_sole_proprietor(
        legal_form_code=1, decision_date=date(2026, 9, 24), entity_id=[1],
        entity_relationship_type_code=[vocab.SOLE_PROPRIETOR_PRINCIPAL], entity_key=["PERSON-A"],
        declared_annual_turnover=800_000.0, requested_amount=300_000.0,
    )
    assert ran is True
    assert verdict in (1, 2, 3, 4)  # project 02's PASS/MARGINAL/FAIL/INDETERMINATE
    assert max_instalment >= 0.0
