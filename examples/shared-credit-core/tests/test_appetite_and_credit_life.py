import pytest

from credit_core.appetite import MAX_AMOUNT, MAX_INSTALMENT_RATIO, MAX_TERM, binding_appetite_limit, build_appetite_table
from credit_core.credit_life import JOINT, SINGLE, build_credit_life_table, credit_life_cap_applied, credit_life_premium


def test_appetite_limits_tighten_with_each_worse_grade(score):
    table = build_appetite_table()
    grade_1 = score(table, {"product_code": 10, "segment_code": 1, "risk_grade": 1})
    grade_12 = score(table, {"product_code": 10, "segment_code": 1, "risk_grade": 12})
    assert (grade_1["max_amount"], grade_1["max_term"], grade_1["max_ratio"]) == (465_000.0, 84, 0.42)
    assert (grade_12["max_amount"], grade_12["max_ratio"]) == (80_000.0, 0.09)
    assert grade_1["cell_id"] == "appetite@appetite-2026.09#10|1|1"


def test_a_combination_without_a_row_has_no_appetite(score):
    out = score(build_appetite_table(), {"product_code": 40, "segment_code": 1, "risk_grade": 1})
    assert (out["max_amount"], out["max_term"], out["cell_id"]) == (0.0, 0, None)


@pytest.mark.parametrize("request_,binding", [
    ({"requested_amount": 500_000.0, "term_months": 96.0, "instalment_to_income_ratio": 0.9}, MAX_AMOUNT),
    ({"requested_amount": 100_000.0, "term_months": 96.0, "instalment_to_income_ratio": 0.9}, MAX_TERM),
    ({"requested_amount": 100_000.0, "term_months": 60.0, "instalment_to_income_ratio": 0.9}, MAX_INSTALMENT_RATIO),
    ({"requested_amount": 100_000.0, "term_months": 60.0, "instalment_to_income_ratio": 0.1}, ""),
])
def test_the_binding_limit_is_the_first_one_broken_in_order(request_, binding):
    assert binding_appetite_limit(max_amount=465_000.0, max_term=84.0, max_ratio=0.42, **request_) == binding


@pytest.mark.parametrize("age,term,cover,rate", [
    (25.0, 12.0, SINGLE, 3.20), (30.0, 24.0, SINGLE, 4.70), (40.0, 70.0, SINGLE, 5.60),
    (60.0, 70.0, JOINT, 15.40), (25.0, 12.0, JOINT, 5.40)])
def test_the_premium_rate_depends_on_age_term_and_cover_type(score, age, term, cover, rate):
    out = score(build_credit_life_table(), {"applicant_age_years": age, "term_months": term, "cover_type_code": cover})
    assert out["rate"] == rate


def test_the_premium_is_the_rate_per_thousand_of_cover_up_to_the_cap():
    assert credit_life_premium(65_000.0, 3.2) == 208.0
    assert not credit_life_cap_applied(65_000.0, 3.2)
    assert credit_life_premium(100_000.0, 15.4) == 350.0  # 1 540 before the cap
    assert credit_life_cap_applied(100_000.0, 15.4)
