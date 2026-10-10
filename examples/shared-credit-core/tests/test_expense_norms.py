import math
from datetime import date

import pytest
from decikit import flow

from credit_core.expense_norms import (
    DECLARED, NORM_FLOOR, STATEMENT_DERIVED, build_internal_table, build_statutory_table, expense_basis_code,
    internal_norm_amount, internal_version_step, living_expenses, norm_floor, norm_table_version, statutory_version_step,
)

NORMS = flow(statutory_version_step, internal_version_step, build_statutory_table(), build_internal_table(),
             internal_norm_amount, norm_floor, norm_table_version, expense_basis_code, living_expenses, name="norms").emit(
    "statutory_norm_amount", "internal_norm_amount", "norm_floor", "norm_table_version")

# Income 5 000 with no dependants is the second income band, first dependants band: statutory cell 3 (R1 800 in 2025).
BASE = {"decision_date": date(2026, 9, 30), "gross_monthly_income": 5000.0, "dependants_count": 0, "product_code": 10}


def norms(score, **changes):
    return score(NORMS, {**BASE, **changes})


def test_the_floor_is_the_higher_of_the_statutory_and_internal_norm(score):
    out = norms(score)
    assert out["statutory_norm_amount"] == 1800.0
    assert out["internal_norm_amount"] == 2016.0  # 1 800 * 1.12 for a Flex Loan in 2026
    assert out["norm_floor"] == 2016.0
    assert out["norm_table_version"] == "norms.internal@2026"


def test_the_floor_binds_when_the_applicant_claims_less(score):
    out = norms(score, declared_living_expenses=1000.0)
    assert (out["living_expenses"], out["expense_basis_code"]) == (2016.0, NORM_FLOOR)


def test_a_claim_above_the_floor_stands(score):
    out = norms(score, declared_living_expenses=3000.0)
    assert (out["living_expenses"], out["expense_basis_code"]) == (3000.0, DECLARED)


def test_statement_expenses_take_precedence_over_declared_ones(score):
    out = norms(score, declared_living_expenses=1000.0, statement_living_expenses=2500.0)
    assert (out["living_expenses"], out["expense_basis_code"]) == (2500.0, STATEMENT_DERIVED)


def test_no_claim_at_all_is_floored_not_zero(score):
    out = norms(score)
    assert (out["living_expenses"], out["expense_basis_code"]) == (2016.0, NORM_FLOOR)


def test_gazetted_statutory_norms_apply_from_their_effective_date_only(score):
    before = norms(score, decision_date=date(2026, 9, 30))
    after = norms(score, decision_date=date(2026, 10, 1))
    assert before["statutory_norm_amount"] == 1800.0
    assert after["statutory_norm_amount"] == 1890.0  # 5% higher
    assert before["statutory_norm_cell_id"].endswith("norms.statutory@2025#3")
    assert after["statutory_norm_cell_id"].endswith("norms.statutory@2026#3")


def test_the_internal_multiplier_depends_on_the_product(score):
    assert norms(score, product_code=20)["internal_norm_amount"] == 2124.0  # 1 800 * 1.18


def test_dependants_and_income_select_the_cell(score):
    assert norms(score, dependants_count=3)["statutory_norm_amount"] == 2700.0
    assert norms(score, gross_monthly_income=2000.0)["statutory_norm_amount"] == 1500.0
    assert norms(score, gross_monthly_income=20000.0, dependants_count=3)["statutory_norm_amount"] == 4200.0


def test_living_expenses_do_not_treat_nan_as_a_claim():
    assert living_expenses(2016.0, math.nan, math.nan) == 2016.0
    assert expense_basis_code(2016.0, math.nan, math.nan) == NORM_FLOOR
