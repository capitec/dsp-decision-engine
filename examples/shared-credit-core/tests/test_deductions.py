from datetime import date

import pytest
from decider import flow

from credit_core.deductions import build_tax_table, net_monthly_income, statutory_deductions, tax_table_version_step

TAX = flow(tax_table_version_step, build_tax_table(), statutory_deductions, net_monthly_income, name="tax").emit(
    "tax_table_version", "tax_base", "tax_rate", "tax_band_lo", "statutory_deductions")


def tax(score, gross, employment=1, on=date(2026, 9, 1)):
    return score(TAX, {"gross_monthly_income": gross, "employment_type_code": employment, "decision_date": on})


@pytest.mark.parametrize("gross,expected", [
    (8000.0, 0.0),                    # under the threshold
    (8900.0, 0.0),                    # a bracket starts inclusive: nothing above it yet
    (10000.0, 198.0),                 # (10 000 - 8 900) * 18%
    (30000.0, 4518.0),                # 2 178 + (30 000 - 21 000) * 26%
    (50000.0, 9868.0),                # 8 938 + (50 000 - 47 000) * 31%
])
def test_tax_follows_the_progressive_brackets_of_the_year(score, gross, expected):
    assert tax(score, gross)["statutory_deductions"] == expected


def test_a_decision_before_the_new_tax_year_uses_the_old_brackets(score):
    assert tax(score, 10000.0, on=date(2026, 2, 28))["statutory_deductions"] == 270.0  # (10 000 - 8 500) * 18%
    assert tax(score, 10000.0, on=date(2026, 3, 1))["statutory_deductions"] == 198.0


def test_the_bracket_read_is_recorded(score):
    out = tax(score, 30000.0)
    assert (out["tax_table_version"], out["tax_band_lo"], out["tax_rate"], out["tax_base"]) == ("tax@2026", 21000.0, 0.26, 2178.0)
    assert out["tax_table_cell_id"] == "tax_table@tax@2026#2"


@pytest.mark.parametrize("employment", [3, 6])
def test_earners_without_tax_withheld_have_no_statutory_deduction(score, employment):
    assert tax(score, 30000.0, employment)["statutory_deductions"] == 0.0


def test_net_income_is_gross_less_deductions(score):
    assert tax(score, 30000.0)["net_monthly_income"] == 25482.0


def test_a_date_before_any_tax_table_cannot_be_resolved(score):
    with pytest.raises(LookupError):
        tax(score, 10000.0, on=date(2025, 2, 1))
