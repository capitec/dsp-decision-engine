import math

from assessment import household, verdict

CARD = {"account_type_code": 10, "balance": 8500.0, "limit": 15000.0, "instalment": 450.0,
        "months_in_arrears": 0, "opened_date": "2022-03-01", "closed": False, "is_internal": False}
SECOND = {
    "is_joint_application": True, "applicant2_dependants_count": 3, "applicant2_employment_type_code": 2,
    "applicant2_payslip_income": 9000.0,
    "applicant2_declared_expenses": {"accommodation": 3000.0, "food": 2000.0, "transport": 600.0},
    "applicant2_bureau_accounts": [CARD],
}


def test_a_joint_application_adds_incomes_after_each_applicants_own_haircut(assess):
    out = assess(**SECOND)
    assert out["gross_monthly_income"] == 26125.0          # 17 575 + 8 550
    assert out["income_haircut_applied"] == 0.05


def test_tax_is_charged_on_each_applicant_not_on_the_household_sum(assess):
    out = assess(**SECOND)
    assert out["statutory_deductions"] == 1561.5           # applicant 2's R8 550 is below the first bracket
    assert out["net_monthly_income"] == 24563.5            # taxed as a sum it would be R3 510.50


def test_a_joint_household_takes_the_higher_dependants_count_and_one_norm_on_combined_income(assess):
    out = assess(**SECOND)
    assert out["dependants_count"] == 3.0
    assert out["internal_norm_cell_id"].endswith("#10|11")  # combined income band, three-or-more dependants


def test_joint_expenses_take_the_higher_shared_cost_and_add_personal_costs(assess):
    # accommodation max(3500, 3000) + food max(1800, 2000) + transport 900 + 600
    assert assess(**SECOND)["living_expenses"] == 7000.0


def test_a_cost_outside_the_shared_list_adds_up(assess, sample):
    first = {**sample["applicant1_declared_expenses"], "education": 700.0}
    out = assess(**{**SECOND, "applicant1_declared_expenses": first, "applicant2_declared_expenses": {"education": 300.0}})
    assert out["living_expenses"] == 3500.0 + 1800.0 + 900.0 + 1000.0


def test_a_card_on_both_applicants_profiles_counts_once(assess):
    out = assess(**SECOND)
    assert out["obligation_account_type_codes"] == [1, 10]
    assert out["existing_obligations"] == 1600.0


def test_a_solo_application_ignores_whatever_is_sent_for_applicant_two(assess):
    solo = assess(**{**SECOND, "is_joint_application": False})
    assert solo["gross_monthly_income"] == 17575.0
    assert solo["existing_obligations"] == 1600.0
    assert solo["living_expenses"] == 6200.0
    assert solo["dependants_count"] == 2.0


def test_a_household_is_not_assessed_on_one_applicants_income_alone(assess):
    out = assess(is_joint_application=True, applicant2_employment_type_code=1)
    assert out["applicant_income_evidence_gap"] == 2
    assert out["evidence_sufficiency_code"] == verdict.APPLICANT2_INCOME_UNESTABLISHED
    assert out["affordability_verdict_code"] == verdict.INDETERMINATE
    assert out["decline_reason_codes"] == [202]


def test_neither_applicants_income_established_is_its_own_code(assess):
    out = assess(is_joint_application=True, applicant1_payslip_income=None)
    assert (out["applicant_income_evidence_gap"], out["evidence_sufficiency_code"]) == (3, 3)


def test_a_small_side_income_at_a_weak_tier_does_not_set_the_households_tier(assess):
    small = assess(**{**SECOND, "applicant2_payslip_income": None, "applicant2_declared_income": 1000.0})
    assert small["income_verification_tier"] == 2          # 800 of 18 375 is under 10%: payslip tier stands
    large = assess(**{**SECOND, "applicant2_payslip_income": None, "applicant2_declared_income": 5000.0})
    assert large["income_verification_tier"] == 4          # 4 000 of 21 575 is over 10%: the weakest counts


def test_the_blended_haircut_weights_each_applicant_by_income_before_haircut(assess):
    out = assess(**{**SECOND, "applicant2_payslip_income": None, "applicant2_declared_income": 10000.0})
    # 18 500 at 5% and 10 000 at 20%: 1 - (17 575 + 8 000) / 28 500
    assert out["income_haircut_applied"] == round(1 - 25575 / 28500, 4)


def test_zero_declared_expenses_leave_the_norm_to_bind(assess):
    out = assess(applicant1_declared_expenses={})
    assert out["expense_basis_code"] == 2
    assert out["living_expenses"] == 3920.0                # the internal norm: 3 500 x 1.12


def test_no_expenses_supplied_is_not_a_zero_claim():
    assert math.isnan(household._household_expenses(False, {}, {}))
    assert household._household_expenses(False, {"food": 0.0}, {}) == 0.0
