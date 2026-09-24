"""Stage 1 (spec 02 §5.1): household framing -- standalone, no pipeline (§7.5 style)."""
from datetime import date

from decider import Engine

from assessment import household


def test_dependants_count_takes_the_higher_declaration():
    assert household.dependants_count(applicant1_dependants_count=2, applicant2_dependants_count=4) == 4
    assert household.dependants_count(applicant1_dependants_count=3) == 3


def test_expense_consolidation_takes_higher_of_shared_and_sum_of_personal():
    a1 = {"accommodation": 3000.0, "food": 1500.0, "transport": 800.0}
    a2 = {"accommodation": 3500.0, "food": 1000.0, "transport": 600.0}
    total = household._consolidate(a1, a2)
    # shared: max(3000,3500) + max(1500,1000) = 3500 + 1500 = 5000 (water/insurance both 0)
    # personal: 800 + 600 = 1400 (medical/communication both 0)
    assert total == 6400.0


def test_solo_application_consolidation_equals_the_one_applicants_total():
    a1 = {"accommodation": 3000.0, "food": 1500.0, "transport": 800.0}
    assert household._consolidate(a1, {}) == household._consolidate(a1, None)
    assert household._consolidate(a1, {}) == 5300.0


def test_applicant_income_evidence_gap_names_which_applicant_and_ignores_solo_application():
    from credit_core.income import NONE_ESTABLISHED, PAYSLIP
    assert household.applicant_income_evidence_gap(True, NONE_ESTABLISHED, PAYSLIP) == 1
    assert household.applicant_income_evidence_gap(True, PAYSLIP, NONE_ESTABLISHED) == 2
    assert household.applicant_income_evidence_gap(True, NONE_ESTABLISHED, NONE_ESTABLISHED) == 3
    assert household.applicant_income_evidence_gap(True, PAYSLIP, PAYSLIP) == 0
    # A solo application never blames applicant 2 for having no evidence -- there is no applicant 2.
    assert household.applicant_income_evidence_gap(False, PAYSLIP, NONE_ESTABLISHED) == 0


def test_a_joint_account_reported_on_both_applicants_bureau_profiles_counts_once():
    """§5.1: "the commonest joint-application defect" -- same account, same identity key,
    arriving under both applicants, must not double the obligation."""
    account = {"account_type_code": 10, "balance": 8000.0, "limit": 15000.0, "instalment": 450.0,
               "months_in_arrears": 0, "opened_date": date(2022, 3, 1), "closed": False, "is_internal": False}
    exe = Engine().bind(household.household_obligations, mode="interpreted")
    out = exe.score({
        "applicant1_bureau_accounts": [account], "applicant2_bureau_accounts": [dict(account)],
        "applicant1_internal_accounts": [], "applicant2_internal_accounts": [],
        "assessment_mode_code": 1,
    })
    assert out["obligation_account_type_codes"] == [10]
    assert out["existing_obligations"] == 450.0


def test_settlement_quote_only_zeroes_the_obligation_in_scenario_mode():
    """§5.5.2 `EXCLUDE_ON_QUOTE`: "only in scenario mode... never in new-application mode"."""
    account = {"account_type_code": 20, "balance": 5000.0, "limit": 0.0, "instalment": 800.0,
               "months_in_arrears": 0, "opened_date": date(2021, 1, 1), "closed": False, "is_internal": False,
               "settlement_quote": 2400.0}
    exe = Engine().bind(household.household_obligations, mode="interpreted")
    common = {"applicant1_bureau_accounts": [account], "applicant2_bureau_accounts": [],
              "applicant1_internal_accounts": [], "applicant2_internal_accounts": []}

    new_application = exe.score({**common, "assessment_mode_code": 1})
    scenario = exe.score({**common, "assessment_mode_code": 4})

    assert new_application["obligation_treatment_codes"] == [1]      # USE_STATED_INSTALMENT: quote stripped
    assert new_application["existing_obligations"] == 800.0
    assert scenario["obligation_treatment_codes"] == [6]             # USE_SETTLEMENT_QUOTE
    assert scenario["existing_obligations"] != 800.0


def test_refer_account_type_is_flagged():
    account = {"account_type_code": 99, "balance": 100.0, "instalment": 50.0,
               "months_in_arrears": 0, "opened_date": date(2020, 1, 1), "closed": False, "is_internal": False}
    exe = Engine().bind(household.household_obligations, mode="interpreted")
    out = exe.score({"applicant1_bureau_accounts": [account], "applicant2_bureau_accounts": [],
                      "applicant1_internal_accounts": [], "applicant2_internal_accounts": [],
                      "assessment_mode_code": 1})
    assert out["has_refer_account"] is True
