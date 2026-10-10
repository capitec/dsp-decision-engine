from assessment import verdict

QUOTED = {"account_type_code": 20, "balance": 5000.0, "limit": 0.0, "instalment": 800.0, "months_in_arrears": 0,
          "opened_date": "2021-01-01", "closed": False, "is_internal": False, "settlement_quote": 2400.0}


def test_a_settlement_quote_is_ignored_outside_a_scenario(assess):
    out = assess(applicant1_bureau_accounts=[QUOTED], applicant1_internal_accounts=[])
    assert out["obligation_treatment_codes"] == [1]        # costed at its stated instalment
    assert out["existing_obligations"] == 800.0


def test_a_scenario_assumes_a_quoted_account_is_settled(assess):
    out = assess(assessment_mode_code=4, applicant1_bureau_accounts=[QUOTED, QUOTED | {"account_type_code": 2,
                 "opened_date": "2020-05-05", "settlement_quote": None}], applicant1_internal_accounts=[])
    assert out["obligation_account_type_codes"] == [2]
    assert out["existing_obligations"] == 800.0
    assert out["total_exposure"] == 5000.0


def test_a_bureau_account_the_bank_also_holds_counts_once_with_the_internal_record_winning(assess, sample):
    twin = {**sample["applicant1_internal_accounts"][0], "instalment": 1000.0, "is_internal": False}
    out = assess(applicant1_bureau_accounts=[twin])
    assert out["obligation_monthly_amounts"] == [1150.0]
    assert out["obligations_internal"] == 1150.0 and out["obligations_external"] == 0.0


def test_a_closed_account_costs_nothing_but_stays_in_the_annotation(assess):
    out = assess(applicant1_bureau_accounts=[QUOTED | {"closed": True}], applicant1_internal_accounts=[])
    assert out["obligation_treatment_codes"] == [8] and out["obligation_monthly_amounts"] == [0.0]


def test_arrears_are_reported_but_do_not_change_the_obligation(assess, sample):
    late = sample["applicant1_bureau_accounts"][0] | {"months_in_arrears": 3}
    out = assess(applicant1_bureau_accounts=[late])
    assert (out["worst_arrears_months"], out["accounts_in_arrears_count"]) == (3, 1)
    assert out["existing_obligations"] == 1600.0


def test_an_account_type_nobody_can_cost_makes_the_verdict_indeterminate_not_a_fail(assess):
    mystery = {"account_type_code": 99, "balance": 100.0, "instalment": 50.0, "months_in_arrears": 0,
               "opened_date": "2020-01-01", "closed": False, "is_internal": False}
    out = assess(applicant1_bureau_accounts=[mystery], proposed_instalment=1_000_000.0)
    assert out["has_refer_account"] is True
    assert out["evidence_sufficiency_code"] == verdict.REFER_ACCOUNT
    assert out["affordability_verdict_code"] == verdict.INDETERMINATE


def test_no_accounts_at_all_is_a_clean_zero(assess):
    out = assess(applicant1_bureau_accounts=[], applicant1_internal_accounts=[])
    assert out["existing_obligations"] == 0.0 and out["obligation_treatment_codes"] == []
