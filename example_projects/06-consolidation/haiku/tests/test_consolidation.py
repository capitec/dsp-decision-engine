"""Tests for consolidation assessment."""
import pytest
from datetime import date
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consolidation import (
    classify_settleability,
    derive_settlement_amount,
    baseline_assessment,
    generate_candidate_scenarios,
    assess_consolidation,
    Account,
    SettleabilityCode,
    PRODUCT_FLEX_CONSOLIDATION,
    PRODUCT_BALANCE_TRANSFER,
)


def test_settleability_internal():
    """Internal accounts should be settleable."""
    acc = Account(
        account_ref=1,
        account_type_code=10,
        provider_code=1,
        is_internal=True,
        balance=10000.0,
        instalment=300.0,
        nominal_annual_rate=12.0,
        remaining_term_months=24,
        months_in_arrears=0,
        opened_date=date(2023, 1, 1),
        is_secured=False,
        account_status_code=1,
        is_disputed=False,
    )
    code = classify_settleability(acc, date(2024, 9, 24))
    assert code == SettleabilityCode.INTERNAL


def test_settleability_disputed():
    """Disputed accounts should be blocked."""
    acc = Account(
        account_ref=1,
        account_type_code=10,
        provider_code=1,
        is_internal=False,
        balance=10000.0,
        instalment=300.0,
        nominal_annual_rate=12.0,
        remaining_term_months=24,
        months_in_arrears=0,
        opened_date=date(2023, 1, 1),
        is_secured=False,
        account_status_code=1,
        is_disputed=True,
    )
    code = classify_settleability(acc, date(2024, 9, 24))
    assert code == SettleabilityCode.BLOCKED_BY_STATUS


def test_settleability_new_account():
    """Newly opened accounts should be blocked."""
    acc = Account(
        account_ref=1,
        account_type_code=10,
        provider_code=1,
        is_internal=False,
        balance=10000.0,
        instalment=300.0,
        nominal_annual_rate=12.0,
        remaining_term_months=24,
        months_in_arrears=0,
        opened_date=date(2024, 8, 1),  # Less than 3 months old
        is_secured=False,
        account_status_code=1,
        is_disputed=False,
    )
    code = classify_settleability(acc, date(2024, 9, 24))
    assert code == SettleabilityCode.BLOCKED_BY_POLICY


def test_settleability_revolving():
    """Revolving accounts should be partially settleable."""
    acc = Account(
        account_ref=1,
        account_type_code=20,  # Revolving
        provider_code=1,
        is_internal=False,
        balance=5000.0,
        instalment=150.0,
        nominal_annual_rate=22.0,
        remaining_term_months=None,
        months_in_arrears=0,
        opened_date=date(2023, 1, 1),
        is_secured=False,
        account_status_code=1,
        is_disputed=False,
    )
    code = classify_settleability(acc, date(2024, 9, 24))
    assert code == SettleabilityCode.PARTIALLY_SETTLEABLE


def test_settlement_amount():
    """Settlement amount should include balance + buffer."""
    acc = Account(
        account_ref=1,
        account_type_code=10,
        provider_code=1,
        is_internal=True,
        balance=10000.0,
        instalment=300.0,
        nominal_annual_rate=12.0,
        remaining_term_months=24,
        months_in_arrears=0,
        opened_date=date(2023, 1, 1),
        is_secured=False,
        account_status_code=1,
        is_disputed=False,
    )
    amount = derive_settlement_amount(acc, date(2024, 9, 24))
    # Should be: 10000 + (10000 * 0.015) = 10150
    assert amount == pytest.approx(10150.0, rel=0.01)


def test_settlement_amount_high_rate():
    """Settlement amount should include early settlement charge for high-rate accounts."""
    acc = Account(
        account_ref=1,
        account_type_code=10,
        provider_code=1,
        is_internal=False,
        balance=10000.0,
        instalment=300.0,
        nominal_annual_rate=28.5,  # High rate
        remaining_term_months=24,
        months_in_arrears=0,
        opened_date=date(2023, 1, 1),
        is_secured=False,
        account_status_code=1,
        is_disputed=False,
    )
    amount = derive_settlement_amount(acc, date(2024, 9, 24))
    # Should include 2% early settlement charge
    # 10000 + 150 + 200 = 10350
    assert amount > 10150.0


def test_baseline_assessment():
    """Baseline should compute current state correctly."""
    accounts = [
        Account(
            account_ref=1,
            account_type_code=10,
            provider_code=1,
            is_internal=True,
            balance=25000.0,
            instalment=800.0,
            nominal_annual_rate=12.5,
            remaining_term_months=24,
            months_in_arrears=0,
            opened_date=date(2023, 1, 1),
            is_secured=False,
            account_status_code=1,
            is_disputed=False,
        ),
        Account(
            account_ref=2,
            account_type_code=10,
            provider_code=2,
            is_internal=False,
            balance=15000.0,
            instalment=450.0,
            nominal_annual_rate=22.0,
            remaining_term_months=36,
            months_in_arrears=0,
            opened_date=date(2022, 6, 1),
            is_secured=False,
            account_status_code=1,
            is_disputed=False,
        ),
    ]

    baseline = baseline_assessment(
        accounts,
        gross_income=18000.0,
        net_income=13500.0,
        existing_obligations=2500.0,
        decision_date=date(2024, 9, 24),
        requested_amount=60000.0,
    )

    assert baseline["current_instalment"] == pytest.approx(1250.0, rel=0.01)
    assert baseline["debt_service_ratio"] == pytest.approx(2500.0 / 13500.0, rel=0.01)


def test_candidate_generation():
    """Should generate scenarios within budget."""
    accounts = [
        Account(
            account_ref=i,
            account_type_code=10,
            provider_code=i,
            is_internal=False,
            balance=10000.0,
            instalment=300.0,
            nominal_annual_rate=12.0 + i,
            remaining_term_months=24,
            months_in_arrears=0,
            opened_date=date(2023, 1, 1),
            is_secured=False,
            account_status_code=1,
            is_disputed=False,
        )
        for i in range(1, 5)
    ]

    # Classify
    for acc in accounts:
        acc.settleability_code = SettleabilityCode.QUOTATION_OBTAINABLE
        acc.settlement_amount = 10150.0

    candidates = generate_candidate_scenarios(accounts, accounts, budget=100)

    assert len(candidates) <= 100
    assert len(candidates) > 0
    # Should include empty set (baseline)
    assert any(len(s.settlement_set) == 0 for s in candidates)


def test_assess_consolidation_basic():
    """Integration test: assess_consolidation should return valid result."""
    request = {
        "client_id": 12345,
        "application_id": 54321,
        "decision_date": date(2024, 9, 24),
        "channel_code": 1,
        "requested_amount": 60000.0,
        "assessment_mode_code": 1,
        "client_nominated_settle": [],
        "client_excluded_settle": [],
        "hardship_declared": False,
        "gross_monthly_income": 18000.0,
        "net_monthly_income": 13500.0,
        "living_expenses": 5000.0,
        "accounts": [
            {
                "account_ref": 1001,
                "account_type_code": 10,
                "provider_code": 100,
                "is_internal": True,
                "balance": 25000.0,
                "instalment": 800.0,
                "nominal_annual_rate": 12.5,
                "remaining_term_months": 24,
                "months_in_arrears": 0,
                "opened_date": "2022-01-15",
                "is_secured": False,
                "account_status_code": 1,
                "is_disputed": False,
            },
            {
                "account_ref": 1002,
                "account_type_code": 20,
                "provider_code": 101,
                "is_internal": False,
                "balance": 15000.0,
                "instalment": 450.0,
                "nominal_annual_rate": 22.0,
                "remaining_term_months": 36,
                "months_in_arrears": 0,
                "opened_date": "2021-06-10",
                "is_secured": False,
                "account_status_code": 1,
                "is_disputed": False,
            },
        ],
        "scenario_budget": 100,
        "objective_id": 2,
        "policy_thresholds": {
            "con_int_01_max_accounts": 8,
            "con_int_02_months_since_open": 3,
        },
    }

    result = assess_consolidation(request)

    assert "decision_id" in result
    assert result["client_id"] == 12345
    assert result["is_eligible"] == True
    assert result["settleable_count"] >= 1
    assert result["scenarios_evaluated"] <= result["scenarios_budget"]
    assert "winner" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
