"""Credit limit management pipeline (project 07).

Stages 5.1–5.8: population, exclusions, scoring, matrix, caps, affordability, ranking, allocation.
"""
from __future__ import annotations
from datetime import date
from typing import Optional
from decider import flow, param, missing_as
import sys
import os

# Add project 00 and 02 to path for consumption
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../00-shared-credit-core'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../02-affordability'))

from limit_management import (
    Account, evaluate_hard_exclusions, evaluate_behaviour_score,
    determine_utilisation_band, determine_mob_band, lookup_matrix_cell,
    apply_cycle_dial_overlay, apply_caps
)


def limit_decision(
    account_id: int = param(1, ge=1),
    product_code: int = param(20, ge=20, le=21),
    client_id: int = param(1, ge=1),
    current_limit: float = param(15000.0, ge=1000),
    current_balance: float = param(7500.0, ge=0),
    months_on_book: int = param(24, ge=6),
    revolving_utilisation_6m: float = param(0.50, ge=0, le=1),
    arrears_worst_3m: int = param(0, ge=0),
    arrears_worst_24m: int = param(0, ge=0),
    cycles_ever_past_due: int = param(0, ge=0),
    over_limit_count_12m: int = param(0, ge=0),
    cash_withdrawal_ratio_12m: float = param(0.20, ge=0, le=1),
    spend_percentile_90_6m: float = param(800.0, ge=0),
    has_auto_increase_consent: bool = param(True),
    treatment_state_code: int = param(0, ge=0),
    gross_monthly_income: float = param(10000.0, ge=0),
    existing_obligations: float = param(2000.0, ge=0),
    cycle_dial_multiplier: float = param(1.0, ge=0.5, le=1.0),
    income_multiple_cap_factor: float = param(4.0, ge=1),
    spend_cap_multiple: float = param(3.5, ge=1),
    budget_amount: float = param(2.4e9, ge=0),
    over_allocation_factor: float = param(1.38, ge=1),
    decision_date: date = missing_as(date.today()),
) -> dict:
    """Orchestrate all stages of limit management decision (§5.1–5.8)."""

    # Stage 5.1: Population and account state
    account = Account(
        account_id=account_id,
        product_code=product_code,
        client_id=client_id,
        current_limit=current_limit,
        current_balance=current_balance,
        months_on_book=months_on_book,
        revolving_utilisation_6m=revolving_utilisation_6m,
        arrears_worst_3m=arrears_worst_3m,
        arrears_worst_24m=arrears_worst_24m,
        cycles_ever_past_due=cycles_ever_past_due,
        over_limit_count_12m=over_limit_count_12m,
        cash_withdrawal_ratio_12m=cash_withdrawal_ratio_12m,
        spend_percentile_90_6m=spend_percentile_90_6m,
        has_auto_increase_consent=has_auto_increase_consent,
        treatment_state_code=treatment_state_code,
    )

    # Stage 5.2: Hard exclusions
    exclusion_result = evaluate_hard_exclusions(account, decision_date)
    if exclusion_result.is_excluded:
        return {
            'account_id': account_id,
            'is_excluded': True,
            'exclusion_codes': exclusion_result.exclusion_codes,
            'behaviour_grade': 0,
            'proposed_limit': current_limit,
            'affordability_verdict_code': 0,
            'allocation_outcome_code': 0,
            'funded': False,
        }

    # Stage 5.3: Behaviour scoring
    scoring_result = evaluate_behaviour_score(account, product_code, decision_date)

    # Stage 5.4: Matrix lookup
    util_band = determine_utilisation_band(revolving_utilisation_6m * 100)
    mob_band = determine_mob_band(months_on_book)
    matrix_data = _get_matrix_data()
    cell, cell_id = lookup_matrix_cell(scoring_result.behaviour_grade, util_band, mob_band, product_code, matrix_data)

    # Apply cycle dial
    cell_adjusted = apply_cycle_dial_overlay(cell, cycle_dial_multiplier)
    uncapped_target = current_limit * cell_adjusted.limit_multiplier

    # Stage 5.5: Caps
    params = {
        'income_multiple_cap_factor': income_multiple_cap_factor,
        'spend_cap_multiple': spend_cap_multiple,
    }
    caps_result = apply_caps(current_limit, uncapped_target, account, gross_monthly_income, params)

    # Stage 5.6: Affordability (stub)
    min_payment_rate = 0.035 if product_code == 20 else 0.05
    notional_instalment = caps_result.proposed_limit * min_payment_rate
    discretionary = gross_monthly_income - existing_obligations - notional_instalment
    affordability_verdict = 1 if discretionary >= 0 else 3  # 1=pass, 3=fail

    # Stage 5.8: Ranking (simplified)
    proposed_increase = caps_result.proposed_limit - current_limit
    if affordability_verdict == 1 and proposed_increase > 0:
        ccf = 0.42 if product_code == 20 else 0.55
        incremental_drawn = proposed_increase * ccf
        revenue = incremental_drawn * 0.09  # 8% NIM + 1% fees
        lgd = 0.74 if product_code == 20 else 0.79
        expected_loss = proposed_increase * scoring_result.probability_of_default * lgd
        ranking_value = (revenue - expected_loss) / proposed_increase if proposed_increase > 0 else 0
        can_rank = True
    else:
        ranking_value = -999999.0
        can_rank = False

    # Stage 5.8: Allocation (simplified single-account)
    offer_envelope = budget_amount * over_allocation_factor
    estimated_consumed = proposed_increase
    if can_rank and estimated_consumed <= offer_envelope:
        allocation_outcome = 1  # Funded
        funded = True
    else:
        allocation_outcome = 2  # Below line
        funded = False

    return {
        'account_id': account_id,
        'is_excluded': False,
        'exclusion_codes': [],
        'behaviour_score': scoring_result.behaviour_score,
        'behaviour_grade': scoring_result.behaviour_grade,
        'probability_of_default': scoring_result.probability_of_default,
        'matrix_cell_id': cell_id,
        'matrix_multiplier_unadjusted': cell.limit_multiplier,
        'matrix_multiplier_adjusted': cell_adjusted.limit_multiplier,
        'proposed_limit': caps_result.proposed_limit,
        'binding_cap_code': caps_result.binding_cap_code,
        'affordability_verdict_code': affordability_verdict,
        'ranking_value': round(ranking_value, 6),
        'allocation_rank': 1,
        'allocation_outcome_code': allocation_outcome,
        'funded': funded,
        'decision_date': decision_date.isoformat() if isinstance(decision_date, date) else decision_date,
    }


def build():
    """Build the credit limit management pipeline."""
    return flow(limit_decision, name="credit_limit_management")


def _get_matrix_data():
    """Return stub matrix data for cell lookups (§5.4)."""
    matrix = {}

    for grade in range(1, 13):
        for util_band in range(8):
            for mob_band in range(1, 7):
                for product in [20, 21]:
                    # Conservative scaling: higher grade = lower multiplier
                    base_mult = 1.6 - (grade * 0.05)
                    base_mult = max(1.0, min(1.75, base_mult))

                    # Adjust for utilisation: low util gets better terms
                    if util_band <= 3:
                        mult_adjustment = 0.1
                    elif util_band <= 5:
                        mult_adjustment = 0.0
                    else:
                        mult_adjustment = -0.2

                    multiplier = max(1.0, min(1.75, base_mult + mult_adjustment))

                    key = (grade, util_band, mob_band, product)
                    matrix[key] = {
                        'multiplier': round(multiplier, 2),
                        'max_absolute': 30000 if multiplier > 1.2 else 20000,
                        'min_increment': 500 if product == 20 else 500,
                        'cell_id': hash(key) & 0x7FFFFFFF,
                    }

    return matrix
