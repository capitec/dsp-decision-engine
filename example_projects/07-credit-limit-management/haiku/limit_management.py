"""Credit limit management core logic.

Stages 5.1–5.8 of the limit management programme:
- Population and account state
- Hard exclusions
- Behaviour scoring and grading
- Assignment matrix lookup with overlays
- Caps application
- Affordability reassessment
- Portfolio budget allocation

Simulation uses the same implementation as production (§5.10, AC3).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, List, Dict
import math


@dataclass
class Account:
    """Per-account state snapshot (§5.1)."""
    account_id: int
    product_code: int  # 20=Card, 21=Facility
    client_id: int
    current_limit: float
    current_balance: float
    months_on_book: int
    revolving_utilisation_6m: float  # 6-month mean
    arrears_worst_3m: int  # worst delinquency in 3 months (0 = none)
    arrears_worst_24m: int
    cycles_ever_past_due: int
    over_limit_count_12m: int
    cash_withdrawal_ratio_12m: float
    spend_percentile_90_6m: float  # trailing 90th-percentile monthly spend
    last_limit_change_date: Optional[date] = None
    has_auto_increase_consent: bool = True
    treatment_state_code: int = 0  # 0 = performing, stub for X16 exclusion


@dataclass
class ExclusionResult:
    """Hard exclusion verdict (§5.2)."""
    is_excluded: bool
    exclusion_codes: List[int] = field(default_factory=list)  # X01–X16


@dataclass
class ScoringResult:
    """Behaviour scoring and grading (§5.3)."""
    behaviour_score: float
    probability_of_default: float
    behaviour_grade: int  # 1..12, lower is better
    score_unadjusted: float
    probability_of_default_unadjusted: float


def determine_utilisation_band(utilisation: float) -> int:
    """Map 6-month mean utilisation to band 0..7.

    Bands: [0%], [0-10%), [10-25%), [25-40%), [40-55%), [55-70%), [70-90%), [90%+]
    Lower bound closed; upper open.
    """
    if utilisation < 0:
        return 0
    elif utilisation < 10:
        return 1
    elif utilisation < 25:
        return 2
    elif utilisation < 40:
        return 3
    elif utilisation < 55:
        return 4
    elif utilisation < 70:
        return 5
    elif utilisation < 90:
        return 6
    else:
        return 7


def determine_mob_band(months: int) -> int:
    """Map months on book to band 1..6.

    Bands: [6-11], [12-17], [18-23], [24-35], [36-59], [60+]
    """
    if months < 6:
        return 0  # ineligible
    elif months < 12:
        return 1
    elif months < 18:
        return 2
    elif months < 24:
        return 3
    elif months < 36:
        return 4
    elif months < 60:
        return 5
    else:
        return 6


def evaluate_hard_exclusions(account: Account, decision_date: date) -> ExclusionResult:
    """Evaluate 16 hard exclusion rules (§5.2, simplified to X01–X05, X08–X09, X12–X14).

    Stub X16 (arrangement state) with stub treatment_state_code.
    """
    codes = []

    # X01: In arrears now
    if account.arrears_worst_3m > 0:
        codes.append(1)

    # X02: Arrears within 6 months (approximated via worst in 24m with age check)
    if account.arrears_worst_24m > 0 and account.cycles_ever_past_due > 0:
        codes.append(2)

    # X03–X07: Debt review, insolvency, deceased, fraud, dispute - stub as not present
    # (would come from marker feeds)

    # X08: Dormant (balance = 0 for 6+ months) - stub as not present

    # X09: Cooling-off period - stub as not active

    # X12: No consent for automatic increases
    if not account.has_auto_increase_consent:
        codes.append(12)

    # X13: Staff account - stub as not present

    # X14: Too young (< 6 months on book)
    if account.months_on_book < 6:
        codes.append(14)

    # X15: Already at product maximum (Card R300k, Facility R150k)
    product_max = 300000 if account.product_code == 20 else 150000
    if account.current_limit >= product_max:
        codes.append(15)

    # X16: Arrangement state - stub with treatment_state_code
    if account.treatment_state_code > 0:
        codes.append(16)

    return ExclusionResult(
        is_excluded=len(codes) > 0,
        exclusion_codes=codes
    )


def evaluate_behaviour_score(
    account: Account, product_code: int, decision_date: date
) -> ScoringResult:
    """Evaluate behaviour score and derive grade (§5.3, simplified).

    Stub scorecard with synthetic deterministic scoring based on account
    characteristics. Real implementation would use pre-fitted model.
    """
    # Synthetic score: base 600 + characteristics
    score = 600.0

    # Delinquency: worst 24m delinquency
    score -= min(account.arrears_worst_24m * 40, 300)

    # Payment behaviour: cycles past due
    score -= min(account.cycles_ever_past_due * 20, 150)

    # Utilisation: high utilisation is risk
    util = account.revolving_utilisation_6m
    if util > 90:
        score -= 100
    elif util > 70:
        score -= 60
    elif util > 50:
        score -= 20
    else:
        score += 50 if util < 20 else 20

    # Tenure: longer tenure is safer
    mob = account.months_on_book
    if mob < 12:
        score -= 80
    elif mob < 24:
        score -= 40
    elif mob > 60:
        score += 40

    # Over-limit: persistent over-limit is risk
    if account.over_limit_count_12m > 3:
        score -= 80

    # Cash withdrawal: high cash ratio is risk
    if account.cash_withdrawal_ratio_12m > 0.45:
        score -= 60

    # Bound score to [300, 850]
    score = max(300, min(850, score))

    # Calibrate to PD (simplified: score to probability mapping)
    # Higher score = lower PD
    pd_base = math.exp(-(score - 500) / 100) / (1 + math.exp(-(score - 500) / 100))
    pd_scaled = 0.001 + (pd_base * 0.15)  # Scale to [0.1%, 15%] range
    pd_scaled = max(0.0001, min(0.9999, pd_scaled))

    # Grade: 1 (best) to 12 (worst), based on PD buckets
    if pd_scaled < 0.005:
        grade = 1
    elif pd_scaled < 0.01:
        grade = 2
    elif pd_scaled < 0.02:
        grade = 3
    elif pd_scaled < 0.04:
        grade = 4
    elif pd_scaled < 0.07:
        grade = 5
    elif pd_scaled < 0.10:
        grade = 6
    elif pd_scaled < 0.15:
        grade = 7
    elif pd_scaled < 0.25:
        grade = 8
    elif pd_scaled < 0.40:
        grade = 9
    elif pd_scaled < 0.60:
        grade = 10
    elif pd_scaled < 0.80:
        grade = 11
    else:
        grade = 12

    return ScoringResult(
        behaviour_score=round(score, 2),
        probability_of_default=round(pd_scaled, 6),
        behaviour_grade=grade,
        score_unadjusted=round(score, 2),
        probability_of_default_unadjusted=round(pd_scaled, 6)
    )


@dataclass
class MatrixCell:
    """Assignment matrix cell (§5.4)."""
    limit_multiplier: float  # 1.00–1.75
    max_absolute_increase: float  # R0–R60k
    minimum_increment: float  # R500–R2.5k


def lookup_matrix_cell(
    behaviour_grade: int, utilisation_band: int, mob_band: int, product_code: int,
    matrix_data: Dict
) -> tuple[MatrixCell, int]:
    """Look up single matrix cell (§5.4).

    Returns (cell, cell_id).
    """
    # Stub matrix: use dict keyed by (grade, util_band, mob_band, product_code)
    key = (behaviour_grade, utilisation_band, mob_band, product_code)

    if key in matrix_data:
        cell_dict = matrix_data[key]
        cell = MatrixCell(
            limit_multiplier=cell_dict['multiplier'],
            max_absolute_increase=cell_dict['max_absolute'],
            minimum_increment=cell_dict['min_increment']
        )
        cell_id = cell_dict.get('cell_id', hash(key) & 0x7FFFFFFF)
    else:
        # Default conservative cell if missing
        cell = MatrixCell(
            limit_multiplier=1.0,
            max_absolute_increase=0,
            minimum_increment=500
        )
        cell_id = 0

    return cell, cell_id


def apply_cycle_dial_overlay(
    cell: MatrixCell, dial_multiplier: float
) -> MatrixCell:
    """Apply cycle dial overlay (§5.4 "The cycle dial").

    Applied to excess over 1.00: cell.multiplier of 1.50 becomes 1.40 at 80% dial.
    """
    excess = cell.limit_multiplier - 1.0
    adjusted_excess = excess * dial_multiplier
    adjusted_multiplier = 1.0 + adjusted_excess

    return MatrixCell(
        limit_multiplier=adjusted_multiplier,
        max_absolute_increase=cell.max_absolute_increase,
        minimum_increment=cell.minimum_increment
    )


@dataclass
class CapsResult:
    """Caps computation result (§5.5)."""
    proposed_limit: float
    binding_cap_code: int  # 0=none, 1..7
    cap_values: Dict[str, float]  # all cap computed values


def apply_caps(
    current_limit: float,
    uncapped_target: float,
    account: Account,
    income: float,
    params: Dict
) -> CapsResult:
    """Apply cap waterfall (§5.5): six caps, lowest wins.

    Caps: C1 product max, C2 income-multiple, C3 exposure, C4 group, C5 spend, C6 matrix, C7 affordability.
    Simplified to C1, C2, C5 for this implementation.
    """
    caps = {}

    # C1: Product maximum
    product_max = 300000 if account.product_code == 20 else 150000
    caps['C1_product_max'] = product_max

    # C2: Income-multiple (simplified: no segment logic)
    income_mult_factor = params.get('income_multiple_cap_factor', 4.0)
    caps['C2_income_multiple'] = income * income_mult_factor

    # C5: Observed spend cap (§5.5)
    spend_multiple = params.get('spend_cap_multiple', 3.5)
    spend_cap = max(current_limit, spend_multiple * account.spend_percentile_90_6m, 2000)
    caps['C5_observed_spend'] = spend_cap

    # Find binding cap (lowest)
    binding_value = min(product_max, caps['C2_income_multiple'], spend_cap)
    binding_cap = 0

    if binding_value == product_max:
        binding_cap = 1
    elif binding_value == caps['C2_income_multiple']:
        binding_cap = 2
    elif binding_value == spend_cap:
        binding_cap = 5

    # Proposed limit is minimum of uncapped and binding cap, rounded down to R500
    proposed = min(uncapped_target, binding_value)
    proposed = math.floor(proposed / 500) * 500

    return CapsResult(
        proposed_limit=proposed,
        binding_cap_code=binding_cap,
        cap_values=caps
    )


@dataclass
class AllocationResult:
    """Allocation outcome (§5.8)."""
    allocation_rank: int  # 0 if not ranked, else position
    ranking_value: float
    allocation_outcome_code: int  # 1=funded, 2=below line, 3=fairness-capped, 4=tail-skipped
    reason_codes: List[int] = field(default_factory=list)


def allocate_budget(
    ranked_accounts: List[Dict],
    budget: float,
    over_allocation_factor: float,
    params: Dict
) -> Dict[int, AllocationResult]:
    """Allocate portfolio budget to accounts (§5.8).

    Simplified: rank by ranking_value, fund in order until budget exhausted.
    """
    offer_envelope = budget * over_allocation_factor
    consumed = 0.0
    funded = 0
    results = {}

    for rank, account_data in enumerate(ranked_accounts, 1):
        account_id = account_data['account_id']
        proposed_increase = account_data['proposed_increase']
        ranking_value = account_data['ranking_value']

        # Check if this account fits in budget
        if consumed + proposed_increase <= offer_envelope:
            # Funded
            consumed += proposed_increase
            funded += 1
            outcome = 1  # Funded
        else:
            # Below the funding line
            outcome = 2

        results[account_id] = AllocationResult(
            allocation_rank=rank,
            ranking_value=ranking_value,
            allocation_outcome_code=outcome
        )

    return results
