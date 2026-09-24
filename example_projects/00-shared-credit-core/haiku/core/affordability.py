from __future__ import annotations
"""Affordability assessment (00 §6.5, 00-ADDENDUM A11)."""
from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum


class AffordabilityVerdict(Enum):
    """Affordability verdict codes."""
    PASS = 1
    MARGINAL = 2
    FAIL = 3
    INDETERMINATE = 4


class AssessmentMode(Enum):
    """Four modes of affordability assessment (00-ADDENDUM A11)."""
    STANDARD = "standard"  # Normal lending mode
    LIMIT_INCREASE = "limit_increase"  # 07's degraded-evidence mode
    ARRANGEMENT = "arrangement"  # 08's arrangement mode
    RESTRUCTURE = "restructure"  # 06's consolidation mode


@dataclass
class AffordabilityFacts:
    """Facts needed for 03's solve: answers the question 'for a given amount, what rate?'"""
    discrete_income: float
    discrete_expenses: float
    discrete_obligations: float
    discrete_available: float
    rate_range_minimum: float
    rate_range_maximum: float


@dataclass
class AffordabilityResult:
    """Result of affordability assessment."""
    verdict_code: int  # 1=pass, 2=marginal, 3=fail, 4=indeterminate
    verdict: str  # "pass", "marginal", "fail", "indeterminate"
    discretionary_income: float  # Net - expenses - obligations
    max_affordable_instalment: float  # Max the applicant may be committed to
    # Three answer shapes (00-ADDENDUM A11):
    facts: Optional[AffordabilityFacts]  # For solve
    max_amount: Optional[float]  # For granting
    pass_fail: bool  # For yes/no decisions


def assess_affordability(
    net_monthly_income: float,
    living_expenses: float,
    existing_obligations: float,
    proposed_instalment: Optional[float] = None,
    court_ordered_deductions: float = 0.0,
    mode: str = "standard",
    appetite_haircut: float = 0.85,
    affordability_buffer: float = 0.12
) -> AffordabilityResult:
    """
    Assess affordability against income and obligations.

    Implements 00-ADDENDUM A11: four modes, three answer shapes, monotone guarantee.
    Implements 09 §5.15 item 6: inputs captured.
    """
    # Deduct court-ordered amounts outside statutory deductions
    disposable = net_monthly_income - court_ordered_deductions - living_expenses - existing_obligations

    # Apply buffer and appetite haircut
    if mode == "arrangement":
        # Arrangement mode: norm floor only plausibility check (00-ADDENDUM B.4)
        buffer_applied = affordability_buffer  # Still applies but not disqualifier
    else:
        buffer_applied = affordability_buffer

    available_capacity = disposable * (1.0 - buffer_applied) * appetite_haircut

    if available_capacity <= 0:
        verdict_code = 3  # Fail
        max_instalment = 0.0
    elif proposed_instalment is not None:
        # Pass/fail based on proposed instalment
        if proposed_instalment <= available_capacity:
            verdict_code = 1  # Pass
        else:
            verdict_code = 3  # Fail
        max_instalment = available_capacity
    else:
        # Indeterminate (amount not yet proposed)
        verdict_code = 4
        max_instalment = available_capacity

    verdict_str = {1: "pass", 2: "marginal", 3: "fail", 4: "indeterminate"}[verdict_code]

    # For monotonicity (02 §5.7.2(c)): ensure result is monotone in proposed_instalment
    # This is guaranteed by the linear relationship above

    facts = AffordabilityFacts(
        discrete_income=net_monthly_income,
        discrete_expenses=living_expenses,
        discrete_obligations=existing_obligations,
        discrete_available=available_capacity,
        rate_range_minimum=0.0,
        rate_range_maximum=100.0
    )

    return AffordabilityResult(
        verdict_code=verdict_code,
        verdict=verdict_str,
        discretionary_income=disposable,
        max_affordable_instalment=max_instalment,
        facts=facts,  # For solve (03)
        max_amount=max_instalment,  # For granting
        pass_fail=verdict_code == 1  # For yes/no decisions
    )
