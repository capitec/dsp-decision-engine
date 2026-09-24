"""`core.income` -- income determination (spec 00 §6.1).

The evidence waterfall (which tier of income evidence wins, and the
haircut per tier) is a policy artefact Credit Risk Policy changes without
an engineer; the arithmetic once a tier is chosen is not. This module
keeps the waterfall's *order and haircuts* as `param()`s and the tier
*selection logic* as plain Python, which is 00 §6.1's "Hard part" answered
directly: the line falls between a param table (what wins, what the
haircut is) and code (how a haircut is applied), not between two
capabilities.

Missing evidence is `float("nan")`, distinct from a verified R0 income
(00 §7.4 / vocab.NullKind: `NOT_COLLECTED` vs. `COLLECTED_ZERO`) -- a
consumer that reads `gross_monthly_income` never sees the difference, but
`income_source_code` and this module's internal tier check do.
"""
from __future__ import annotations

import math
import statistics

from decider import missing_as, param

# income_source_code
EMPLOYER_CONFIRMED = 1
PAYSLIP = 2
BANK_STATEMENT = 3
DECLARED = 4
BUREAU_ESTIMATED = 5
NONE_ESTABLISHED = 0

_NA = float("nan")


def _present(x: float) -> bool:
    return not math.isnan(x)


def gross_monthly_income(
    declared_income: float = missing_as(_NA),
    payslip_income: float = missing_as(_NA),
    statement_income: float = missing_as(_NA),
    statement_confidence: float = missing_as(_NA),
    employer_confirmed_income: float = missing_as(_NA),
    bureau_estimated_income: float = missing_as(_NA),
    statement_confidence_threshold: float = param(0.7, ge=0.0, le=1.0),
    haircut_payslip: float = param(0.05, ge=0.0, le=1.0),
    haircut_statement_high_confidence: float = param(0.10, ge=0.0, le=1.0),
    haircut_statement_low_confidence: float = param(0.25, ge=0.0, le=1.0),
    haircut_declared: float = param(0.20, ge=0.0, le=1.0),
    haircut_bureau_estimated: float = param(0.30, ge=0.0, le=1.0),
) -> float:
    raw, _tier, haircut = _select_tier(
        declared_income, payslip_income, statement_income, statement_confidence, employer_confirmed_income,
        bureau_estimated_income, statement_confidence_threshold, haircut_payslip,
        haircut_statement_high_confidence, haircut_statement_low_confidence, haircut_declared,
        haircut_bureau_estimated,
    )
    return round(raw * (1.0 - haircut), 2)


def _select_tier(declared, payslip, statement, statement_confidence, employer_confirmed, bureau_estimated,
                  confidence_threshold, hc_payslip, hc_stmt_hi, hc_stmt_lo, hc_declared, hc_bureau):
    """The waterfall, best evidence first. Returns (raw_value, tier_code, haircut)."""
    if _present(employer_confirmed):
        return employer_confirmed, EMPLOYER_CONFIRMED, 0.0
    if _present(payslip):
        return payslip, PAYSLIP, hc_payslip
    if _present(statement):
        confident = _present(statement_confidence) and statement_confidence >= confidence_threshold
        return statement, BANK_STATEMENT, (hc_stmt_hi if confident else hc_stmt_lo)
    if _present(declared):
        return declared, DECLARED, hc_declared
    if _present(bureau_estimated):
        return bureau_estimated, BUREAU_ESTIMATED, hc_bureau
    return 0.0, NONE_ESTABLISHED, 0.0


def income_source_code(
    declared_income: float = missing_as(_NA), payslip_income: float = missing_as(_NA),
    statement_income: float = missing_as(_NA), statement_confidence: float = missing_as(_NA),
    employer_confirmed_income: float = missing_as(_NA), bureau_estimated_income: float = missing_as(_NA),
    statement_confidence_threshold: float = param(0.7, ge=0.0, le=1.0),
) -> int:
    _, tier, _ = _select_tier(
        declared_income, payslip_income, statement_income, statement_confidence, employer_confirmed_income,
        bureau_estimated_income, statement_confidence_threshold, 0.0, 0.0, 0.0, 0.0, 0.0,
    )
    return tier


def income_verification_tier(income_source_code: int) -> int:
    """Alias kept distinct from `income_source_code` for consumers that key evidence on it directly."""
    return income_source_code


def income_haircut_applied(
    declared_income: float = missing_as(_NA), payslip_income: float = missing_as(_NA),
    statement_income: float = missing_as(_NA), statement_confidence: float = missing_as(_NA),
    employer_confirmed_income: float = missing_as(_NA), bureau_estimated_income: float = missing_as(_NA),
    statement_confidence_threshold: float = param(0.7, ge=0.0, le=1.0),
    haircut_payslip: float = param(0.05, ge=0.0, le=1.0),
    haircut_statement_high_confidence: float = param(0.10, ge=0.0, le=1.0),
    haircut_statement_low_confidence: float = param(0.25, ge=0.0, le=1.0),
    haircut_declared: float = param(0.20, ge=0.0, le=1.0),
    haircut_bureau_estimated: float = param(0.30, ge=0.0, le=1.0),
) -> float:
    _, _, haircut = _select_tier(
        declared_income, payslip_income, statement_income, statement_confidence, employer_confirmed_income,
        bureau_estimated_income, statement_confidence_threshold, haircut_payslip,
        haircut_statement_high_confidence, haircut_statement_low_confidence, haircut_declared,
        haircut_bureau_estimated,
    )
    return haircut


def income_variability_ratio(variable_pay_history: list[float] = missing_as([])) -> float:
    """Coefficient of variation of up to 12 monthly variable-pay values (a ragged, ≤12-element list).

    `variable_pay_history` arrives as a numpy array in compiled/kernel
    paths, so it is checked with `is None`, never a bare truthiness test
    (`array or []` raises "truth value of an array... is ambiguous").
    """
    history = [] if variable_pay_history is None else variable_pay_history
    values = [v for v in history if v is not None]
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return round(statistics.pstdev(values) / mean, 4)
