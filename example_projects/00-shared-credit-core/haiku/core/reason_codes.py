"""Reason code registry with ranking and primary reason (00 §6.21)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReasonCode:
    """A registered reason code with metadata."""
    code: int
    description: str
    severity: int  # 1 (highest) to 10 (lowest)
    category: str  # e.g., "affordability", "fraud", "credit_history"


# Registry mapping code → ReasonCode
REGISTRY: dict[int, ReasonCode] = {
    1001: ReasonCode(1001, "Insufficient income", 1, "affordability"),
    1002: ReasonCode(1002, "Excessive existing obligations", 1, "affordability"),
    1003: ReasonCode(1003, "Low credit score", 2, "credit_history"),
    1004: ReasonCode(1004, "Delinquency history", 1, "credit_history"),
    1005: ReasonCode(1005, "Insufficient employment tenure", 3, "employment"),
    1006: ReasonCode(1006, "Adverse credit events", 2, "credit_history"),
    1007: ReasonCode(1007, "Bureau data not available", 4, "bureau"),
    1008: ReasonCode(1008, "Income verification failed", 3, "income"),
    2001: ReasonCode(2001, "Approved", 10, "outcome"),
    2002: ReasonCode(2002, "Refer for manual review", 5, "outcome"),
    2003: ReasonCode(2003, "Approved with conditions", 9, "outcome"),
}


def rank_reasons(reason_codes: list[int]) -> list[int]:
    """
    Return reason codes ranked by severity (highest first).
    Implements 09 §5.15 item 11: declared reason codes in severity order.
    """
    reasons = [(REGISTRY[code].severity, code) for code in reason_codes if code in REGISTRY]
    reasons.sort()  # Sort by severity (lower severity value = more severe)
    return [code for _, code in reasons]


def get_primary_reason(reason_codes: list[int]) -> Optional[int]:
    """
    Return the primary (most severe) reason code.
    """
    ranked = rank_reasons(reason_codes)
    return ranked[0] if ranked else None
