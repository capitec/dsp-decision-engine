from __future__ import annotations
"""Risk grade: 1 (best) to 12 (worst) (00 §6.12)."""
from dataclasses import dataclass


@dataclass
class RiskGradeResult:
    """Result of risk grading."""
    risk_grade: int  # 1 (best) to 12 (worst)
    grade_bucket: str  # "prime", "near_prime", "subprime", "deep_subprime"
    segment_code: str  # Pricing and appetite key


def assign_risk_grade(
    probability_of_default: float,
    segment_code: str,
    grade_boundaries: dict[str, list[float]]
) -> RiskGradeResult:
    """
    Assign a risk grade based on PD and segment.

    Implements 00-ADDENDUM A2: publish segment_code.
    Grades key both pricing and appetite (00 §4).
    """
    # Get boundaries for this segment
    boundaries = grade_boundaries.get(segment_code, [
        0.01, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.50, 0.70, 0.85, 1.0
    ])

    # Find grade (1-12) based on PD
    grade = 12  # Worst
    for i, boundary in enumerate(boundaries):
        if probability_of_default <= boundary:
            grade = i + 1
            break

    # Bucket assignment
    if grade <= 3:
        bucket = "prime"
    elif grade <= 6:
        bucket = "near_prime"
    elif grade <= 9:
        bucket = "subprime"
    else:
        bucket = "deep_subprime"

    return RiskGradeResult(
        risk_grade=grade,
        grade_bucket=bucket,
        segment_code=segment_code
    )
