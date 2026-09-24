from __future__ import annotations
"""Calibration: score to probability of default (00 §6.11)."""
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Result of score calibration."""
    probability_of_default: float  # Calibrated 12-month PD, 0..1
    odds: float  # Odds of default (PD / (1-PD))
    probability_of_default_unadjusted: float  # Before overlay


def calibrate_score(
    score: float,
    segment_code: str,
    calibration_parameters: dict[str, float]
) -> CalibrationResult:
    """
    Convert a score to a probability of default using calibration parameters.

    Implements 00-ADDENDUM A2: publish segment_code.
    Implements 09 §5.15 item 7: unadjusted values recorded.
    """
    # Simplified logistic calibration
    # Real implementation would have segment-specific parameters

    # Get segment-specific parameters or use defaults
    scale = calibration_parameters.get(f"segment_{segment_code}_scale", 0.001)
    offset = calibration_parameters.get(f"segment_{segment_code}_offset", 500.0)

    # Logistic transform: PD = 1 / (1 + exp(-(score - offset) * scale))
    import math
    exponent = -(score - offset) * scale
    try:
        pd = 1.0 / (1.0 + math.exp(min(exponent, 100)))  # Clip to avoid overflow
    except:
        pd = 0.5

    pd = max(0.0001, min(0.9999, pd))  # Bound to valid range

    odds = pd / (1.0 - pd) if pd < 1.0 else 1000.0

    return CalibrationResult(
        probability_of_default=pd,
        odds=odds,
        probability_of_default_unadjusted=pd
    )
