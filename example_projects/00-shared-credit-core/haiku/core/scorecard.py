from __future__ import annotations
"""Scorecard scoring with per-characteristic contributions (00 §6.10)."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class CharacteristicContribution:
    """Contribution of one scorecard characteristic."""
    characteristic_id: str
    characteristic_name: str
    raw_value: float  # The input value
    bin: Optional[str]  # The bin it fell into (including "null bin")
    points: float  # Points awarded
    contribution: float  # Signed contribution to score


@dataclass
class ScorecardResult:
    """Result of scorecard evaluation."""
    scorecard_id: int
    score: float  # Scaled score
    contributions: list[CharacteristicContribution]
    raw_score: float  # Before scaling
    score_unadjusted: float  # Before overlay (09 §5.15 item 7)


def evaluate_scorecard(
    scorecard_id: int,
    characteristics: dict[str, Optional[float]],
    bins: dict[str, list[tuple[float, float]]]  # characteristic -> [(low, high), ...]
) -> ScorecardResult:
    """
    Evaluate a scorecard and return score with per-characteristic contributions.

    Implements 00 §6.10 and 09 §5.15 items 14 & 16:
    - Per-characteristic, signed contributions
    - Null bin tracking (the null bin is a bin, not an error)
    - Used for adverse-action explanation
    """
    contributions_list = []
    total_points = 0.0

    for characteristic_id, value in characteristics.items():
        if value is None:
            # Null bin
            bin_name = "null"
            points = 0.0
        else:
            # Find which bin the value falls into
            bin_name = None
            points = 0.0
            if characteristic_id in bins:
                for i, (low, high) in enumerate(bins[characteristic_id]):
                    if low <= value < high:
                        bin_name = f"bin_{i}"
                        points = float(i)
                        break
            if bin_name is None:
                bin_name = "out_of_range"
                points = -10.0

        total_points += points
        contributions_list.append(CharacteristicContribution(
            characteristic_id=characteristic_id,
            characteristic_name=characteristic_id,
            raw_value=value or 0.0,
            bin=bin_name,
            points=points,
            contribution=points
        ))

    # Scale to 0-1000 range (typical scorecard)
    raw_score = total_points
    scaled_score = min(1000.0, max(0.0, raw_score * 10 + 500))

    return ScorecardResult(
        scorecard_id=scorecard_id,
        score=scaled_score,
        contributions=contributions_list,
        raw_score=raw_score,
        score_unadjusted=scaled_score  # Would be set before overlay
    )
