from __future__ import annotations
"""Post-model adjustments/overlays (00 §6.22, 09 §5.14)."""
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Optional


class OverlayKind(Enum):
    """Types of overlays (00 §6.22 property 2)."""
    SCORE_SHIFT = "score_shift"  # Numeric: add/subtract from score
    MULTIPLIER = "multiplier"  # Numeric: multiply value
    THRESHOLD_SHIFT = "threshold_shift"  # Numeric: shift a cut-off
    MATRIX_DIAL = "matrix_dial"  # Numeric: dial intensity up/down
    GRADE_BOUNDARY_MOVE = "grade_boundary_move"  # Numeric: move grade cutoff
    TIGHTEN_ONLY_FLAG = "tighten_only"  # Constraint: may only reduce capacity


class AdjustmentDirection(Enum):
    """Direction of overlay effect (00-ADDENDUM B.2)."""
    BOTH = "both"  # Can loosen or tighten
    TIGHTEN_ONLY = "tighten_only"  # May only reduce (lower score, higher rate, etc.)


@dataclass
class Adjustment:
    """A named, approved, effective-dated overlay (00 §6.22)."""
    adjustment_id: int
    kind: OverlayKind
    scope: str  # e.g., "product:10,channel:2" or "all"
    magnitude: float  # The adjustment value
    position: int  # Stack position (lower = applied first)
    effective_from: date
    effective_to: Optional[date]  # None if open-ended
    review_date: date  # Mandatory (00-ADDENDUM B.2)
    direction: AdjustmentDirection  # Asymmetry: tighten-only? (09 §5.14.6)
    enabled: bool = True  # Lapse state (00-ADDENDUM B.2)


@dataclass
class AdjustmentSetResult:
    """Result of applying a set of adjustments."""
    base_value: float
    adjusted_value: float
    applied_adjustments: list[int]  # IDs of adjustments that changed the value
    unadjusted_inputs: dict[str, Any]  # Original values before overlay (09 §5.15 item 7)


def apply_adjustments(
    base_value: float,
    adjustments: list[Adjustment],
    decision_date: date
) -> AdjustmentSetResult:
    """
    Apply a stack of adjustments to a base value.
    Implements 09 §5.15 item 7: overlay stack recorded.
    """
    # Filter to active adjustments on decision_date
    active = [
        a for a in adjustments
        if a.enabled and a.effective_from <= decision_date and
           (a.effective_to is None or decision_date <= a.effective_to)
    ]

    # Sort by stack position
    active.sort(key=lambda a: a.position)

    current = base_value
    applied = []

    for adj in active:
        if adj.kind == OverlayKind.SCORE_SHIFT:
            current += adj.magnitude
            applied.append(adj.adjustment_id)
        elif adj.kind == OverlayKind.MULTIPLIER:
            current *= adj.magnitude
            applied.append(adj.adjustment_id)
        elif adj.kind == OverlayKind.THRESHOLD_SHIFT:
            current += adj.magnitude
            applied.append(adj.adjustment_id)
        elif adj.kind == OverlayKind.MATRIX_DIAL:
            current *= (1.0 + adj.magnitude)
            applied.append(adj.adjustment_id)

    return AdjustmentSetResult(
        base_value=base_value,
        adjusted_value=current,
        applied_adjustments=applied,
        unadjusted_inputs={"value": base_value}
    )
