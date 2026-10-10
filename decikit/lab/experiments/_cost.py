from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from decikit._engine.wiring import resolve

__all__ = ["CostEstimate", "estimate_cost"]


class CostEstimate(BaseModel):
    """A first-order run-cost estimate: rows times non-frame calls, plus one run per frame call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: int = 0
    calls: int = 0
    frame_calls: int = 0

    @property
    def row_operations(self) -> int:
        return self.rows * (self.calls - self.frame_calls)


def estimate_cost(step: Any, rows: int) -> CostEstimate:
    calls = resolve(step).calls
    return CostEstimate(rows=rows, calls=len(calls), frame_calls=sum(c.node.kind == "frame" for c in calls))
