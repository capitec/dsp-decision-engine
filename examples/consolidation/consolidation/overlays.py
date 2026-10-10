"""Overlays on the values a Python step reads: the anti-harm ceiling and the objective weights.

`credit_core.adjustments.AdjustmentStack` overlays one column of a flow. These overlays tighten a threshold or a weight
inside the search and the selection, so they are documents the steps read, not steps themselves.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Literal, Mapping

from credit_core.adjustments import Overlay, evaluate
from decikit import ConfigurableStep
from decikit.exceptions import IRError
from pydantic import model_validator


def _pack(a: Overlay) -> tuple:
    # Plain str/int/float/bool tuples with dates as ordinals: the diff hashes those by content, and governance fields
    # (owner, approval, rationale, review date) stay out, as they are not behaviour.
    return (a.adjustment_id, a.effect.op, a.effect.value, tuple(sorted(a.scope.items())), a.effective_from.toordinal(),
            a.effective_to.toordinal() if a.effective_to else 0, a.enabled)


class OverlaySet(ConfigurableStep):
    """Approved, scoped, dated overlays, each naming the `target` value it changes, applied in `stack_position` order.

    A step reads `packed()` as data and calls `apply`; the set cannot be put in a flow.

    Example::

        overlays = OverlaySet(name="o", set_id="AS-1", adjustments=[Overlay(
            adjustment_id="ADJ-1", kind="cap_adjustment", target="ceiling", effect=Effect(op="multiply", value=0.8),
            scope={"product_code": 11}, stack_position=1, owner="Risk", approval_reference="CRC-1",
            rationale="Tighter pending review", effective_from=date(2026, 1, 1), review_date=date(2026, 12, 31))])
        apply(overlays.packed(), "ceiling", 0.15, {"product_code": 11}, date(2026, 6, 1), True)   # (0.12, ["ADJ-1"])
    """

    type: Literal["overlay_set"] = "overlay_set"
    set_id: str
    adjustments: tuple[Overlay, ...]

    @model_validator(mode="after")
    def _ids_unique(self) -> OverlaySet:
        ids = [a.adjustment_id for a in self.adjustments]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate adjustment ids: {sorted({i for i in ids if ids.count(i) > 1})}")
        return self

    def packed(self) -> tuple[tuple[str, tuple], ...]:
        """`(target, overlays in stack order)` for each target, as plain tuples a step can close over."""
        ordered = sorted(self.adjustments, key=lambda a: a.stack_position)
        return tuple((t, tuple(_pack(a) for a in ordered if a.target == t)) for t in sorted({a.target for a in ordered}))

    def to_ir(self, ctx: Any) -> Any:
        raise IRError(f"{self.name}: an overlay_set is read by the step that uses its values, it is not a step in a flow")


def apply(packed: tuple, target: str, base: float, provenance: Mapping[str, object], on: date, enabled: bool,
          ) -> tuple[float, list[str]]:
    """`base` after the overlays for `target` that are in scope and in force on `on`, and the ids of those applied.

    `enabled=False` goes through the same code and returns `base`.
    """
    overlays = next((o for t, o in packed if t == target), ())
    value, applied, _ = evaluate(overlays, base, provenance, on.toordinal(), enabled)
    return value, applied
