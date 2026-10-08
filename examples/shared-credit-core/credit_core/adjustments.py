"""Overlays: approved, scoped, dated changes to a model's answer, recorded beside the answer they changed.

An overlay never edits the model. The unadjusted value survives, the stacking order is declared, and
every overlay considered is recorded, whether it applied or not.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Mapping, Sequence

from decider import allow_fallback, param, step

Op = Literal["add", "multiply", "set"]


@dataclass(frozen=True)
class AdjustmentEffect:
    op: Op
    value: float


def _apply_effect(value: float, effect: AdjustmentEffect) -> float:
    if effect.op == "add":
        return value + effect.value
    if effect.op == "multiply":
        return value * effect.value
    return effect.value


# What counts as tightening, per kind. A kind not listed here cannot be declared tighten-only.
_TIGHTENS = {
    "score_shift": lambda e: e.op == "add" and e.value <= 0,
    "odds_multiplier": lambda e: e.op == "multiply" and e.value >= 1.0,
    "rate_addon": lambda e: e.op == "add" and e.value >= 0,
    "cap_adjustment": lambda e: e.op == "multiply" and e.value <= 1.0,
    "buffer_adjustment": lambda e: e.op == "add" and e.value >= 0,
}


@dataclass(frozen=True)
class Adjustment:
    """One overlay. `scope` maps provenance fields to the values it applies to; `stack_position` orders stacking.

    A `tighten_only` overlay whose effect loosens is rejected when defined. It lapses at `effective_to`
    and is surfaced for review from `review_date`.
    """

    adjustment_id: str
    kind: str  # e.g. "score_shift", "odds_multiplier", "rate_addon"
    target: str  # the value it overlays, e.g. "score"
    effect: AdjustmentEffect
    scope: Mapping[str, object]
    stack_position: int
    owner: str
    approval_reference: str
    rationale: str
    effective_from: date
    effective_to: date | None
    review_date: date
    tighten_only: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.tighten_only:
            return
        tightens = _TIGHTENS.get(self.kind)
        if tightens is None:
            raise ValueError(f"kind {self.kind!r} has no declared tighten direction; cannot be tighten_only")
        if not tightens(self.effect):
            raise ValueError(f"{self.adjustment_id}: {self.kind} is tighten-only but its effect loosens")

    def in_scope(self, provenance: Mapping[str, object]) -> bool:
        return all(provenance.get(k) == v for k, v in self.scope.items())

    def lapsed(self, on: date) -> bool:
        return self.effective_to is not None and on >= self.effective_to

    def in_force(self, on: date) -> bool:
        return self.enabled and self.effective_from <= on and not self.lapsed(on)

    def due_for_review(self, on: date) -> bool:
        return on >= self.review_date and not self.lapsed(on)

    def apply(self, base_value: float, provenance: Mapping[str, object], on: date) -> float:
        """Apply this one overlay. Outside its scope or window is an error, never a silent no-op."""
        if not self.in_scope(provenance):
            raise ValueError(f"{self.adjustment_id} applied to a population outside its declared scope")
        if not self.in_force(on):
            raise ValueError(f"{self.adjustment_id} is not in force on {on}")
        return _apply_effect(base_value, self.effect)


@dataclass(frozen=True)
class AdjustmentEvaluation:
    adjustment_id: str
    applied: bool
    reason: str  # "applied", "out_of_scope", "not_yet_effective", "lapsed" or "stack_disabled"


@dataclass(frozen=True)
class AdjustmentResult:
    unadjusted_value: float
    adjusted_value: float
    adjustment_set_id: str
    adjustments_applied: tuple[str, ...]  # in application order
    evaluations: tuple[AdjustmentEvaluation, ...]  # every overlay considered for the target


class AdjustmentRegister:
    """Every overlay defined for a set of targets.

    Example::

        register = AdjustmentRegister([Adjustment("ADJ-1", "score_shift", "score", AdjustmentEffect("add", -10.0),
                                                  {"segment_code": 3}, 1, "Credit Risk", "CRC-1", "Defaults above model",
                                                  date(2026, 1, 1), None, date(2026, 12, 31), tighten_only=True)])
        register.apply_stack("score", 600.0, {"segment_code": 3}, date(2026, 6, 1), "AS-1").adjusted_value  # 590.0
    """

    def __init__(self, adjustments: Sequence[Adjustment]):
        ids = [a.adjustment_id for a in adjustments]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate adjustment ids: {sorted({i for i in ids if ids.count(i) > 1})}")
        self._all = tuple(adjustments)

    def __len__(self) -> int:
        return len(self._all)

    def for_target(self, target: str) -> tuple[Adjustment, ...]:
        return tuple(a for a in self._all if a.target == target)

    def due_for_review(self, on: date) -> tuple[Adjustment, ...]:
        """Overlays past their review date and not yet lapsed, oldest review date first."""
        return tuple(sorted((a for a in self._all if a.due_for_review(on)), key=lambda a: a.review_date))

    def apply_stack(
        self, target: str, base_value: float, provenance: Mapping[str, object], on: date, adjustment_set_id: str,
        stack_enabled: bool = True,
    ) -> AdjustmentResult:
        """Apply every in-scope, in-force overlay for `target` in stack order.

        `stack_enabled=False` goes through the same code and returns the unadjusted value.
        """
        value, applied, evaluations = base_value, [], []
        for a in sorted(self.for_target(target), key=lambda a: a.stack_position):
            if not a.in_scope(provenance):
                reason = "out_of_scope"
            elif not a.in_force(on):
                reason = "lapsed" if a.lapsed(on) else "not_yet_effective"
            elif not stack_enabled:
                reason = "stack_disabled"
            else:
                value, reason = _apply_effect(value, a.effect), "applied"
                applied.append(a.adjustment_id)
            evaluations.append(AdjustmentEvaluation(a.adjustment_id, reason == "applied", reason))
        return AdjustmentResult(base_value, value, adjustment_set_id, tuple(applied), tuple(evaluations))

    def apply_stack_step(
        self, target: str, adjustment_set_id: str, *,
        base_field: str | None = None, adjusted_output: str | None = None, unadjusted_output: str | None = None,
    ):
        """A step that overlays the `base_field` column (default `target`).

        It writes the adjusted value (default `target`), `<target>_unadjusted`, `adjustment_set_id` and
        `adjustments_applied`; when `base_field` is the target itself, put the step in a `flow`. Scope is matched
        on `product_code`, `segment_code`, `channel_code` and `scorecard_id`. The param `adjustment_stack_enabled`
        switches the whole stack off.
        """

        @allow_fallback
        def apply_adjustments(
            adjustment_base_value: float, decision_date: date,
            adjustment_stack_enabled: bool = param(True),
            product_code: float | None = None, segment_code: float | None = None,
            channel_code: int | None = None, scorecard_id: int | None = None,
        ) -> tuple[float, float, str, list[str]]:
            provenance = {k: v for k, v in (("product_code", product_code), ("segment_code", segment_code),
                                            ("channel_code", channel_code), ("scorecard_id", scorecard_id))
                          if v is not None}
            result = self.apply_stack(target, adjustment_base_value, provenance, decision_date, adjustment_set_id,
                                      stack_enabled=adjustment_stack_enabled)
            return (result.adjusted_value, result.unadjusted_value, result.adjustment_set_id,
                    list(result.adjustments_applied))

        apply_adjustments.__name__ = f"apply_{target}_adjustments"
        s = step(apply_adjustments, outputs=(adjusted_output or target, unadjusted_output or f"{target}_unadjusted",
                                             "adjustment_set_id", "adjustments_applied"))
        return s.relabel(reads={"adjustment_base_value": base_field or target})
