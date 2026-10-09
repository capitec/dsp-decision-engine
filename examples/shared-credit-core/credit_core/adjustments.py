"""Overlays: approved, scoped, dated changes to a model's answer, recorded beside the answer they changed.

An overlay never edits the model. The unadjusted value survives, the stacking order is declared, and
every overlay considered is recorded, whether it applied or not.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, model_validator

from decider import ConfigurableStep, ParamRef, Value, allow_fallback

Op = Literal["add", "multiply", "set"]


# What counts as tightening, per kind. A kind not listed here cannot be declared tighten-only.
_TIGHTENS = {
    "score_shift": lambda e: e.op == "add" and e.value <= 0,
    "odds_multiplier": lambda e: e.op == "multiply" and e.value >= 1.0,
    "rate_addon": lambda e: e.op == "add" and e.value >= 0,
    "cap_adjustment": lambda e: e.op == "multiply" and e.value <= 1.0,
    "buffer_adjustment": lambda e: e.op == "add" and e.value >= 0,
}


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


class Effect(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    op: Op
    value: float


class Overlay(BaseModel):
    """One overlay in a stack document. The governance fields (owner, approval, rationale, review date) are not behaviour.

    A `tighten_only` overlay whose effect loosens is rejected when the document loads.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    adjustment_id: str
    kind: str  # e.g. "score_shift", "odds_multiplier", "rate_addon"
    target: str  # the value it overlays, e.g. "score"
    effect: Effect
    scope: dict[str, int | float | str]  # provenance field -> the value it applies to; empty applies to everyone
    stack_position: int
    owner: str
    approval_reference: str
    rationale: str
    effective_from: date
    effective_to: date | None = None
    review_date: date
    tighten_only: bool = False
    enabled: bool = True

    @model_validator(mode="after")
    def _tighten_only_tightens(self) -> Overlay:
        if self.tighten_only:
            tightens = _TIGHTENS.get(self.kind)
            if tightens is None:
                raise ValueError(f"kind {self.kind!r} has no declared tighten direction; cannot be tighten_only")
            if not tightens(self.effect):
                raise ValueError(f"{self.adjustment_id}: {self.kind} is tighten-only but its effect loosens")
        return self

    def lapsed(self, on: date) -> bool:
        return self.effective_to is not None and on >= self.effective_to

    def due_for_review(self, on: date) -> bool:
        return on >= self.review_date and not self.lapsed(on)


def _pack(a: Overlay) -> tuple:
    # Plain str/int/float/bool tuples with dates as ordinals: the diff hashes those by content and cannot hash a date.
    return (a.adjustment_id, a.effect.op, a.effect.value, tuple(sorted(a.scope.items())), a.effective_from.toordinal(),
            a.effective_to.toordinal() if a.effective_to else 0, a.enabled)


def evaluate(overlays: tuple, base_value: float, provenance: Mapping[str, object], on: int, stack_enabled: bool = True,
             ) -> tuple[float, list[str], list[tuple[str, bool, str]]]:
    """The adjusted value, the ids applied and `(id, applied, reason)` for each overlay, given overlays in stack order."""
    value, applied, evaluations = base_value, [], []
    for adjustment_id, op, amount, scope, start, end, enabled in overlays:
        lapsed = end != 0 and on >= end
        if not all(provenance.get(k) == v for k, v in scope):
            reason = "out_of_scope"
        elif not (enabled and start <= on and not lapsed):
            reason = "lapsed" if lapsed else "not_yet_effective"
        elif not stack_enabled:
            reason = "stack_disabled"
        else:
            value = value + amount if op == "add" else value * amount if op == "multiply" else amount
            reason = "applied"
            applied.append(adjustment_id)
        evaluations.append((adjustment_id, reason == "applied", reason))
    return value, applied, evaluations


# These two columns have always been read as whole numbers; any other numeric scope column is a float unless the
# stack's `scope_types` says otherwise, because a column has one type across the steps that read it.
_INT_COLUMNS = frozenset({"channel_code", "scorecard_id"})
_TYPES = {"int": int, "float": float, "str": str}


def _scope_function(columns: dict[str, type]):
    """`apply_adjustments` with one optional argument per scope column, so the step reads exactly those columns."""

    @allow_fallback
    def apply_adjustments(adjustment_base_value, decision_date, overlays, set_id, adjustment_stack_enabled, **scope):
        provenance = {k: v for k, v in scope.items() if v is not None}
        value, applied, _ = evaluate(overlays, adjustment_base_value, provenance, decision_date.toordinal(),
                                     adjustment_stack_enabled)
        return value, adjustment_base_value, set_id, applied

    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    required = [inspect.Parameter(n, kind, annotation=t) for n, t in
                (("adjustment_base_value", float), ("decision_date", date), ("overlays", tuple), ("set_id", str),
                 ("adjustment_stack_enabled", bool))]
    scoped = [inspect.Parameter(n, kind, default=None, annotation=t | None) for n, t in columns.items()]
    apply_adjustments.__signature__ = inspect.Signature(
        required + scoped, return_annotation=tuple[float, float, str, list[str]])
    return apply_adjustments


class AdjustmentStack(ConfigurableStep):
    """Approved, scoped, dated overlays on one value, applied in `stack_position` order.

    The step overlays the `base_field` column (default `target`) and writes the adjusted value (default `target`),
    `<target>_unadjusted`, `adjustment_set_id` and `adjustments_applied`; when `base_field` is the target itself, put the
    step in a `flow`. The step reads one column per scope field the overlays name (`segment_code`, `sector_code`, ...); a
    record whose column is missing or null is out of scope of the overlays that name it. A scope column is read as a
    float (a string if its values are strings); `scope_types` sets `int`, `float` or `str` where another step reads the same
    column as another type. The param
    `adjustment_stack_enabled` switches the whole stack off.

    Example::

        stack = AdjustmentStack(target="score", set_id="AS-1", base_field="score_raw", adjustments=[
            Overlay(adjustment_id="ADJ-1", kind="score_shift", target="score", effect=Effect(op="add", value=-10.0),
                    scope={"segment_code": 3}, stack_position=1, owner="Credit Risk", approval_reference="CRC-1",
                    rationale="Defaults above model", effective_from=date(2026, 1, 1), review_date=date(2026, 12, 31),
                    tighten_only=True)])
        stack.apply(600.0, {"segment_code": 3}, date(2026, 6, 1)).adjusted_value  # 590.0
    """

    type: Literal["adjustment_stack"] = "adjustment_stack"
    target: str
    set_id: str
    base_field: str | None = None
    adjusted_output: str | None = None
    unadjusted_output: str | None = None
    adjustments: tuple[Overlay, ...]
    scope_types: dict[str, Literal["int", "float", "str"]] = {}
    stack_enabled: Value[bool] = ParamRef(param="adjustment_stack_enabled", default=True)

    @model_validator(mode="before")
    @classmethod
    def _default_name(cls, data: Any) -> Any:
        if isinstance(data, dict) and "target" in data:
            return {"name": f"apply_{data['target']}_adjustments", **data}
        return data

    @model_validator(mode="after")
    def _ids_unique_and_for_target(self) -> AdjustmentStack:
        ids = [a.adjustment_id for a in self.adjustments]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate adjustment ids: {sorted({i for i in ids if ids.count(i) > 1})}")
        stray = [a.adjustment_id for a in self.adjustments if a.target != self.target]
        if stray:
            raise ValueError(f"overlays {stray} are not for target {self.target!r}")
        return self

    def packed(self) -> tuple[tuple, ...]:
        """The overlays in stack order as plain tuples, the form `evaluate` takes, for a step that applies a stack itself.

        Example::

            evaluate(stack.packed(), 600.0, {"segment_code": 3}, date(2026, 6, 1).toordinal())[0]  # 590.0
        """
        return tuple(_pack(a) for a in sorted(self.adjustments, key=lambda a: a.stack_position))

    def due_for_review(self, on: date) -> tuple[Overlay, ...]:
        """Overlays past their review date and not yet lapsed, oldest review date first."""
        return tuple(sorted((a for a in self.adjustments if a.due_for_review(on)), key=lambda a: a.review_date))

    def apply(self, base_value: float, provenance: Mapping[str, object], on: date, stack_enabled: bool = True,
              ) -> AdjustmentResult:
        """Apply the stack outside a flow, to a value whose provenance fields are in `provenance`."""
        value, applied, evaluations = evaluate(self.packed(), base_value, provenance, on.toordinal(), stack_enabled)
        return AdjustmentResult(base_value, value, self.set_id, tuple(applied),
                                tuple(AdjustmentEvaluation(*e) for e in evaluations))

    def scope_columns(self) -> dict[str, type]:
        """The columns the overlays are scoped on, which the step reads, with the type each is read as."""
        values: dict[str, list] = {}
        for a in self.adjustments:
            for k, v in a.scope.items():
                values.setdefault(k, []).append(v)
        return {k: _TYPES[self.scope_types[k]] if k in self.scope_types else int if k in _INT_COLUMNS
                else str if all(isinstance(v, str) for v in vs) else float for k, vs in sorted(values.items())}

    def to_ir(self, ctx: Any) -> Any:
        return ctx.call(
            self, _scope_function(self.scope_columns()), inputs={"adjustment_base_value": self.base_field or self.target},
            values={"overlays": self.packed(), "set_id": self.set_id, "adjustment_stack_enabled": self.stack_enabled},
            outputs=(self.adjusted_output or self.target, self.unadjusted_output or f"{self.target}_unadjusted",
                     "adjustment_set_id", "adjustments_applied"))
