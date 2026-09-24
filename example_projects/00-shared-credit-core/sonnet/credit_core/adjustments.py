"""`core.adjustments` -- post-model overlays (spec 00 §6.22, §7.6; addendum B1-B2).

The mechanism every other capability's "adjusted vs. unadjusted" pair is
built on. An adjustment is an overlay over an already-approved artefact,
never an edit to it (property 1): the base scorecard, calibration or rate
card stays exactly as validated, and this module composes named, scoped,
effective-dated overlays over its output at runtime.

Six properties from 00 §6.22 this module is built to hold:
  1. overlay, not edit           -- `apply_stack` never mutates the base value's source
  2. unadjusted value survives   -- `AdjustmentResult.unadjusted_value`
  3. stacking order is declared  -- `Adjustment.stack_position`, not insertion order
  4. identity + justification    -- id, owner, approval_reference, rationale
  5. expiry is mandatory         -- `review_date` (surfaced) and `effective_to` (auto-lapses)
  6. scope is declared           -- `Adjustment.scope`; applying outside it is an error

Addendum B1: identifiers are `adjustment_set_id` (str) and
`adjustments_applied` (list[str]) -- the one pair every consumer keys on.
Addendum B2: expiry lapses automatically *and* surfaces at the review date;
some kinds are tighten-only, rejected at definition if they would loosen.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Mapping, Sequence

from decider import missing_as, param, step

Op = Literal["add", "multiply", "set"]


@dataclass(frozen=True)
class AdjustmentEffect:
    op: Op
    value: float | str


def _apply_effect(value: float, effect: AdjustmentEffect) -> float:
    if effect.op == "add":
        return value + effect.value
    if effect.op == "multiply":
        return value * effect.value
    if effect.op == "set":
        return effect.value
    raise ValueError(f"unknown op {effect.op!r}")


# Direction convention per kind (spec 00 §6.22 table + examples), used only
# when an adjustment declares `tighten_only=True`. A kind not listed here
# cannot be declared tighten-only -- the direction would be a guess.
_TIGHTEN_RULES = {
    "score_shift": lambda e: e.op == "add" and float(e.value) <= 0,          # a lower score is worse
    "odds_multiplier": lambda e: e.op == "multiply" and float(e.value) >= 1.0,  # PD x1.35 etc.
    "rate_addon": lambda e: e.op == "add" and float(e.value) >= 0,
    "cap_adjustment": lambda e: e.op == "multiply" and float(e.value) <= 1.0,
    "buffer_adjustment": lambda e: e.op == "add" and float(e.value) >= 0,
}


def _is_tightening(kind: str, effect: AdjustmentEffect) -> bool:
    rule = _TIGHTEN_RULES.get(kind)
    if rule is None:
        raise ValueError(f"kind {kind!r} has no declared tighten direction; cannot be tighten_only")
    return rule(effect)


@dataclass(frozen=True)
class Adjustment:
    """One named, approved, scoped overlay. Immutable once defined -- a change is a new id."""

    adjustment_id: str
    kind: str                       # e.g. "score_shift", "odds_multiplier", "cap_adjustment", ...
    target: str                     # the value it overlays, e.g. "score", "probability_of_default"
    effect: AdjustmentEffect
    scope: Mapping[str, object]     # e.g. {"product_code": 10, "segment_code": 3}
    stack_position: int             # declared composition order -- not insertion order
    owner: str
    approval_reference: str
    rationale: str
    effective_from: date
    effective_to: date | None
    review_date: date
    tighten_only: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.tighten_only and not _is_tightening(self.kind, self.effect):
            raise ValueError(f"{self.adjustment_id}: {self.kind} is tighten-only but its effect loosens")

    def in_scope(self, provenance: Mapping[str, object]) -> bool:
        return all(provenance.get(k) == v for k, v in self.scope.items())

    def in_force(self, decision_date: date) -> bool:
        if not self.enabled:
            return False
        if decision_date < self.effective_from:
            return False
        return self.effective_to is None or decision_date < self.effective_to

    def lapsed(self, decision_date: date) -> bool:
        return self.effective_to is not None and decision_date >= self.effective_to

    def due_for_review(self, as_of: date) -> bool:
        return as_of >= self.review_date and not self.lapsed(as_of)

    def apply(self, base_value: float, provenance: Mapping[str, object], decision_date: date) -> float:
        """Apply this one adjustment directly. Outside its scope or window is an error, never a no-op (property 6)."""
        if not self.in_scope(provenance):
            raise ValueError(f"{self.adjustment_id} applied to a population outside its declared scope")
        if not self.in_force(decision_date):
            raise ValueError(f"{self.adjustment_id} is not in force on {decision_date}")
        return _apply_effect(base_value, self.effect)


@dataclass(frozen=True)
class AdjustmentEvaluation:
    adjustment_id: str
    applied: bool
    reason: str                     # "applied" | "out_of_scope" | "not_yet_effective" | "lapsed" | "stack_disabled"


@dataclass(frozen=True)
class AdjustmentResult:
    unadjusted_value: float
    adjusted_value: float
    adjustment_set_id: str
    adjustments_applied: tuple[str, ...]      # ids that actually applied, in application order
    evaluations: tuple[AdjustmentEvaluation, ...]  # every overlay *considered*, fired or not (09 §5.15 item 14)


class AdjustmentRegister:
    """The live register: every overlay ever defined, append-only (property: never merged, never overwritten).

    Testable standalone (§7.5): build one from a plain list of `Adjustment`s
    and call `apply_stack` with no pipeline.
    """

    def __init__(self, adjustments: Sequence[Adjustment]):
        ids = [a.adjustment_id for a in adjustments]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate adjustment ids: {dupes}")
        self._all = tuple(adjustments)

    def __len__(self) -> int:
        return len(self._all)

    def for_target(self, target: str) -> tuple[Adjustment, ...]:
        return tuple(a for a in self._all if a.target == target)

    def due_for_review(self, as_of: date) -> tuple[Adjustment, ...]:
        """Every overlay past its review date and not yet lapsed -- the ageing report (§6.22 property 5)."""
        return tuple(sorted((a for a in self._all if a.due_for_review(as_of)), key=lambda a: a.review_date))

    def apply_stack(
        self,
        target: str,
        base_value: float,
        provenance: Mapping[str, object],
        decision_date: date,
        adjustment_set_id: str,
        stack_enabled: bool = True,
    ) -> AdjustmentResult:
        """Compose every in-scope, in-force overlay for `target`, in declared stack order.

        `stack_enabled=False` runs the same code path and returns the
        unadjusted answer -- the "stack off" run (§7.6, acceptance §10 item
        8) is this call with one flag flipped, not a second implementation.
        """
        value = base_value
        applied: list[str] = []
        evaluations: list[AdjustmentEvaluation] = []
        candidates = sorted(self.for_target(target), key=lambda a: a.stack_position)
        for a in candidates:
            if not a.in_scope(provenance):
                evaluations.append(AdjustmentEvaluation(a.adjustment_id, False, "out_of_scope"))
                continue
            if not a.in_force(decision_date):
                reason = "lapsed" if a.lapsed(decision_date) else "not_yet_effective"
                evaluations.append(AdjustmentEvaluation(a.adjustment_id, False, reason))
                continue
            if not stack_enabled:
                evaluations.append(AdjustmentEvaluation(a.adjustment_id, False, "stack_disabled"))
                continue
            value = _apply_effect(value, a.effect)
            applied.append(a.adjustment_id)
            evaluations.append(AdjustmentEvaluation(a.adjustment_id, True, "applied"))
        return AdjustmentResult(
            unadjusted_value=base_value,
            adjusted_value=value,
            adjustment_set_id=adjustment_set_id,
            adjustments_applied=tuple(applied),
            evaluations=tuple(evaluations),
        )

    def apply_stack_step(
        self, target: str, adjustment_set_id: str, *,
        base_field: str | None = None, adjusted_output: str | None = None, unadjusted_output: str | None = None,
    ):
        """A decider step wrapping `apply_stack` for one `target`, so a pipeline can run with
        the stack disabled through a `param()` flip -- the same code path, not a second
        implementation (§7.6; acceptance §10 item 8).

        `base_field` (default: `target` itself) is the column holding the value
        to overlay; provenance is read from the four scope keys every kind in
        this project's overlay taxonomy is declared against: `product_code`,
        `segment_code`, `channel_code`, `scorecard_id`. A consumer needing a
        wider scope (sector, campaign, grade range) builds its own step from
        `apply_stack` directly -- the mechanism, not this convenience wrapper,
        is what's frozen.
        """
        base_field = base_field or target
        adjusted_output = adjusted_output or target
        unadjusted_output = unadjusted_output or f"{target}_unadjusted"
        register = self

        def apply_adjustments(
            adjustment_base_value: float, decision_date: date,
            adjustment_stack_enabled: bool = param(True),
            product_code: int = missing_as(-1), segment_code: int = missing_as(-1),
            channel_code: int = missing_as(-1), scorecard_id: int = missing_as(-1),
        ) -> tuple[float, float, str, list[str]]:
            provenance = {"product_code": product_code, "segment_code": segment_code,
                          "channel_code": channel_code, "scorecard_id": scorecard_id}
            provenance = {k: v for k, v in provenance.items() if v != -1}
            result = register.apply_stack(target, adjustment_base_value, provenance, decision_date,
                                           adjustment_set_id, stack_enabled=adjustment_stack_enabled)
            return (result.adjusted_value, result.unadjusted_value, result.adjustment_set_id,
                    list(result.adjustments_applied))

        apply_adjustments.__name__ = f"apply_{target}_adjustments"
        s = step(apply_adjustments, outputs=(adjusted_output, unadjusted_output, "adjustment_set_id",
                                              "adjustments_applied"))
        if base_field != "adjustment_base_value":
            s = s.relabel(reads={"adjustment_base_value": base_field})
        return s
