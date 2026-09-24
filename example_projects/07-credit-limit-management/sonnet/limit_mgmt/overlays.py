"""§4.3, §6.6 -- the overlays specific to this project, all built on `core.adjustments`
(`credit_core.adjustments`), "the library's mechanism... pointed at a project-owned
artefact" (§4.3), never a second overlay vocabulary.

Two of the four kinds this project needs do not fit `core.adjustments`'
three effects (`add`/`multiply`/`set`) directly, and both are solved by
choosing a different *target* rather than extending the shared mechanism
(00 NOTES.md's own guidance: "a consumer needing a wider scope builds its
own step from `apply_stack` directly -- the mechanism... is frozen"):

- **The cycle dial** (§5.4: "applied to the multiplier's excess over 1.00,
  so a cell multiplier of 1.50 becomes 1.40 at an 80% dial, not 1.20") is
  a plain `multiply` overlay -- but on `matrix_multiplier_excess`
  (`matrix_multiplier_unadjusted - 1.0`), not on the multiplier itself.
- **The cycle cap** ("no increase above R5 000, whatever the cell says")
  needs a *ceiling*, which none of `add`/`multiply`/`set` expresses (`set`
  would force every cell to the cap, not only the ones above it). See
  `apply_cycle_cap` below -- kept in the same evidence shape (id, owner,
  approval, scope, dates, review date) as a real `Adjustment`, applied by
  a project-owned function instead of `AdjustmentEffect`. Flagged under
  NOTES.md "Framework friction".

The score shift and PD multiplier (§5.3, §6.6) fit the shared mechanism
exactly (`score_shift` = add, `odds_multiplier` = multiply) and need no
workaround.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

from decider import missing_as, param, step

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister

ADJUSTMENT_SET_ID = "AS-07-2026.09"

# --- Score shift and PD multiplier (§5.3, §6.6) -----------------------------

BEHAVIOUR_ADJUSTMENTS = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-07-001", kind="score_shift", target="behaviour_score",
        effect=AdjustmentEffect("add", -15.0), scope={"channel_code": 5}, stack_position=1,
        owner="Model Risk", approval_reference="MRC-2026-041",
        rationale="Partner-originated channel running worse than model prediction in early life",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31), tighten_only=True,
    ),
    Adjustment(
        adjustment_id="ADJ-07-002", kind="odds_multiplier", target="probability_of_default",
        effect=AdjustmentEffect("multiply", 1.25), scope={"product_code": 21}, stack_position=1,
        owner="Model Risk", approval_reference="MRC-2026-044",
        rationale="Access Facility observed default running above prediction",
        effective_from=date(2026, 3, 1), effective_to=date(2027, 3, 1), review_date=date(2026, 12, 1),
        tighten_only=True,
    ),
])

behaviour_score_adjustment_step = BEHAVIOUR_ADJUSTMENTS.apply_stack_step(
    "behaviour_score", ADJUSTMENT_SET_ID, base_field="behaviour_score_raw",
).relabel(writes={"adjustment_set_id": "score_adjustment_set_id", "adjustments_applied": "score_adjustments_applied"})
pd_adjustment_step = BEHAVIOUR_ADJUSTMENTS.apply_stack_step(
    "probability_of_default", ADJUSTMENT_SET_ID, base_field="probability_of_default_raw",
).relabel(writes={"adjustment_set_id": "pd_adjustment_set_id", "adjustments_applied": "pd_adjustments_applied"})


# --- Cycle dial: a `multiply` overlay on the multiplier's excess over 1.00 -

MATRIX_ADJUSTMENTS = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-07-DIAL-2026-03", kind="cap_adjustment", target="matrix_multiplier_excess",
        effect=AdjustmentEffect("multiply", 0.80), scope={}, stack_position=1,
        owner="Credit Risk Policy", approval_reference="CRC-2026-088",
        rationale="Funding environment tightened; run the programme at 80% of the matrix this cycle",
        effective_from=date(2026, 9, 1), effective_to=date(2026, 11, 1), review_date=date(2026, 10, 15),
        tighten_only=True,
    ),
])


def matrix_multiplier_excess(matrix_multiplier_unadjusted: float) -> float:
    return round(matrix_multiplier_unadjusted - 1.0, 6)


matrix_multiplier_excess_step = step(matrix_multiplier_excess)

matrix_dial_step = MATRIX_ADJUSTMENTS.apply_stack_step(
    "matrix_multiplier_excess", ADJUSTMENT_SET_ID,
    adjusted_output="matrix_multiplier_excess_adjusted", unadjusted_output="matrix_multiplier_excess_unadjusted",
).relabel(writes={"adjustment_set_id": "matrix_adjustment_set_id", "adjustments_applied": "matrix_adjustments_applied"})


def matrix_multiplier(matrix_multiplier_excess_adjusted: float) -> float:
    return round(1.0 + matrix_multiplier_excess_adjusted, 4)


matrix_multiplier_step = step(matrix_multiplier)


# --- Cycle cap: a ceiling, expressed with the same governance shape as an
# `Adjustment` but applied by a bespoke function (see module docstring). ----

@dataclass(frozen=True)
class CycleCap:
    cap_id: str
    amount: float
    scope: Mapping[str, object]
    owner: str
    approval_reference: str
    rationale: str
    effective_from: date
    effective_to: date | None
    review_date: date
    enabled: bool = True

    def in_scope(self, provenance: Mapping[str, object]) -> bool:
        return all(provenance.get(k) == v for k, v in self.scope.items())

    def in_force(self, decision_date: date) -> bool:
        if not self.enabled:
            return False
        if decision_date < self.effective_from:
            return False
        return self.effective_to is None or decision_date < self.effective_to

    def due_for_review(self, as_of: date) -> bool:
        return as_of >= self.review_date and self.in_force(as_of)


# Empty by default (§11 change scenario 3: set for a cycle, then it lapses). Non-empty
# here only to exercise the mechanism end to end in tests and simulation.
ACTIVE_CYCLE_CAP: CycleCap | None = None


def apply_cycle_cap(
    matrix_max_increase_unadjusted: float, decision_date: date, product_code: int,
    cycle_cap_enabled: bool = param(False),
    cycle_cap_amount: float = param(5_000.0, ge=0.0),
) -> tuple[float, bool]:
    """Whatever the cell says, `min(cell_ceiling, cap)` when a cap is in force this
    cycle (§6.6: "Nothing above R5 000 this cycle, whatever the cell says") -- the cap
    itself is a `param()` flip so it composes with simulation/stack-off the same way
    every other overlay in this project does, even though its arithmetic (a ceiling)
    doesn't fit `core.adjustments`' three ops."""
    if not cycle_cap_enabled:
        return matrix_max_increase_unadjusted, False
    capped = min(matrix_max_increase_unadjusted, cycle_cap_amount)
    return capped, capped < matrix_max_increase_unadjusted


apply_cycle_cap_step = step(apply_cycle_cap, outputs=("matrix_max_increase", "cycle_cap_bound"))


# --- 07's own buffer, layered on 02's shared assessment as an overlay ------
# (§5.6: "18% against origination's 12%... a parameter of the assessment, supplied by
# this project, not a fork of the shared capability... itself overlay-adjustable").

BUFFER_ADJUSTMENTS = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-07-BUFFER-BASE", kind="buffer_adjustment", target="affordability_buffer_applied",
        effect=AdjustmentEffect("add", 0.06), scope={}, stack_position=1,
        owner="Credit Risk Policy", approval_reference="CRC-2026-091",
        rationale="Programme affordability underwrites against a salary pattern and a bureau "
                   "file, not verified evidence collected days earlier (§5.6): a wider buffer "
                   "than origination's, expressed as this project's own overlay over the shared "
                   "buffer grid rather than a fork of project 02's capacity calculation.",
        effective_from=date(2026, 1, 1), effective_to=None, review_date=date(2026, 12, 31), tighten_only=True,
    ),
])

buffer_adjustment_step = BUFFER_ADJUSTMENTS.apply_stack_step(
    "affordability_buffer_applied", ADJUSTMENT_SET_ID,
    base_field="affordability_buffer_applied", unadjusted_output="affordability_buffer_unadjusted",
).relabel(writes={"adjustment_set_id": "buffer_adjustment_set_id", "adjustments_applied": "buffer_adjustments_applied"})
