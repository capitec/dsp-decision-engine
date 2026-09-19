"""The overlay operator registry, and the conservative-only proof.

Spec 5.6.2(1): "An overlay that would increase capacity is INVALID AT DEFINITION
TIME, not rejected at runtime. The statutory calculation is a floor on
conservatism, and a mechanism that can breach it is a reckless-lending mechanism
regardless of intent." Acceptance criterion 8 repeats it.

A runtime sign check is the obvious implementation and it fails three ways:
it is bypassable by a negative magnitude; it fires on an applicant rather than
on an approval; and it cannot be shown to the Credit Committee, who approve the
overlay months before any applicant meets it.

So the asymmetry is carried by the OPERATOR, in three layers, and all three run
when `register.json` is validated:

  1. Each operator declares its monotone direction in `base`, and its magnitude
     domain. There is no operator with a free-signed magnitude. `scale_down`
     takes (0, 1]; `scale_up` takes [1, inf). An author who wants to loosen by
     writing `scale_down(magnitude=1.4)` gets a pydantic bounds error on the
     magnitude, not a policy error on the effect -- which is a better error,
     because it fires at the field they typed.

  2. The overlay target declares which direction tightens it
     (`tightens_when(DECREASES)` on capacity, `tightens_when(INCREASES)` on the
     buffer and on expenses). Composing (2) with (1) gives admissibility, and
     the register refuses an inadmissible pair by name:

         overlay 'ADJ-2026-031' is not admissible: operator 'scale_up' is
         non-decreasing in its base, and target 'max_affordable_instalment_cents'
         tightens when it DECREASES. This overlay would increase capacity.
         Admissible operators for this target: scale_down, subtract_cents,
         cap_at_cents.

  3. Where a target declares a FLOOR that is itself a computed value --
     `living_expenses_cents` may never go below the statutory norm -- the
     operator must be floor-preserving. Every admissible operator for such a
     target is annotated `preserves_floor=True`, and one that is not cannot be
     named at all.

None of this can be evaluated on an applicant, which is the point: the
Committee's approval and the framework's check happen on the same artefact, at
the same moment, before anybody is declined by it.
"""

from decider2 import NON_DECREASING, NON_INCREASING, overlay_op
from pydantic import Field


@overlay_op(direction=NON_INCREASING, preserves_floor=True)
def scale_down(base: int, magnitude: float = Field(..., gt=0.0, le=1.0)) -> int:
    pass  # round_half_up(base * magnitude)


@overlay_op(direction=NON_DECREASING, preserves_floor=True)
def scale_up(base: int, magnitude: float = Field(..., ge=1.0, le=5.0)) -> int:
    pass  # round_half_up(base * magnitude)


@overlay_op(direction=NON_INCREASING, preserves_floor=False)
def subtract_cents(base: int, magnitude: int = Field(..., ge=0)) -> int:
    pass  # base - magnitude, floored at zero


@overlay_op(direction=NON_DECREASING, preserves_floor=True)
def add_cents(base: int, magnitude: int = Field(..., ge=0)) -> int:
    pass  # base + magnitude


@overlay_op(direction=NON_DECREASING, preserves_floor=True)
def add_pp(base: float, magnitude: float = Field(..., ge=0.0, le=0.50)) -> float:
    pass  # base + magnitude, for percentage-point targets such as the buffer


@overlay_op(direction=NON_INCREASING, preserves_floor=False)
def cap_at_cents(base: int, magnitude: int = Field(..., ge=0)) -> int:
    pass  # min(base, magnitude)


@overlay_op(direction=NON_DECREASING, preserves_floor=True)
def force_basis(base: int, magnitude: str = Field(..., pattern="^[A-D]_")) -> int:
    pass  # forces a named expense basis to bind; admissible only where the forced
          # basis is provably >= the contest's current winner, which the register
          # checks from the contest's declared `monotone=True`


# There is deliberately no `set_to`, no `multiply` and no `adjust`. Each would
# take a free-signed magnitude and each would move the check from definition
# time to runtime. The registry is closed; widening it is a code change with
# the same approval as any other, which is the right price for a new way to
# change an approved answer.

ADMISSIBLE = {
    "max_affordable_instalment_cents": ("scale_down", "subtract_cents", "cap_at_cents"),
    "retained_pct": ("add_pp", "scale_up"),
    "floor_cents": ("add_cents", "scale_up"),
    "living_expenses_cents": ("add_cents", "scale_up", "force_basis"),
}
# Derived, not written. `ADMISSIBLE` is what `decider overlays explain` prints
# for the Credit Committee, and it is computed by composing each operator's
# declared direction with each target's declared `tightens_when`. It appears
# here as a materialised artefact so that a reviewer can read it without a tool,
# and a CI check asserts it still matches what the composition produces.
