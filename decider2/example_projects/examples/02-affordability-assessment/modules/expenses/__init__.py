"""Stage 4 assembly, including the one overlay that applies before capacity.

The stringency overlay changes which basis binds -- typically forcing the
internal norm on a named channel, or uplifting it for a period. It may only
RAISE the applied figure, and it may never reduce it below basis C, "because
the statutory floor is not adjustable" (spec 5.4).

That second clause is what makes `living_expenses_cents` an interesting overlay
target. The direction rule alone (`tightens_when(INCREASES)`) closes the
loosening case. The statutory floor needs a second declaration: an overlay
target may carry a FLOOR that is itself a computed value, and the register
validator checks that no admissible op can take the target below it. Both
checks run when the register is validated, not when an assessment runs.
"""

from decider2 import INCREASES, module, overlay_target, rung, tightens_when

from modules.expenses.bases import (
    DeclaredCategories,
    ExpenseBasis,
    declared_expenses_cents,
    exception_admitted,
    non_declaration_flag,
    statement_expenses_cents,
)
from modules.expenses.norms import (
    dependants_bucket,
    internal_norm_cents,
    norm_band_index,
    statutory_norm_cents,
    statutory_norm_ceiling_applied,
)

# --------------------------------------------------------------------------
# The overlay target declaration. This is the whole conservative-only
# mechanism at this site, and it is four lines.
# --------------------------------------------------------------------------
living_expenses_cents = overlay_target(
    "living_expenses_cents",
    base="living_expenses_unadjusted_cents",
    direction=tightens_when(INCREASES),
    floor="statutory_norm_cents",        # never below basis C, and the floor is a VALUE, not a constant
    scope_keys=("product_code", "channel_code", "risk_grade", "segment_code"),
    rung_section="overlays",
)

Expenses = module(
    dependants_bucket,
    norm_band_index,
    statutory_norm_cents,
    statutory_norm_ceiling_applied,
    internal_norm_cents,
    DeclaredCategories,
    declared_expenses_cents,
    non_declaration_flag,
    statement_expenses_cents,
    exception_admitted,
    ExpenseBasis,
    living_expenses_cents,
    name="expenses",
    contract="contracts/expenses.json",
    taps=[
        "expense_basis_code",
        "living_expenses_unadjusted_cents",
        # Every losing candidate, by name. `contest(retain_losers=True)` makes
        # these tap-addressable; a `max()` would have discarded them.
        "living_expenses_unadjusted_cents@A_declared",
        "living_expenses_unadjusted_cents@B_statement",
        "living_expenses_unadjusted_cents@C_statutory",
        "living_expenses_unadjusted_cents@D_internal",
        # And every overlay that moved it. Qualification by OVERLAY ID, not by
        # producing module -- there is only one producing module and a stack of
        # up to twelve overlays inside it. See FRAMEWORK-DEMANDS #8.
        "living_expenses_cents@overlay:*",
    ],
)
