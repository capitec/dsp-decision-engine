"""The five overlay points, declared in code.

THE STRUCTURAL ANSWER to §13.17 / core-library Q11 -- "what is an overlay,
structurally?  It changes values, which makes it look like a parameter.  It
composes in a declared order that changes the answer, which makes it look like
logic.  It is separately owned, approved, versioned and expiring, which makes
it look like neither."

It looks like all three because it IS two artefacts, one of each class, and the
split falls at the point:

    the POINT is skeleton   -- code, in the pipeline expression, engineer-owned,
                               declares WHAT may be moved and IN WHICH DIRECTION
    the SET  is values      -- a table, Credit-Committee-owned, ad-hoc cadence,
                               declares WHETHER, BY HOW MUCH, FOR WHOM, IN WHAT
                               ORDER, UNTIL WHEN

Everything the spec demands falls out of that split, and none of it needs a
third mechanism:

  * "the order is declared, not emergent" -- `position` is a column of the set,
    and the set is sorted by it before application.  Composition order is data.
  * "the unadjusted value survives" -- an overlay point is `x -> x` and always
    also writes `x_unadjusted`.  The author cannot forget; there is no code.
  * "runs with the stack disabled through the same implementation" -- an empty
    set.  Every point becomes the identity, `x == x_unadjusted`, and the
    kernel is byte-identical because the set is a runtime array, not a
    compile-time constant.  Acceptance criterion 20, for free.
  * "an overlay evaluated outside its scope is an error, not a silent no-op" --
    scope evaluation is the framework's, at the point, and its outcome is a
    recorded THREE-valued fact: APPLIED / IN_FORCE_BUT_OUT_OF_SCOPE /
    NOT_IN_FORCE.  A silent no-op is unrepresentable.
  * "an overlay may tighten and may not loosen" (§13.19) -- `direction` is on
    the POINT, in code, reviewed by an engineer under a release.  A negative
    magnitude in the set is rejected at set validation against the point's
    declared direction, before anything runs.  The set cannot grant itself the
    right to loosen because the right is not in the set.
  * "it is never merged into the base artefact" -- there is nothing to merge
    into.  The point reads the base value as an input.

See FRAMEWORK-DEMANDS #1 (the largest deviation from doc 03 in this project).
"""

from __future__ import annotations

from decider2.overlay import Direction, OverlayKind, OverlayPoint, adjustment_set

# ---------------------------------------------------------------------------
# The set.  One artefact, one version, its own approval, its own expiry --
# owned by NONE of the artefacts it modifies.  Resolved as-of `decision_date`
# exactly like a table, and stamped into every decision record as
# `adjustment_set_id`.
# ---------------------------------------------------------------------------

FLEX_ADJUSTMENTS = adjustment_set(
    name="flex_loan_adjustments",
    source="config/flex_loan/adjustments/",
    owner="credit_committee",
    effective_dated=True,
    capacity=32,               # live count runs 6-20; declared, like every capacity
    # Required fields.  `rationale` and `review_date` are required *by the
    # schema*, which is the only way they stop being a courtesy (§6.3).
    requires=["rationale", "owner", "approval_reference", "effective_from",
              "effective_to", "review_date", "position", "enabled"],
    # An overlay past its review date without renewal SURFACES.  It does not
    # silently stop applying (that would change prices without an approval) and
    # it does not silently continue (that is the four-year-old tightening).
    on_review_date_passed="warn_and_apply_until_effective_to",
)

# ---------------------------------------------------------------------------
# The five points.  Three sit on VALUES.  One sits on a PARAM.  One sits on a
# TABLE CELL.  That a point can target all three is the thing doc 03 cannot do
# at all -- see FRAMEWORK-DEMANDS #2.
# ---------------------------------------------------------------------------

SCORE_SHIFT = OverlayPoint(
    id="risk.score",
    target="score",                                   # a VALUE
    kind=OverlayKind.ADDITIVE,                        # points
    direction=Direction.EITHER,                       # a shift may go both ways
    scope_axes=["scorecard_id", "segment_code", "channel_code", "product_code"],
    set=FLEX_ADJUSTMENTS,
    writes_unadjusted="score_unadjusted",
)

SCALING_CHANGE = OverlayPoint(
    id="risk.scaling",
    target=param("calibration.points_to_double_odds"),  # a PARAM, not a value
    kind=OverlayKind.REPLACE,
    direction=Direction.EITHER,
    scope_axes=["scorecard_id", "segment_code"],
    set=FLEX_ADJUSTMENTS,
    # Resolved BEFORE the params bundle is built, so the bundle's type is
    # unchanged and nothing recompiles.  The resolved value and the base value
    # are both stamped in the record.
    resolve_at="params_binding",
    writes_unadjusted="points_to_double_odds_unadjusted",
)

ODDS_MULTIPLIER = OverlayPoint(
    id="risk.pd",
    target="probability_of_default",
    kind=OverlayKind.MULTIPLICATIVE,
    direction=Direction.INCREASE_ONLY,                # PD may only be made worse
    scope_axes=["scorecard_id", "segment_code", "channel_code", "grade_range"],
    set=FLEX_ADJUSTMENTS,
    writes_unadjusted="probability_of_default_unadjusted",
)

BOUNDARY_SHIFT = OverlayPoint(
    id="risk.grade_boundary",
    target=table("grade_boundaries").cells,           # TABLE CELLS, not the grade
    kind=OverlayKind.ADDITIVE,                        # percentage points of PD
    direction=Direction.TIGHTEN_ONLY,                 # boundaries may only move down
    scope_axes=["segment_code", "grade_range"],
    set=FLEX_ADJUSTMENTS,
    # The base table is NOT edited.  The overlay is applied to the cell value
    # at read time, and `rate_cell_id`-style attribution records both numbers.
    writes_unadjusted="risk_grade_unadjusted",
)

CAP_REDUCTION = OverlayPoint(
    id="policy.ceilings",
    target=["amount_cap", "term_cap"],
    kind=OverlayKind.MULTIPLICATIVE | OverlayKind.ABSOLUTE,
    direction=Direction.REDUCE_ONLY,                  # §5.5: "may only reduce"
    scope_axes=["channel_code", "segment_code", "grade_range", "product_code"],
    set=FLEX_ADJUSTMENTS,
    # A cap overlay enters the waterfall's CHAIN as its own row, attributed to
    # the adjustment set rather than to a rule -- because "reduced to R76 000
    # by a policy overlay approved under CC-2026-31, expiring 2027-01-31" is a
    # different answer to the client than "a rule bound it".
    joins_chain=["amount_cap_chain", "term_cap_chain"],
    chain_attribution="adjustment_set",
    writes_unadjusted=["amount_cap_unadjusted", "term_cap_unadjusted"],
)

RATE_ADD_ON = OverlayPoint(
    id="pricing.rate",
    target="nominal_annual_rate",
    kind=OverlayKind.ADDITIVE,                        # basis points
    direction=Direction.INCREASE_ONLY,                # never below the published card
    scope_axes=["grade_range", "term_range", "amount_band_range", "channel_code"],
    set=FLEX_ADJUSTMENTS,
    writes_unadjusted="nominal_annual_rate_card_value",
    # Two properties this point must carry that the others do not.
    #
    # (a) The statutory ceiling is re-checked AFTER the add-on, not only at
    #     card validation (§5.7(e), acceptance criterion 22).  `post_assert` is
    #     evaluated on the adjusted value and a failure is a hard failure.
    post_assert="nominal_annual_rate <= tables.statutory_rate_ceiling.value",
    # (b) The add-on participates in the solve like any other rate movement,
    #     INCLUDING its effect on band-edge non-monotonicity.  So its scope
    #     boundaries are breakpoints of the search domain, and it says so.
    #     Without this the declared inversion list is incomplete and the search
    #     is wrong in a way no test would find (§5.8.1(2)).
    contributes_breakpoints="amount_band_range",
    # (c) Never merged into the card.  "Ever."  A `merge_into=` argument does
    #     not exist on OverlayPoint, which is the only enforcement that works.
)

ALL_POINTS = [SCORE_SHIFT, SCALING_CHANGE, ODDS_MULTIPLIER, BOUNDARY_SHIFT,
              CAP_REDUCTION, RATE_ADD_ON]
