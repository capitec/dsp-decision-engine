"""Overlays over a published tree (spec 04 §5.3.4). Two mechanisms, one governance shape:

  - **Threshold shift** (volume dial, cut-off shift) -- a named tree param moves, through
    `TreeConfig`'s own `params={tree_name: {param_name: value}}` (see `campaign23.py`'s
    `param_threshold` calls and `pipeline.py`, which runs the tree twice, once with the
    overlay params and once without, to get `leaf` and `unadjusted_leaf` from the *same*
    `TreeConfig` -- §5.3.4 requirement 2, §5.14.3's "one implementation").
  - **Cap reduction** -- a scalar value (the tier ceiling, after the tree has answered)
    overlaid through `credit_core.adjustments.AdjustmentRegister`, reused unmodified: its
    `kind="cap_adjustment"` is already declared tighten-only for a `multiply <= 1.0` effect
    (00's own `_TIGHTEN_RULES`), which is exactly "amount overlays may only reduce" (§5.5
    requirement 6) enforced at *definition* time, for free.

Both kinds share one identity shape (`TreeOverlay`, below) -- id, scope, owner, approval
reference, rationale, effective window, review date (§5.3.4 requirement 4) -- but threshold
shifts are **not** built on `credit_core.adjustments.Adjustment`: that class's tighten-only
check (`_is_tightening`) only recognises five closed kinds (`score_shift`,
`odds_multiplier`, `rate_addon`, `cap_adjustment`, `buffer_adjustment`), none of which is "set
a tree param to a new literal value" -- constructing `Adjustment(kind="volume_dial",
tighten_only=True, ...)` raises `ValueError` at definition, before any cycle runs, because
`_TIGHTEN_RULES` has no entry for it. Project 03 hit the identical wall over its own
"boundary shift" and "scaling change" overlay kinds (03 NOTES.md "Gaps in what I consumed") --
two independent projects needing the same closed registry extended is a real finding, not a
one-off; see NOTES.md "Framework friction". `TreeOverlay` below is a small, independent
dataclass with the same identity/governance fields and its own tighten-only check scoped to
this project's two threshold-shift kinds.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from decider import missing_as, param, step

from credit_core.adjustments import Adjustment, AdjustmentEffect, AdjustmentRegister

VOLUME_DIAL = "volume_dial"
CUTOFF_SHIFT = "cutoff_shift"

# A floor threshold ("x >= thresh") is tighter the higher it moves; a ceiling ("x <= thresh")
# is tighter the lower it moves. Declared per kind, exactly as 00's own `_TIGHTEN_RULES` is
# declared per kind -- an undeclared kind cannot be marked tighten-only, on purpose.
_TREE_OVERLAY_DIRECTION = {VOLUME_DIAL: "floor", CUTOFF_SHIFT: "floor"}


@dataclass(frozen=True)
class TreeOverlay:
    """A threshold-shift overlay's identity and governance record (§5.3.4 requirement 4):
    id, the tree param it targets, its scope, owner, approval, rationale, effective window
    and review date. Application is `pipeline.py` passing `new_value` as that param's
    override -- this object never touches the tree itself (§5.3.4 requirement 1: "an overlay
    is not an edit")."""

    overlay_id: str
    kind: str                      # VOLUME_DIAL | CUTOFF_SHIFT
    tree_param: str
    published_value: float
    new_value: float
    scope: dict                    # e.g. {"campaign_id": 23}
    owner: str
    approval_reference: str
    rationale: str
    effective_from: date
    effective_to: date | None
    review_date: date
    enabled: bool = True

    def __post_init__(self) -> None:
        direction = _TREE_OVERLAY_DIRECTION[self.kind]
        tightened = self.new_value > self.published_value if direction == "floor" else self.new_value < self.published_value
        if not tightened:
            raise ValueError(
                f"{self.overlay_id}: {self.kind} on {self.tree_param!r} loosens the {direction} "
                f"({self.published_value} -> {self.new_value}); threshold-shift overlays are tighten-only (§5.3.4)")

    def in_scope(self, provenance: dict) -> bool:
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


def resolve_tree_params(overlays: list[TreeOverlay], provenance: dict, decision_date: date,
                         ) -> tuple[dict[str, float], tuple[str, ...]]:
    """Every in-scope, in-force overlay's `{tree_param: new_value}`, ready to pass as
    `TreeConfig.run(df, params={tree_name: resolve_tree_params(...)})`, plus the ids that
    actually applied (in declared stack order -- `overlay_stack_id`, §4.4). An overlay
    applied outside its declared scope, or past its review date, is an **error that fails
    the cycle** (§5.3.4 requirements 5-6), never a silent no-op -- callers must check
    `overlay_errors` first."""
    resolved: dict[str, float] = {}
    applied: list[str] = []
    for ov in overlays:
        if not ov.in_scope(provenance):
            continue  # not this overlay's campaign/segment/etc. -- correctly not applied, not an error
        if not ov.in_force(decision_date):
            continue  # expired or not yet effective -- correctly not applied
        resolved[ov.tree_param] = ov.new_value
        applied.append(ov.overlay_id)
    return resolved, tuple(applied)


def overlay_stack_id(applied_overlay_ids: tuple[str, ...]) -> str:
    """§4.4: "the ordered set of adjustments in force for a (campaign, cycle), as resolved
    at cycle_date" -- stable and content-derived, so two cycles that resolved the same
    overlays in the same order carry the identical id (§5.3.4 requirement 3: "the
    composition order changes the answer... it is recorded in overlay_stack_id"), and two
    that did not are visibly different, which is what §5.3.4(a)'s node-volume decomposition
    needs to detect an overlay-caused movement at all."""
    return "OS-NONE" if not applied_overlay_ids else "OS-" + "+".join(applied_overlay_ids)


def overlay_errors(overlays: list[TreeOverlay], as_of: date) -> list[str]:
    """§5.3.4 requirement 5: an overlay past its review date without renewal must surface
    before the cycle that would run under it, and the cycle must refuse to apply it."""
    return [f"{ov.overlay_id} is past its review date ({ov.review_date}) without renewal"
            for ov in overlays if ov.due_for_review(as_of) and ov.enabled]


# --- Cap reduction: a scalar overlay over the tree's `tier_ceiling` output, through
# `credit_core.adjustments` unmodified (see module docstring). ---

CAP_ADJUSTMENT_SET_ID = "AS-CAMPAIGN23-2026.09"

from campaign_trees.campaign23 import NODE3_BEHAVIOUR_SCORE_PARAM, NODE6_DISCRETIONARY_INCOME_PARAM

# The two threshold-shift worked examples from §5.3.4's own table, over campaign 23's TREE_V1
# (published defaults 2 200 / 620.0 -- see campaign23.py).
TREE_OVERLAYS: list[TreeOverlay] = [
    TreeOverlay(
        overlay_id="ADJ-C23-VOL-001", kind=VOLUME_DIAL, tree_param=NODE6_DISCRETIONARY_INCOME_PARAM,
        published_value=2200.0, new_value=2600.0, scope={"campaign_id": 23},
        owner="Campaign owner (Flex Loan top-up)", approval_reference="FORUM-2026-09-14",
        rationale="Bring targeted volume from 412 000 to the 350 000 the contact centre has "
                   "committed to staff (§5.3.4 worked example)",
        effective_from=date(2026, 9, 1), effective_to=date(2026, 12, 1), review_date=date(2026, 11, 15),
    ),
    TreeOverlay(
        overlay_id="ADJ-C23-CUT-001", kind=CUTOFF_SHIFT, tree_param=NODE3_BEHAVIOUR_SCORE_PARAM,
        published_value=620.0, new_value=640.0, scope={"campaign_id": 23},
        owner="Credit Risk Policy", approval_reference="CRC-2026-038",
        rationale="Tighten campaign 23 by one notch after early-arrears deterioration on the "
                   "August cohort (§5.3.4 worked example)",
        effective_from=date(2026, 10, 1), effective_to=date(2026, 12, 1), review_date=date(2027, 4, 1),
    ),
]

CAP_ADJUSTMENTS = AdjustmentRegister([
    Adjustment(
        adjustment_id="ADJ-C23-CAP-001", kind="cap_adjustment", target="tier_ceiling",
        effect=AdjustmentEffect("multiply", 0.80), scope={"campaign_id": 23, "offer_tier_code": 1},
        stack_position=1, owner="Credit Risk Policy", approval_reference="CRC-2026-041",
        rationale="Tier A funding constrained for the October cycle; 20% off while reviewed",
        effective_from=date(2026, 10, 1), effective_to=date(2026, 11, 1), review_date=date(2026, 10, 25),
        tighten_only=True,
    ),
])


def apply_tier_ceiling_cap_adjustment(
    tier_ceiling: float, campaign_id: int, offer_tier_code: int, decision_date: date,
    adjustment_stack_enabled: bool = param(True),
) -> tuple[float, float, str, list[str]]:
    """Not `AdjustmentRegister.apply_stack_step` (its provenance is fixed to
    `product_code`/`segment_code`/`channel_code`/`scorecard_id` -- see 00's own NOTES.md on
    that wrapper's convenience-vs-generality trade-off); this project's cap reduction scopes
    to `campaign_id` and `offer_tier_code` instead, so it calls `apply_stack` directly, per
    that wrapper's own docstring ("a consumer needing a wider scope... builds its own step").
    `adjustment_stack_enabled=False` is the 5% overlays-off shadow evaluation (§5.3.4
    requirement 2, §00 §7.6) -- the same code path, not a second implementation."""
    result = CAP_ADJUSTMENTS.apply_stack(
        "tier_ceiling", tier_ceiling, {"campaign_id": campaign_id, "offer_tier_code": offer_tier_code},
        decision_date, CAP_ADJUSTMENT_SET_ID, stack_enabled=adjustment_stack_enabled,
    )
    return result.adjusted_value, result.unadjusted_value, result.adjustment_set_id, list(result.adjustments_applied)


apply_tier_ceiling_cap_adjustment_step = step(
    apply_tier_ceiling_cap_adjustment,
    # "tier_ceiling_before_cap_overlay", not "unadjusted_tier_ceiling": the tree's own
    # `tier_ceiling` output is already the *threshold-overlay-adjusted* value (from
    # `tree_adjusted`, §5.3.4's other overlay mechanism) -- this step's "unadjusted" is a
    # second, independent axis (cap-reduction on/off), not the same one the tree's own
    # `unadjusted_leaf` pair names. See NOTES.md "Spec problems".
    outputs=("advertised_tier_ceiling", "tier_ceiling_before_cap_overlay", "cap_adjustment_set_id",
             "cap_adjustments_applied"),
)
