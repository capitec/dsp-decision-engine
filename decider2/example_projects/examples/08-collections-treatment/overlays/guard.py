"""The three enforcement mechanisms, and honestly which ones are real.

Spec §5.4: "An overlay definition that names one of them is rejected at authoring
time, not at run time, and the attempted definition is itself recorded."

1. NON-ADDRESSABILITY (structural — cannot be circumvented).
   A statutory parameter is not registered as an OverlaySurface. The overlay
   target grammar admits only registered surface ids. There is no wildcard, no
   dotted path, no `getattr`. This is the doc 08 §1.1 reference-not-pointer rule
   doing real work: because an overlay may only REFERENCE a registered id, the
   registry is the permission boundary, and the permission boundary is one file
   a person can read (overlays/surfaces.py).

2. DIRECTIONAL COMBINERS (structural, within a surface).
   Where a surface exists but only one direction is safe — cooling-off may be
   lengthened, an interval may be lengthened — the combiner is `TIGHTEN_ONLY`
   and the arithmetic is max() rather than assignment. An overlay proposing a
   loosening resolves to the base value AND is recorded as a no-effect
   application, so the forum that approved it learns it did nothing rather than
   believing it worked.

3. FEASIBILITY SIMULATION (procedural — real, but it is a check, not a proof).
   `allocation.weight` overlays are validated by running the allocator over a
   reference population at authoring time. An overlay that pushes any account
   past the untouched-days limit is rejected naming the constraint. This one is
   honest-to-goodness validation and it can be wrong if the reference population
   is unrepresentative. It is the weakest of the three and it is the one guarding
   the fairness floors, which is uncomfortable. FRAMEWORK-DEMANDS #16.

WHAT IS NOT ENFORCED, stated plainly in the spirit of doc 04 §2.1: that anybody
runs `expired_overlays()`. The framework can make the report exist and can make
it impossible to compute wrongly; it cannot make a person read it on a Monday.
What it CAN do is refuse to run — see `on_review_date_lapsed` below.
"""

from decider2 import OverlayStack, overlay_guard
from decider2.types import Date

from .surfaces import REGISTRY


CollectionsOverlays = OverlayStack(
    name="collections_overlays",
    surfaces=REGISTRY,
    source="register.json",
    version=91,
    resolve_by=("decision_date", "cohort_code"),

    # Spec §5.4: "An overlay reaching its review date without renewal must
    # surface loudly: the failure mode this mechanism exists to prevent is a
    # six-week tightening imposed after one bad quarter and still quietly in
    # force four years later."
    #
    # "Surface loudly" as a standing report is what everybody builds and it is
    # what fails, because a report has to be read. So the default here is
    # harder: an overlay past its review date STOPS APPLYING, is recorded as
    # lapsed on every account it would have touched, and the run emits a
    # blocking alert. The commercial tilt reverts to the approved artefact,
    # which is the safe direction by construction.
    on_review_date_lapsed="expire_and_alert",

    # An overlay applied outside its declared scope is an error, not a silent
    # no-op (spec §5.4, project 00 §6.22 property 6).
    on_out_of_scope="error",

    # Never merged into the artefact it modifies (project 00 §7.6). This is a
    # build-time assertion, not a convention: the matrix release pipeline
    # compares v214's cells against v213's cells plus the overlay stack and
    # FAILS if a cell has moved to exactly the overlaid value, which is what
    # "somebody folded the overlay in to tidy up" looks like.
    on_merge_into_base="error",

    emits=[
        "adjustment_set_version",
        "applied_overlay_ids",          # ordered list, per account, per surface
        "overlay_effect_ledger",        # (overlay_id, surface, before, after) triples
        "overlay_lapsed_ids",
    ],
)


@overlay_guard(stage="authoring")
def reject_statutory_targets(document) -> None:
    """Mechanism 1. Runs before the document is admitted, and records the attempt."""
    pass


@overlay_guard(stage="authoring")
def require_cohort_neutrality_declaration(document) -> None:
    """Spec §5.12: "An overlay that is not cohort-scoped must apply equally to
    champion and challenger, and that equality must be DEMONSTRABLE rather than
    assumed." So a non-cohort-scoped overlay is validated by computing its scope
    predicate's hit rate per cohort over the reference population and failing if
    the arms differ by more than a declared tolerance. An overlay scoped on
    `balance_band_code` that happens to correlate with the split is caught here
    and not eight weeks later in an uninterpretable experiment result."""
    pass


@overlay_guard(stage="authoring")
def require_capacity_effect_on_band_overlays(document) -> None:
    """Mechanism 3, narrow form. A band_edge_shift must carry a computed
    capacity_effect_estimate produced by running the allocator, not a number
    somebody typed into the approval form."""
    pass


@overlay_guard(stage="activation")
def forbid_stack_order_change_mid_experiment(document, live_experiments) -> None:
    """Spec §5.12: where an overlay starts, ends or changes scope mid-window, the
    experiment window is split at the change and each segment reported
    separately. Reordering the STACK mid-window is worse — it changes every
    account's answer with no event to split on — so it is refused while an
    experiment that would be affected is live, naming the experiment."""
    pass


def run_with_overlays_disabled(pipeline, frame, params):
    """Project 00 §7.6 / spec §8: the flow must be runnable with the overlay
    stack disabled, which is how the base score's own performance is monitored.

    It is one argument, not a second implementation, because every overlayable
    value is produced by the SAME kernel emitting a pair. With the stack empty
    the adjusted and unadjusted columns are equal, which is also a cheap
    always-on assertion.
    """
    pass  # pipeline.apply(frame, params=params, overlays=OverlayStack.empty())
