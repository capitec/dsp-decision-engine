"""What may be overlaid, declared in code, once.

An overlay is a fourth change class. Doc 08 §2 gives three — values, interiors,
skeleton — and an overlay is none of them:

  * it is not a VALUE, because it is effective-dated, ordered in a declared
    stack, scoped by a predicate over record attributes, and individually
    attributable in the output. A params bundle is a flat set of scalars with a
    fixed type; it has no room for any of that.
  * it is not an INTERIOR, because it does not change what a module does. It
    changes the ANSWER a module already gave, from outside, under a different
    approval, with the pre-overlay answer preserved beside it.
  * it is emphatically not SKELETON.

So: `overlay` is class four. Free at runtime like a value (no compile — the
stack is arrays), approved like an interior, and audited like neither.

THE MECHANISM. A value becomes overlayable by being declared an OverlaySurface.
Declaration is the ONLY route: the overlay resolver resolves a target id against
the surface registry, and a target that is not in the registry does not fail at
run time, it fails at AUTHORING time with the owner's name in the message. Every
statutory parameter in this project — suspension rules, notice periods, contact
hours, frequency caps, prescription handling, the discount authority grid — is
simply never registered. That is the asymmetry spec §5.4 demands be "enforced,
not documented and trusted": an overlay cannot weaken a statutory protection
because there is no name it can write down that reaches one.

Below is the complete list. It is deliberately short and deliberately in one
file, because the security property is "read this file and you know the blast
radius of every overlay the Bank can approve".
"""

from decider2 import OverlaySurface, combiner
from decider2.types import f8, i1, i2

from ..matrix.dimensions import COLLECTIONS_BAND
from ..matrix.grid import TREATMENT_MATRIX
from ..path.sequence import CONTACT_INTERVALS

# ------------------------------------------------------------- score layer ---

SCORE = OverlaySurface(
    id="collections.score",
    target="collections_score",
    kinds=("score_shift", "scaling_change", "odds_multiplier"),
    combiner=combiner.ORDERED,          # stack order is declared, never emergent
    unadjusted_name="collections_score_unadjusted",
    owner="collections_strategy",
    approver="model_risk_and_validation",
    max_live=9,
    requires=("rationale", "approval_reference", "effective_from", "effective_to",
              "review_date", "volume_effect_estimate"),
)

BAND = OverlaySurface(
    id="collections.band",
    target="collections_band_code",
    kinds=("band_edge_shift",),
    combiner=combiner.ORDERED,
    unadjusted_name="collections_band_code_unadjusted",
    owner="collections_strategy",
    approver="model_risk_and_validation",
    max_live=4,
    requires=("rationale", "approval_reference", "effective_from", "effective_to",
              "review_date", "volume_effect_estimate", "capacity_effect_estimate"),
    # ^ capacity_effect_estimate is MANDATORY on this surface and nowhere else.
    #   Spec §5.3: "a one-notch band shift on the unsecured book moves roughly
    #   74 000 accounts and adds about 9 000 agent call attempts to a pool of
    #   46 000. The overlay's approval must therefore show its volume effect."
    #   A required field is how "must show" becomes true.
)

# ------------------------------------------------------------ matrix layer ---

MATRIX_INTENSITY = OverlaySurface(
    id="matrix.intensity",
    target=(TREATMENT_MATRIX, "treatment_intensity"),
    kinds=("intensity_dial",),
    combiner=combiner.ORDERED_CLAMPED(lo=1, hi=5),
    unadjusted_name="treatment_intensity_unadjusted",
    scope_dimensions=("arrears_bucket_code", "collections_band_code", "balance_band_code",
                      "contact_band_code", "product_family_code", "channel_code",
                      "cohort_code"),
    owner="collections_strategy",
    approver="collections_credit_forum",
    max_live=15,
)

MATRIX_RETRIES = OverlaySurface(
    id="matrix.retries",
    target=(TREATMENT_MATRIX, "permitted_retries"),
    kinds=("intensity_dial",),
    combiner=combiner.ORDERED_CLAMPED(lo=0, hi=4),
    unadjusted_name="permitted_retries_unadjusted",
    scope_dimensions=MATRIX_INTENSITY.scope_dimensions,
    owner="collections_strategy",
    approver="collections_credit_forum",
)

MATRIX_COOLING_OFF = OverlaySurface(
    id="matrix.cooling_off",
    target=(TREATMENT_MATRIX, "cooling_off_days"),
    kinds=("intensity_dial",),
    combiner=combiner.TIGHTEN_ONLY(direction="increase"),
    # ^ cooling-off may only ever be LENGTHENED by an overlay. The combiner is
    #   max(base, overlaid), so an overlay proposing 2 days over a base of 7
    #   resolves to 7 and the attempt is recorded as a no-effect application
    #   rather than silently discarded. Spec §5.4: overlays "may only ever be
    #   MORE conservative than a statutory rule, never less."
    unadjusted_name="cooling_off_days_unadjusted",
    scope_dimensions=MATRIX_INTENSITY.scope_dimensions,
    owner="collections_strategy",
    approver="collections_credit_forum",
)

TREATMENT_AVAILABILITY = OverlaySurface(
    id="matrix.treatment_availability",
    target=(TREATMENT_MATRIX, "treatment_code"),
    kinds=("suppression", "substitution"),
    # NOT a free write. `suppression` removes a treatment over a scope;
    # `substitution` replaces it with a NAMED alternative that must already be
    # permitted for that product family. Neither can introduce a treatment the
    # matrix does not permit in that cell (spec §5.4).
    combiner=combiner.SUPPRESS_THEN_SUBSTITUTE,
    unadjusted_name="treatment_code_unadjusted",
    scope_dimensions=MATRIX_INTENSITY.scope_dimensions,
    owner="collections_strategy",
    approver="collections_credit_forum",
    validate=("substitute_permitted_for_family", "substitute_not_higher_intensity"),
)

ALLOCATION_WEIGHT = OverlaySurface(
    id="allocation.weight",
    target="pool_reserved_share",
    kinds=("allocation_weighting",),
    combiner=combiner.ORDERED,
    unadjusted_name="pool_reserved_share_unadjusted",
    scope_dimensions=MATRIX_INTENSITY.scope_dimensions + ("pool_code",),
    owner="collections_strategy",
    approver="collections_credit_forum",
    validate=("feasible_against_fairness_floors",),
    # ^ spec §5.9: "an allocation weighting dial is scored against the fairness
    #   floors of this stage, not over them. An overlay that would push any
    #   account past the untouched-days limit is rejected at authoring with the
    #   constraint it would breach named." That is a SIMULATION run at authoring
    #   time, which means the surface validator needs the allocator. See
    #   FRAMEWORK-DEMANDS #16 — this is the ugliest coupling in the sketch.
)

INTERVAL_TIGHTENING = OverlaySurface(
    id="path.min_interval",
    target=(CONTACT_INTERVALS, "min_interval_days"),
    kinds=("intensity_dial",),
    combiner=combiner.TIGHTEN_ONLY(direction="increase"),
    unadjusted_name="min_interval_days_unadjusted",
    owner="collections_strategy",
    approver="collections_credit_forum",
    co_sign="regulatory_compliance",
)

# ===========================================================================
# DELIBERATELY ABSENT, AND THE ABSENCE IS THE POINT
#
#   suspension rules (all 20)          permitted contact hours
#   notice periods                     contact frequency caps
#   prescription period / handling     the discount authority grid
#   arrangement minimums               the absolute caps in path/intervals.yaml
#   the business-day calendar          authority levels L1-L5
#
# None of these is an OverlaySurface. There is no flag to set and no authority
# level that unlocks them. They change through Compliance's or Credit Risk
# Policy's own release path, with their own approval, as effective-dated
# artefacts. An overlay document naming one is rejected at load with:
#
#   overlay OV-2026-091 targets 'suspension.117.cap_per_7d', which is not an
#   overlay surface. That parameter is owned by regulatory_compliance and
#   changes only through its own release path. Overlay surfaces are:
#   collections.score, collections.band, matrix.intensity, matrix.retries,
#   matrix.cooling_off, matrix.treatment_availability, allocation.weight,
#   path.min_interval.
#   This attempt has been recorded as OVERLAY-REJECT-2026-09-19-0004.
# ===========================================================================

REGISTRY = (SCORE, BAND, MATRIX_INTENSITY, MATRIX_RETRIES, MATRIX_COOLING_OFF,
            TREATMENT_AVAILABILITY, ALLOCATION_WEIGHT, INTERVAL_TIGHTENING)
