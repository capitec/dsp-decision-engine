"""The declared overlay stack, and the unadjusted spine.

s5.10: "Overlays compose, and the composition order changes the answer, so the
order is part of the definition rather than an emergent property of where the
code happens to call `core.adjustments`."

That sentence is a demand on the framework, not on this project, and it is why
the stack is declared here as data and attached to the pipeline in
`pipelines/business_facility.py` with `.with_overlay_stack(STACK)` -- rather
than being wherever eleven call sites happened to land.

Eleven positions, four grains. Position 1 applies at the Event grain, positions
2-4 at the Entity grain, 5-8 at the Application grain, 9-11 at the decision and
pricing stages. The grain is part of the position, which is how PP-11's
double-count is *detected*: "a sector PD multiplier scoped to juristic entities
and a sector PD multiplier scoped to the business are two different overlays
with two different scopes, and stacking both over a group company that is also
in that sector double-counts."
"""

from decider2 import overlay, Shadow, overlays_off
from grains import Application, Entity, Event, Candidate

STACK = overlay.stack(
    name="business_facility_overlay_stack",

    positions=[
    # position           grain        target                        scope axes
        overlay.position(1,  Event,       "event_amount_thresholds",
                         scopes=("sector_code", "criticality_class")),
        overlay.position(2,  Entity,      "score",
                         scopes=("scorecard_id", "segment_code", "is_new_to_bank")),
        overlay.position(3,  Entity,      "probability_of_default",
                         scopes=("scorecard_id", "sector_grouping_code")),
        overlay.position(4,  Entity,      "risk_grade",
                         scopes=("is_natural_person",)),
        overlay.position(5,  Application, "financial_pd",
                         scopes=("segment_code", "financial_confidence_code")),
        overlay.position(6,  Application, "sector_pd_multiplier",
                         scopes=("sector_code", "sector_grouping_code")),
        overlay.position(7,  Application, "probability_of_default",
                         scopes=("segment_code", "channel_code", "product_code")),
        overlay.position(8,  Application, "risk_grade",
                         scopes=("product_code", "segment_code")),
        overlay.position(9,  Application, "minimum_acceptable_grade",
                         scopes=("product_code", "sector_code")),
        overlay.position(10, Application, ("appetite_maximum_facility",
                                           "surety_cover", "group_exposure_caps"),
                         scopes=("risk_grade", "sector_code")),
        overlay.position(11, Candidate,   "nominal_annual_rate",
                         scopes=("rate_cell_range", "risk_grade", "security_type")),
    ],

    # -----------------------------------------------------------------------
    # Three checks that run when an ADJUSTMENT SET IS PUBLISHED, not when an
    # application meets it. s5.10: "Two overlays occupying the same position
    # with overlapping scope is a conflict and must be detected when the
    # adjustment set is published, not when an application encounters it."
    # -----------------------------------------------------------------------
    on_publish=[
        overlay.no_scope_overlap_within_position(),
        overlay.no_cross_position_double_count(
            # PP-11's named collision: position 3 (entity, sector grouping) and
            # position 6 (business, sector) both reach a group company that is
            # itself in the sector. Declared as a pair to check, because the
            # framework cannot infer that two scope axes denote the same sector.
            pairs=[(3, 6)],
            shared_axis="sector_grouping_code",
        ),
        overlay.expiry_required(),          # core.adjustments property 5
        overlay.scope_is_reachable(),       # an overlay scoped to a value the
                                                # pipeline never produces is an error
                                                # at publish, not a silent no-op
    ],
)


# --------------------------------------------------------------------------
# The unadjusted spine.
#
# PP-11 requires `people_pd_unadjusted` -- "the same blend over
# probability_of_default_unadjusted, with identical weights" -- and s5.10
# requires `risk_grade_unadjusted` and `probability_of_default_unadjusted` at
# the business level. `core.adjustments` acceptance criterion 8 generalises it:
# "Any flow can be run with its adjustment stack disabled, producing the
# unadjusted answer alongside the adjusted one, WITHOUT A SEPARATE
# IMPLEMENTATION."
#
# `Shadow` is that. It re-evaluates a named sub-pipeline under a declared
# perturbation and suffixes the outputs. It is declared, so:
#   * it renders and diffs like any other node;
#   * it cannot drift from the primary, because it IS the primary;
#   * `lineage("people_pd_unadjusted")` answers.
#
# The cost is real and is stated rather than hidden. The perturbation reaches
# position 1, which is at the Event grain, so the shadow re-runs event
# classification -- the most expensive stage in the flow. Measured shape: the
# shadow roughly doubles the nested section. `share_prefix=True` asks the
# framework to prove from static lineage which prefix the perturbation cannot
# reach and to share it; for an application with no position-1 overlay in scope,
# that proof is total and the shadow collapses to a copy. FRAMEWORK-DEMANDS D18.
# --------------------------------------------------------------------------
from entities.adverse.classify import ClassifyEvent
from entities.adverse.gather import EntityAdverseFacts
from entities.adverse.verdict import EntityAdverseVerdict
from entities.scoring.families import EntityScoring
from people.blend import PeopleBlend
from people.caps import PeopleCaps

UnadjustedSpine = Shadow(
    ClassifyEvent | EntityAdverseFacts | EntityAdverseVerdict
    | EntityScoring | PeopleBlend | PeopleCaps,
    perturb=overlays_off(positions=[1, 2, 3, 4, 5, 6, 7, 8]),
    suffix="_unadjusted",
    share_prefix=True,
    name="unadjusted_spine",
    emits=[
        "entity_adverse_verdict_code_unadjusted",   # s5.6's required counterpart
        "risk_grade_unadjusted",
        "probability_of_default_unadjusted",
        "people_pd_unadjusted",
        "people_grade_unadjusted",
    ],
)

# --------------------------------------------------------------------------
# What this does NOT do, deliberately: it does not answer "why is this business
# a grade 7 this quarter when it was a grade 6 last quarter?" That question has
# three candidate answers -- the entity's data changed, the model changed, an
# overlay changed -- and the shadow separates only the third from the other two.
#
# Separating the first two needs the PREVIOUS assessment, which is project 09's
# replay harness, not this flow's. What this flow owes project 09 is the
# material to do it with: `adjustment_set_id`, the per-entity
# `adjustments_applied` list, the scorecard versions resolved at decision_date,
# and the per-entity input hashes. All four are in the audit record
# (doc 08 s8). The flow's job is to make the three causes SEPARABLE; deciding
# which one moved is a diff between two records.
# --------------------------------------------------------------------------
