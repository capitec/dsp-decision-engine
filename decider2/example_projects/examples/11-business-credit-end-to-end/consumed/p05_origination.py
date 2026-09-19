"""Project 05's thirteen stages, bound as this project's phases O2-O14, O16.

Spec 5.3: "This is project 05's flow. It is referenced, not restated." The whole
of the origination assessment -- 1 280 of 1 900 decision points -- arrives
through this file, and this file contains no credit logic at all. What it
contains is the five things a consumption site has to say:

    which component, at which pinned major, under which roles, in which temporal
    view, with which gaps declared.

The fifth is the one doc 03 has no place for. Spec 2.2 consequence 1 -- "no
consumed component may be re-specified here" -- is an obligation on the
implementation, and the only way to hold it is to make re-specification visibly
expensive and mechanically detectable. `gaps=` plus the build check in
consumed/__init__.py is that mechanism.
"""

from decider2 import consume, Branch
from consumed.manifest import MANIFEST, p05_major
from roles import DIRECTOR, SURETY, CORPORATE_GUARANTOR, SOLE_PROPRIETOR, PERIPHERAL, GROUP
from time.bitemporal import KNOWN

p05 = consume("credit.business_origination", manifest=MANIFEST, major=p05_major)

# --------------------------------------------------------------------------
# O2 -- structure resolution. 05 5.1, unchanged: 3 levels, 40 entities, 5.0%
# materiality floor, path products to 4dp, cycle truncation, de-duplication by
# entity_key with summed ownership and most-senior role.
#
# What changes: it now runs AS AT A DATE over a bi-temporal fact store. The
# component takes a structure; this project hands it one resolved from history,
# and the `view=` argument is what makes which history visible at the use site.
# --------------------------------------------------------------------------
StructureResolution = p05.structure_resolution.over(
    "known.entities", view=KNOWN,
).emits("structure_delta")            # gap `structure_delta`, resolution COMPOSE

# --------------------------------------------------------------------------
# O4 -- the 20 business absolute rules, B-AROD-01..20. Consumed unchanged.
#
# What changes: seven of the twenty MEAN something different when the facility
# already exists (spec 5.3 O4). A rule that declines a new application cannot
# decline a facility the Bank has already written -- it produces a finding.
#
# The spec's requirement is that this mapping be DECLARED DATA in one place and
# not re-derived by each entry point. So it is a disposition matrix, in exactly
# the shape 05's own 14x3 criticality matrix already has, and the component's
# rule set is consumed untouched beside it.
# --------------------------------------------------------------------------
AbsoluteRules = p05.business_absolute_rules.dispositioned_by(
    "tables/governance/arod_lifecycle_disposition.csv",   # 20 rules x 9 entry points
    owner="business_credit_risk_policy",
    # A rule with no disposition for an entry point that runs it is a build
    # error. 180 cells, zero blanks permitted -- which is how the seven
    # lifecycle-sensitive rules stay seven and do not quietly become five.
    total=True,
)

# --------------------------------------------------------------------------
# O5 -- per-entity rules, event classification, entity scoring. 05 5.4-5.7 in
# full: 14 entity rules x 3 criticality classes, the 84-cell threshold table,
# 34 classification rules, the 84-cell ageing decay, two scorecard families
# with thin-file variants. None of it restated.
#
# The roles are where the reuse actually happens. Five role scopes, one
# component, and the vocabulary mapping declared in roles.py rather than here.
# --------------------------------------------------------------------------
EntityAssessment = p05.entity_assessment.under(
    [DIRECTOR, SURETY, CORPORATE_GUARANTOR, SOLE_PROPRIETOR, PERIPHERAL],
).with_partial(
    # Spec 5.17.2 / 5.17.5: the daily pass needs ONE event on ONE entity, and
    # one entity's verdict re-rolled -- 9 400 and 48 000 times a day. Project 05
    # publishes this as an OPTIMISATION (its 8, 500 ms partial re-assessment),
    # not as an interface. Gap `single_entity_rescore`, resolution EXTEND,
    # owner project 05, review 2027-03-31. Until it lands, this project waits
    # rather than forking -- fork pressure #1 and #2.
    unit="entity",
    equivalent_to="whole",
    tested_over=5_000,     # spec 10 acceptance 27
)

# --------------------------------------------------------------------------
# O6 -- the people blend, PP-01..PP-11. Consumed. The partial re-blend is not.
# --------------------------------------------------------------------------
PeopleBlend = p05.people_blend.under([DIRECTOR, SOLE_PROPRIETOR])
PartialReBlend = PeopleBlend.partial(over="one entity's pd", weights="unchanged")

# --------------------------------------------------------------------------
# O7 -- spreading, haircuts, sector benchmarking. 05 5.9's ~600-label mapping,
# annualisation, audit-level haircuts, statement-age haircuts, the bank-turnover
# alternative with its R1 500 000 cap and grade cap of 6.
#
# Three changes, and the third is the hardest thing in the project:
#   1..6 periods instead of 1..3;
#   periods may be on two accounting bases (spec 11 change scenario 4);
#   a COVENANT freezes its own spreading rules -- the review's DSCR and the
#   covenant's DSCR are two different numbers for the same business in the same
#   year, both correct (spec 5.3 O7 item 3).
#
# `.scoped(...)` is fork pressure #3: O7 spreads 48 lines and computes 11
# measures; a DSCR covenant needs two lines and one ratio, 2.4 M times a year.
# The requirement is that the SAME implementation serve both, invoked for a
# declared subset, without a second mapping table appearing.
# --------------------------------------------------------------------------
Spreading = p05.financial_spreading.with_periods(1, 6).basis_aware()
CovenantScopedSpread = Spreading.scoped(
    lines="from the bound covenant definition",
    mapping_version="from the bound covenant definition's closure",
    equivalent_to="whole",
)

# --------------------------------------------------------------------------
# O9 -- combined grade and the 11-position overlay stack. Consumed.
# What changes: `master_scale_version` becomes a REQUIRED output (spec 5.3 O9),
# and the overlay decomposition must be differenced against the predecessor.
# --------------------------------------------------------------------------
CombinedGrade = p05.combined_grade.emits("master_scale_version")

# O10 / O12 / O13 / O16 -- group exposure, collateral, pricing search,
# conditions. Consumed, each extended by a declared gap. See consumed/GAPS.toml.
GroupExposure = p05.group_exposure.under(GROUP)
CollateralCover = p05.collateral_cover          # per-facility. Allocation is ours.
PricingSearch = p05.structuring_search.with_candidate_spaces(
    "tables/facility_types.toml",               # fork pressure #5: 9 spaces, not 2
)
Conditions = p05.conditions.extended_to(61)     # precedent + subsequent
