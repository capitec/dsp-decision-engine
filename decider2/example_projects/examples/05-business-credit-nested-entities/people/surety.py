"""Stage 5.8 PP-08/PP-09 and stage 5.11 -- sureties, guarantors, security and
the ceilings.

The fourth and fifth roll-ups of the same Entity collection, and the one that is
*not* a blend at all: "A surety's grade does not contribute to the people PD. It
determines how much cover that surety provides, which caps the facility."

s5.8 PP-08's last sentence is the one that catches implementations out: "A
surety who is also an owner contributes to the blend as an owner *and* provides
cover as a surety, and the two must not be conflated in the record." With
`Gather` that is automatic -- the same entity row satisfies `in_blend` and
`is_required_surety`, and the two folds read different predicates. There is no
classification step where an entity is assigned to one bucket, so there is
nothing to get wrong.
"""

from decider2 import module, param, step, table, Gather, sum_of, max_of, count, overlay
from grains import Entity, Application

SURETY_COVER = table(
    "surety_cover",
    key="risk_grade",
    columns=("cover_multiple", "absolute_ceiling"),
    source="tables/surety_cover.csv",
    effective_dated=True,
    owner="Credit Committee",
    cadence="annual",
    overlay=overlay.position(10, scopes=("risk_grade", "sector_code")),
)


def surety_cover(
    is_required_surety: bool,
    risk_grade: int,
    assessed_net_worth: float,
    is_non_resident: bool,
    requires_co_surety: bool,
) -> float:
    """PP-08 per surety. E-AROD-10 halves it where the surety is over 75;
    E-AROD-11 does not count a non-resident's cover at all."""
    pass


def guarantee_cover(
    relationship_type_code: int,
    guaranteed_amount: float,
    guarantor_unused_appetite: float,
    guarantor_tangible_net_worth: float,
    guarantor_tnw_pct: float = param(40.0, ge=0.0, le=100.0),
) -> float:
    """PP-09: min(guaranteed amount, the guarantor's own unused appetite at its
    own grade, 40% of tangible net worth). The guarantor is scored as a juristic
    entity in its own right at s5.7, including its own adverse events -- which is
    free here, because it is simply another row of the Entity grain."""
    pass


CoverTerms = module(surety_cover, guarantee_cover,
                    name="cover_terms", grain=Entity,
                    taps=["surety_cover", "guarantee_cover"])


# --------------------------------------------------------------------------
# PP-08's correlation cap: "cover is the SUM of individual covers, capped at
# 1.5x the largest single cover, because correlated sureties from one household
# are not independent."
#
# A sum and a max over the same subset, and a step at the Application grain that
# combines them. Two folds, not a bespoke aggregate -- which matters because the
# 1.5 is a tunable and the folds are not.
# --------------------------------------------------------------------------
SuretyCoverFacts = (
    CoverTerms
    | Gather(Entity, into=Application, name="surety_cover_facts",
             surety_cover_sum     = sum_of("surety_cover", where="is_required_surety"),
             surety_cover_largest = max_of("surety_cover", where="is_required_surety"),
             surety_count         = count(where="is_required_surety"),
             surety_count_counted = count(where="surety_cover_is_counted"))
    | total_surety_cover          # a bare function is a pipeline element (doc 03 s5.3)
)


def total_surety_cover(
    surety_cover_sum: float,
    surety_cover_largest: float,
    correlation_cap_multiple: float = param(
        1.5, ge=1.0, le=5.0,
        description="PP-08: correlated sureties from one household are not "
                    "independent"),
) -> float:
    """min(sum, 1.5 x largest)."""
    pass


GuaranteeCoverFacts = Gather(
    Entity, into=Application, name="guarantee_cover_facts",
    total_guarantee_cover = sum_of("guarantee_cover", where="is_corporate_guarantor"),
    guarantor_count       = count(where="is_corporate_guarantor"),
    guarantee_not_counted = count(where="guarantee_disqualified"),   # E-AROD-13
)


# --------------------------------------------------------------------------
# Stage 5.11 -- the ceilings, and the circularity that makes pricing hard.
#
# `security_type` depends on the OFFERED amount, and the rate depends on the
# security type, and the affordable amount depends on the rate. s5.11 names this
# as "a second source of non-monotonicity in s5.12, on top of the amount bands".
#
# This file computes everything that does NOT depend on the amount. Everything
# that does -- the cover ratio, the security type, the rate -- is computed at the
# Candidate grain in pricing/price_one.py, once per candidate. The circularity
# dissolves because the candidate grain enumerates the amounts, so nothing has to
# be solved for. Spec s13 Q11 -- "where does the circularity get resolved?" --
# is answered: it does not exist once the amount is a coordinate rather than an
# unknown.
# --------------------------------------------------------------------------
COLLATERAL_ADVANCE = table(
    "collateral_advance_rates",
    key="collateral_class_code",
    columns=("advance_rate", "max_valuation_age_months", "requires_registration"),
    source="tables/collateral_advance.csv",
    effective_dated=True,
)

APPETITE_MAXIMUM = table(
    "appetite_maximum_facility",
    key=("risk_grade", "sector_appetite_class", "security_type"),
    columns=("maximum_facility",),
    source="tables/appetite_maximum.csv",
    effective_dated=True,
    overlay=overlay.position(10, scopes=("risk_grade", "sector_code")),
)

GROUP_EXPOSURE_CAP = table(
    "group_exposure_caps",
    key="risk_grade",
    columns=("group_cap",),
    source="tables/group_exposure_caps.csv",
    effective_dated=True,
    overlay=overlay.position(10, scopes=("risk_grade",)),
)


def adjusted_collateral_value(
    declared_value: float,
    collateral_class_code: int,
    prior_encumbrances: float,
    valuation_age_months: float,
) -> float:
    """Declared value x advance rate, less prior encumbrances.

    NOTE: this is written at the Entity grain's sibling -- `security_offered` is
    0..12 per application (s4.5), which is a sixth ragged collection and a sixth
    grain. It is declared in pricing/candidates.py rather than grains.py because
    it never crosses a grain shift the reader needs to follow; the Gather is a
    plain sum. Left here as a signpost: a project of this size has more ragged
    collections than the two the spec headlines, and a design that only handles
    two is not enough. FRAMEWORK-DEMANDS D03.
    """
    pass


def group_exposure_headroom(
    group_total_exposure: float,
    risk_grade: int,
    pending_application_exposure: float,
) -> float:
    """core.exposure at the Application grain, with the group defined as every
    business sharing a CRITICAL entity with the applicant -- a graph query, run
    in the frame tier, on `entity_key`. Change scenario 13 widens that query to
    common trust beneficiaries, which widens the join and nothing else: the
    assessment structure is untouched because exposure is set-shaped work joined
    in, not a sixth grain."""
    pass


SecurityPosition = module(
    adjusted_collateral_value, name="security_position", grain=Application)

AppetiteAndExposure = module(
    total_surety_cover, group_exposure_headroom,
    name="appetite_and_exposure", grain=Application,
    taps=["total_surety_cover", "group_exposure_headroom"],
)
