"""Stage 2.1 -- the six-tier evidence waterfall, per income source.

The waterfall is a `contest`: every candidate tier is evaluated, the strongest
*available* one binds, and the losers survive as evidence. That last part is
why it is not a chain of `if`s. Spec 5.2.1 requires that "weaker contradictory
evidence is recorded but does not change the tier", and an `if/elif` ladder
throws that away at the first `return`.
"""

from decider2 import STRONGEST, behaviour_table, contest, dated_table, local, policy, statutory, step
from decider2.domains import Switch

from modules.income.haircuts import haircut_pct

# --------------------------------------------------------------------------
# What a mode does when the strongest available tier is below the product floor.
# A closed enum, not a bool: `CONDITIONAL` is project 07's third answer and the
# cheapest implementation of the whole stage is a bool that cannot hold it.
# --------------------------------------------------------------------------
class StaleEvidence(Switch):
    INDETERMINATE = 1        # new application, scenario
    CONDITIONAL = 2          # limit increase: offer subject to confirmation
    PERMIT_WEAKER_TIER = 3   # arrangement: sustainability, not appetite


# Effective-dated because Compliance moved product 21 from tier 4 to tier 5 in
# 2025 and every assessment before that date must keep resolving to 4, forever.
MIN_EVIDENCE_TIER = dated_table(
    "minimum_evidence_tier",
    key=("product_code", "tier_source"),
    columns={"weakest_permitted_tier": "int8"},
    owner=statutory,                       # Compliance. Unreachable from a product config.
    versions="tables/statutory/minimum_evidence_tier/",
)


# --------------------------------------------------------------------------
# One candidate per tier. Each returns (available, amount_cents, evidence_id).
# `available` is not "non-zero": a tier that exists and reports zero is
# different from a tier that does not exist, and both are different from a tier
# that exists but is not permitted for this employment type (the `X` cells in
# the haircut matrix, spec 5.2.2).
# --------------------------------------------------------------------------

@step(description="Tier 1: written employer confirmation, verified register.")
def tier1_employer_confirmed(
    employer_confirmation_amount_cents: int | None,
    employer_is_on_verified_register: bool,
    employment_type_code: int,
) -> tuple[bool, int, int]:
    pass  # available iff confirmation present AND employer on register AND tier 1 permitted for this employment type; NOT_PERMITTED for pensioners (employment_type_code=4) effective 2026-09-23


@step(description="Tier 2: latest three consecutive payslips, or two with a letter under 3 months' tenure.")
def tier2_payslips(
    payslip_count: int,
    payslips_consecutive: bool,
    payslip_gross_cents: int | None,
    months_employed: float,
    employment_letter_present: bool,
    employment_type_code: int,
) -> tuple[bool, int, int]:
    pass  # available at 3 consecutive payslips, or 2 + letter when months_employed < 3


@step(description="Tier 3: six months of salary-classified credits from a consistent originator at the Bank.")
def tier3_internal_deposits(
    internal_salary_deposit_count: int,
    internal_originator_consistent: bool,
    internal_salary_deposit_mean_cents: int | None,
    employment_type_code: int,
) -> tuple[bool, int, int]:
    pass  # available at >= 6 classified deposits from one originator


@step(description="Tier 4: aggregated statement inflows, up to twelve months, with aggregator confidence.")
def tier4_statement_inflows(
    statement_inflow_months: int,
    statement_inflow_mean_cents: int | None,
    statement_confidence: float,
    min_statement_confidence: float = policy(0.55, ge=0.0, le=1.0),
) -> tuple[bool, int, int]:
    pass  # UNAVAILABLE below min_statement_confidence -- spec 5.2.2; falls through to tier 5


@step(description="Tier 5: bureau-modelled income estimate.")
def tier5_bureau_estimate(
    bureau_income_estimate_cents: int | None,
    employment_type_code: int,
) -> tuple[bool, int, int]:
    pass  # available iff the bureau supplied an estimate and tier 5 is permitted for the type


@step(description="Tier 6: unverified client declaration.")
def tier6_declared(
    declared_income_cents: int | None,
) -> tuple[bool, int, int]:
    pass  # available iff a declaration was made; zero-declared is available-at-zero, absent is unavailable


# --------------------------------------------------------------------------
# The contest. Third use of this combinator in the project (see
# modules/expenses/bases.py and modules/capacity/__init__.py); here `select` is
# STRONGEST rather than HIGHEST, which is the point -- the shape recurs, the
# selection rule does not.
# --------------------------------------------------------------------------
SourceTier = contest(
    "income_source_tier",
    candidates={
        1: tier1_employer_confirmed,
        2: tier2_payslips,
        3: tier3_internal_deposits,
        4: tier4_statement_inflows,
        5: tier5_bureau_estimate,
        6: tier6_declared,
    },
    select=STRONGEST,                 # lowest tier number that reports available
    availability="available",         # field 0 of each candidate's tuple
    binds="source_gross_cents",       # field 1
    emits_basis="income_source_tier",
    # Every candidate's figure is retained, whether or not it bound. An
    # adjudicator's first question is "what else did you have, and why did you
    # not use it"; a chain of `if`s cannot answer it and this is why the four
    # rejected candidates are not a debugging convenience.
    retain_losers=True,
)


@step(description="Is the established tier weaker than this product's floor?")
def tier_shortfall(
    income_source_tier: int,
    product_code: int,
    min_tier = MIN_EVIDENCE_TIER.asof,       # resolves on decision_date. There is no `.latest`.
    tier_source: str = local("product"),     # profile-set: "product" or "arrangement"
) -> bool:
    pass  # income_source_tier > min_tier[product_code, tier_source]


@step(
    output="evidence_sufficiency_code",
    description="Why this source cannot support a grant, if it cannot. Zero when it can.",
)
def source_sufficiency(
    tier_shortfall: bool,
    statement_confidence_below_threshold: bool,
    cash_income_no_banking_footprint: bool,
    on_stale_evidence: StaleEvidence = policy(StaleEvidence.INDETERMINATE),
) -> int:
    pass  # returns a SufficiencyCode; 0 when sufficient. PERMIT_WEAKER_TIER suppresses the shortfall code only.


# --------------------------------------------------------------------------
# Haircut application, per source. `haircut_pct` (haircuts.py) is the matrix
# cell plus the three additive modifiers, capped at 55%.
# --------------------------------------------------------------------------

@step(description="Apply the verification haircut to this source's gross figure.")
def source_gross_after_haircut_cents(
    source_gross_cents: int,
    haircut_pct: float,
) -> int:
    pass  # round_half_up(source_gross_cents * (1 - haircut_pct)); never bare round()


@step(description="Social grant income is capped at the published statutory figure.")
def social_grant_cap_cents(
    source_gross_after_haircut_cents: int,
    employment_type_code: int,
    grant_type_code: int,
    grant_schedule = dated_table(
        "social_grant_amounts",
        key=("grant_type_code",),
        columns={"monthly_amount_cents": "int64"},
        owner=statutory,
        versions="tables/statutory/social_grants/",
    ).asof,
) -> int:
    pass  # min(figure, published amount); a declared excess is reduced AND flagged, never silently kept
