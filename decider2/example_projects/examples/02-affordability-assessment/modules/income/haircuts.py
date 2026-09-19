"""Stage 2.2 -- the haircut matrix and its three additive modifiers.

36 cells, six of which are not a number at all: `X` means the (tier,
employment type) combination is not permitted and forces a fall-through to a
weaker tier. A table of floats cannot hold `X` without a sentinel, and a
sentinel in a money-adjacent table is how -1 becomes a 100% haircut in
somebody's reconciliation. So the matrix is a `behaviour_table`: one column
selects an outcome from a closed set, the rest carry coefficients.
"""

from decider2 import behaviour_table, dated_table, policy, rung, statutory, step

# --------------------------------------------------------------------------
# 6 tiers x 6 employment types. Credit Risk Policy, quarterly, effective-dated.
#
# `behaviour` is one of two registered outcomes. `NOT_PERMITTED` is not a
# haircut of 100%; it removes the candidate from the contest in waterfall.py,
# which is a different thing and produces a different evidence trail.
# --------------------------------------------------------------------------
HAIRCUTS = behaviour_table(
    "income_haircuts",
    key=("income_source_tier", "employment_type_code"),
    behaviours={
        "PCT": lambda cell: cell.haircut_pct,
        "NOT_PERMITTED": None,          # removes the tier from the contest
    },
    coefficients=("haircut_pct",),
    owner=policy,
    versions="tables/policy/income_haircuts/",
    # Change scenario: a new employment type appears. Spec 11.4 demands the
    # analogous obligation case produce `indeterminate` rather than a silent
    # zero, and the same reasoning applies here -- a missing cell is not a
    # zero haircut.
    unknown_key="NOT_PERMITTED",
)

MODIFIERS = dated_table(
    "haircut_modifiers",
    key=("modifier_code",),
    columns={
        "threshold": "float64",
        "increment_pp": "float64",
    },
    owner=policy,
    versions="tables/policy/haircut_modifiers/",
)


@step(description="Statement-confidence modifier: (1 - confidence) x 20 percentage points, tier 4 only.")
def modifier_confidence_pp(
    income_source_tier: int,
    statement_confidence: float,
    confidence_scale_pp: float = policy(20.0, ge=0.0, le=60.0),
) -> float:
    pass  # 0.0 unless tier == 4


@step(description="Tenure modifier: +5pp on contract under six months, +3pp on permanent under three.")
def modifier_tenure_pp(
    employment_type_code: int,
    months_employed: float,
    modifiers = MODIFIERS.asof,
) -> float:
    pass  # reads the CONTRACT_TENURE and PERMANENT_TENURE rows; thresholds are data, not literals


@step(description="Variability modifier: +5pp above a variability ratio of 0.35, +10pp above 0.60.")
def modifier_variability_pp(
    income_variability_ratio: float,
    modifiers = MODIFIERS.asof,
) -> float:
    pass  # reads VARIABILITY_LOW and VARIABILITY_HIGH rows


@rung(
    section="income",
    order=25,
    says=(
        "A {haircut_base_pct:pct} verification haircut applied for tier "
        "{income_source_tier} evidence on {employment_type_code:employment}, "
        "from matrix version {income_haircuts@version}, cell "
        "({income_source_tier}, {employment_type_code}). "
        "Modifiers added {modifier_confidence_pp:pp} for statement confidence, "
        "{modifier_tenure_pp:pp} for employment tenure and "
        "{modifier_variability_pp:pp} for income variability, "
        "giving {haircut_pct:pct}{haircut_cap_note}."
    ),
)
@step(description="The effective haircut: matrix cell plus modifiers, capped at 55 percent in total.")
def haircut_pct(
    haircut_base_pct: float,
    modifier_confidence_pp: float,
    modifier_tenure_pp: float,
    modifier_variability_pp: float,
    total_cap_pct: float = policy(0.55, ge=0.0, le=1.0),
) -> float:
    pass  # min(base + sum(modifiers)/100, total_cap_pct)


@step(description="Whether the 55 percent total cap bound, recorded because it is a departure in the applicant's favour.")
def haircut_cap_applied(
    haircut_base_pct: float,
    modifier_confidence_pp: float,
    modifier_tenure_pp: float,
    modifier_variability_pp: float,
    total_cap_pct: float = policy(0.55, ge=0.0, le=1.0),
) -> bool:
    pass  # uncapped total > total_cap_pct


@step(
    output="income_haircut_applied",
    description="The effective blended reduction across all sources, to four decimal places.",
)
def blended_haircut(
    gross_monthly_income_cents: int,
    gross_pre_haircut_cents: int,
) -> float:
    pass  # 1 - post/pre, quantised to 4dp with round_half_up
