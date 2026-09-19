"""Stage 2 -- income determination. Assembly only; the logic is in the siblings.

Two levels of `over()`:

    applicants (1..2)
      -> income sources (0..6 each)
           -> variable-pay component months (0..12 each)

Doc 02's tier split says group-bys are frame-tier work. All three of these
collections live *inside one record* and none of them can leave the record
tier: an application is one row, and a per-application list has no independent
row identity to group on. `over()` is the record-tier answer, and
FRAMEWORK-DEMANDS #6 is about what it costs.
"""

from decider2 import MAX_SEVERITY, MIN, OF_MAX, SUM, contest, module, over, policy, rung, step

from modules.income.haircuts import (
    HAIRCUTS,
    blended_haircut,
    haircut_cap_applied,
    haircut_pct,
    modifier_confidence_pp,
    modifier_tenure_pp,
    modifier_variability_pp,
)
from modules.income.variable_pay import (
    MonthQualification,
    bonus_monthly_cents,
    component_monthly_cents,
    inclusion_rate,
    variability_ratio,
)
from modules.income.waterfall import (
    SourceTier,
    social_grant_cap_cents,
    source_gross_after_haircut_cents,
    source_sufficiency,
    tier_shortfall,
)

# --------------------------------------------------------------------------
# One income source, end to end.
# --------------------------------------------------------------------------
IncomeSource = module(
    MonthQualification,
    inclusion_rate,
    component_monthly_cents,
    bonus_monthly_cents,
    variability_ratio,
    SourceTier,
    HAIRCUTS.lookup(output="haircut_base_pct"),
    modifier_confidence_pp,
    modifier_tenure_pp,
    modifier_variability_pp,
    haircut_pct,
    haircut_cap_applied,
    source_gross_after_haircut_cents,
    social_grant_cap_cents,
    tier_shortfall,
    source_sufficiency,
    name="income_source",
)


@step(description="Does this source contribute more than the materiality threshold of the household total?")
def source_is_material(
    source_gross_after_haircut_cents: int,
    household_gross_cents: int,
    materiality_pct: float = policy(0.10, ge=0.0, le=1.0),
) -> bool:
    pass  # a 4% side income at tier 6 must not drag a salaried applicant to tier 6...


@step(description="Tier 5 and 6 sources are capped as a proportion of the household total; the excess is discarded, not the source.")
def weak_tier_excess_cents(
    weak_tier_total_cents: int,
    household_gross_cents: int,
    weak_tier_max_share: float = policy(0.15, ge=0.0, le=1.0),   # 0.30 on products 20 and 21
) -> int:
    pass  # max(0, weak_tier_total - household_gross * share); recorded as its own line


# --------------------------------------------------------------------------
# Over the sources. Note `income_verification_tier` is the WEAKEST tier among
# MATERIAL sources -- an aggregate with a predicate, which is why `over()`
# takes `where=` on every reducer rather than expecting the caller to filter.
# --------------------------------------------------------------------------
Sources = over(
    "income_sources",
    IncomeSource,
    aggregate={
        "household_gross_cents": SUM("source_gross_after_haircut_cents"),
        "gross_pre_haircut_cents": SUM("source_gross_cents"),
        "income_verification_tier": MIN("income_source_tier", where="source_is_material", sign=-1),
        "income_source_code": OF_MAX("income_source_tier", by="source_gross_after_haircut_cents"),
        "weak_tier_total_cents": SUM("source_gross_after_haircut_cents", where="income_source_tier >= 5"),
        "source_sufficiency_worst": MAX_SEVERITY("evidence_sufficiency_code"),
    },
    annotate=[
        "income_source_tier", "source_gross_cents", "haircut_pct",
        "haircut_base_pct", "modifier_confidence_pp", "modifier_tenure_pp",
        "modifier_variability_pp", "qualifying_month_count",
        "evidence_sufficiency_code", "evidence_reference",
    ],
    max_elements=6,
)


@rung(
    section="income",
    order=10,
    says=(
        "Gross monthly income of {gross_monthly_income_cents:money} was "
        "established from {income_source_count} income "
        "source{income_source_count:plural}. The weakest tier among sources "
        "contributing more than {materiality_pct:pct} of the household total "
        "was tier {income_verification_tier}; the largest source was at tier "
        "{income_source_code}. The effective blended haircut was "
        "{income_haircut_applied:pct}, reducing "
        "{gross_pre_haircut_cents:money} to {gross_monthly_income_cents:money}."
    ),
)
@step(
    output="gross_monthly_income_cents",
    description="Household gross monthly income, post-haircut, less any discarded weak-tier excess.",
)
def gross_monthly_income_cents(
    household_gross_cents: int,
    weak_tier_excess_cents: int,
) -> int:
    pass  # household_gross - weak_tier_excess


Income = module(
    Sources,
    source_is_material,
    weak_tier_excess_cents,
    gross_monthly_income_cents,
    blended_haircut,
    name="income",
    contract="contracts/income.json",
    # `taps=` is not where table versions go. Every `dated_table` read emits its
    # resolved version automatically (FRAMEWORK-DEMANDS #3); acceptance 12 is
    # not something an author can forget to ask for.
    taps=["income_verification_tier", "income_haircut_applied", "branch_path"],
)
