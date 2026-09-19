"""Stage 3 -- income tax.

Four lines of arithmetic sitting on top of the single most dangerous defect
class in the project: selecting a table by today's date. That failure produces a
plausible wrong answer on every historical record at once and stays invisible
until somebody reconciles (spec 5.3).

The design response is that there is no spelling of the wrong thing.
`dated_table(...).asof` is the only accessor; `.latest`, `.today` and
`.version("2026-03-01")` do not exist on a table whose `owner=statutory`. The
pinned form used by the replay harness is a property of the *resolver*, not of
the call site, so a replay cannot be faked by editing a step.
"""

from decider2 import dated_table, rung, statutory, step

# --------------------------------------------------------------------------
# 8 brackets x 4 rebate classes. Annual, occasionally mid-year (spec 11.2).
# Owned by Credit Systems transcribing the gazette; owner class `statutory`,
# which means no config document written by Credit Risk Policy or by a product
# team can reach it. `decider export --params --owner=product` does not emit a
# single field from this file.
# --------------------------------------------------------------------------
PAYE = dated_table(
    "paye_brackets",
    key=("bracket_index",),
    columns={
        "floor_cents": "int64",
        "base_tax_cents": "int64",
        "marginal_rate": "float64",
    },
    owner=statutory,
    versions="tables/statutory/paye/",
)

REBATES = dated_table(
    "paye_rebates",
    key=("rebate_class",),
    columns={"annual_rebate_cents": "int64"},
    owner=statutory,
    versions="tables/statutory/paye_rebates/",
)


@step(
    description=(
        "The rebate class depends on age AT decision_date: primary below 65, "
        "primary plus secondary from 65, plus tertiary from 75. An applicant "
        "who turns 65 in the month of assessment is a real case, so the "
        "boundary is stated here rather than discovered."
    ),
)
def rebate_class(applicant_age_years: float) -> int:
    pass  # 1 below 65.0, 2 from 65.0 inclusive, 3 from 75.0 inclusive; age is computed at decision_date


@step(description="Annualise the household gross for bracket selection.")
def annual_gross_cents(gross_monthly_income_cents: int) -> int:
    pass  # gross * 12; int64 is safe here -- R90 trillion of headroom against a monthly figure


@step(description="Locate the applicable tax bracket and record its index for the evidence ladder.")
def paye_bracket_index(
    annual_gross_cents: int,
    paye = PAYE.asof,
) -> int:
    pass  # largest index whose floor_cents <= annual_gross_cents


@rung(
    section="deductions",
    order=40,
    says=(
        "Income tax of {income_tax_cents:money} per month. Gross of "
        "{gross_monthly_income_cents:money} annualised to "
        "{annual_gross_cents:money}, placing the applicant in bracket "
        "{paye_bracket_index} of table version {paye_brackets@version} "
        "(base {paye_base_tax_cents:money} plus {paye_marginal_rate:pct} of "
        "the excess over {paye_floor_cents:money}). Rebate class "
        "{rebate_class:rebate} applied, being the class for age "
        "{applicant_age_years:years} at {decision_date:date}, worth "
        "{annual_rebate_cents:money} a year."
    ),
)
@step(description="Monthly income tax: annual liability less rebates, divided by twelve.")
def income_tax_cents(
    annual_gross_cents: int,
    paye_bracket_index: int,
    rebate_class: int,
    paye = PAYE.asof,
    rebates = REBATES.asof,
) -> int:
    pass  # ((base + (gross - floor) * rate) - rebate) / 12, floored at zero, round_half_up
