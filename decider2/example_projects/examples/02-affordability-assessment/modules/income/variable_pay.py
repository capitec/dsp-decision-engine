"""Stage 2.3 -- variable pay averaging.

This is a ragged collection nested inside a ragged collection: 0..6 income
sources per applicant, each with 0..12 monthly values per variable component.
`over()` composes, so the nesting is expressed rather than flattened.

The whole difficulty is the denominator, and the spec says so: a month that was
payable and paid zero *counts as zero*; a month with no record *does not count*
and shrinks the denominator. Those are two different nulls (doc 00 7.4), and
the cheapest implementation conflates them by filling both with 0.0. So the
month record is `int | None` and the step is written in doc 03's tier-3 null
style -- explicit `None` -- because here the distinction IS the logic.
"""

from decider2 import MAX, SUM, over, policy, rung, step

# --------------------------------------------------------------------------
# Per month, over 0..12 months of one component.
# --------------------------------------------------------------------------

@step(description="A month qualifies when the component was payable, whether or not it paid anything.")
def month_qualifies(
    component_amount_cents: int | None,
    component_was_payable: bool,
) -> bool:
    pass  # component_was_payable and component_amount_cents is not None


@step(description="A month above three times the window median is a once-off and is excluded.")
def month_is_once_off(
    component_amount_cents: int | None,
    window_median_cents: int,
    once_off_multiple: float = policy(3.0, ge=1.0, le=10.0),
) -> bool:
    pass  # amount > median * once_off_multiple; back-pay and settlements


@step(
    output="month_exclusion_reason_code",
    description="Why a month did not count. Zero when it did.",
)
def month_exclusion_reason(
    component_was_payable: bool,
    component_amount_cents: int | None,
    month_is_once_off: bool,
) -> int:
    pass  # NOT_YET_EMPLOYED / STATEMENT_GAP / PAYSLIP_NOT_SUPPLIED / ONCE_OFF / 0


# The exclusion reason is emitted as a POSITIVE record, not left as an absence.
# An adjudicator asks "why is October missing", and an absent row cannot answer.
# Every "why not" question in spec 9.1 is answered by an emitted annotation
# somewhere in this project; that is a design rule, not a coincidence.
MonthQualification = over(
    "component_months",
    steps=[month_qualifies, month_is_once_off, month_exclusion_reason],
    aggregate={
        "qualifying_month_count": SUM("month_qualifies"),
        "component_total_cents": SUM("component_amount_cents", where="month_qualifies"),
        "lowest_qualifying_month_cents": MAX("component_amount_cents", where="month_qualifies", sign=-1),
    },
    annotate=["month_qualifies", "month_exclusion_reason_code"],
    max_elements=12,
)


@step(description="Inclusion rate by qualifying months: 0/50/75/90/100 percent at 0-2/3-5/6-8/9-11/12.")
def inclusion_rate(
    qualifying_month_count: int,
    rates = policy.table(
        "variable_pay_inclusion",
        key=("min_months",),
        columns={"rate": "float64"},
    ),
) -> float:
    pass  # step function over the count; a table because Credit Risk Policy argues about the 3-5 band


@rung(
    section="income",
    order=22,
    says=(
        "{component_code:component} averaged over {qualifying_month_count} "
        "qualifying months of a {averaging_window_months}-month window. "
        "{excluded_month_count} months excluded: {month_exclusion_summary}. "
        "Included at {inclusion_rate:pct} of the average."
    ),
)
@step(description="The component's monthly contribution, after averaging, capping and the inclusion rate.")
def component_monthly_cents(
    component_total_cents: int,
    qualifying_month_count: int,
    lowest_qualifying_month_cents: int,
    inclusion_rate: float,
    income_variability_ratio: float,
    volatility_floor_ratio: float = policy(0.60, ge=0.0, le=1.0),
) -> int:
    pass  # above volatility_floor_ratio the LOWEST qualifying month binds, not the average


@step(description="An annual or thirteenth cheque: divided by twelve, included at 60 percent, twelve months' history required.")
def bonus_monthly_cents(
    bonus_amount_cents: int | None,
    bonus_history_months: int,
    inclusion_pct: float = policy(0.60, ge=0.0, le=1.0),
    required_history_months: int = policy(12, ge=1, le=24),
) -> int:
    pass  # zero unless bonus_history_months >= required_history_months


@step(
    output="income_variability_ratio",
    description="Coefficient of variation of the qualifying monthly values.",
)
def variability_ratio(
    component_total_cents: int,
    qualifying_month_count: int,
    component_sum_squares: float,      # float64 accumulator -- an int64 cent square wraps at 2 667 rows
) -> float:
    pass  # stddev / mean over qualifying months, 0.0 when count < 2
