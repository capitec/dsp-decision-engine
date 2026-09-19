"""Stage 5.9 -- the business financial assessment.

Included in the sketch for three reasons, none of them the arithmetic:

1. **The spread is a third ragged collection** -- 1..3 periods, of unequal
   length. A fifth grain (`Period`), 48 lines each. It is shown here as a grain
   because the alternative -- 144 columns named `turnover_p1`, `turnover_p2`,
   `turnover_p3` -- is what actually gets written when a framework has no answer
   for ragged data, and it makes "three periods exist so compute a CAGR"
   unexpressible without a null check per column.

2. **Haircuts and overlays must not be implemented as each other.** s5.9 is
   unusually explicit: "a haircut is a data-quality adjustment to an *input*,
   applied inside the assessment by rule; an overlay is a policy adjustment to
   the *output*, applied on top of a validated model under a separate approval.
   Both may apply to one application; each must be separately visible; neither
   may be implemented as the other." The framework can enforce that, and here it
   does: a haircut is an ordinary step, an overlay is `overlay.position(5, ...)`,
   and the lint rule is that no step named `*_haircut` may write a value that an
   overlay position also writes.

3. **`trend_unavailable` is not zero.** One more instance of doc 03 s1 tier 3.
"""

from decider2 import module, param, step, table, grain, Capacity, Gather
from decider2 import count, sum_of, best_of, asc, desc, missing_as
from grains import Application

Period = grain(
    "period",
    key=("application_id", "period_index"),
    parent=Application,
    identity="period_end_date",
    capacity=Capacity(3, on_exceed="truncate", flag="excess_periods_ignored"),
    order=(desc("period_end_date"),),
)

SECTOR_BENCHMARKS = table(
    "sector_ratio_benchmarks",
    key=("sector_code", "ratio_code"),
    columns=("p25", "median", "p75", "observation_count"),
    source="tables/sector_benchmarks.csv",
    effective_dated=True,
    fallback_key=("sector_grouping_code", "ratio_code"),
    fallback_when="observation_count < min_observations",
    records_fallback=True,           # s5.9: "and that fallback is recorded"
)

STATEMENT_LINE_MAP = table(
    "statement_line_mapping",
    key="source_label_hash",
    columns=("standard_line_code",),
    source="tables/statement_line_mapping.csv",
    effective_dated=True,
    owner="Business Credit Risk Policy",
    cadence="continuous",
    on_miss="flag:unmapped_label",
)


# -- Period grain -----------------------------------------------------------

def period_months(period_start_date: int, period_end_date: int) -> float:
    """Length of the reporting period."""
    pass


def period_annualised(
    period_months: float,
    short_period_months: float = param(10.0, ge=1.0, le=12.0),
    long_period_months: float = param(14.0, ge=12.0, le=24.0),
) -> bool:
    """s4.5: a nine-month first period after incorporation occurs on ~6% and must
    be annualised WITH A FLAG, not silently compared against twelve-month
    benchmarks."""
    pass


def annualised_turnover(revenue: float, period_months: float) -> float:
    """Revenue net of indirect tax, scaled to twelve months where needed."""
    pass


def annualised_ebitda(
    operating_profit: float,
    depreciation: float,
    amortisation: float,
    non_recurring_items: float = missing_as(0.0),
    period_months: float = 12.0,
    non_recurring_floor: float = param(50_000.0, ge=0.0),
) -> float:
    """Operating profit + D&A, adjusted for declared non-recurring items above
    the floor."""
    pass


PeriodMeasures = module(
    period_months, period_annualised, annualised_turnover, annualised_ebitda,
    name="period_measures", grain=Period,
)


# -- Period -> Application --------------------------------------------------

FinancialFacts = Gather(
    Period, into=Application, name="financial_facts",
    period_count      = count(),
    any_annualised    = count(where="period_annualised"),
    latest_period     = best_of("period_end_date", tie_break=(asc("period_index"),),
                                lift=["annualised_turnover", "annualised_ebitda",
                                      "period_end_date", "audit_level_code",
                                      "audit_opinion_code", "period_annualised"]),
    earliest_period   = best_of("period_end_date", direction="min",
                                tie_break=(desc("period_index"),),
                                lift=["annualised_turnover", "period_end_date"]),
)


def turnover_trend(
    period_count: int,
    latest_period_annualised_turnover: float,
    earliest_period_annualised_turnover: float,
    earliest_period_end_date: int,
    latest_period_end_date: int,
) -> float | None:
    """Period-on-period growth; CAGR where three periods exist.

    Returns None for a single-period business. s5.9: "single-period businesses
    get `trend_unavailable`, not zero". A `missing_as(0.0)` here reads as flat
    growth, which is a materially better financial picture than "we do not know",
    and it is the kind of default that survives a code review because it looks
    harmless.
    """
    pass


def ebitda_after_haircuts(
    latest_period_annualised_ebitda: float,
    audit_level_code: int,
    audit_opinion_code: int,
    months_since_year_end: float,
) -> float:
    """The five audit-level haircuts and the four age haircuts, composed in the
    declared order. An INPUT adjustment. Not an overlay -- see the module
    docstring."""
    pass


def debt_service_coverage(
    ebitda_after_haircuts: float,
    tax_paid: float,
    maintenance_capex: float,
    subordinated_director_loan_movement: float,
    existing_interest: float,
    existing_scheduled_principal: float,
    proposed_annual_debt_service: float,
) -> float:
    """s5.9's DSCR. Consumed at the Candidate grain once per candidate, because
    `proposed_annual_debt_service` is a candidate property -- so this step is
    written at the Application grain and RE-PINNED to Candidate at its use site
    in pricing/price_one.py. See FRAMEWORK-DEMANDS D08: one step, two grains,
    with the grain chosen where it is used rather than where it is defined."""
    pass


def financial_confidence_code(
    period_count: int,
    audit_level_code: int,
    months_since_year_end: float,
    uses_bank_statements: bool,
    unmapped_label_pct: float,
) -> int:
    """high / medium / low / none. Drives the blend weights at s5.10 and the
    statement-based caps."""
    pass


def statement_based_caps_apply(uses_bank_statements: bool) -> bool:
    """s5.9: caps the facility at R1 500 000 and the business grade at 6.
    "It cannot produce a top grade, ever." """
    pass


FinancialAssessment = (
      PeriodMeasures
    | FinancialFacts
    | module(turnover_trend, ebitda_after_haircuts, debt_service_coverage,
             financial_confidence_code, statement_based_caps_apply,
             name="financial_assessment", grain=Application,
             taps=["financial_confidence_code", "ebitda_after_haircuts",
                   "turnover_trend", "statement_based_caps_apply"],
             contract="contracts/financial_assessment.json")
)
