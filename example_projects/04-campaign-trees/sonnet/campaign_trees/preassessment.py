"""Stage 5 -- eligibility and amount determination (spec 04 §5.5): "the amount the Bank
would actually grant today", backed by project 03 running in batch.

**Why this is not a real call into project 03's pipeline.** DEPS.md names 04's dependency on
03 as "hard, but 03 can be stubbed" and lists this as the SCOPE.md-sanctioned wave-reducing
variant ("Stub 03's batch output for 04. 04 moves to wave 2"). 04's own hard dependencies
(DEPS.md) are 00 and 03 only -- **not** 02 -- and 03's real pre-assessment amount is itself
the output of 03's full pipeline, which consumes 02's full affordability assessment. Wiring
that whole chain into this project would mean re-importing 02 (not a declared dependency of
04) just to approximate what a real monthly batch run's *stored output file* would already
contain. Given DEPS.md's own sanctioned shortcut, this module instead:

  - **reuses 03's actual, published solve and pricing** (`loan_granting.solve.solve_term`,
    `loan_granting.pricing.PriceEvaluator`/`RateCardIndex`) unmodified -- the same bounded
    search and the same rate card 03's own real-time granting uses, not a re-implementation;
  - **reuses 00's `core.appetite`** for `amount_cap` (§4.3: "core.appetite | Maximum amount...
    bounding what any campaign may advertise" -- a direct, named match);
  - **approximates** the one figure that would otherwise require 02's full assessment,
    `max_affordable_instalment`, as a fixed fraction of a mart-supplied discretionary-income
    estimate. This is the one deliberate shortcut in this module.

`ponytail: max_affordable_instalment is `estimated_discretionary_income * serviceability_ratio`,
not project 02's real affordability verdict; upgrade path is to replace this function's body
with a read from 03's actual stored batch output (or a real call into 03's pipeline) once
that artefact exists to read from.`

Every other figure in the §5.5 "Emits" contract (`pre_assessed_amount`, `pre_assessed_term`,
the binding constraint, `risk_grade`, the validity window) is produced by real arithmetic,
not invented.
"""
from __future__ import annotations

from datetime import date, timedelta

from decider import frame_step, missing_as, param

from loan_granting.pricing import CreditLifeIndex, PriceEvaluator, RateCardIndex
from loan_granting.solve import BIND_AFF, BIND_EXH, solve_term

PRODUCT_CODE = 10  # Flex Loan
PRE_ASSESSMENT_TERM_MONTHS = 36  # campaign trees offer one standard term, not a term menu (§5.5: an amount, not an offer set)
MONTHLY_VALIDITY_DAYS = 35
DAILY_VALIDITY_DAYS = 10


def estimate_max_affordable_instalment(
    estimated_discretionary_income: float = missing_as(0.0),
    serviceability_ratio: float = param(0.30, ge=0.05, le=0.60),
) -> float:
    """`ponytail:` see module docstring -- a stand-in for project 02's real affordability
    verdict via project 03's batch output, not a re-implementation of either."""
    return round(max(0.0, estimated_discretionary_income) * serviceability_ratio, 2)


def build_preassessment_step(rate_card_index: RateCardIndex, credit_life_index: CreditLifeIndex,
                              statutory_ceiling: float):
    """A `frame_step` closing over the rate card and credit life indexes built once at
    pipeline-build time (never rebuilt per record -- 03's own `granting.py` makes the same
    choice, for the same reason: batch throughput)."""
    evaluator = PriceEvaluator(rate_card_index, credit_life_index, statutory_ceiling,
                                credit_life_substitution_declared=False)

    @frame_step(
        reads=["risk_grade", "applicant_age_years", "max_affordable_instalment",
               "appetite_max_amount", "cycle_date", "is_daily_delta"],
        writes=["pre_assessed_amount", "pre_assessed_term", "binding_constraint_code", "pre_assessment_evaluations",
                "pre_assessment_valid_from", "pre_assessment_valid_to"],
    )
    def preassess(df):
        # A campaign offer is unsolicited (§1: "which live credit campaigns that client
        # should be targeted for") -- there is no client-requested amount to cap against, so
        # the solve's domain top is the appetite limit alone (`requested_amount=None`, the
        # same "no cap from that side" the solve already supports for project 03's own
        # walk-in applications that name no amount).
        amounts, terms, codes, evals, valid_from, valid_to = [], [], [], [], [], []
        for row in df.iter_rows(named=True):
            cycle_date: date = row["cycle_date"]
            result = solve_term(
                evaluator, PRE_ASSESSMENT_TERM_MONTHS, int(row["risk_grade"]), row["applicant_age_years"],
                is_joint=False, amount_cap=row["appetite_max_amount"], requested_amount=None,
                max_affordable_instalment=row["max_affordable_instalment"],
            )
            amounts.append(result.amount if result.amount is not None else 0.0)
            terms.append(PRE_ASSESSMENT_TERM_MONTHS)
            codes.append(result.binding_constraint_code)
            evals.append(result.evaluation_count)
            valid_from.append(cycle_date)
            days = DAILY_VALIDITY_DAYS if row["is_daily_delta"] else MONTHLY_VALIDITY_DAYS
            valid_to.append(cycle_date + timedelta(days=days))
        import polars as pl
        return df.with_columns(
            pl.Series("pre_assessed_amount", amounts, dtype=pl.Float64),
            pl.Series("pre_assessed_term", terms, dtype=pl.Int64),
            pl.Series("binding_constraint_code", codes, dtype=pl.String),
            pl.Series("pre_assessment_evaluations", evals, dtype=pl.Int64),
            pl.Series("pre_assessment_valid_from", valid_from, dtype=pl.Date),
            pl.Series("pre_assessment_valid_to", valid_to, dtype=pl.Date),
        )

    return preassess


def is_preassessment_expired(cycle_date: date, pre_assessment_valid_to: date) -> bool:
    """§5.5 requirement 3: "An offer whose pre-assessment has expired must not be
    presentable, and the app must be able to tell." """
    return cycle_date > pre_assessment_valid_to


__all__ = ["BIND_AFF", "BIND_EXH", "build_preassessment_step",
           "estimate_max_affordable_instalment", "is_preassessment_expired", "PRODUCT_CODE"]
