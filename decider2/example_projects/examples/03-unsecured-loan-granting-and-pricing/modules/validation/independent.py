"""Stage 5.10 -- final validation, and the only interesting question about it.

§13.14: "How is the final validation stage expressed so that it is GENUINELY
INDEPENDENT of the stages it validates, rather than re-running the same code
and agreeing with itself?"

Independence cannot be achieved by discipline, and it cannot be achieved by
writing the arithmetic twice (two annuity implementations is two bug sources
and the second one gets less scrutiny than the first).  It has to be a
CHECKABLE PROPERTY OF THE GRAPH -- and the graph is already data, and static
lineage already answers "what can affect z" without running anything.  So:

    independent_of=[...]   is a NEGATIVE LINEAGE ASSERTION.

The framework computes `lineage(FinalValidation.outputs)` and asserts it is
disjoint from `steps(Solve) union steps(CapWaterfall) union steps(PriceCandidate)`.
A single shared step is a build error naming it.  Doc 03 §9 has only positive
lineage queries; the negative form is the same machinery with the result
inverted, and it is the single cheapest governance feature in this sketch.
See FRAMEWORK-DEMANDS #22.

That is too strict on its own, and the honest version has two tiers.

  TIER A -- INDEPENDENT DERIVATION.  No value computed by a validated stage may
  be an input.  The validator's entire input set is: the chosen offer's
  (amount, term), `risk_grade`, `decision_date`, and the raw tables.  It
  re-reads the card, recomputes the fee, re-looks-up the premium, recomputes
  the instalment and compares.  Enforced by lineage; no exceptions.

  TIER B -- SHARED PRIMITIVES, DECLARED AND ORACLE-TESTED.  The annuity itself
  is shared, because writing it twice is worse.  Every shared primitive must be
  NAMED in `shares_primitives=` and must carry a `corpus=` of expected
  input/output pairs supplied by Compliance -- core-library §7.5's "Credit Risk
  Policy supplies a spreadsheet and it runs as a test".  An unlisted shared
  step is a build error.

What this does NOT catch, stated plainly rather than hidden: a bug inside a
shared primitive is invisible to this stage.  The annuity is checked against a
Compliance-owned corpus, not against the validator.  That residual is real, it
is bounded to four named functions, and it is the correct place to spend audit
attention -- which is a better answer than a validator that re-runs 140 steps
and agrees with all of them.
"""

from __future__ import annotations

from decider2 import assert_that, module, validator
from decider2.money import Money

from modules.caps.register import CapWaterfall
from modules.pricing.probe import PriceCandidate
from modules.solve.search import MaximumAffordableAmount

FinalValidation = validator(
    name="final_validation",
    # -- Tier A ------------------------------------------------------------
    independent_of=[MaximumAffordableAmount, PriceCandidate, CapWaterfall],
    reads=["offered_amount", "term_months", "risk_grade", "decision_date",
           "max_affordable_instalment", "amount_cap", "term_cap",
           "worst_acceptable_grade", "credit_life_substitution_declared",
           "applicant_age_years", "employment_type_code", "segment_code"],
    # -- Tier B ------------------------------------------------------------
    shares_primitives={
        "core.instalment.annuity": "tests/corpora/annuity_compliance.csv",
        "core.fees.piecewise": "tests/corpora/initiation_fee_compliance.csv",
        "core.rounding.round_half_up": "tests/corpora/rounding.csv",
        "core.dates.resolve_asof": "tests/corpora/effective_dating.csv",
    },
    # -- the 14 assertions -------------------------------------------------
    assertions=[
        assert_that("nominal_annual_rate").equals_cell(
            "flex_rate_card", keys=["amount_band_of(offered_amount)", "term_months",
                                    "risk_grade"], asof="decision_date"),
        assert_that("nominal_annual_rate").at_most("statutory_rate_ceiling", asof="decision_date"),
        assert_that("initiation_fee").equals_recomputed().and_at_most("initiation_fee_cap"),
        assert_that("monthly_service_fee").equals("service_fee_cap", asof="decision_date"),
        assert_that("credit_life_premium").equals_cell(
            "credit_life_rates", keys=["age_band", "term_band", "employment_type_code"],
        ).and_at_most(Money("4.50"), per=Money("1000.00")),
        assert_that("instalment").equals_recomputed(tolerance=Money("0.00")),  # to the cent
        assert_that("instalment").at_most("max_affordable_instalment"),
        assert_that("total_cost_of_credit").satisfies("scheduled_in_duplum"),
        assert_that("total_cost_ratio").at_most("total_cost_ratio_threshold"),
        assert_that("offered_amount").at_most_every_binding_ceiling().and_at_least("product_minimum"),
        assert_that("term_months").at_most("term_cap").and_in("permitted_term_list"),
        assert_that("risk_grade").at_most("worst_acceptable_grade"),
        assert_that("offered_amount").is_multiple_of(Money("100.00")),
        assert_that_every_table_version_referenced().resolves_from("decision_date"),
    ],
    # -- the consequence ---------------------------------------------------
    on_failure="withdraw_offer",
    failure_outcome="refer",
    failure_queue=7,
    raise_incident=True,
    # Any mismatch is a HARD failure.  Not a warning, not a logged anomaly, not
    # a value quietly corrected.  `on_failure="correct"` is not an available
    # value, which is the only way to make sure nobody chooses it at 4pm on a
    # Friday.
    #
    # Every assertion records its COMPUTED and EXPECTED values whether it
    # passes or fails (§7.2: "all 14 final validation assertions with their
    # computed and expected values"), so Internal Audit's quarterly re-derivation
    # of 200 approvals reads them rather than recomputing them.
    records="validation_assertions",
)
