"""The degraded review ladder. Five MODES, not one path and four error branches.

Spec 5.4.3, H9. 30% of reviews run on stale or incomplete financials, and:

    "if the complete-evidence path is the main path and the other four are
     branches off it, the main path is used 44% of the time and four-fifths of
     the logic lives in exception handling that nobody reviews."

This is the same construct as project 02's four assessment modes (its 5.8), and
this project deliberately uses the same one rather than inventing a second --
which is a small reuse win worth noticing, because the shape transferred even
though not a line of the logic did.

The rule that makes a mode a mode: **A MODE MAY NOT CHANGE THE ARITHMETIC.**
It selects evidence rules, parameter sets and which outputs are produced. The
DSCR formula is the same in basis 1 and basis 5. What differs is what feeds it,
what caps the answer, and what the next review date is.
"""

from decider2 import modes, mode, param, module

REVIEW_BASIS = modes(
    "review_basis",
    arithmetic_invariant=True,      # <- the load-bearing declaration
    exhaustive=True,                # every review lands on exactly one. No default.
    modes=[
        mode(1, "complete",
             when="year_end_months <= 9 and mgmt_accounts_months <= 3",
             share=0.44,
             caps=None, next_review_months=12),
        mode(2, "late",
             when="9 < year_end_months <= 15",
             share=0.26,
             haircut_pct=5.0, next_review_months=12,
             also="information undertaking breach recorded if the covenant deadline has passed"),
        mode(3, "stale",
             when="15 < year_end_months <= 21 and mgmt_accounts_present",
             share=0.18,
             haircut_pct=15.0, grade_cap=7, next_review_months=6,
             watchlist_floor=1,
             forbids=["limit_increase", "term_extension", "new_facility"]),
        mode(4, "turnover_only",
             when="year_end_months > 21 and turnover_months >= 6",
             share=0.09,
             grade_cap=6, amount_cap_cents=1_500_000_00, next_review_months=3,
             watchlist_floor=2,
             freezes="undrawn",
             also="covenant breach on the information undertaking, cure clock running"),
        mode(5, "not_performed",
             when="no financials, no turnover, no response",
             share=0.03,
             grade="unchanged, flagged grade_stale",
             freezes="facility", watchlist_floor=2,
             next_review_months=2,
             mandates="exit assessment within 60 days",
             records="performed, evidence absent"),
    ],
)

# --------------------------------------------------------------------------
# Three rules on the ladder, and the third is the one auditors ask about.
# --------------------------------------------------------------------------

REVIEW_NEVER_DEFERRED = REVIEW_BASIS.assert_total(
    deadline="review_date + 30 days",
    forbidden_outcomes=["awaiting_financials", "deferred", "pending"],
)
"""Spec 5.4.3 rule 1 and spec 10 acceptance 6. Every facility in the cohort
produces a decision of record by its review date plus 30 days. "Awaiting
financials" is not an outcome; basis 4 or 5 is.

`assert_total` on a `modes` declaration generates two tests: that the `when`
predicates partition the input space with no gap and no overlap, and that no
outcome code outside the declared set is reachable. A gap in mode predicates is
how "pending" gets invented at 2am.
"""


def next_review_date(review_basis_code: int, review_date,
                     by_basis: tuple = param((12, 12, 6, 3, 2),
                         description="Months to next review, by basis")) -> object:
    """Spec 5.4.3 rule 2: the basis is an OUTPUT and it drives the next review
    date. A degraded review buys less time than a complete one, which is the
    mechanism by which the book does not silently drift onto stale information.
    """
    pass  # review_date + by_basis[review_basis_code - 1] months


def degradation_is_not_decline(review_basis_code: int) -> int:
    """Rule 3. Basis 4 and 5 RESTRICT, they do not terminate. A facility is only
    exited through L6. Stated as a step because the assertion has to appear in
    the rendered artefact a credit committee reads, not only in a test.
    """
    pass  # assert outcome_code != DECLINE for any basis


ReviewBasis = module(next_review_date, degradation_is_not_decline,
                     name="review_basis", owner="business_credit_risk_policy",
                     taps=["review_basis_code", "branch_path"])
