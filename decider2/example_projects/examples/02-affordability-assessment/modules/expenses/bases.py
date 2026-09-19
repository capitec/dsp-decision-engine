"""Stage 4 -- the four candidate bases, and the contest between them.

`max(a, b, c, d)` is the obvious implementation and it is wrong in three ways
the spec explicitly forbids:

  * it discards the three losers, and spec 5.4 requires ALL FOUR candidate
    figures, "not only the winner";
  * it cannot emit `expense_basis_code`, so the basis has to be re-derived by
    a second comparison somewhere else, which is a second code path that can
    disagree;
  * it has no tie-break, and the spec has one -- where two bases tie to the
    cent the STATUTORY basis is reported as binding, "because that is the one
    the adjudicator will ask about".

`contest` is the shape. This is its second of three uses; the others are the
income tier waterfall (STRONGEST) and the capacity constraint pair (LOWEST).
"""

from decider2 import HIGHEST, SUM, contest, over, policy, rung, step

from modules.expenses.norms import (
    internal_norm_cents,
    statutory_norm_ceiling_applied,
    statutory_norm_cents,
)
from modules.framing import EXPENSE_CLASSIFICATION


# --------------------------------------------------------------------------
# Basis A -- declared. 8..14 categories by product and channel.
#
# "A category declared as zero and a category not asked are different and must
# not be conflated" (spec 5.4). So the category amount is `int | None` in doc
# 03's explicit tier-3 null style, and `missing_as(0)` is FORBIDDEN on this
# input by a declared rule -- the fill that makes every other step simpler is
# precisely the bug here.
# --------------------------------------------------------------------------

@step(description="One declared expense category, consolidated across the household by its classification.")
def category_household_cents(
    applicant_a_category_cents: int | None,
    applicant_b_category_cents: int | None,
    expense_category_code: int,
    combine = EXPENSE_CLASSIFICATION.behaviour,     # SHARED -> max, PERSONAL -> sum
) -> int | None:
    pass  # None only when BOTH applicants' values are absent


@step(description="Whether this category was asked of the applicant at all. Absence is not zero.")
def category_was_asked(
    expense_category_code: int,
    product_code: int,
    channel_code: int,
) -> bool:
    pass  # the prescribed questionnaire varies by product and channel, 8..14 categories


DeclaredCategories = over(
    "expense_categories",
    steps=[category_household_cents, category_was_asked],
    aggregate={
        "declared_total_cents": SUM("category_household_cents", where="category_household_cents is not None"),
        "categories_asked": SUM("category_was_asked"),
        "categories_answered": SUM("category_household_cents is not None"),
    },
    annotate=["expense_category_code", "category_household_cents", "category_was_asked", "combine_behaviour"],
    max_elements=14,
)


@rung(
    section="expenses",
    order=70,
    says=(
        "Declared living expenses of {declared_total_cents:money} across "
        "{categories_answered} of {categories_asked} categories asked"
        "{non_declaration_note}."
    ),
)
@step(description="Basis A: declared expenses, plus any undecomposed non-statutory payslip deductions.")
def declared_expenses_cents(
    declared_total_cents: int,
    undecomposed_payslip_deductions_cents: int,
) -> int:
    pass  # the payslip remainder belongs in living expenses, not in statutory deductions


@step(
    output="non_declaration_flag",
    description=(
        "Zero across all categories is not a low-expense applicant; it is a "
        "non-declaration, and it sets an evidence flag."
    ),
)
def non_declaration_flag(
    declared_total_cents: int,
    categories_answered: int,
) -> bool:
    pass  # answered > 0 and total == 0, or answered == 0


# --------------------------------------------------------------------------
# Basis B -- statement-derived. Debt service outflows are EXCLUDED; that is
# stage 5's business and including them double-counts.
# --------------------------------------------------------------------------

@rung(
    section="expenses",
    order=72,
    says=(
        "Statement-derived living expenses of {statement_expenses_cents:money}, "
        "covering {statement_category_coverage:pct} of the expense categories "
        "over {statement_month_count} months. Debt service outflows are "
        "excluded from this figure and are assessed separately as obligations."
    ),
)
@step(description="Basis B: statement-derived expenses, excluding debt service outflows.")
def statement_expenses_cents(
    statement_category_totals_cents: "Ragged[int]",
    statement_debt_service_cents: int,
) -> int:
    pass  # sum of categorised outflows less debt service; partial coverage is normal and is recorded


# --------------------------------------------------------------------------
# The contest.
# --------------------------------------------------------------------------
ExpenseBasis = contest(
    "living_expenses_unadjusted_cents",
    candidates={
        "A_declared": declared_expenses_cents,
        "B_statement": statement_expenses_cents,
        "C_statutory": statutory_norm_cents,
        "D_internal": internal_norm_cents,
    },
    select=HIGHEST,
    # Declared, not emergent. `max()` resolves a tie by argument order, which is
    # an accident of how somebody typed the call.
    tie_break=("C_statutory", "D_internal", "B_statement", "A_declared"),
    emits_basis="expense_basis_code",
    retain_losers=True,          # all four figures survive; spec 5.4 requires it
    monotone=True,               # each candidate is non-decreasing in its inputs;
                                 # feeds the pipeline-level monotonicity proof
)


# --------------------------------------------------------------------------
# The exceptional basis. The regulation permits a claim BELOW the norm where
# justified and where the prescribed questionnaire is completed. Bank policy
# makes this route available on products 20 and 21 only.
#
# This is the one place a declared figure may beat the statutory norm, and it
# is modelled as an OVERRIDE ON THE CONTEST with a required evidence reference
# rather than as an extra candidate -- because a candidate would compete on
# magnitude, and this one competes on paperwork.
# --------------------------------------------------------------------------

@rung(
    section="expenses",
    order=78,
    says=(
        "An exceptional-basis claim below the statutory norm was accepted under "
        "questionnaire {exception_questionnaire_ref}, approved by "
        "{exception_approver_id:staff}: {exception_justification}. "
        "Expenses of {declared_expenses_cents:money} applied in place of the "
        "norm of {statutory_norm_cents:money}."
    ),
)
@step(description="Whether the exceptional-basis route is available and its evidence is complete.")
def exception_admitted(
    product_code: int,
    exception_questionnaire_ref: str | None,
    exception_approver_id: int | None,
    exception_justification: str | None,
    eligible_products: tuple[int, ...] = policy((20, 21)),
) -> bool:
    pass  # all three references present AND product in eligible_products; without a
          # questionnaire the claim is unavailable and the norm stands
