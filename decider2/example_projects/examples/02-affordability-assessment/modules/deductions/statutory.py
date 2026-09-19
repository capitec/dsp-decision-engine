"""Stage 3 -- unemployment insurance, retirement, and court-ordered deductions.

Three separate traps in one file.

1. The unemployment insurance ceiling applies PER EMPLOYER. With two employers
   the household contribution can legitimately exceed the single-employer
   maximum. The naive implementation computes it on the household gross and is
   wrong by up to one ceiling. Expressed as a reduction over the employer list,
   not as one multiplication.

2. Retirement may be OBSERVED on a payslip or IMPUTED from the employer's fund
   register. Both produce a number; only one is evidence. The distinction is an
   emitted value, not a comment.

3. A court-ordered deduction in respect of a credit agreement appears BOTH as a
   payslip deduction and as a bureau account. Counting both double-charges the
   applicant by a full instalment. The match is made here, upstream of the
   evidence cut, but its *effect* -- suppressing the matching account -- lands
   in modules/obligations/assemble.py, downstream of it. So this step emits
   match keys and not a decision, which is the same cut discipline as framing.
"""

from decider2 import SUM, dated_table, over, policy, rung, statutory, step

UIF = dated_table(
    "uif",
    key=(),                                      # a singleton row
    columns={"rate": "float64", "monthly_ceiling_cents": "int64"},
    owner=statutory,
    versions="tables/statutory/uif/",
)


@step(description="Unemployment insurance for one employer: a fixed rate of remuneration, subject to a monthly ceiling.")
def employer_uif_cents(
    employer_remuneration_cents: int,
    employment_type_code: int,
    uif = UIF.asof,
) -> int:
    pass  # 0 for pension and social grant income; else min(remuneration * rate, ceiling)


EmployerUIF = over(
    "employers",
    steps=[employer_uif_cents],
    aggregate={"uif_cents": SUM("employer_uif_cents")},
    annotate=["employer_uif_cents", "employer_remuneration_cents"],
    max_elements=4,
)
# Spec 11.7: the ceiling rises and an applicant with two employers is found to
# have been under-deducted for eight months. Answering "which assessments" is a
# query over the emitted annotation plus the resolved `uif@version`, which is
# why the per-employer figure is annotated and not summed away.


@step(description="Compulsory retirement contribution, taken from the payslip where present.")
def retirement_observed_cents(payslip_retirement_cents: int | None) -> int | None:
    pass  # passes through None when the payslip does not show one


@step(
    description=(
        "Where no payslip figure exists and the employer is on the register of "
        "funds with a known mandatory rate, that rate is applied."
    ),
)
def retirement_imputed_cents(
    gross_monthly_income_cents: int,
    employer_fund_mandatory_rate: float | None,
) -> int | None:
    pass  # None when the employer is not on the register


@rung(
    section="deductions",
    order=50,
    says=(
        "Compulsory retirement contribution of "
        "{retirement_cents:money}, {retirement_basis:observed_or_imputed}."
    ),
)
@step(description="The applied retirement figure, and whether it was observed or imputed.")
def retirement_cents(
    retirement_observed_cents: int | None,
    retirement_imputed_cents: int | None,
) -> int:
    pass  # observed wins; imputed second; zero third


@step(output="retirement_basis", description="Observed, imputed, or absent.")
def retirement_basis(
    retirement_observed_cents: int | None,
    retirement_imputed_cents: int | None,
) -> int:
    pass  # OBSERVED / IMPUTED / ABSENT


@rung(
    section="deductions",
    order=30,
    says=(
        "Statutory deductions of {statutory_deductions_cents:money}: income tax "
        "{income_tax_cents:money}, unemployment insurance {uif_cents:money} "
        "across {employer_count} employer{employer_count:plural}, compulsory "
        "retirement {retirement_cents:money}. Net monthly income "
        "{net_monthly_income_cents:money}."
    ),
)
@step(output="statutory_deductions_cents", description="Tax, unemployment insurance and compulsory retirement. Nothing else.")
def statutory_deductions_cents(
    income_tax_cents: int,
    uif_cents: int,
    retirement_cents: int,
) -> int:
    pass  # the sum of exactly three components


# Medical scheme, union dues, group life, staff loans and savings clubs are NOT
# statutory and are not deducted here. Payslips lump them into one total; where
# it cannot be decomposed the whole non-statutory remainder goes to living
# expenses and the failure to decompose is recorded.
@step(description="Non-statutory payslip deductions that could not be decomposed, routed to living expenses.")
def undecomposed_payslip_deductions_cents(
    payslip_total_deductions_cents: int | None,
    income_tax_cents: int,
    uif_cents: int,
    retirement_cents: int,
) -> int:
    pass  # max(0, total - the three statutory components); consumed by modules/expenses


@step(output="payslip_decomposition_failed", description="Whether the payslip could not be decomposed. Recorded.")
def payslip_decomposition_failed(undecomposed_payslip_deductions_cents: int) -> bool:
    pass  # > 0


# --------------------------------------------------------------------------
# Court-ordered deductions.
# --------------------------------------------------------------------------

@rung(
    section="court_orders",
    order=60,
    says=(
        "Court-ordered deductions of {court_ordered_deductions_cents:money} "
        "under {court_order_count} instrument{court_order_count:plural}: "
        "{court_order_instruments}."
    ),
)
@step(
    output="court_ordered_deductions_cents",
    description="Emoluments attachment, garnishee, maintenance and administration order payments.",
)
def court_ordered_deductions_cents(
    court_order_amounts_cents: "Ragged[int]",
) -> int:
    pass  # sum over 0..6 orders


@step(
    description=(
        "Match keys for court orders taken in respect of a credit agreement. "
        "Consumed downstream of the evidence cut to suppress the matching "
        "bureau account, because counting both double-charges the applicant by "
        "the full instalment."
    ),
)
def court_order_credit_agreement_keys(
    court_order_provider_ids: "Ragged[int]",
    court_order_is_credit_agreement: "Ragged[bool]",
) -> "IdSet":
    pass  # the provider ids of orders flagged as credit-agreement attachments
