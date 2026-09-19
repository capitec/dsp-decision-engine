"""Stage 5.3 -- the scalar and the per-account annotation, from one computation.

Spec 5.5.3 and acceptance 10: both outputs are required, neither is a
convenience, and "the scalar must never be computed by a second code path that
could disagree with the annotation it summarises".

The mechanism is that the scalar is DEFINED AS a reduction over the annotation.
There is no expression in this project that computes `existing_obligations_cents`
from anything other than the `obligation_cents` column of the annotation. It is
therefore not merely true that they agree; there is no arrangement of the code
in which they could differ.

What the caller chooses is whether the annotation is MATERIALISED, not whether
it is computed. Project 06 orders its consolidation search on it and cannot run
without it; projects 03 and 07 take only the scalar and would be handed 14
million x 80 rows they never asked for. So:

    Assessment.apply(frame, params=p)                       # scalar only
    Assessment.apply(frame, params=p, materialise=["accounts_annotated"])

and `assert_materialisation_neutral` (pipelines/affordability.py) is the test
that the two agree on every scalar. Doc 03 has no notion of a value that is
always computed and sometimes written back; see FRAMEWORK-DEMANDS #7.
"""

from decider2 import MAX, SUM, module, over, rung, step

from modules.obligations.assemble import (
    AccountList,
    bureau_is_stale,
    enquiry_velocity_uplift_cents,
)
from modules.obligations.treatments import TREATMENTS, QuoteHandling, account_arrears_months


@step(description="The monthly obligation for one account, from the treatment its type selects.")
def obligation_cents(
    account: "Account",
    treatment = TREATMENTS.behaviour,      # the switch; one arm per registered behaviour
    cell = TREATMENTS.asof,                # the coefficients, effective-dated
    quote_handling: QuoteHandling = policy(QuoteHandling.IGNORE_QUOTES),
) -> int:
    pass  # dispatches to stated / pct_limit / ... ; every arm is in the compiled kernel


@step(
    output="obligation_basis_code",
    description="Which basis produced the figure: stated, imputed from limit, imputed from balance, formula, contingent.",
)
def obligation_basis(account: "Account", treatment_code: int) -> int:
    pass  # not the treatment code -- GREATER_OF resolves to STATED or IMPUTED depending on which won


@step(
    output="exclusion_reason_code",
    description="Why an account contributed nothing. Zero when it contributed.",
)
def exclusion_reason(
    closure_established: bool,
    suppressed_by_court_order: bool,
    treatment_code: int,
    quote_handling: QuoteHandling = policy(QuoteHandling.IGNORE_QUOTES),
) -> int:
    pass  # CLOSED_TWO_SOURCE / COURT_ORDER_MATCH / SETTLEMENT_QUOTE / 0
          # An exclusion is an emitted positive fact. "Every account excluded with its
          # reason" (spec 9.1.5) cannot be answered by an account that is simply absent.


@step(description="An account maturing inside the proposed term emits its maturity date. It does not reduce the obligation.")
def maturity_within_term(
    account: "Account",
    proposed_term_months: int | None,
    decision_date: "Date",
) -> bool:
    pass  # emitted to the caller as an appetite input; acted on here, never


@step(description="Whether this account is a Bank facility, for the internal/external split.")
def is_internal(account: "Account") -> bool:
    pass


@step(description="Drawn over limit, for revolving accounts only.")
def account_utilisation(account: "Account") -> float:
    pass  # balance / limit where the account is revolving and limit > 0, else 0.0


# --------------------------------------------------------------------------
# The single `over()`. Every scalar below is a reduction over a column of the
# annotation, named by that column. There is no second arithmetic.
# --------------------------------------------------------------------------
Accounts = over(
    "accounts",
    steps=[
        obligation_cents,
        obligation_basis,
        exclusion_reason,
        maturity_within_term,
        is_internal,
        account_arrears_months,
        account_utilisation,
    ],
    aggregate={
        "obligations_from_accounts_cents": SUM("obligation_cents"),
        "obligations_internal_cents": SUM("obligation_cents", where="is_internal"),
        "obligations_external_cents": SUM("obligation_cents", where="not is_internal"),
        "total_exposure_cents": SUM("account.balance_cents", where="exclusion_reason_code == 0"),
        "revolving_drawn_cents": SUM("account.balance_cents", where="account.is_revolving"),
        "revolving_limit_cents": SUM("account.limit_cents", where="account.is_revolving"),
        "worst_arrears_months": MAX("account_arrears_months"),
        "accounts_in_arrears_count": SUM("account_arrears_months > 0"),
        "refer_account_count": SUM("treatment_code == REFER"),
        "maturing_account_count": SUM("maturity_within_term"),
    },
    annotate=[
        "account_reference",
        "account_type_code",
        "treatment_code",
        "obligation_cents",
        "obligation_basis_code",
        "dedup_source_won",
        "dedup_discrepancy",
        "exclusion_reason_code",
        "maturity_within_term",
        "account_maturity_date",
        "account_arrears_months",
        "obligation_treatment@version",     # the matrix version, per account, automatically
    ],
    max_elements=80,
    # Materialisation is a per-call decision and the default is off. At 80 rows
    # x 12 annotated fields the sidecar is ~7.7 KB per assessment, which is
    # 108 GB across project 07's 14 M monthly records. The scalar path writes
    # none of it. See FRAMEWORK-DEMANDS #7 for what this costs in the kernel.
    materialise_by_default=False,
)


@rung(
    section="obligations",
    order=85,
    says=(
        "Existing monthly debt obligations of {existing_obligations_cents:money} "
        "across {included_account_count} of {considered_account_count} accounts "
        "considered, from treatment matrix version "
        "{obligation_treatment@version}. "
        "{obligations_internal_cents:money} on facilities with the Bank and "
        "{obligations_external_cents:money} elsewhere. "
        "{excluded_account_count} accounts were excluded: "
        "{exclusion_summary}. "
        "Each account, its treatment and the figure used, is listed overleaf."
    ),
)
@step(
    output="existing_obligations_cents",
    description="The obligation total, being the account reduction plus any named policy uplift.",
)
def existing_obligations_cents(
    obligations_from_accounts_cents: int,
    enquiry_velocity_uplift_cents: int,
) -> int:
    pass  # the uplift is a separate addend so that analysis can subtract it


@step(output="revolving_utilisation", description="Drawn over limit across revolving accounts.")
def revolving_utilisation(
    revolving_drawn_cents: int,
    revolving_limit_cents: int,
) -> float:
    pass  # 0.0 when there is no revolving limit; NOT a null, and NOT an error


Obligations = module(
    AccountList,
    Accounts,
    existing_obligations_cents,
    revolving_utilisation,
    enquiry_velocity_uplift_cents,
    bureau_is_stale,
    name="obligations",
    contract="contracts/obligations.json",
    taps=["existing_obligations_cents", "worst_arrears_months", "refer_account_count"],
)

# Spec change scenario 6: project 06 needs the three most expensive obligations
# by instalment, which no other consumer wants. That is not a new output on this
# module and not a fork of it. It is a reduction the caller declares over the
# annotation it already receives:
#
#     top3 = Obligations.reduce("accounts_annotated",
#                               TOP_N("obligation_cents", n=3))
#
# A consumer-declared reduction over a published annotation is the seam that
# stops "one more output for one more consumer" from widening this interface
# once a quarter.

# `modules/obligations/` deliberately has no `__init__.py` assembly of its own.
# `Obligations` is exported from this file because the aggregation and the
# annotation are the same object, and a separate assembly point would be the
# first step towards two of them.
