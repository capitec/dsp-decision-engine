"""Stage 5.2 -- the obligation treatment matrix.

45 account types. Each cell selects a BEHAVIOUR and carries its coefficients.
`PCT_LIMIT` at 5% and `PCT_LIMIT` at 3% are the same behaviour with different
parameters; a quarterly retune changes the coefficient, not the behaviour, and
behaviour changes are approved differently (spec 5.5.2).

Spec question 5 asks what this is, structurally -- a table, a rule set, or
something the vocabulary does not have. It is the third. Doc 08 3.4 offers two
kinds: `decision_table` (uniform operators, generic kernel, free to change) and
`ruleset` (heterogeneous predicates, codegen, staged compile). This is neither.
Every row applies the same *shape* of rule -- one behaviour from a closed set,
with coefficients -- so it is not a ruleset; but the behaviour column is a
discriminant over code, not a value, so it is not a decision table either.

`behaviour_table` compiles to a `switch` with one arm per registered behaviour.
Consequences worth being precise about, because they are not what one would
guess:

  * changing a coefficient is free (a value);
  * changing a CELL from STATED to GREATER_OF is ALSO free, because both arms
    are already in the compiled switch -- it is a data change;
  * adding an eleventh behaviour is a code change, because it is a new step.

So the compile cost splits at a different place from where the approval splits.
Doc 08 2's table assumes those two axes co-vary. They do not. See
FRAMEWORK-DEMANDS #5.
"""

from decider2 import behaviour_table, policy, round_half_up, step
from decider2.domains import Switch


class QuoteHandling(Switch):
    """Whether a live settlement quotation zeroes an account. Scenario mode only."""
    IGNORE_QUOTES = 1          # new application, limit increase, arrangement
    EXCLUDE_ON_QUOTE = 2       # scenario (project 06) ONLY


# --------------------------------------------------------------------------
# The ten behaviours. Each is a registered step with the same signature, which
# is what lets the matrix cell be a discriminant rather than a pointer -- doc
# 08 1.1's reference-not-pointer rule, applied to a table column.
# --------------------------------------------------------------------------

@step(description="The reported contractual instalment.")
def stated(account: "Account", cell: "TreatmentCell") -> int:
    pass  # account.instalment_cents


@step(description="A percentage of the facility limit, for revolving facilities where the limit is the exposure.")
def pct_limit(account: "Account", cell: "TreatmentCell") -> int:
    pass  # round_half_up(account.limit_cents * cell.rate)


@step(description="A percentage of the drawn balance, floored at an absolute minimum.")
def pct_balance(account: "Account", cell: "TreatmentCell") -> int:
    pass  # max(round_half_up(account.balance_cents * cell.rate), cell.floor_cents)


@step(description="The greater of the stated instalment and the imputed figure.")
def greater_of(account: "Account", cell: "TreatmentCell") -> int:
    pass  # max(stated(...), pct_limit(...))


@step(description="A contractual minimum-payment formula: a percentage of balance plus interest and fees, floored.")
def min_payment(account: "Account", cell: "TreatmentCell") -> int:
    pass  # max(balance * cell.min_pct + account.monthly_interest_cents + account.monthly_fees_cents, cell.min_floor_cents)


@step(description="Settled, closed, written-off and paid-up accounts contribute nothing.")
def exclude_closed(account: "Account", cell: "TreatmentCell") -> int:
    pass  # 0, subject to the two-source closure rule in assemble.py


@step(description="An account with a live settlement quotation, zeroed in scenario mode only.")
def exclude_on_quote(account: "Account", cell: "TreatmentCell") -> int:
    pass  # 0 when quote_handling is EXCLUDE_ON_QUOTE, else falls through to the account's
          # underlying behaviour -- it does NOT fall through to zero


@step(description="Surety, guarantor and co-signatory exposure: a percentage of the principal's instalment.")
def contingent(account: "Account", cell: "TreatmentCell") -> int:
    pass  # instalment * cell.contingent_pct, rising to 100% where the principal account is in arrears


@step(
    description=(
        "An account maturing inside the proposed term. The instalment is taken "
        "IN FULL and the maturity date is emitted. Anticipating expiry here "
        "would be lending against money the applicant does not yet have, and "
        "whether it may be anticipated is an appetite question belonging to the "
        "caller."
    ),
)
def term_aware(account: "Account", cell: "TreatmentCell") -> int:
    pass  # identical arithmetic to STATED; the difference is the emitted maturity date


@step(description="An account type the matrix cannot treat mechanically. No figure; forces indeterminate.")
def refer(account: "Account", cell: "TreatmentCell") -> int:
    pass  # raises no value; sets evidence_sufficiency_code = UNTREATABLE_ACCOUNT_TYPE


TREATMENTS = behaviour_table(
    "obligation_treatment",
    key=("account_type_code",),
    behaviours={
        "STATED": stated,
        "PCT_LIMIT": pct_limit,
        "PCT_BALANCE": pct_balance,
        "GREATER_OF": greater_of,
        "MIN_PAYMENT": min_payment,
        "EXCLUDE_CLOSED": exclude_closed,
        "EXCLUDE_ON_QUOTE": exclude_on_quote,
        "CONTINGENT": contingent,
        "TERM_AWARE": term_aware,
        "REFER": refer,
    },
    coefficients=("rate", "floor_cents", "contingent_pct", "min_pct", "min_floor_cents"),
    owner=policy,
    versions="tables/policy/obligation_treatment/",
    # Spec change scenario 4: a new account type appears at the bureau and needs
    # a treatment. Until it has one it must produce `indeterminate`, not
    # silently score zero. That is one line, and it is the default rather than
    # something an author remembers.
    unknown_key="REFER",
    # The behaviour column and the coefficient columns have different change
    # classes and different approvers. Declared per column, because the
    # document-level classes in doc 08 2 cannot express it.
    change_class={
        "behaviour": "interior",       # Credit Risk Policy + Credit Committee, reviewed
        "*": "value",                  # Credit Risk Policy, quarterly
    },
)


@step(description="Arrears do not change the treatment. They are recorded, and they travel to the caller.")
def account_arrears_months(account: "Account") -> int:
    pass  # catch-up amounts are NOT added to the obligation; the contractual instalment is the obligation
