"""Settleability classification: nine classes, ordered, first match wins.

This stage sets the size of the search space. Shrinking 61 accounts to 14
settleable ones is worth more than any cleverness downstream - it takes 2^61 off
the table before a single scenario is generated - and it is the cheapest thing
in the whole flow.

Structurally this is an ordered `ruleset` (doc 08 3): interface in code, nine
rules in an interior document. The ordering is the semantics, so `first_match`
is declared rather than implied, and the rule that classified each account is an
output rather than a debugging nicety - the contact centre question "why was my
store card not even CONSIDERED" is a different question from "why was it not
settled", with a different answer and a different remedy (spec 5.2).

Projects 07 (limit management) and 08 (collections) both want this module and
settlement_amount.py. They are written here first, which is precisely how the
previous generation ended up with the same logic in three places. The answer in
this sketch is that both live in a `consol_core/` package this project owns and
publishes, with a frozen `contract=`, and projects 07 and 08 import it - a
project-owned library beside the Bank's credit-core library. See FRAMEWORK-
DEMANDS D16 for why the framework has nothing to say about that and should.
"""

from decider2 import param, step
from decider2.values import missing_as, na
from decider2.rules import ruleset


Settleability = ruleset(
    name="settleability",
    reads=[
        "account_type_code",
        "provider_code",
        "is_internal",
        "account_status_code",
        "is_disputed",
        "debt_review_status_account",
        "opened_date",
        "decision_date",
        "is_secured",
        "security_type_code",
        "is_revolving",
        "provider_issues_third_party_quotation",
        "provider_accepts_third_party_settlement",
        "has_unexpired_quotation",
        "is_client_excluded",
        "is_client_nominated",
    ],
    writes=[
        "settleability_code",
        "settleability_rule_id",
        "settleability_reason_code",
        "quotation_turnaround_days",
        "security_release_days",
        "security_releases",
        "security_transfers",
        "early_settlement_rule_code",
    ],
    prioritisation="first_match",
    interior="config/settleability/classification_rules.json",
    params=None,
    contract="contracts/settleability.json",
)


# The nine classes, in evaluation order, with what each one costs the search:
#
#   7  UNKNOWN                      provider unknown, or attributes absent.
#                                   DISTINCT FROM "not settleable" - library 7.4.
#                                   Excluded from settlement sets but REMAINS IN
#                                   THE OBLIGATION FIGURE, and the count is
#                                   reported, because a scenario built on an
#                                   inventory with six unknowns is a weaker
#                                   scenario and the consultant should be told.
#   6  BLOCKED BY STATUS            disputed, under debt review at account level,
#                                   handed over, in legal process, ceded, written
#                                   off.
#   5  BLOCKED BY POLICY            opened within 3 months (CON-INT-02); account
#                                   type on the non-consolidatable list - court-
#                                   ordered maintenance, emoluments attachment,
#                                   tax debt, municipal accounts, student loans
#                                   on concessionary terms.
#   4  BLOCKED BY PROVIDER          17 of ~200 providers; 2.3% of external
#                                   accounts.
#   3  SETTLEABLE WITH SECURITY     settleable only if the security releases or
#      RELEASE                      transfers. Carries security_release_days and
#                                   a release cost. Vehicle 10 days; residential
#                                   bond cancellation 45-90.
#   8  PARTIALLY SETTLEABLE         revolving. "Settlement" is a paydown plus a
#                                   limit action, and the amount is the balance
#                                   at a FUTURE date, which the client can change
#                                   by spending. The only class whose settlement
#                                   amount the client controls.
#   2  SETTLEABLE, QUOTATION        settlement_amount is ESTIMATED; the outcome
#      OBTAINABLE                   is conditional on the actual quotation.
#   1  SETTLEABLE, QUOTATION HELD   unexpired quotation, firm amount and expiry.
#   0  SETTLEABLE, INTERNAL         derivable instantly and exactly.
#
# The class numbering is NOT the evaluation order, and that is deliberate: 0..3
# are the settleable classes and sort naturally for H6's amount_basis_rank, while
# 4..8 are the blocked and awkward ones. An interior whose rule order matched its
# code order would be a coincidence waiting to be broken by an insertion.


@step(output="amount_basis_rank")
def amount_basis_rank(settleability_code: int) -> int:
    """Rank settleability by how firm the amount is. H6 orders on this.

    0 internal (exact) < 1 quoted (firm) < 2 estimated < 3 partial (client-
    controlled). A scenario built on held quotations is executable this week; one
    built on estimates is conditional; one built on a revolving paydown can be
    invalidated by the client buying groceries on Saturday.
    """
    pass  # small lookup, settleability_code -> firmness rank


@step(output="nomination_conflict")
def nomination_conflict(
    is_client_nominated: bool,
    settleability_code: int,
) -> int:
    """A mandatory account that is not settleable is a hard conflict, not a drop.

    Spec 5.2: it "must be surfaced to the consultant, not silently dropped". The
    conflict code carries WHY the nominated account cannot be settled, because
    the conversation the consultant then has is "your furniture account is in
    dispute and we cannot settle a disputed account until the dispute closes",
    not "the system would not let me".
    """
    pass  # 0 none, else CON-CONF-01..07 mirroring the blocking class


@step(output="inventory_quality")
def inventory_quality(
    unknown_count: int,
    settleable_count: int,
    total_account_count: int,
    settleable_proportion_of_obligation: float,
    bureau_is_stale: bool,
) -> dict:
    """Inventory-level counts by class, and how much of the debt is reachable.

    Reported to the consultant BEFORE the search runs, because "we can only see
    nine of your eleven accounts clearly" changes the conversation and changes
    what the client should be told about the answer's reliability.
    """
    pass  # counts by class, proportions, staleness flags


# --- where client nominations and exclusions are applied ----------------------
#
# Here, not in the search. A surprise to most people who read spec 5.2, and
# right: an excluded account is removed from CANDIDACY, which shrinks the mask
# index, which shrinks the space, which is the only place shrinking is free. An
# exclusion applied in the search would be a filter over already-generated
# candidates - the same answer at several hundred times the cost, and with the
# excluded account occupying bit positions the ordering rules then waste.
#
# The refusal is recorded either way. "You told us not to settle your employer
# loan" is an answer the contact centre needs a week later, and it is not
# derivable from the absence of the account in the result.
