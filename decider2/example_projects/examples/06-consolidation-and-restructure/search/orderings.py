"""The eight candidate-ordering rules, as an artefact Credit Risk Policy owns.

Spec question 2: "How is the candidate ordering expressed so that Credit Risk
Policy can author, read, reorder and retire the eight rules without an engineer,
while the ordering stays total and the search stays deterministic?"

The answer is a DATA-SHAPED MODULE (doc 08 3) of a new kind - `ordering` - whose
interior is config/search/ordering_rules.json. Three things make it work:

1.  The interface is code. `reads` below is the closed set of account attributes
    an ordering rule may key on. A UI edit cannot make the search depend on
    something the inventory does not carry, and static lineage survives an edit
    nobody reviewed (doc 08 3, property 1).

2.  The vocabulary is closed. A rule is {key, direction, filter, prefix} over
    `reads` plus registered feature ids. There is no expression language here
    and there will not be one - doc 08 3.2 is the governing rule, and an
    ordering that could compute is an ordering nobody can validate.

3.  THE TIE-BREAK IS APPENDED BY THE FRAMEWORK, NOT BY THE AUTHOR. Spec 5.5.4
    requires a total order. An analyst writing "sort by rate descending" has not
    written a total order and cannot be expected to know that. `ordering(...)`
    appends the declared tie-break to every rule's key and REFUSES an ordering
    whose final key is not unique over the account frame. That refusal is a
    validation error at config time, before a compile, with the offending rule
    named. It is the single most valuable thing this module kind does.

Change scenario 6 - "a ninth ordering rule is added, authored by Credit Risk
Policy, live the following Monday" - is H9 in the JSON, `enabled: false` until
its date, `enabled_from: 2026-06-01`, staged and activated. No engineer.
"""

from decider2 import param, step
from decider2.search import ordering
from decider2.frame import Aggregate, Join


# --- the ordering module kind ------------------------------------------------

CandidateOrderings = ordering(
    name="search.orderings",
    # What an ordering rule may key on. An interior naming anything else fails
    # validation with a did-you-mean, before any compile.
    reads=[
        "account_ref",
        "provider_code",
        "settlement_amount",
        "instalment",
        "balance",
        "credit_limit",
        "remaining_term_months",
        "months_in_arrears",
        "is_secured",
        "is_revolving",
        "security_releases",
        "security_transfers",
        "amount_basis_rank",
        "quotation_turnaround_days",
        "provider_worst_arrears_months",
        # registered features - steps below, referenced from the interior by id
        "settle.effective_rate",
        "settle.relief_per_rand",
        "settle.provider_relief_total",
        "settle.reaccumulation_risk",
    ],
    writes=["ordering_rank"],
    tie_break=["account_ref"],
    require_total=True,
    interior="config/search/ordering_rules.json",
    params=None,  # prefix depths live in the plan params namespace, not here
    contract="contracts/orderings.json",
)


# --- the registered features an ordering rule may reference -------------------
#
# Doc 08 3.2: a derived value is a STEP, written and registered, referenced from
# the interior by id. Not an expression string. Each of these is separately
# testable and appears in lineage and in the reviewable artefact, where an
# expression string would hide inside a rule.


@step(output="settle.effective_rate", register=True)
def effective_rate(
    nominal_annual_rate: float,
    monthly_fee: float = missing_as(0.0),
    balance: float = missing_as(0.0),
) -> float:
    """All-in annualised cost of carrying this account, including monthly fees.

    H1 says "settle the highest EFFECTIVE rate first" and means it: a R180
    monthly service fee on a R6 000 store card is 36 percentage points of
    effective rate that a nominal-rate ordering cannot see. Every lender that
    has ordered on nominal rate has settled the wrong accounts first.
    """
    pass  # nominal + annualised fee burden over balance, floored at nominal


@step(output="settle.relief_per_rand", register=True)
def relief_per_rand(instalment: float, settlement_amount: float) -> float:
    """Monthly instalment released per rand of settlement. H2's whole content."""
    pass  # instalment / max(settlement_amount, 1.0)


@step(output="settle.reaccumulation_risk", register=True)
def reaccumulation_risk(
    is_revolving: bool,
    credit_limit: float = missing_as(0.0),
    balance: float = missing_as(0.0),
    months_since_last_advance: float = missing_as(0.0),
    closure_permitted: bool = missing_as(False),
) -> float:
    """How likely this facility is to be re-used after it is settled.

    A store card settled and left open will be re-used. Where the facility
    cannot be closed, H8 prefers settling it early AND the execution package
    carries a closure condition - which is why `closure_permitted` is read here
    and again in products/card_20/pricing.py. Same fact, two consequences.
    """
    pass  # utilisation-weighted limit headroom, zero for non-revolving


@step(output="settle.provider_relief_total", register=True)
def provider_relief_total(instalment: float, provider_code: int) -> float:
    """Total instalment released by exiting this provider relationship entirely.

    Read by H7, which is the only GROUPED rule: its prefix k means "the top k
    provider relationships, entire", not "the top k accounts". Leaving a R380
    balance open at a provider the client has otherwise exited produces a
    forgotten account and a default listing.
    """
    pass  # frame-tier: sum of instalment over provider_code, broadcast back


# --- ordering ranks are frame-shaped, deliberately ---------------------------
#
# Ranking 18 accounts eight ways is a sort, and a sort is set-shaped work. It
# belongs in the frame tier (doc 02 1) and it is here, not inside a kernel.
#
# The determinism requirement this creates is NOT in the framework docs and is
# FRAMEWORK-DEMANDS D13: polars' sort must be declared STABLE, and the join that
# broadcasts `provider_relief_total` back must have a declared row order. An
# unstable sort produces a different plan, which produces a different winner,
# from identical inputs. That is the exact failure the record-tier determinism
# story protects against and the frame tier currently does not.

RankAccounts = (
    Join(CandidateOrderings.feature_frame, on="account_ref", how="left", stable=True)
    | Aggregate(by="provider_code", metrics={"provider_relief_total": "sum(instalment)"}, stable=True)
    | CandidateOrderings
)
