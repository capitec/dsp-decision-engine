"""Which products may carry a given settlement set. A table, not a chain of ifs.

Spec 5.6.3. Four eligibility predicates, evaluated per settlement set, producing
zero to four rows out of one. The `may_carry` predicates are business rules with
owners, so they are a `decision_table` interior (doc 08 3.4): N rows x M
condition columns, uniform operators, GENERIC KERNEL, and therefore an interior
change is FREE - no staged compile.

That matters for change scenario 1 more than it looks. "A fifth product joins the
search" splits cleanly into two changes of two different classes:

    adding product 21 TO ROUTING       an interior row. Free. Product team.
    adding product 21's PRICING ARM    a Python module and a Branch arm.
                                       Skeleton. Engineer. Redeploy.

Doc 08 2's three change classes handle each half correctly and have nothing to
say about the fact that ONE business change straddles two of them. The routing
row must not activate before the arm exists, and nothing in the framework stops
it: a routing table naming product 21 with no arm registered is a config that
validates and then fails at the Branch. See FRAMEWORK-DEMANDS D17.
"""

from decider2 import param, step
from decider2.rules import decision_table
from decider2.values import na


ProductRouting = decision_table(
    name="product_routing",
    reads=[
        "settlement_set_mask",
        "settleable_count_in_set",
        "contains_secured_still_secured",
        "contains_existing_vehicle_finance",
        "all_accounts_revolving_transferable",
        "contains_term_loan",
        "client_holds_bond_with_bank",
        "bond_in_good_standing",
        "available_equity",
        "required_advance_estimate",
        "warning_acknowledged",
        "unencumbered_vehicle_offered",
        "vehicle_passes_age_mileage_gates",
        "vehicle_guide_available",
        "assessment_mode_code",
    ],
    writes=["product_code", "may_carry_reason_code", "may_not_carry_reason_code"],
    prioritisation="all",  # ALL matching products, not the first. The whole point.
    interior="config/products/routing_table.json",
    contract="contracts/product_routing.json",
)


# The four rules, as they read in the interior document:
#
#   11 FLEX LOAN CONSOLIDATION
#       may carry: any settleable set of 2 or more accounts, provided no secured
#       account remains secured. The universal fallback - if any product can
#       carry a set, this one usually can.
#
#   30 DRIVE FINANCE REFINANCE
#       may carry: the set contains the client's existing vehicle finance
#       account, OR the client offers an unencumbered vehicle passing the age and
#       mileage gates. WHERE THE VEHICLE FINANCE ACCOUNT EXISTS IT IS MANDATORY
#       IN THE SET - a vehicle cannot be refinanced without settling what is
#       currently secured on it. That is a constraint on the SET, expressed here
#       rather than in the plan, and it is the only place a product's rules reach
#       backwards into candidate generation.
#
#   20 EVERYDAY CARD BALANCE TRANSFER
#       may carry: every account in the set is a card or revolving facility on
#       the transferable account type list. A set containing one term loan cannot
#       be routed here. Note the quantifier: EVERY, not any. A rule written with
#       `any` routes a set containing one card and four personal loans to a
#       balance transfer, prices it, and the rejection only arrives at the
#       product's own gate - four wasted evaluations out of a budget of 400.
#
#   40 HOME LOAN FURTHER ADVANCE
#       may carry: the client holds a bond with the Bank, available equity covers
#       the advance, and the client HAS ACKNOWLEDGED THE SECURITY WARNING.
#
# The warning condition in routing is deliberate and it is the mechanism behind
# acceptance criterion 12 ("a product 40 scenario cannot be selected without a
# recorded warning acknowledgement, demonstrated by a test that attempts it").
# Putting it here rather than at selection means an unacknowledged product 40
# scenario is never GENERATED, so there is no path by which one can be selected.
# A check at selection time would be a guard that a later refactor can remove;
# a routing condition is load-bearing and its removal breaks the fan-out.


@step(output="withdraw_product")
def withdraw_product(
    product_code: int,
    vehicle_guide_available: bool,
    avm_age_days: int,
    rate_card_resolved: bool,
) -> int:
    """Withdraw a product from routing where its data is unavailable, and RECORD IT.

    Spec 8, Degradation: "Where the vehicle guide is unavailable, product 30 is
    withdrawn from routing and the withdrawal is recorded - the assessment does
    not fail."

    This is the shape of degradation that the framework's per-node fallback (doc
    02 3.2) cannot express, because it is not about compilation: it is a BUSINESS
    degradation with a recorded consequence. A withdrawn product must appear in
    the output as withdrawn, because "why was I not offered a vehicle refinance"
    has an answer and the answer is "the valuation guide was unavailable on the
    day", not silence.
    """
    pass  # 0 available, else WD-30-GUIDE / WD-40-AVM / WD-ANY-CARD


@step(output="product_terms_axis")
def product_terms_axis(
    product_code: int,
    product_min_term: int,
    product_max_term: int,
) -> list:
    """Each product declares its OWN candidate axis. The plan does not assume terms.

    Products 11, 30 and 40 return a term list. Product 20 returns its
    PROMOTIONAL DURATION list - [0, 6, 12, 18, 24] - because it has no term, and
    the plan expands whatever axis it is handed.

    This is the smallest possible version of the four-heterogeneous-products
    problem and it appears two stages before anyone expects it: candidate
    generation, not pricing. A plan that hardcoded `for term in terms` would
    already have decided that every product has a term.
    """
    pass  # per-product axis from the product's own params namespace
