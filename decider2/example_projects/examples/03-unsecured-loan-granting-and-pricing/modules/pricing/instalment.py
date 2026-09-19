"""Amortisation, and its inverse.

§13.10 asks whether `core.instalment` can be run in reverse -- solve the
advance that produces a given instalment -- and whether using the inverse as a
STARTING POINT for the search, rather than as the answer, changes what the
library needs to publish.

It does, and this is the answer: the inverse is published as a BOUND, not as a
result.  `advance_for_instalment_relaxed` deliberately computes an OVER-ESTIMATE
by using the cheapest rate available anywhere in the search domain and the
smallest possible loadings.  Because it over-estimates, no amount above it can
be feasible at any rate -- which makes it a PROVABLE upper bound on the search
domain rather than a guess that happens to be close.  That distinction is the
difference between a heuristic and a correctness argument, and it is why the
bracket earns its place in the solve's static probe-count proof.

`invertible_pair` ties the forward and reverse directions together so that
their agreement is a framework-generated test over a declared corpus, and so
that a change to one that is not reflected in the other fails the build rather
than drifting.
"""

from __future__ import annotations

from decider2 import invertible_pair, module, monotone_in, param, step
from decider2.money import Money


@monotone_in("amount_financed", direction="increasing")
@monotone_in("nominal_annual_rate", direction="increasing")
@invertible_pair(inverse="advance_for_instalment", agree_to=Money("0.01"),
                 corpus="tests/corpora/annuity.csv")
@step(output="instalment")
def instalment(
    amount_financed: Money,
    term_months: int,
    nominal_annual_rate: int,
    monthly_service_fee: Money,
    credit_life_premium: Money,
) -> Money:
    """Ordinary monthly annuity on the amount financed, plus the service fee, plus
    the credit life premium, rounded to the cent with `round_half_up`.

    Three declarations, not one implementation detail:
      * monotone increasing in `amount_financed` at a FIXED rate -- this is the
        theorem the solve's within-segment bisection rests on;
      * monotone increasing in the rate -- this is what makes the cheapest-rate
        bracket a valid bound;
      * invertible, with a declared tolerance and a declared corpus.
    """
    pass


@step(output="advance_for_instalment")
def advance_for_instalment(
    target_instalment: Money, term_months: int, nominal_annual_rate: int,
    monthly_service_fee: Money, credit_life_premium: Money,
) -> Money:
    """Closed-form inverse annuity. The advance whose instalment is the target."""
    pass


@step(output="advance_upper_bound")
def advance_for_instalment_relaxed(
    max_affordable_instalment: Money,
    term_months: int,
    cheapest_rate_in_domain: int,
    tighten: int = param(2, ge=1, le=4, owner="credit_systems"),
) -> Money:
    """A PROVABLE upper bound on the feasible advance.

    Computed at the cheapest rate anywhere in the current domain, with the
    smallest possible fee and premium loading.  `tighten=2` re-runs it once
    against the cheapest rate in the reduced range, which typically halves the
    number of bands the segment scan has to walk.  The worst-case probe count
    is a closed-form function of `tighten` and the card's axis, computed at
    card validation -- see rate_card.search_budget_is_satisfiable.
    """
    pass


@step(output="total_cost_of_credit")
def total_cost_of_credit(instalment: Money, term_months: int) -> Money:
    """Instalment times the term."""
    pass


@step(output="effective_annual_rate")
def effective_annual_rate(
    offered_amount: Money, instalment: Money, term_months: int,
) -> int:
    """Annualised IRR on the amount ADVANCED against the full instalment stream.

    Note the asymmetry: the annuity is on the amount FINANCED, the effective
    rate is on the amount ADVANCED.  That is why the effective rate exceeds the
    nominal rate substantially at short terms, and why the 6-month offer for
    client W is suppressed at 76.6% while its total-cost ratio of 1.18 is the
    lowest in the set.  Two rules measuring two different things.

    This is a bounded Newton iteration -- `Loop(max_iterations=32)` with a
    declared convergence tolerance -- and it is the one place in this project
    where doc 03 §8.3's `Loop` is exactly the right tool with no extension
    needed.  Worth saying, because most of this sketch is complaint.
    """
    pass


Instalment = module(instalment, advance_for_instalment, advance_for_instalment_relaxed,
                    total_cost_of_credit, effective_annual_rate, name="instalment")
