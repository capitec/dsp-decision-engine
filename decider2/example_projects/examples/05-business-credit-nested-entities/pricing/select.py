"""Stage 5.12c -- choose the offer. Candidate -> Application.

One fold. s5.12's requirement is stated as a property:

    "The Bank must offer the LARGEST admissible amount. Ties are broken by the
     shortest term, then by the lowest total_cost_of_credit."

which is a `best_of` with a three-level total tie-break, and the tie-break's
last level is a stable identity so that the order is total rather than merely
usually-total. That last level is what makes the outcome reproducible when two
candidates agree on all three declared criteria -- which happens, because 40
amount bands and 55 terms produce ties.
"""

from decider2 import module, param, step, Gather, best_of, count, asc, desc
from grains import Candidate, Application


SelectOffer = Gather(
    Candidate, into=Application, name="select_offer",

    chosen = best_of(
        "candidate_amount",
        where="is_admissible",
        tie_break=(asc("term_months"),
                   asc("total_cost_of_credit"),
                   asc("candidate_id")),          # total, and stable
        lift=["candidate_id", "candidate_amount", "term_months",
              "nominal_annual_rate", "rate_cell_id", "rate_card_version",
              "security_type", "instalment", "initiation_fee",
              "monthly_service_fee", "total_cost_of_credit",
              "effective_annual_rate", "candidate_dscr",
              "binding_constraint_code"],
    ),

    # The ledger statistics the caller and the committee pack need.
    candidates_evaluated  = count(),
    candidates_admissible = count(where="is_admissible"),

    # "Why not larger?" -- the best INADMISSIBLE candidate above the chosen one,
    # and what stopped it. This is s5.12's "Records" clause turned into a value,
    # and it is also two thirds of change scenario 15 ("the smallest change that
    # would have made this approvable") for free.
    next_larger_blocked = best_of(
        "candidate_amount",
        where="is_inadmissible",
        tie_break=(asc("term_months"), asc("candidate_id")),
        lift=["candidate_id", "candidate_amount", "term_months",
              "binding_constraint_code", "candidate_dscr", "security_type"],
    ),
)


def offered_amount(chosen_candidate_amount: float, candidates_admissible: int) -> float:
    """The offer, or zero where no candidate was admissible."""
    pass


def offer_outcome_code(
    offered_amount: float,
    requested_amount: float,
    candidates_admissible: int,
    binding_constraint_code: int,
    product_minimum_amount: float,
    reduced_offer_floor_pct: float = param(
        60.0, ge=0.0, le=100.0,
        description="s6.1 reduced-offer floor. Product, quarterly."),
) -> int:
    """s5.12's five outcomes: full, reduced, reduced-and-referred, decline at the
    product minimum, decline with reason 5501."""
    pass


def search_truncated(candidates_evaluated: int) -> bool:
    """s5.12: a declared maximum, a declared ordering, and a declared outcome on
    truncation -- the best admissible candidate found so far, flagged, and
    referred if the truncation occurred before any admissible candidate.

    With an enumerated grain this can only fire when the ENUMERATION exceeds
    `Candidate.capacity`, which is a property of the product and term tables
    rather than of the search. That is a much better failure mode than a
    search that stopped early: it is detectable when Treasury publishes a rate
    card with more bands, not when an application meets it.
    """
    pass


SelectedOffer = module(
    offered_amount, offer_outcome_code, search_truncated,
    name="selected_offer", grain=Application,
    taps=["offered_amount", "offer_outcome_code", "binding_constraint_code",
          "candidates_evaluated", "candidates_admissible"],
)
