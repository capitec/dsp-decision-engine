"""P09 policy gates and the cap waterfall — 196 decision points, the largest
phase after fraud. Spec 5.10. Four bodies of work, four names, one phase.

The 118-entry register itself is DATA — tables/cap_register/cap_register.toml
— not Python. `Register` is the interface: it declares the five ceilings it
may narrow (values/ceilings.py), the closed predicate vocabulary an entry may
use, and the two-pass split (O-10, ordering.py) that the entries themselves
must never be aware of. This mirrors doc 08 §3's `ruleset` shape, widened:
each entry targets a `ceiling()` object instead of writing an arbitrary
output, which is what makes direction-checking (values/ceilings.py's
`direction=REDUCE_ONLY`) a build-time property of the register rather than a
convention 118 authors are trusted to keep.

Ownership crosses this one file five ways (OWNERS.toml): T4, T7, T8, T9, T3,
T1, T10 each own a slice of the 196 decision points here, none of them own
the whole file, and the register's declaration is what makes "T7 shipped 14
entries without touching T4's 89" a checkable claim rather than a hope.
"""

from __future__ import annotations

from decider2 import module, param, ruleset

# ---------------------------------------------------------------------------
# Regulatory seeds and ceilings.  12 decision points.  The ONLY ones entry
# point 7 runs (entrypoints/ep07_quotation.toml [derived.reduced] P09).
# ---------------------------------------------------------------------------

def statutory_rate_ceiling(reference_rate: float, decision_date: str) -> float:
    """A margin over the reference rate. Re-evaluated on every repo move and
    every card version. class=regulatory; O-12 pins CAP-0118 after this."""
    pass  # reference_rate-derived ceiling per the statutory formula in force

def statutory_fee_and_premium_caps(decision_date: str) -> dict:
    pass  # initiation fee cap, service fee cap, credit life R4.25/R1000 ceiling

def in_duplum_test(decision_date: str) -> dict:
    pass  # the scheduled in duplum bound

def statutory_min_max_amount_and_term(product_code: int, decision_date: str) -> dict:
    pass  # the seven disclosure-bearing statutory limits

RegulatorySeeds = module(statutory_rate_ceiling, statutory_fee_and_premium_caps,
                         in_duplum_test, statutory_min_max_amount_and_term,
                         name="regulatory_seeds")

# ---------------------------------------------------------------------------
# The register.  118 decision points.  Ordered rules, each narrowing one
# ceiling, declining outright, or doing nothing.  DATA: tables/cap_register/.
# ---------------------------------------------------------------------------

CapRegister = ruleset(
    name="cap_register",
    reads=["amount_cap", "term_cap", "limit_cap", "worst_acceptable_grade",
           "instalment_cap", "risk_grade", "segment_code", "product_code",
           "channel_code", "campaign_id", "employer_sector_code"],
    writes=["amount_cap", "term_cap", "limit_cap", "worst_acceptable_grade",
           "instalment_cap", "decline_code"],
    # 118 entries, 26 product-agnostic, 92 product-specific. Interior document:
    # tables/cap_register/cap_register.toml. Every entry declares `narrows=`
    # (which of the five `ceiling()` objects it targets), `direction` is
    # CHECKED against the ceiling's own declared direction at load, and
    # coincidence resolution (earlier binds) and the evaluated/did-not-bind
    # distinction (values/ceilings.py) are properties of the ceiling, not of
    # any one entry — so the 118 authors never implement either.
)

def register_pass(cap_register: dict, loop_pass_index: int) -> dict:
    """O-10: the two-pass split. Entries acting on amount/term/grade run
    before P10; entries narrowing instalment_cap run after it. The PASS an
    entry belongs to is derived from which ceiling it narrows
    (`pass_derived_from="narrows"` in values/ceilings.py) — never declared per
    entry, which is what keeps the split invisible to the register's authors."""
    pass  # partition register evaluation by ceiling.pass_derived_from

Register = module(register_pass, name="register")

# ---------------------------------------------------------------------------
# Exposure and concentration.  24 decision points.
# ---------------------------------------------------------------------------

def group_exposure_headroom(related_party_set: list[int], group_limit: float,
                            existing_obligations: float = param(0.0)) -> float:
    """CAP-0301. O-11 pins P02's related-party set (expensive, set-shaped)
    before this (cheap, record-shaped): computing it here would spend the
    graph query inside a 7.5 ms budget."""
    pass  # group_limit - sum(exposure across the related-party set)

def employer_sector_concentration(employer_sector_code: int,
                                  concentration_watchlist_hit: bool) -> bool:
    """CAP-0266. Degrades to a blanket R60 000 cap when the watchlist is
    unreachable (degradation/sources.py)."""
    pass  # check the 3 150-identifier concentration watchlist

def single_name_limit(client_id: int) -> float:
    pass  # single-name exposure ceiling

def in_flight_application_aggregation(client_id: int) -> dict:
    """O-24: serialised per client. The one concurrency constraint in the
    flow, not merely an ordering one."""
    pass  # aggregate in-flight exposure across simultaneous applications

ExposureAndConcentration = module(group_exposure_headroom, employer_sector_concentration,
                                  single_name_limit, in_flight_application_aggregation,
                                  name="exposure_and_concentration")

# ---------------------------------------------------------------------------
# Policy gates.  42 decision points.  Non-narrowing: decline or refer only.
# ---------------------------------------------------------------------------

def purpose_exclusion_gate(loan_purpose_code: int, product_code: int) -> bool:
    pass  # declared purpose exclusions per product

def product_pair_restriction_gate(product_code: int, existing_product_holdings: list[int]) -> bool:
    pass  # maximum permitted number of the product family already held

def campaign_validity_gate(campaign_id: int | None, decision_date: str) -> bool:
    pass  # campaign still authorised and unexpired at decision_date

def cooling_off_gate(product_code: int, last_decline_date: str | None,
                     decision_date: str) -> bool:
    pass  # cooling-off window since the last decline on this product

PolicyGates = module(purpose_exclusion_gate, product_pair_restriction_gate,
                     campaign_validity_gate, cooling_off_gate, name="policy_gates")
