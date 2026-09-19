"""P17 final validation — 61 assertions across six products, 14 for product 10.
Spec 5.18. Re-derives the recommended offer end to end from its OWN amount,
term and product, carrying nothing forward — O-18 (ordering.py) makes this an
isolation constraint the build enforces, not a discipline this file keeps.

Every parameter here that names a multi-basis shared value must carry a
`basis_of(...)` marker (values/bases.py) rather than a bare name — a bare
`instalment_cap: Money` parameter in this file does not compile, because O-18
declares this phase may read only `("offered_amount", "term_months",
"product_code")` plus effective-dated artefacts plus the shipped offer's
PROVENANCE.
"""

from __future__ import annotations

from decider2 import module, basis_of

def assertion_01_rate_matches_cell(offered_amount: float, term_months: int,
                                   product_code: int, decision_date: str) -> bool:
    pass  # re-derive the rate from the card cell in force at decision_date; compare

def assertion_02_rate_within_statutory_ceiling(offered_amount: float, term_months: int,
                                               product_code: int, decision_date: str) -> bool:
    pass  # re-derive nominal_annual_rate incl. add-on; compare to the ceiling in force

def assertion_03_fee_matches_piecewise(offered_amount: float) -> bool:
    pass  # re-derive the initiation fee from the piecewise formula; compare, check the cap

def assertion_04_service_fee_capped(decision_date: str) -> bool:
    pass  # re-derive the capped service fee; compare

def assertion_05_premium_matches_table(offered_amount: float, term_months: int,
                                       product_code: int) -> bool:
    pass  # re-derive the credit life premium; compare, check the R4.25 ceiling

def assertion_06_instalment_recomputes(offered_amount: float, term_months: int,
                                       product_code: int) -> bool:
    """To the cent."""
    pass  # re-derive the full instalment chain from scratch; compare exactly

def assertion_07_within_affordability(offered_amount: float, term_months: int,
                                      max_affordable_instalment: float = basis_of("shipped_offer")
                                      ) -> bool:
    """The subtle one, spec 5.18: re-derives against the obligations basis the
    offer was PRICED ON, identified by the shipped offer's provenance
    (loop_pass_index, value_basis_code) — not against "the" affordability
    answer, of which there are up to four live in this decision."""
    pass  # re-derive the instalment ceiling under the shipped offer's own basis; compare

def assertion_08_in_duplum(offered_amount: float, term_months: int, product_code: int) -> bool:
    pass  # re-derive the scheduled in duplum test

def assertion_09_total_cost_ratio(offered_amount: float, term_months: int,
                                  product_code: int) -> bool:
    pass  # re-derive total cost ratio; compare to 1.92

def assertion_10_amount_within_ceilings(offered_amount: float, product_code: int) -> bool:
    pass  # re-check against every ceiling that bound, and the product minimum

def assertion_11_term_within_cap(term_months: int, product_code: int) -> bool:
    pass  # re-check term_cap and the permitted term list for the segment

def assertion_12_grade_meets_floor(product_code: int) -> bool:
    pass  # re-check worst_acceptable_grade

def assertion_13_amount_multiple_of_grid(offered_amount: float) -> bool:
    pass  # offered_amount % 250 == 0

def assertion_14_table_versions_current(decision_date: str) -> bool:
    pass  # every table version referenced resolves to decision_date's version

def assertion_set(product_code: int) -> dict:
    """Which of the 14 apply for product_code — derived (entrypoints/manifest.py's
    applicability rule), never an `if product_code == 10` here. On entry point 7
    only 1-6 are applicable, because 7-14 read values P10 did not produce."""
    pass  # collect the applicable assertions' results; ANY failure is a hard fail

AssertionSet = module(assertion_01_rate_matches_cell, assertion_02_rate_within_statutory_ceiling,
                      assertion_03_fee_matches_piecewise, assertion_04_service_fee_capped,
                      assertion_05_premium_matches_table, assertion_06_instalment_recomputes,
                      assertion_07_within_affordability, assertion_08_in_duplum,
                      assertion_09_total_cost_ratio, assertion_10_amount_within_ceilings,
                      assertion_11_term_within_cap, assertion_12_grade_meets_floor,
                      assertion_13_amount_multiple_of_grid, assertion_14_table_versions_current,
                      assertion_set, name="assertions")
