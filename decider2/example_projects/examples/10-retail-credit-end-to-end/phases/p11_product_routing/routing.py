"""P11 product routing — 52 decision points. Spec 5.12. Elided from
phases/__init__.py "for length"; this package is the real file.

The phase most entangled in O-09, the second genuine cycle (ordering.py):
product eligibility is amount-keyed, the amount comes from the solve, the
solve is bounded by affordability, and affordability's buffer is grade- AND
product-keyed. This file is called TWICE per decision by flow.py — once by
`P09.at_site("product_neutral")`'s caller before this runs, and once after —
never duplicated as a second implementation.
"""

from __future__ import annotations

from decider2 import module, param

def product_eligibility(product_code: int, requested_amount: float, term_months: int,
                        evidence_tier: int, risk_grade: int, channel_code: int,
                        existing_product_holdings: list[int]) -> bool:
    """31 decision points. Amount/term in range, evidence tier met, security
    valued (30/40), existing-relationship (11/40), channel permits it, grade
    at or better than the product floor (40 <= grade 7, 30 <= grade 9)."""
    pass  # evaluate all per-product eligibility conditions

def substitution(product_eligibility: bool, requested_product: int,
                 candidate_products: list[int]) -> dict | None:
    """13 decision points. A policy decision with a DISCLOSURE consequence —
    the client asked for one thing and is offered another. Every substitution
    is recorded with the rule that made it."""
    pass  # where the requested product is ineligible, find a carrying alternative

def preference_order(candidate_products: list[int], substitution: dict | None) -> list[int]:
    """8 decision points. Owned jointly by T7/T8/T9 — "exactly as comfortable
    as it sounds" (spec 5.12)."""
    pass  # rank the eligible product set for P16's later arbitration

def routed_products(product_eligibility: bool, substitution: dict | None,
                    preference_order: list[int]) -> list[int]:
    pass  # the final routed set, feeding flow.py's routed_products Collection

Route = module(product_eligibility, substitution, preference_order, routed_products,
               name="routing")
