"""The facility catalogue (spec 11 §4.2, table 29) and facility identity.

Only products 50 (term) and 51 (revolving) are built (SCOPE.md); 52-58 are
named in `vocab.PRODUCT_CATALOGUE_ONLY` but not implemented.

`facility_id` (spec 11 §4.8) is this project's own name -- the library has
nothing like it (a retail account or a business application does not need an
identity stable across a five-year sequence of re-decisions). Minted once at
origination and carried, unchanged, through every later decision of record for
that facility -- the "same subject" requirement of §5.10.2 comparability
requirement 1. Minted by the **caller**, the same pattern project 00's
`credit_core.evidence.new_decision_id()` documents for `decision_id` ("assigned
once before any logic runs... generating it inside a step would make it a
second, later assignment") -- so `mint_facility_id()` is a plain function, not
a decider step, called before a new-facility request is built (see
`tests/test_origination.py`, `inference.py`).
"""
from __future__ import annotations

import itertools
from datetime import date

from business_credit_e2e import vocab

_facility_ids = itertools.count(500_001)


def mint_facility_id() -> int:
    """A new facility's identity, minted once at EP-1/EP-2 and never reassigned.
    A real deployment would use a durable sequence (or the persisted decision
    store's own key); this project's in-memory counter is a placeholder for that
    -- see NOTES.md "What I would do next"."""
    return next(_facility_ids)


def facility_shape(product_code: int, offered_amount: float, term_months: int | None) -> dict:
    """Product 50 (term) carries an amount and a term; product 51 (revolving)
    carries a limit and no term -- spec 11 §4.2's own distinction ("n/a" in the
    term column for 51). 05's `pricing.py` computes `offered_amount` the same
    way for both (a single-lookup ceiling, SCOPE.md); this project decides what
    that number *means* per product, which 05 has no reason to know."""
    is_revolving = product_code in vocab.REVOLVING_PRODUCTS
    return {
        "product_code": product_code,
        "is_revolving": is_revolving,
        "approved_limit": offered_amount if is_revolving else None,
        "approved_amount": None if is_revolving else offered_amount,
        "term_months": None if is_revolving else term_months,
        "status": vocab.FACILITY_STATUS_LIVE,
    }


def next_review_date(decision_date: date, review_basis_code: int) -> date:
    """§5.4.3's ladder: a degraded review buys less time than a complete one.
    Complete/late -> 12 months; stale -> 6 months; turnover-only -> 3 months;
    not-performed -> 2 months (§5.4.3 basis 5's "exit assessment mandatory
    within 60 days" stands in for this project's own not-performed cadence)."""
    months = {
        vocab.REVIEW_COMPLETE: 12, vocab.REVIEW_LATE: 12, vocab.REVIEW_STALE: 6,
        vocab.REVIEW_TURNOVER_ONLY: 3, vocab.REVIEW_NOT_PERFORMED: 2,
    }[review_basis_code]
    year = decision_date.year + (decision_date.month - 1 + months) // 12
    month = (decision_date.month - 1 + months) % 12 + 1
    day = min(decision_date.day, 28)  # ponytail: avoids month-length edge cases; exact day is not load-bearing here
    return date(year, month, day)
