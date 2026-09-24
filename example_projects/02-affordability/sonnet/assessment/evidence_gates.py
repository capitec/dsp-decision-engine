"""The checks that decide `evidence_sufficiency_code` (spec 02 §5.7.1; §4.2).

`indeterminate` "is a distinct outcome and conflating it with `fail` is the
most consequential error available in this project" (§5.7.1). This is the
bureau-freshness check; the REFER-account check lives in `household.py`
(computed inline while the merged account list is already in hand -- see
that module's docstring), and the applicant-income-evidence-gap check also
lives in `household.py`, next to the income combination it reads.
"""
from __future__ import annotations

from datetime import date

from decider import missing_as, param

# Products with the longer, mortgage-secured freshness window (§4.2: "fourteen for
# mortgage-secured products"). `credit_core.vocab.PRODUCT_HOME_LOAN_FURTHER_ADVANCE`.
MORTGAGE_PRODUCTS = frozenset({40})


def bureau_is_stale(
    decision_date: date, product_code: float = missing_as(-1.0), bureau_as_of_date: date | None = None,
    staleness_window_days: int = param(7, ge=1, le=60),
    staleness_window_days_mortgage: int = param(14, ge=1, le=60),
) -> bool:
    """§4.2: "the debt repayment history... must have been obtained within a short window
    before approval -- illustratively seven business days for non-mortgage credit and
    fourteen for mortgage-secured products." No `bureau_as_of_date` at all is stale by
    construction -- there is nothing to be fresh."""
    if bureau_as_of_date is None:
        return True
    window = staleness_window_days_mortgage if product_code in MORTGAGE_PRODUCTS else staleness_window_days
    return (decision_date - bureau_as_of_date).days > window
