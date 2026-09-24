from __future__ import annotations
"""Bureau data normalization (00 §6.14)."""
from dataclasses import dataclass
from datetime import date


@dataclass
class BureauResult:
    """Result of bureau fetch."""
    bureau_as_of_date: date
    bureau_is_stale: bool
    account_count: int
    accounts: list[dict]


def normalize_bureau_response(
    response: dict,
    decision_date: date,
    staleness_threshold_days: int = 90
) -> BureauResult:
    """
    Normalize bureau response (consumer and commercial formats).
    Implements 09 §5.15 item 8: mutable state captured as at decision date.
    """
    bureau_date = response.get("as_at_date", decision_date)
    days_old = (decision_date - bureau_date).days
    is_stale = days_old > staleness_threshold_days

    accounts = response.get("accounts", [])

    return BureauResult(
        bureau_as_of_date=bureau_date,
        bureau_is_stale=is_stale,
        account_count=len(accounts),
        accounts=accounts
    )
