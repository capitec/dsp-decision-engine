"""Bureau view normalisation: counts, worst status, enquiry velocity and staleness from one bureau response."""
from __future__ import annotations

from datetime import date
from typing import TypedDict

from decikit import allow_fallback, param, step


class BureauAccount(TypedDict, total=False):
    status_code: int


class BureauEnquiry(TypedDict, total=False):
    enquiry_date: date


class BureauResponse(TypedDict, total=False):
    accounts: list[BureauAccount]
    enquiries: list[BureauEnquiry]
    public_records: list[dict]
    as_of_date: date


def _within_months(event: date, decision_date: date, months: int) -> bool:
    return 0 <= (decision_date.year - event.year) * 12 + (decision_date.month - event.month) <= months


@allow_fallback
def normalise_bureau_response(
    bureau_response: BureauResponse, decision_date: date, staleness_days: int = param(90, ge=0),
) -> tuple[int, int, int, int, int, date | None, bool, str]:
    accounts = bureau_response.get("accounts") or []
    dates = [e["enquiry_date"] for e in bureau_response.get("enquiries") or [] if e.get("enquiry_date")]
    as_of = bureau_response.get("as_of_date")
    stale = as_of is None or (decision_date - as_of).days > staleness_days
    quality = "no_record" if not bureau_response else ("stale" if stale else "ok")
    return (len(accounts), sum(_within_months(d, decision_date, 3) for d in dates),
            sum(_within_months(d, decision_date, 12) for d in dates),
            len(bureau_response.get("public_records") or []),
            max((a.get("status_code") or 0 for a in accounts), default=0), as_of, stale, quality)


normalise_bureau = step(normalise_bureau_response, outputs=(
    "bureau_account_count", "bureau_enquiry_count_3m", "bureau_enquiry_count_12m", "bureau_public_record_count",
    "bureau_worst_status", "bureau_as_of_date", "bureau_is_stale", "bureau_data_quality"))
