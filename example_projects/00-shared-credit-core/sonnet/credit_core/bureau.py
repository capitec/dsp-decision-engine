"""`core.bureau` -- bureau view normalisation (spec 00 §6.13). Thin: one bureau schema (SCOPE.md).

Normalises the raw response (accounts, enquiries, public records, all
ragged) into counts, worst-status indicators, adverse-item ages, enquiry
velocity, `bureau_as_of_date` and `bureau_is_stale`. Only the one incoming
schema this slice's `sample_request.json` uses -- the three-bureaux
normalisation layer (00 §6.13 "Hard part") is out of scope here.
"""
from __future__ import annotations

from decider import frame_step

import polars as pl

BUREAU_STALENESS_DAYS = 90  # the product's tolerance (00 §4: `bureau_is_stale`); a param would need a Step, not a FrameStep


def _normalise(response: dict, decision_date, staleness_days: int) -> dict:
    accounts = response.get("accounts") or []
    enquiries = response.get("enquiries") or []
    public_records = response.get("public_records") or []
    worst_status = max((a.get("status_code") or 0) for a in accounts) if accounts else 0
    recent_enquiries_3m = sum(1 for e in enquiries if _within_months(e.get("enquiry_date"), decision_date, 3))
    recent_enquiries_12m = sum(1 for e in enquiries if _within_months(e.get("enquiry_date"), decision_date, 12))
    as_of = response.get("as_of_date")
    is_stale = _age_days(as_of, decision_date) > staleness_days if as_of else True
    quality = "no_record" if not response else ("stale" if is_stale else "ok")
    return {
        "bureau_account_count": len(accounts),
        "bureau_enquiry_count_3m": recent_enquiries_3m,
        "bureau_enquiry_count_12m": recent_enquiries_12m,
        "bureau_public_record_count": len(public_records),
        "bureau_worst_status": worst_status,
        "bureau_as_of_date": as_of,
        "bureau_is_stale": is_stale,
        "bureau_data_quality": quality,
    }


def _age_days(as_of, decision_date) -> int:
    return (decision_date - as_of).days if as_of else 10**6


def _within_months(event_date, decision_date, months: int) -> bool:
    if event_date is None:
        return False
    months_diff = (decision_date.year - event_date.year) * 12 + (decision_date.month - event_date.month)
    return 0 <= months_diff <= months


@frame_step(reads=["bureau_response", "decision_date"],
            writes=["bureau_account_count", "bureau_enquiry_count_3m", "bureau_enquiry_count_12m",
                    "bureau_public_record_count", "bureau_worst_status", "bureau_as_of_date", "bureau_is_stale",
                    "bureau_data_quality"])
def normalise_bureau(df: pl.DataFrame) -> pl.DataFrame:
    results = [
        _normalise(row.get("bureau_response") or {}, row.get("decision_date"), BUREAU_STALENESS_DAYS)
        for row in df.select("bureau_response", "decision_date").to_dicts()
    ]
    return df.with_columns(pl.DataFrame(results))
