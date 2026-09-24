"""Stage 5.1 -- intake and eligibility (spec 06 §5.1).

Seven gates, evaluated in the declared order, every one recorded (not only the
failing one -- 09 §5.15 item 14: "evaluation recorded, not only firing").
CON-ELIG-04 depends on §5.2's settleable count, so this step runs *after*
`inventory.settleability` in the pipeline, even though the spec lists it first --
`decider`'s `dag` resolves that from reads/writes, not list order (00/03 precedent).
"""
from __future__ import annotations

from datetime import date
from typing import Any

from decider import frame_step

import polars as pl

CON_ELIG_01 = "CON-ELIG-01"  # debt review -> route to restructure, not a decline
CON_ELIG_02 = "CON-ELIG-02"  # administration / sequestration -> terminal decline
CON_ELIG_03 = "CON-ELIG-03"  # consolidated within 6 months -> refer
CON_ELIG_04 = "CON-ELIG-04"  # fewer than 2 settleable accounts -> not applicable
CON_ELIG_05 = "CON-ELIG-05"  # no verified income -> terminal decline
CON_ELIG_06 = "CON-ELIG-06"  # bureau unobtainable -> terminal decline
CON_ELIG_07 = "CON-ELIG-07"  # active reckless-lending allegation -> refer to Compliance

ROUTE_PROCEED = "proceed"
ROUTE_DEBT_REVIEW = "debt_review_path"
ROUTE_DECLINE = "decline"
ROUTE_REFER = "refer"
ROUTE_NOT_APPLICABLE = "not_applicable"

_MIN_SETTLEABLE_ACCOUNTS = 2
_SERIAL_CONSOLIDATION_MONTHS = 6


def _months_since(decision_date, prior_date) -> int | None:
    if prior_date is None:
        return None
    return (decision_date.year - prior_date.year) * 12 + (decision_date.month - prior_date.month)


def _evaluate(row: dict) -> dict:
    gate_ids, verdicts = [], []

    def record(gate_id: str, fired: bool) -> None:
        gate_ids.append(gate_id)
        verdicts.append(fired)

    under_debt_review = bool(row.get("client_under_debt_review"))
    record(CON_ELIG_01, under_debt_review)

    under_admin = bool(row.get("client_under_administration"))
    record(CON_ELIG_02, under_admin)

    months = _months_since(row["decision_date"], row.get("last_consolidation_date"))
    serial = months is not None and months < _SERIAL_CONSOLIDATION_MONTHS
    record(CON_ELIG_03, serial)

    settleable_count = row.get("settleable_count") or 0
    too_few = settleable_count < _MIN_SETTLEABLE_ACCOUNTS
    record(CON_ELIG_04, too_few)

    no_income = not row.get("income_verified", False)
    record(CON_ELIG_05, no_income)

    bureau_unobtainable = bool(row.get("bureau_unobtainable"))
    record(CON_ELIG_06, bureau_unobtainable)

    reckless_allegation = bool(row.get("active_reckless_lending_allegation"))
    record(CON_ELIG_07, reckless_allegation)

    # First-match routing, in the priority the gate table implies (terminal and
    # cheap gates first; §5.1 "the ordering matters").
    if under_debt_review:
        route = ROUTE_DEBT_REVIEW
    elif under_admin:
        route = ROUTE_DECLINE
    elif bureau_unobtainable:
        route = ROUTE_DECLINE
    elif no_income:
        route = ROUTE_DECLINE
    elif serial or reckless_allegation:
        route = ROUTE_REFER
    elif too_few:
        route = ROUTE_NOT_APPLICABLE
    else:
        route = ROUTE_PROCEED

    return {
        "eligibility_gate_ids": gate_ids,
        "eligibility_gate_verdicts": verdicts,
        "eligibility_route": route,
        "is_eligible_for_search": route == ROUTE_PROCEED,
    }


# Typed so a served JSON request's date strings arrive as dates.
@frame_step(
    reads={"decision_date": date, "client_under_debt_review": Any, "client_under_administration": Any,
           "last_consolidation_date": date | None, "settleable_count": Any, "income_verified": Any,
           "bureau_unobtainable": Any, "active_reckless_lending_allegation": Any},
    writes=["eligibility_gate_ids", "eligibility_gate_verdicts", "eligibility_route", "is_eligible_for_search"],
)
def eligibility_gates(df: pl.DataFrame) -> pl.DataFrame:
    results = [
        _evaluate(row) for row in df.select(
            "decision_date", "client_under_debt_review", "client_under_administration", "last_consolidation_date",
            "settleable_count", "income_verified", "bureau_unobtainable", "active_reckless_lending_allegation",
        ).to_dicts()
    ]
    return df.with_columns(pl.DataFrame(results))
