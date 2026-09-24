"""`core.exposure` -- group exposure aggregation (spec 00 §6.16). Thin, no graph (SCOPE.md).

Takes an already-resolved list of related-facility balances (the caller
does the "related party" graph query -- out of scope here, SCOPE.md)
and sums them against a cap, *as at* a date (addendum A8: `core.exposure`
aggregates as-at a date, including contingent exposure at declared
conversion factors).
"""
from __future__ import annotations

from decider import frame_step, param, step

import polars as pl


def _as_at(facilities: list, as_of) -> float:
    total = 0.0
    for f in facilities or []:
        opened = f.get("opened_date")
        closed = f.get("closed_date")
        if opened is not None and str(opened) > str(as_of):
            continue
        if closed is not None and str(closed) <= str(as_of):
            continue
        balance = f.get("balance") or 0.0
        conversion_factor = f.get("conversion_factor", 1.0) if f.get("is_contingent") else 1.0
        total += balance * conversion_factor
    return round(total, 2)


@frame_step(reads=["related_facilities", "exposure_as_of_date"], writes=["total_exposure"])
def total_exposure(df: pl.DataFrame) -> pl.DataFrame:
    values = [
        _as_at(row.get("related_facilities") or [], row.get("exposure_as_of_date"))
        for row in df.select("related_facilities", "exposure_as_of_date").to_dicts()
    ]
    return df.with_columns(pl.Series("total_exposure", values))


def exposure_headroom(total_exposure: float, group_exposure_cap: float = param(5_000_000.0, ge=0.0)) -> float:
    return round(max(0.0, group_exposure_cap - total_exposure), 2)


exposure_headroom_step = step(exposure_headroom)
