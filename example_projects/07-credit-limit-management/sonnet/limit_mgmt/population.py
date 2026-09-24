"""§5.1 -- population and account state: the derived quantities later stages key off.

`revolving_utilisation` (spot) is the library's own name (used unchanged);
this module adds the 6-month mean the matrix actually keys on (§5.1: "the
matrix keys off the 6-month mean, not the spot value, precisely to damp the
oscillation in §5.8") and the trailing 90th-percentile observed spend the
spend cap (C5) needs. Both are genuinely row-shaped over a ragged 6-24
bucket history, so they are `frame_step`s, not scalar steps -- decider's
documented escape hatch for exactly this shape (BRIEF: "frame_step").
"""
from __future__ import annotations

import statistics

import polars as pl
from decider import frame_step, missing_as, step

# utilisation_band (§6.2): 8 bands, lower closed, upper open, on the 6-month mean.
_UTIL_BANDS = [0.0, 0.10, 0.25, 0.40, 0.55, 0.70, 0.90, float("inf")]

# mob_band (§6.2): 6 bands.
_MOB_BANDS = [6, 12, 18, 24, 36, 60, 10_000]


def utilisation_band(revolving_utilisation_6m: float) -> int:
    """Exactly one band is read; band edges are closed at the lower bound, open at the
    upper, without exception (§5.4: "whichever the code does is not an answer")."""
    u = max(0.0, revolving_utilisation_6m)
    for i in range(8):
        lo = _UTIL_BANDS[i]
        hi = _UTIL_BANDS[i + 1] if i + 1 < len(_UTIL_BANDS) else float("inf")
        if lo <= u < hi:
            return i + 1
    return 8


def mob_band(months_on_book: int) -> int:
    if months_on_book < 6:
        return 1  # X14 excludes these; banding still resolves for completeness of the record
    for i, edge in enumerate(_MOB_BANDS):
        if months_on_book < edge:
            return i if i > 0 else 1
    return 6


utilisation_band_step = step(utilisation_band)
mob_band_step = step(mob_band)


@frame_step(reads=["cycle_balances", "cycle_limits"], writes=["revolving_utilisation_6m", "revolving_utilisation_3m"])
def utilisation_means(df: pl.DataFrame) -> pl.DataFrame:
    """3- and 6-month mean utilisation from the ragged cycle history (§5.1). An account
    with fewer than 6 buckets is scored on what it has -- raggedness is data, not a
    defect (§5.1: "the fact that it had 7 must be visible afterwards")."""
    rows = df.select("cycle_balances", "cycle_limits").to_dicts()
    six, three = [], []
    for r in rows:
        balances = r["cycle_balances"] or []
        limits = r["cycle_limits"] or []
        n = min(len(balances), len(limits))
        pairs = [(balances[i], limits[i]) for i in range(n) if limits[i]]
        six.append(_mean_util(pairs[:6]))
        three.append(_mean_util(pairs[:3]))
    return df.with_columns(pl.Series("revolving_utilisation_6m", six), pl.Series("revolving_utilisation_3m", three))


def _mean_util(pairs: list[tuple[float, float]]) -> float:
    if not pairs:
        return 0.0
    ratios = [b / l for b, l in pairs if l]
    return round(statistics.mean(ratios), 4) if ratios else 0.0


@frame_step(reads=["cycle_purchase_values"], writes=["observed_spend_p90"])
def observed_spend(df: pl.DataFrame) -> pl.DataFrame:
    """The trailing 6-month 90th-percentile monthly purchase value (§5.1), which drives
    the spend cap C5 -- "a client whose highest month in six was R400 does not get a
    R30 000 limit"."""
    values = df.get_column("cycle_purchase_values").to_list()
    out = [_p90(v[:6] if v else []) for v in values]
    return df.with_columns(pl.Series("observed_spend_p90", out))


def _p90(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(0.90 * (len(ordered) - 1)))
    return round(ordered[idx], 2)


def cycle_bucket_count(cycle_balances: list[float] = missing_as([])) -> int:
    """§5.1 "Recorded": the count of cycle buckets actually present, per account."""
    return len(cycle_balances) if cycle_balances is not None else 0


cycle_bucket_count_step = step(cycle_bucket_count)
