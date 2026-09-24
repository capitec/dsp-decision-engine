"""Operational capacity and allocation (spec 08 §5.9) -- answers §13 Q5.

**Q5 (how does a population-level constraint compose with per-account logic?).**
This slice's answer: a `frame_step`. Every other step in this pipeline is a
plain `Step` (one row in, one row out); ranking and rationing are not --
"whether an account is touched depends on every other account that ran
today" is exactly what `frame_step`'s `DataFrame -> DataFrame` signature is
for. One `frame_step`, run once over the whole day's population, is the
same shape project 07's portfolio budget wants (§13 Q5's own question),
so this is offered as the answer for that project too, not reinvented.

A single real-time record (the sample request, or the 400ms arrangement
path) is a population of one: it ranks 1st of 1 in whatever pool its
treatment lands in, and is allocated if that pool's daily supply is at
least 1 -- degenerate, not special-cased, and never wrong (SCOPE.md skips
the real-time path itself; this is only about not crashing on it).

# ponytail: pure-Python per-pool ranking (`.to_dicts()`, sort, allocate),
not a vectorised polars window-function pipeline. Correct and fully
deterministic (stable sort key `account_id`) at the few-thousand-row
volumes this slice's tests run; not tuned for 2.3M rows in a 90-minute
batch window (§8). Upgrade path: replace the per-pool Python loop with
`pl.col(...).rank()` over `.over("pool")` if this shows up in a profile --
project 00's `core.obligations` sets the same precedent (its own
ponytail note) for "pure Python now, vectorise if it's ever the
bottleneck".
"""
from __future__ import annotations

import polars as pl

from decider import frame_step

from collections_treatment import vocab

# Pool -> (daily supply, reserved share for balance bands 6-7). Illustrative, working-depth
# subset of §5.9's 12 pools -- the ones this slice's treatment codes actually route to.
# A real deployment reads this from the daily-revised capacity feed (§4.2), not a constant;
# left as a constant here since `frame_step` has no `param()`/params-document mechanism the
# way `DecisionTableConfig` does (see NOTES.md "Framework friction").
CAPACITY_SUPPLY = {
    "sms": (900_000, 0.0),
    "email": (2_400_000, 0.0),
    "in_app": (414_000, 0.0),
    "voice_message": (350_000, 0.0),
    "early_agents": (27_000, 0.08),
    "late_agents": (11_000, 0.08),
    "field": (180, 0.0),
    "agency": (1_300, 0.0),   # 40 000/month, paced daily
    "notices": (12_000, 0.0),
    "legal": (80, 0.0),      # 2 500/month, paced daily
}

BASIS_VALUE = "value"
BASIS_PRIORITY = "priority"

_HIGH_BALANCE_BANDS = (6, 7)


def _basis_score(row: dict, ranking_basis: str) -> float:
    if ranking_basis == BASIS_PRIORITY:
        score = float(row.get("arrears_bucket_code") or 0) * 10.0
        if row.get("pre_prescription_flag"):
            score += 1_000.0
        if row.get("is_new_bucket3_entrant"):
            score += 500.0
        return score
    return float(row.get("recovery_estimate") or 0.0) - float(row.get("cost_to_collect") or 0.0)


def _non_selection_before_ranking(row: dict) -> int | None:
    """Reasons that pre-empt ranking entirely (§5.9 codes 200/205/210/220/230/240):
    evaluated in the priority order the spec's own table lists them."""
    if row.get("suppression_adjustments_applied"):
        return vocab.NS_OVERLAY_SUPPRESSED
    if row.get("treatment_code") == vocab.NO_ACTION:
        return vocab.NS_MATRIX_NO_ACTION
    if row.get("suspended_blocks_all"):
        return vocab.NS_SUSPENDED
    if "cooling_off" in (row.get("interval_or_cap_detail") or ""):
        return vocab.NS_COOLING_OFF
    if row.get("interval_or_cap_blocked"):
        return vocab.NS_INTERVAL_OR_CAP
    if 119 in (row.get("suspension_codes") or []):
        return vocab.NS_PROMISE_IN_FORCE
    return None


def allocate(df: pl.DataFrame, ranking_basis: str = BASIS_VALUE) -> pl.DataFrame:
    rows = df.to_dicts()
    for row in rows:
        row["pool"] = vocab.TREATMENT_POOL.get(row.get("treatment_code"))
        row["non_selection_reason_code"] = _non_selection_before_ranking(row)
        row["allocated"] = False
        row["allocation_pool"] = row["pool"] if row["non_selection_reason_code"] is None else None
        row["allocation_rank"] = None
        row["allocation_cutoff_rank"] = None
        row["allocation_ranking_basis"] = ranking_basis

    by_pool: dict[str, list[dict]] = {}
    for row in rows:
        if row["non_selection_reason_code"] is None and row["pool"] is not None:
            by_pool.setdefault(row["pool"], []).append(row)

    for pool, candidates in by_pool.items():
        supply, reserved_share = CAPACITY_SUPPLY.get(pool, (len(candidates), 0.0))
        reserved_capacity = int(supply * reserved_share)

        high = [r for r in candidates if r.get("balance_band_code") in _HIGH_BALANCE_BANDS]
        high.sort(key=lambda r: (-_basis_score(r, ranking_basis), r["account_id"]))
        reserved_allocated = high[:reserved_capacity]
        reserved_ids = {r["account_id"] for r in reserved_allocated}

        for i, r in enumerate(reserved_allocated, start=1):
            r["allocated"] = True
            r["allocation_rank"] = i
            r["allocation_cutoff_rank"] = min(reserved_capacity, len(high))

        remaining_supply = max(0, supply - len(reserved_allocated))
        remainder = [r for r in candidates if r["account_id"] not in reserved_ids]
        remainder.sort(key=lambda r: (-_basis_score(r, ranking_basis), r["account_id"]))
        for i, r in enumerate(remainder, start=1):
            r["allocation_rank"] = len(reserved_allocated) + i
            r["allocation_cutoff_rank"] = len(reserved_allocated) + min(remaining_supply, len(remainder))
            if i <= remaining_supply:
                r["allocated"] = True
            else:
                r["non_selection_reason_code"] = vocab.NS_BELOW_CUTOFF

    return pl.DataFrame(rows, infer_schema_length=None)


def allocate_capacity(ranking_basis: str = BASIS_VALUE):
    def _run(df: pl.DataFrame) -> pl.DataFrame:
        return allocate(df, ranking_basis)

    return frame_step(
        _run, name="allocate_capacity",
        reads=["account_id", "treatment_code", "arrears_bucket_code", "balance_band_code",
               "recovery_estimate", "cost_to_collect", "suspended_blocks_all",
               "suppression_adjustments_applied", "interval_or_cap_blocked", "interval_or_cap_detail",
               "suspension_codes", "pre_prescription_flag", "is_new_bucket3_entrant"],
        writes=["pool", "non_selection_reason_code", "allocated", "allocation_pool", "allocation_rank",
                "allocation_cutoff_rank", "allocation_ranking_basis"],
    )
