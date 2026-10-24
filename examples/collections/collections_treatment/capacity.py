"""Capacity: which of today's treatments actually happen, when the pools they draw on are finite.

The pools' daily supply lives in the params document (it is revised through the day), so it changes with no
rebuild. The ranking basis is a param too. Ties break on `account_id`, so the same book and the same capacity give
the same allocation on every run.
"""
from __future__ import annotations

import polars as pl

from decikit import DecisionTableConfig, allow_fallback, frame_step, missing_as, param

from collections_treatment import vocab as v

BASIS_VALUE = "value"
BASIS_PRIORITY = "priority"

# pool_code -> (daily supply, share reserved for balance bands 6-7). Agency and legal are monthly volumes, paced daily.
DEFAULT_SUPPLY = {
    v.POOL_SMS: (900_000, 0.0), v.POOL_EMAIL: (2_400_000, 0.0), v.POOL_IN_APP: (414_000, 0.0),
    v.POOL_VOICE_MESSAGE: (350_000, 0.0), v.POOL_EARLY_AGENTS: (27_000, 0.08), v.POOL_LATE_AGENTS: (11_000, 0.08),
    v.POOL_FIELD: (180, 0.0), v.POOL_AGENCY: (1_300, 0.0), v.POOL_NOTICES: (12_000, 0.0), v.POOL_LEGAL: (80, 0.0),
}
_HIGH_BALANCE_BANDS = (6, 7)


def default_pool_rows() -> list[dict]:
    """The params-document rows of the `pools` table: one per treatment that draws on a pool."""
    return [{"treatment": treatment, "pool_code": pool, "pool_supply": DEFAULT_SUPPLY[pool][0],
             "pool_reserved_share": DEFAULT_SUPPLY[pool][1]} for treatment, pool in v.TREATMENT_POOL.items()]


def build_pool_table() -> DecisionTableConfig:
    """Writes `pool_code`, `pool_supply` and `pool_reserved_share` for the account's `treatment_code`.

    The rows come from the params document (`default_pool_rows()` is a starting point), so a revised supply is a
    new params document, not a release.
    """
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "capacity",
        "columns": {"treatment": "Int64", "pool_code": "Int64", "pool_supply": "Float64", "pool_reserved_share": "Float64"},
        "rows": {"table": "pools"},
        "expression": {"type": "eq", "variable": "treatment_code", "value_column": "treatment"},
        "outputs": ["pool_code", "pool_supply", "pool_reserved_share"], "default": [v.NO_POOL, 0.0, 0.0],
    })


@allow_fallback
def selection_blocker_code(
    matrix_treatment_code: int, overlaid_treatment_code: int, suspension_blocked: bool,
    suppression_adjustments_applied: list[str], suspension_codes: list[int], interval_or_cap_blocked: bool,
    interval_or_cap_detail: str,
) -> int:
    """The non-selection reason (200..240) that stops an account before ranking, or 0 when it can be ranked."""
    if suppression_adjustments_applied:
        return v.NS_OVERLAY_SUPPRESSED
    if overlaid_treatment_code == v.NO_ACTION:
        return v.NS_MATRIX_NO_ACTION
    if suspension_blocked:
        return v.NS_PROMISE_IN_FORCE if set(suspension_codes) == {v.SUSPENSION_PROMISE_IN_FORCE} else v.NS_SUSPENDED
    if "cooling_off" in interval_or_cap_detail:
        return v.NS_COOLING_OFF
    return v.NS_INTERVAL_OR_CAP if interval_or_cap_blocked else 0


def allocation_priority(
    arrears_bucket_code: int, pre_prescription_flag: bool, is_new_bucket3_entrant: bool = missing_as(False),
) -> float:
    """The policy-priority ranking value: deeper arrears first, pre-prescription and new bucket-3 entrants above all."""
    return arrears_bucket_code * 10.0 + 1_000.0 * pre_prescription_flag + 500.0 * is_new_bucket3_entrant


@frame_step(
    reads=["account_id", "selection_blocker_code", "pool_code", "pool_supply", "pool_reserved_share",
           "balance_band_code", "recovery_estimate", "cost_to_collect", "allocation_priority"],
    writes=["allocated", "allocation_rank", "allocation_cutoff_rank", "allocation_ranking_basis",
            "non_selection_reason_code"])
def allocate_capacity(df: pl.DataFrame, ranking_basis: str = param(BASIS_VALUE)) -> pl.DataFrame:
    """Rank the day's accounts within each pool and allocate down to its supply.

    `ranking_basis` is "value" (expected recovery net of cost) or "priority" (`allocation_priority`). Balance
    bands 6-7 get their reserved share first, so cheap volume cannot crowd them out. An account past the cut-off
    gets reason 250 with its rank and the cut-off rank; a treatment with no pool is never rationed.

    Example: `allocate_capacity(day_frame)` adds `allocated`, `allocation_rank`, `non_selection_reason_code`.
    """
    if ranking_basis not in (BASIS_VALUE, BASIS_PRIORITY):
        raise ValueError(f"ranking_basis must be {BASIS_VALUE!r} or {BASIS_PRIORITY!r}, not {ranking_basis!r}")
    rows = df.to_dicts()
    score = [r["allocation_priority"] if ranking_basis == BASIS_PRIORITY else r["recovery_estimate"] - r["cost_to_collect"]
             for r in rows]
    allocated = [r["selection_blocker_code"] == 0 and r["pool_code"] == v.NO_POOL for r in rows]
    rank = [None] * len(rows)
    cutoff = [None] * len(rows)
    reason = [r["selection_blocker_code"] or None for r in rows]
    pools: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        if r["selection_blocker_code"] == 0 and r["pool_code"] != v.NO_POOL:
            pools.setdefault(r["pool_code"], []).append(i)

    def ranked(indexes):
        return sorted(indexes, key=lambda i: (-score[i], rows[i]["account_id"]))

    for indexes in pools.values():
        supply, share = int(rows[indexes[0]]["pool_supply"]), rows[indexes[0]]["pool_reserved_share"]
        reserved = ranked([i for i in indexes if rows[i]["balance_band_code"] in _HIGH_BALANCE_BANDS])[:int(supply * share)]
        taken = set(reserved)
        rest = ranked([i for i in indexes if i not in taken])
        room = max(0, supply - len(reserved))
        for place, i in enumerate(reserved, 1):
            allocated[i], rank[i], cutoff[i] = True, place, len(reserved)
        for place, i in enumerate(rest, 1):
            rank[i], cutoff[i] = len(reserved) + place, len(reserved) + min(room, len(rest))
            allocated[i] = place <= room
            if not allocated[i]:
                reason[i] = v.NS_BELOW_CUTOFF

    return df.with_columns(
        pl.Series("allocated", allocated, dtype=pl.Boolean), pl.Series("allocation_rank", rank, dtype=pl.Int64),
        pl.Series("allocation_cutoff_rank", cutoff, dtype=pl.Int64),
        pl.lit(ranking_basis).alias("allocation_ranking_basis"),
        pl.Series("non_selection_reason_code", reason, dtype=pl.Int64))
