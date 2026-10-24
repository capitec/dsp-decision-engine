"""Account state that is computed, not read: the arrears bucket and the balance band."""
from __future__ import annotations

from decikit import DecisionTableConfig, missing_as

from credit_core.evidence import cell_id

BANDING_VERSION = "collections-bands-2026.09"


def arrears_bucket_code(
    days_past_due: int, under_performing_arrangement: bool = missing_as(False),
    bucket_at_arrangement_entry: int = missing_as(0), times_cured_12m: float = missing_as(0.0),
) -> int:
    """1..8 from `days_past_due` (0 when not in arrears).

    An account under a performing arrangement stays in the bucket it entered the arrangement in while its days keep
    ageing; an account cured twice in the year is floored at bucket 3.
    """
    # ponytail: bucket edges are constants; a shared param_table would let Collections Strategy move them
    bucket = ((days_past_due > 0) + (days_past_due > 14) + (days_past_due > 29) + (days_past_due > 59)
              + (days_past_due > 89) + (days_past_due > 119) + (days_past_due > 179) + (days_past_due > 364))
    if under_performing_arrangement and bucket_at_arrangement_entry > 0:
        bucket = bucket_at_arrangement_entry
    if times_cured_12m >= 2:
        bucket = max(bucket, 3)
    return bucket


def arrears_to_balance_ratio(arrears_amount: float, outstanding_balance: float) -> float:
    return round(arrears_amount / outstanding_balance, 4) if outstanding_balance else 0.0


def build_balance_band_table() -> DecisionTableConfig:
    """`balance_band_code` 1..7 from `outstanding_balance`, with the cell it read."""
    inf = float("inf")
    edges = [(-inf, 2_500.0), (2_500.0, 10_000.0), (10_000.0, 25_000.0), (25_000.0, 50_000.0),
             (50_000.0, 100_000.0), (100_000.0, 250_000.0), (250_000.0, inf)]
    rows = [{"lo": lo, "hi": hi, "band": i + 1, "cell_id": cell_id("balance_band", BANDING_VERSION, i + 1)}
            for i, (lo, hi) in enumerate(edges)]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "balance_band",
        "columns": {"lo": "Float64", "hi": "Float64", "band": "Int64", "cell_id": "String"}, "rows": rows,
        "expression": {"type": "between", "variable": "outstanding_balance",
                       "lower_bound_column": "lo", "upper_bound_column": "hi"},
        "outputs": ["band", "cell_id"], "default": [7, None],
    }).relabel(writes={"band": "balance_band_code", "cell_id": "balance_band_cell_id"})

