"""Account state assembly (spec 08 §5.1, working slice).

Windowed aggregates over the 180M-row event log (30/60/90-day payment
counts, 90-day contact history, 12/24-month roll patterns) are assumed
already computed upstream, one column each -- exactly how project 00/02
take `payments_30d`-style scalars rather than raw transactions. What this
module owns is the two policy derivations spec 08 §4.3 says must be
*computed, not read*: `arrears_bucket_code` (arrangement-hold and
cure-floor override `days_past_due`) and `balance_band_code`.
"""
from __future__ import annotations

from decider import missing_as, param, step
from decider.steps.tables import DecisionTableConfig

from credit_core.evidence import cell_id as _cell_id

BANDING_VERSION = "collections-bands-2026.09"


# --- arrears_bucket_code (§4.3: "not a pure function of days_past_due") --------------------

def _bucket_from_days(days_past_due: int) -> int:
    if days_past_due <= 0:
        return 0
    if days_past_due <= 14:
        return 1
    if days_past_due <= 29:
        return 2
    if days_past_due <= 59:
        return 3
    if days_past_due <= 89:
        return 4
    if days_past_due <= 119:
        return 5
    if days_past_due <= 179:
        return 6
    if days_past_due <= 364:
        return 7
    return 8


def arrears_bucket_code(
    days_past_due: int,
    under_performing_arrangement: bool = param(False),
    bucket_at_arrangement_entry: int = missing_as(0),
    times_cured_24m: int = param(0, ge=0),
) -> int:
    """§4.3: an account under a performing arrangement is held at the bucket it entered
    the arrangement in (for treatment purposes) while `days_past_due` keeps ageing; an
    account cured twice in six months is floored at bucket 3. Both are policy rules
    Collections Strategy owns, so both are computed here rather than read off the feed."""
    natural = _bucket_from_days(days_past_due)
    if under_performing_arrangement and bucket_at_arrangement_entry > 0:
        bucket = bucket_at_arrangement_entry
    else:
        bucket = natural
    if times_cured_24m >= 2:
        bucket = max(bucket, 3)
    return bucket


arrears_bucket_code_step = step(arrears_bucket_code)


def arrears_to_balance_ratio(arrears_amount: float, outstanding_balance: float) -> float:
    return round(arrears_amount / outstanding_balance, 4) if outstanding_balance else 0.0


arrears_to_balance_ratio_step = step(arrears_to_balance_ratio)


def build_balance_band_table() -> DecisionTableConfig:
    edges = [
        (float("-inf"), 2_500.0), (2_500.0, 10_000.0), (10_000.0, 25_000.0),
        (25_000.0, 50_000.0), (50_000.0, 100_000.0), (100_000.0, 250_000.0),
        (250_000.0, float("inf")),
    ]
    rows = [
        {"lo": lo, "hi": hi, "band": i + 1, "cell_id": _cell_id("balance_band", BANDING_VERSION, i)}
        for i, (lo, hi) in enumerate(edges)
    ]
    return DecisionTableConfig.load({
        "type": "decision_table", "name": "balance_band",
        "columns": {"lo": "Float64", "hi": "Float64", "band": "Int64", "cell_id": "String"},
        "rows": rows,
        "expression": {"type": "between", "variable": "outstanding_balance",
                        "lower_bound_column": "lo", "upper_bound_column": "hi"},
        "outputs": ["band", "cell_id"],
        "default": [7, None],
    }).relabel(writes={"band": "balance_band_code", "cell_id": "balance_band_cell_id"})
