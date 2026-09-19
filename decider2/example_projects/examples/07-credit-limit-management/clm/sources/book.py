"""Frame tier. Everything here is genuinely set-shaped: joins over ragged
histories, per-client roll-ups, the prior cycle's own output joined back in.

Nothing in this file is a decision. Every row it emits is one account, and from
`book` onward the pipeline is per-record until `allocation/`.
"""

import polars as pl
from decider2.frame import Aggregate, Filter, Join, Sort, declares

# --- ragged history -> fixed-width columns ---------------------------------
# 78.4 M cycle buckets, 6-24 per account. The record tier cannot see a ragged
# collection, so the reduction to fixed-width columns happens here and the
# bucket COUNT survives as a column, because "scored on 7 buckets" is data.

CycleHistory = Aggregate(
    source="cycle_buckets",
    by="account_id",
    metrics={
        "cycles_present": pl.len(),
        "mean_utilisation_3m": pl.col("utilisation").tail(3).mean(),
        "mean_utilisation_6m": pl.col("utilisation").tail(6).mean(),
        "mean_utilisation_12m": pl.col("utilisation").tail(12).mean(),
        "over_limit_cycles_12m": pl.col("over_limit").tail(12).sum(),
        "longest_over_limit_run": pl.col("over_limit").rle().struct["len"].max(),
        "min_payment_only_6m": pl.col("paid_minimum_only").tail(6).sum(),
        "observed_spend_p90_c": pl.col("purchase_value_c").tail(6).quantile(0.90, "nearest"),
    },
    declares={"cycles_present": pl.Int16, "observed_spend_p90_c": pl.Int64},
)

ArrearsHistory = Aggregate(source="delinquency", by="account_id", metrics={...},
                           declares={...})

SpendCategories = Aggregate(source="spend_category_aggregates", by="account_id",
                            metrics={...}, declares={...})

# --- per-client roll-up ----------------------------------------------------
# 4.10 M accounts, 3.62 M clients, 480 000 holding both products. The roll-up is
# what C3/C4 and the s5.7 precedence rule read.

ClientExposure = Aggregate(
    source="holdings", by="client_id",
    metrics={"total_unsecured_limit_c": pl.col("limit_c").filter(pl.col("is_unsecured")).sum(),
             "open_account_count": pl.len()},
    declares={"total_unsecured_limit_c": pl.Int64, "open_account_count": pl.Int8},
)

# --- the prior cycle, as an ordinary effective-dated input -----------------
# Hysteresis (s5.8 constraint 6) and the cooling-off windows (s6.5) both need
# "what did we decide last time". That makes this pipeline's own output an
# input to its next run. It is joined like any other source; the consequence
# for replay is spelled out in FRAMEWORK-DEMANDS #17.

PriorCycle = Join(
    source="cycle_records",
    on="account_id",
    how="left",
    select=["prior_allocation_outcome_code", "prior_ranking_value",
            "prior_funded", "prior_decision_date"],
    as_of="shared.decision_date",          # the record in force, not the latest
)

Book = (
    CycleHistory
    | ArrearsHistory
    | SpendCategories
    | Join(source="account_master", on="account_id", how="inner")
    | Join(source="markers", on=["account_id", "client_id"], how="left")
    | Join(source="bureau_refresh", on="client_id", how="left")   # 2.4% miss: null, not zero
    | Join(source="consent_state", on="client_id", how="left")
    | ClientExposure
    | PriorCycle
)


@declares(outputs={"bucket_raggedness_report": pl.DataFrame})
def raggedness_report(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Cycle-level count of accounts by bucket count, for the s5.1 record."""
    pass  # group by cycles_present, count, compare against prior cycle
