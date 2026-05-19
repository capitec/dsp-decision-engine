"""
Churn risk scoring pipeline - DX evaluation script.

Scenario: a telco data scientist scoring customer churn risk.
The pipeline ingests two raw tables (customers + monthly usage) and
produces a per-customer churn probability together with a risk tier.

Pipeline shape
--------------
customers df ──┐
               ├─► JoinModule (enriched) ──► UsageFeatures ──► ChurnSignals ──► ChurnScore
usage df   ────┘

Patterns exercised
------------------
1.  generate_from_functions()          - all three feature modules
2.  JoinModule                         - joining customers + monthly_usage
3.  | operator (Pipeline composition)  - chaining expression modules
4.  ForkPipeline (& operator)          - parallel branches post-features
5.  module.execute() lazy=False        - single-module eager path
6.  pipeline.execute() output_frames   - named intermediate frame retrieval
7.  module.compile() + compiled.execute() - compile-once, execute-many
8.  module.input_names / output_names  - introspection
9.  debug=True                         - trace output
10. config parameter injection         - configurable thresholds via Pydantic
"""

import traceback
import polars as pl
from pydantic import BaseModel
from decider.modules.functional import generate_from_functions
from decider.modules.primitives.join import JoinModule
from decider.pipeline import Pipeline


# =============================================================================
# 1. Test data
# =============================================================================

customers = pl.DataFrame({
    "customer_id":    [1,    2,    3,    4,    5   ],
    "plan":           ["basic", "premium", "basic", "premium", "basic"],
    "months_active":  [2,    12,   1,    8,    6   ],
    "contract_type":  ["monthly", "annual", "monthly", "annual", "monthly"],
})

monthly_usage = pl.DataFrame({
    "customer_id":  [1,    2,    3,    4,    5   ],
    "calls":        [5,    120,  3,    80,   45  ],
    "data_used_gb": [9.5,  3.0,  8.0,  10.0, 5.5 ],
    "data_limit_gb":[10.0, 10.0, 10.0, 10.0, 10.0],
    "support_calls":[0,    3,    1,    0,    2   ],
})

print("=" * 65)
print("CHURN RISK SCORING PIPELINE — DX evaluation")
print("=" * 65)
print("\n--- Input: customers ---")
print(customers)
print("\n--- Input: monthly_usage ---")
print(monthly_usage)
print()


# =============================================================================
# 2. Join customers + monthly_usage (JoinModule)
# =============================================================================

print("=" * 65)
print("STEP 1 — JoinModule: enrich customers with usage data")
print("=" * 65)

join_module = JoinModule(
    name="enrich_join",
    left="customers",
    right="monthly_usage",
    on="customer_id",
    how="left",
    output_frame="enriched",
)

print(f"\njoin_module.input_names : {join_module.input_names}")
print(f"join_module.output_names: {join_module.output_names}")

try:
    join_result = join_module.execute(
        {"customers": customers, "monthly_usage": monthly_usage},
        output_frames=["enriched"],
    )
    enriched_df = join_result["enriched"].collect()
    print(f"\nJoined frame shape: {enriched_df.shape}")
    print(enriched_df)
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
print()


# =============================================================================
# 3. Config-injected feature module (Pattern 10)
# =============================================================================

class ChurnConfig(BaseModel):
    low_calls_threshold: float = 10.0
    near_limit_threshold: float = 0.9


def avg_monthly_calls(calls: pl.Expr, months_active: pl.Expr) -> pl.Expr:
    """Calls per month — rate normalises for tenure."""
    return calls / months_active


def data_usage_pct(data_used_gb: pl.Expr, data_limit_gb: pl.Expr) -> pl.Expr:
    """Fraction of data cap consumed."""
    return data_used_gb / data_limit_gb


def low_usage_flag(avg_monthly_calls: pl.Expr, config: ChurnConfig) -> pl.Expr:
    """1 if call rate is below the configured threshold."""
    return (avg_monthly_calls < config.low_calls_threshold).cast(pl.Int8)


def near_limit_flag(data_usage_pct: pl.Expr, config: ChurnConfig) -> pl.Expr:
    """1 if customer is approaching their data cap."""
    return (data_usage_pct > config.near_limit_threshold).cast(pl.Int8)


def high_support_flag(support_calls: pl.Expr) -> pl.Expr:
    """1 if customer contacted support more than once this period."""
    return (support_calls > 1).cast(pl.Int8)


UsageFeatures = generate_from_functions(
    "usage_features",
    avg_monthly_calls,
    data_usage_pct,
)

ChurnSignals = generate_from_functions(
    "churn_signals",
    low_usage_flag,
    near_limit_flag,
    high_support_flag,
)

usage_features = UsageFeatures(name="usage_features")

# Instantiate with default config (low_calls_threshold=10, near_limit_threshold=0.9)
churn_signals = ChurnSignals(
    name="churn_signals",
    low_calls_threshold=10.0,
    near_limit_threshold=0.9,
)

print("=" * 65)
print("STEP 2 — Module introspection (input_names / output_names)")
print("=" * 65)
print(f"\nusage_features.input_names : {usage_features.input_names}")
print(f"usage_features.output_names: {usage_features.output_names}")
print(f"\nchurn_signals.input_names  : {churn_signals.input_names}")
print(f"churn_signals.output_names : {churn_signals.output_names}")
print()


# =============================================================================
# 4. Churn score (weighted combination of signals)
# =============================================================================

def churn_probability(
    low_usage_flag: pl.Expr,
    near_limit_flag: pl.Expr,
    high_support_flag: pl.Expr,
) -> pl.Expr:
    """Weighted churn probability: low-call disengagement + data-limit frustration + support noise."""
    return (
        low_usage_flag * 0.3
        + near_limit_flag * 0.5
        + high_support_flag * 0.2
    )


def churn_tier(churn_probability: pl.Expr) -> pl.Expr:
    """Segment: 'high' ≥ 0.5, 'medium' ≥ 0.2, else 'low'."""
    return (
        pl.when(churn_probability >= 0.5).then(pl.lit("high"))
        .when(churn_probability >= 0.2).then(pl.lit("medium"))
        .otherwise(pl.lit("low"))
    )


ChurnScore = generate_from_functions("churn_score", churn_probability, churn_tier)
churn_score = ChurnScore(name="churn_score")

print("=" * 65)
print("STEP 3 — Single-module execute on enriched frame (lazy=False)")
print("=" * 65)

try:
    features_df = usage_features.execute({"input": enriched_df}, lazy=False)
    print(f"\nusage_features output (eager DataFrame):")
    print(features_df.select(["customer_id", "avg_monthly_calls", "data_usage_pct"]))
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
print()


# =============================================================================
# 5. Full pipeline via | operator
# =============================================================================

print("=" * 65)
print("STEP 4 — Pipeline composition via | operator")
print("=" * 65)

try:
    pipeline = usage_features | churn_signals | churn_score
    print(f"\nPipeline type : {type(pipeline).__name__}")
    print(f"Pipeline steps: {[s.name for s in pipeline.steps]}")

    result_lf = pipeline.execute({"input": enriched_df})
    result_df = result_lf.collect()
    print(f"\nFull pipeline output:")
    print(
        result_df.select([
            "customer_id", "months_active",
            pl.col("avg_monthly_calls").round(2),
            pl.col("data_usage_pct").round(2),
            "low_usage_flag", "near_limit_flag", "high_support_flag",
            pl.col("churn_probability").round(3),
            "churn_tier",
        ])
    )
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
print()


# =============================================================================
# 6. output_frames — retrieve named intermediates
# =============================================================================

print("=" * 65)
print("STEP 5 — output_frames: retrieve 'usage_features' and 'churn_score'")
print("=" * 65)

try:
    frames = pipeline.execute(
        {"input": enriched_df},
        output_frames=["usage_features", "churn_score"],
    )
    print(f"\nReturned type: {type(frames).__name__}, keys: {list(frames.keys())}")
    for name, lf in frames.items():
        print(f"\n  Frame '{name}':")
        df = lf.collect()
        print(df)
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
print()


# =============================================================================
# 7. compile-once, execute-many
# =============================================================================

print("=" * 65)
print("STEP 6 — compile() once, execute CompiledDag across two batches")
print("=" * 65)

try:
    import time

    # Build a simple standalone scorer for the compile demo
    batch_pipeline = usage_features | churn_signals | churn_score
    # Compile the last module (churn_score) independently to show the pattern
    compiled = churn_score.compile()
    print(f"\nCompiledDag type           : {type(compiled).__name__}")
    print(f"Expression groups          : {len(compiled.expression_groups)}")

    # We need a frame that already has the signals
    # Run the first two stages to get that frame
    signals_df = (usage_features | churn_signals).execute(
        {"input": enriched_df}, lazy=False
    )

    batch_a = signals_df.filter(pl.col("customer_id").is_in([1, 2, 3]))
    batch_b = signals_df.filter(pl.col("customer_id").is_in([4, 5]))

    for label, batch in [("Batch A (customers 1-3)", batch_a), ("Batch B (customers 4-5)", batch_b)]:
        t0 = time.perf_counter()
        result = compiled.execute({"input": batch})
        elapsed_ms = (time.perf_counter() - t0) * 1000
        df = result["input"].collect()
        print(f"\n  {label} ({elapsed_ms:.3f} ms):")
        print(df.select(["customer_id", pl.col("churn_probability").round(3), "churn_tier"]))

except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
print()


# =============================================================================
# 8. ForkPipeline (parallel branches via & operator)
# =============================================================================

print("=" * 65)
print("STEP 7 — ForkPipeline: parallel churn + value branch")
print("=" * 65)


def lifetime_value_estimate(months_active: pl.Expr) -> pl.Expr:
    """Simple LTV proxy: months_active * 50 (arbitrary unit)."""
    return months_active * pl.lit(50.0)


def ltv_tier(lifetime_value_estimate: pl.Expr) -> pl.Expr:
    """'high' ≥ 400, 'medium' ≥ 200, else 'low'."""
    return (
        pl.when(lifetime_value_estimate >= 400).then(pl.lit("high"))
        .when(lifetime_value_estimate >= 200).then(pl.lit("medium"))
        .otherwise(pl.lit("low"))
    )


ValueBranch = generate_from_functions(
    "value_branch", lifetime_value_estimate, ltv_tier
)
value_branch = ValueBranch(name="value_branch")

try:
    # Shared prefix: usage features
    # Then fork into churn scoring vs. LTV scoring
    churn_branch_pipeline = Pipeline([churn_signals, churn_score])
    value_branch_pipeline = Pipeline([value_branch])

    fork = churn_branch_pipeline & value_branch_pipeline
    print(f"\nFork type    : {type(fork).__name__}")
    print(f"Branch count : {len(fork.branches)}")

    # Feed the fork with the usage_features output
    fork_input = usage_features.execute({"input": enriched_df}, lazy=False)

    fork_result = fork.execute({"input": fork_input})
    print(f"\nFork result keys: {list(fork_result.keys())}")

    for branch_name, lf in fork_result.items():
        df = lf.collect()
        print(f"\n  Branch '{branch_name}' columns: {df.columns}")
        print(df.select(["customer_id"] + [c for c in df.columns if c not in enriched_df.columns]))

except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
print()


# =============================================================================
# 9. debug=True trace
# =============================================================================

print("=" * 65)
print("STEP 8 — debug=True execution trace (first 3 customers)")
print("=" * 65)

try:
    debug_input = enriched_df.filter(pl.col("customer_id").is_in([1, 2, 3]))
    _ = (usage_features | churn_signals | churn_score).execute(
        {"input": debug_input}, debug=True
    )
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
print()


# =============================================================================
# 10. Config variant — stricter thresholds
# =============================================================================

print("=" * 65)
print("STEP 9 — Config injection: stricter thresholds")
print("=" * 65)

try:
    strict_signals = ChurnSignals(
        name="strict_signals",
        low_calls_threshold=20.0,   # flag anyone below 20 calls/month
        near_limit_threshold=0.7,   # flag at 70 % data cap
    )

    strict_pipeline = usage_features | strict_signals | churn_score
    strict_result = strict_pipeline.execute({"input": enriched_df}, lazy=False)

    print("\nStrict thresholds (low_calls<20, near_limit>0.7):")
    print(
        strict_result.select([
            "customer_id",
            pl.col("avg_monthly_calls").round(2),
            pl.col("data_usage_pct").round(2),
            "low_usage_flag", "near_limit_flag",
            pl.col("churn_probability").round(3),
            "churn_tier",
        ])
    )
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
print()

print("=" * 65)
print("ALL STEPS COMPLETE")
print("=" * 65)
