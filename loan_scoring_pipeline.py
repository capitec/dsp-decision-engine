"""
Loan creditworthiness scoring pipeline — compile-then-reuse demonstration.

Columns expected: applicant_id, debt, income, credit_used, credit_limit
Derived features:
  dti_ratio            = debt / income
  utilization_rate     = credit_used / credit_limit
  credit_score_estimate = 800 - dti_ratio*200 - utilization_rate*100
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath("."))

import polars as pl
from decider.modules.functional import generate_from_functions

# ---------------------------------------------------------------------------
# 1.  Define feature functions
#     Convention: function name → output column, param names → input columns
# ---------------------------------------------------------------------------

def dti_ratio(debt: pl.Expr, income: pl.Expr) -> pl.Expr:
    """Debt-to-income ratio."""
    return debt / income


def utilization_rate(credit_used: pl.Expr, credit_limit: pl.Expr) -> pl.Expr:
    """Credit utilisation rate."""
    return credit_used / credit_limit


def credit_score_estimate(dti_ratio: pl.Expr, utilization_rate: pl.Expr) -> pl.Expr:
    """Composite credit score estimate (higher = better)."""
    return pl.lit(800) - dti_ratio * 200 - utilization_rate * 100


# ---------------------------------------------------------------------------
# 2.  Generate module and compile ONCE
# ---------------------------------------------------------------------------

print("=" * 60)
print("LOAN CREDITWORTHINESS PIPELINE — compile-then-reuse demo")
print("=" * 60)

CreditScorer = generate_from_functions(
    "credit_scorer",
    dti_ratio,
    utilization_rate,
    credit_score_estimate,
)

scorer = CreditScorer(name="credit_scorer")
print(f"\nModule type   : {type(scorer).__name__}")
print(f"Module name   : {scorer.name}")
print(f"Input columns : {scorer.input_names}")
print(f"Output columns: {scorer.output_names}")

# --- compile once ---
t0 = time.perf_counter()
compiled = scorer.compile()
compile_ms = (time.perf_counter() - t0) * 1000
print(f"\nCompile time  : {compile_ms:.4f} ms")
print(f"CompiledDag type: {type(compiled).__name__}")
print(f"Expression groups: {len(compiled.expression_groups)}")

# ---------------------------------------------------------------------------
# 3.  Three applicant batches
# ---------------------------------------------------------------------------

batch_1 = pl.DataFrame({
    "applicant_id": ["A001", "A002", "A003"],
    "debt":         [25_000.0, 5_000.0,  80_000.0],
    "income":       [50_000.0, 60_000.0, 90_000.0],
    "credit_used":  [4_000.0,  500.0,   18_000.0],
    "credit_limit": [10_000.0, 5_000.0, 20_000.0],
})

batch_2 = pl.DataFrame({
    "applicant_id": ["B001", "B002", "B003"],
    "debt":         [10_000.0, 45_000.0, 3_000.0],
    "income":       [40_000.0, 75_000.0, 35_000.0],
    "credit_used":  [2_500.0,  12_000.0, 300.0],
    "credit_limit": [8_000.0,  15_000.0, 2_000.0],
})

batch_3 = pl.DataFrame({
    "applicant_id": ["C001", "C002", "C003"],
    "debt":         [0.0,     30_000.0, 120_000.0],
    "income":       [55_000.0, 65_000.0, 100_000.0],
    "credit_used":  [100.0,    6_000.0,  25_000.0],
    "credit_limit": [5_000.0,  10_000.0, 25_000.0],
})

batches = [("Batch 1 – mixed risk",      batch_1),
           ("Batch 2 – varied profiles",  batch_2),
           ("Batch 3 – edge cases",       batch_3)]

# ---------------------------------------------------------------------------
# 4.  Reuse compiled plan across all batches; capture execute timing
# ---------------------------------------------------------------------------

execute_times_ms = []

print("\n" + "-" * 60)
print("RESULTS (using compiled plan each time)")
print("-" * 60)

for label, df in batches:
    t0 = time.perf_counter()
    # compiled.execute() returns Dict[str, pl.LazyFrame]
    result_dict = compiled.execute({"input": df})
    # Must .collect() to materialise the LazyFrame
    result_df = result_dict["input"].collect()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    execute_times_ms.append(elapsed_ms)

    print(f"\n{label}  (execute: {elapsed_ms:.4f} ms)")
    print(result_df.select([
        "applicant_id",
        pl.col("dti_ratio").round(3),
        pl.col("utilization_rate").round(3),
        pl.col("credit_score_estimate").round(1),
    ]))

avg_execute_ms = sum(execute_times_ms) / len(execute_times_ms)

# ---------------------------------------------------------------------------
# 5.  Timing summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("TIMING SUMMARY")
print("=" * 60)
print(f"Compile time         : {compile_ms:.4f} ms  (done once)")
for i, (label, _) in enumerate(batches):
    print(f"Execute time batch {i+1} : {execute_times_ms[i]:.4f} ms")
print(f"Average execute time : {avg_execute_ms:.4f} ms")
print(f"Ratio compile/avg-exec: {compile_ms/avg_execute_ms:.1f}x")

# ---------------------------------------------------------------------------
# 6.  Convenience-path comparison (module.execute — compiles internally each time)
# ---------------------------------------------------------------------------

print("\n" + "-" * 60)
print("COMPARISON: module.execute() [compile + execute each call]")
print("-" * 60)

convenience_times_ms = []
for label, df in batches:
    t0 = time.perf_counter()
    # module.execute() returns a LazyFrame by default (lazy=True)
    result_lazy = scorer.execute({"input": df})
    result_df2 = result_lazy.collect()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    convenience_times_ms.append(elapsed_ms)
    print(f"  {label}: {elapsed_ms:.4f} ms  (returns LazyFrame)")

avg_conv_ms = sum(convenience_times_ms) / len(convenience_times_ms)
print(f"\nAverage (convenience): {avg_conv_ms:.4f} ms")
print(f"Average (compiled)   : {avg_execute_ms:.4f} ms")

print("\nDone.")
