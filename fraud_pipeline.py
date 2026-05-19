"""
Transaction fraud-scoring pipeline built with the decider framework.
Attempt log is embedded in comments so the DX report can be honest.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import polars as pl
from decider.modules.functional import generate_from_functions

# ---------------------------------------------------------------------------
# Step 1 – define the three computation functions
# ---------------------------------------------------------------------------
# Convention: function name → output column name
#             parameter name → reads that column OR wires to a sibling function

def amount_zscore(amount: pl.Expr) -> pl.Expr:
    """(amount - mean) / std  — produces column 'amount_zscore'."""
    return (amount - amount.mean()) / amount.std()


def velocity_flag(amount_zscore: pl.Expr) -> pl.Expr:
    """1 if amount_zscore > 2.0, else 0.  Wired to sibling amount_zscore."""
    return (amount_zscore > 2.0).cast(pl.Int32)


def risk_score(velocity_flag: pl.Expr, amount_zscore: pl.Expr) -> pl.Expr:
    """velocity_flag * 50 + amount_zscore * 10, capped at 100."""
    raw = velocity_flag * 50 + amount_zscore * 10
    return raw.clip(upper_bound=100)


# ---------------------------------------------------------------------------
# Step 2 – generate the module class and instantiate it
# ---------------------------------------------------------------------------
FraudScorer = generate_from_functions(
    "fraud_scorer",
    amount_zscore,
    velocity_flag,
    risk_score,
)

scorer = FraudScorer(name="fraud_scorer")

# ---------------------------------------------------------------------------
# Step 3 – build test data (5 rows, one obvious outlier at row index 4)
# ---------------------------------------------------------------------------
df = pl.DataFrame({
    "txn_id": ["T001", "T002", "T003", "T004", "T005"],
    "amount": [50.0, 55.0, 48.0, 52.0, 500.0],   # 500 is the outlier
})

print("=== Input data ===")
print(df)
print()

# ---------------------------------------------------------------------------
# Step 4 – execute (lazy=False returns an eager DataFrame directly)
# ---------------------------------------------------------------------------
result = scorer.execute({"input": df}, lazy=False)

print("=== Pipeline output ===")
print(result)
print()

# ---------------------------------------------------------------------------
# Step 5 – compose two modules with | and verify pipeline produces same result
# ---------------------------------------------------------------------------

def echo_txn_id(txn_id: pl.Expr) -> pl.Expr:
    """Pass-through so txn_id is visibly preserved in stage 2."""
    return txn_id


Passthrough = generate_from_functions("passthrough", echo_txn_id)
passthrough = Passthrough(name="passthrough")

pipeline = scorer | passthrough

print("=== Same result via | composition ===")
pipe_result = pipeline.execute({"input": df}, lazy=False)
print(pipe_result)
print()

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
assert "amount_zscore" in result.columns, "amount_zscore column missing"
assert "velocity_flag" in result.columns, "velocity_flag column missing"
assert "risk_score" in result.columns, "risk_score column missing"

# The outlier (row 4, amount=500) should have velocity_flag=1
outlier_row = result.filter(pl.col("txn_id") == "T005")
assert outlier_row["velocity_flag"][0] == 1, "Outlier should be flagged"
assert outlier_row["risk_score"][0] == 100, "Outlier risk_score should be capped at 100"

# Normal rows should have velocity_flag=0
normal_row = result.filter(pl.col("txn_id") == "T001")
assert normal_row["velocity_flag"][0] == 0, "Normal txn should not be flagged"

print("=== All assertions passed ===")
print()
print("Detailed view of fraud signals:")
print(
    result.select([
        "txn_id", "amount",
        pl.col("amount_zscore").round(3),
        "velocity_flag",
        pl.col("risk_score").round(2),
    ])
)
