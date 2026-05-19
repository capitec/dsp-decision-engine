"""
Inventory Reorder Scoring Pipeline
Supply-chain analyst DX evaluation of the `decider` framework.
"""
import sys
import polars as pl
from decider.modules.functional import generate_from_functions

# ---------------------------------------------------------------------------
# Sample data — 5 rows, one edge case: stock_qty = 0
# ---------------------------------------------------------------------------
df = pl.DataFrame({
    "product_id":     ["SKU-001", "SKU-002", "SKU-003", "SKU-004", "SKU-005"],
    "stock_qty":      [120.0,      0.0,       45.0,       8.0,      200.0],
    "avg_daily_sales":[10.0,       5.0,       3.0,        4.0,       20.0],
})

print("=" * 60)
print("INPUT DATA")
print("=" * 60)
print(df)
print()

# ---------------------------------------------------------------------------
# STEP 1 — BROKEN pipeline (misspelled column: stok_qty instead of stock_qty)
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Deliberately broken pipeline (stok_qty typo)")
print("=" * 60)

def days_of_stock_broken(stok_qty: pl.Expr, avg_daily_sales: pl.Expr) -> pl.Expr:
    """Uses 'stok_qty' (typo) instead of 'stock_qty'."""
    return stok_qty / avg_daily_sales

def reorder_urgency(days_of_stock: pl.Expr) -> pl.Expr:
    return 1 / (days_of_stock + 1)

def should_reorder(days_of_stock: pl.Expr) -> pl.Expr:
    return (days_of_stock < 7).cast(pl.Int8)

BrokenPipeline = generate_from_functions(
    "broken_inventory_scorer",
    days_of_stock_broken,
    reorder_urgency,
    should_reorder,
)
broken = BrokenPipeline(name="broken")

try:
    result = broken.execute({"input": df}, lazy=False)
    print("ERROR: Expected a failure but pipeline succeeded — bug in the DX test!")
except Exception as e:
    print(f"Exception type : {type(e).__name__}")
    print(f"Exception message (verbatim):\n")
    print(str(e))

print()

# ---------------------------------------------------------------------------
# STEP 2 — FIXED pipeline (correct column names)
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 2: Fixed pipeline (correct column names)")
print("=" * 60)

def days_of_stock(stock_qty: pl.Expr, avg_daily_sales: pl.Expr) -> pl.Expr:
    return stock_qty / avg_daily_sales

# reorder_urgency and should_reorder already defined above with correct deps
InventoryScorer = generate_from_functions(
    "inventory_scorer",
    days_of_stock,
    reorder_urgency,
    should_reorder,
)
scorer = InventoryScorer(name="inv_scorer")

result = scorer.execute({"input": df}, lazy=False)
print("Pipeline succeeded. Output:")
print(result)
print()
print("Edge-case row (stock_qty=0):")
print(result.filter(pl.col("product_id") == "SKU-002"))
print()

# ---------------------------------------------------------------------------
# STEP 3 — Compose into a pipeline using | operator (smoke-test)
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 3: Pipeline composition (scorer | scorer) smoke-test")
print("=" * 60)

# A second, downstream module that adds a priority flag
def priority_flag(reorder_urgency: pl.Expr) -> pl.Expr:
    return (reorder_urgency > 0.5).cast(pl.Int8)

PriorityFlagger = generate_from_functions("priority_flagger", priority_flag)
flagger = PriorityFlagger(name="flagger")

composed = scorer | flagger
result2 = composed.execute({"input": df}, lazy=False)
print("Composed pipeline output:")
print(result2)
print()

print("All done.")
