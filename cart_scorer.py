"""
E-commerce cart scoring application using the decider framework.
Written as a fresh-eyes DX exercise.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import polars as pl
from decider.modules.functional import generate_from_functions

# ── Step 1: Define the computation functions ──────────────────────────────────

def cart_value(price: pl.Expr, quantity: pl.Expr) -> pl.Expr:
    """Total value of the cart line: price * quantity."""
    return price * quantity


def is_high_value(cart_value: pl.Expr) -> pl.Expr:
    """Flag rows where cart_value > 100."""
    return cart_value > 100


def discount_rate(is_high_value: pl.Expr) -> pl.Expr:
    """0.1 for high-value carts, 0.0 otherwise."""
    return pl.when(is_high_value).then(pl.lit(0.1)).otherwise(pl.lit(0.0))


# ── Step 2: Generate the module and compose a pipeline ────────────────────────

CartScorer = generate_from_functions(
    "cart_scorer",
    cart_value,
    is_high_value,
    discount_rate,
)

scorer = CartScorer(name="cart_scorer")

# ── Step 3: Build some input data ─────────────────────────────────────────────

cart_df = pl.DataFrame({
    "item_id": ["A001", "A002", "A003", "A004"],
    "price":    [  9.99,  55.00, 120.00,  3.50],
    "quantity": [     2,      3,      1,    10],
})

print("=== Input ===")
print(cart_df)

# ── Step 4: Execute (using | pipeline composition) ───────────────────────────
#
# The task asks for |, so we compose the module with itself (a trivial
# identity pass) just to exercise the operator, then use a single-module
# pipeline via the | operator.  A more natural two-module split is shown
# in the comment below.

# Single-module pipeline built via | operator
from decider.pipeline import Pipeline

# Using | to chain: build a no-op second stage to satisfy "use |"
def final_score(discount_rate: pl.Expr, cart_value: pl.Expr) -> pl.Expr:
    """Discounted cart value."""
    return cart_value * (pl.lit(1.0) - discount_rate)

FinalModule = generate_from_functions("final_scorer", final_score)
final_module = FinalModule(name="final_scorer")

pipeline = scorer | final_module

print("\n=== Pipeline ===")
print(pipeline)

result_df = pipeline.execute({"input": cart_df}, lazy=False)

print("\n=== Output ===")
print(result_df)

# Sanity checks
assert "cart_value" in result_df.columns, "missing cart_value"
assert "is_high_value" in result_df.columns, "missing is_high_value"
assert "discount_rate" in result_df.columns, "missing discount_rate"
assert "final_score" in result_df.columns, "missing final_score"

# Row 0: 9.99*2=19.98 → not high value → discount_rate=0.0
assert abs(result_df["cart_value"][0] - 19.98) < 0.01
assert result_df["is_high_value"][0] == False
assert result_df["discount_rate"][0] == 0.0

# Row 2: 120*1=120 → high value → discount_rate=0.1
assert result_df["is_high_value"][2] == True
assert result_df["discount_rate"][2] == 0.1

print("\n=== All assertions passed ===")
