"""
E-commerce cart scoring pipeline built with the decider framework.
DX exploration script — written naively, then refined.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import polars as pl
from decider.modules.functional import generate_from_functions

# ---------------------------------------------------------------------------
# Step 1: Define the computation functions.
#   Convention: function name  → output column name
#               parameter name → input column name (or sibling function name)
# ---------------------------------------------------------------------------

def cart_value(price: pl.Expr, quantity: pl.Expr) -> pl.Expr:
    """Sum of price * quantity per row."""
    return price * quantity


def is_high_value(cart_value: pl.Expr) -> pl.Expr:
    """True when cart_value > 100."""
    return cart_value > 100


def discount_rate(is_high_value: pl.Expr) -> pl.Expr:
    """10 % discount for high-value carts, 0 % otherwise."""
    return pl.when(is_high_value).then(0.1).otherwise(0.0)


# ---------------------------------------------------------------------------
# Step 2: Build module class + instance.
# ---------------------------------------------------------------------------

CartScorer = generate_from_functions("cart_scorer", cart_value, is_high_value, discount_rate)
scorer = CartScorer(name="cart_scorer")

# ---------------------------------------------------------------------------
# Step 3: Prepare test data (>= 4 rows).
# ---------------------------------------------------------------------------

cart_df = pl.DataFrame({
    "cart_id":  [1,     2,     3,     4,     5   ],
    "price":    [10.0,  50.0,  20.0,  200.0, 5.0 ],
    "quantity": [5,     1,     3,     1,     10  ],
    # expected cart_value: 50, 50, 60, 200, 50
    # expected is_high_value: F, F, F, T, F
    # expected discount_rate: 0, 0, 0, 0.1, 0
})

# ---------------------------------------------------------------------------
# Step 4: Execute using the | composition operator via a two-module pipeline.
#   (Use | to demonstrate composition; split into two modules for clarity.)
# ---------------------------------------------------------------------------

def cart_value_only(price: pl.Expr, quantity: pl.Expr) -> pl.Expr:
    return price * quantity

def is_high_value_only(cart_value_only: pl.Expr) -> pl.Expr:
    return cart_value_only > 100

def discount_rate_only(is_high_value_only: pl.Expr) -> pl.Expr:
    return pl.when(is_high_value_only).then(0.1).otherwise(0.0)

ValueModule = generate_from_functions("value_module", cart_value_only)
FlagsModule = generate_from_functions("flags_module", is_high_value_only, discount_rate_only)

pipeline = ValueModule(name="value") | FlagsModule(name="flags")

print("=== Single-module execute (all three features at once) ===")
result_single = scorer.execute({"input": cart_df}, lazy=False)
print(result_single)

print()
print("=== Pipeline execute via | composition ===")
result_pipeline = pipeline.execute({"input": cart_df}, lazy=False)
print(result_pipeline)
