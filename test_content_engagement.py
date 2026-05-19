"""
Content engagement scoring pipeline — first-timer DX test.

Pipeline:
  1. JoinModule: views + content_metadata -> enriched
  2. ExprModule: enriched -> engagement_score, trending
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import polars as pl
from decider.modules.primitives.join import JoinModule
from decider.modules.functional import generate_from_functions

# ── Test data ────────────────────────────────────────────────────────────────

views = pl.DataFrame({
    "content_id": [1, 2, 3, 4],
    "views":      [1200, 300, 8000, 50],
    "likes":      [400,   80, 3000,  5],
})

content_metadata = pl.DataFrame({
    "content_id":       [1,    2,    3,   4],
    "duration_minutes": [10.0, 2.5,  45.0, 1.0],
})

print("=== Input frames ===")
print("views:")
print(views)
print("\ncontent_metadata:")
print(content_metadata)

# ── Step 1: join ──────────────────────────────────────────────────────────────

join = JoinModule(
    name="content_join",
    left="views",
    right="content_metadata",
    on="content_id",
    how="left",
    output_frame="enriched",
)

# ── Step 2: compute engagement_score and trending flag ────────────────────────

def engagement_score(views: pl.Expr, likes: pl.Expr, duration_minutes: pl.Expr) -> pl.Expr:
    return (views * 0.3 + likes * 0.7) / duration_minutes

def trending(engagement_score: pl.Expr) -> pl.Expr:
    return engagement_score > 2.0

ScoringModule = generate_from_functions("content_scorer", engagement_score, trending)
scorer = ScoringModule(name="scorer")

# ── Compose pipeline ──────────────────────────────────────────────────────────

pipeline = join | scorer

print("\n=== Executing pipeline ===")
result = pipeline.execute(
    {"views": views, "content_metadata": content_metadata},
    lazy=False,
)

print("\n=== Result ===")
print(result)

# ── Assertions ────────────────────────────────────────────────────────────────

assert "engagement_score" in result.columns, "Missing engagement_score"
assert "trending" in result.columns, "Missing trending"

# Row 0: (1200*0.3 + 400*0.7) / 10  = (360 + 280) / 10  = 64.0 -> trending
# Row 1: (300*0.3  + 80*0.7)  / 2.5 = (90  + 56)  / 2.5 = 58.4 -> trending
# Row 2: (8000*0.3 + 3000*0.7)/ 45  = (2400+2100)/ 45   = 100.0 -> trending
# Row 3: (50*0.3  + 5*0.7)   / 1.0  = (15  + 3.5) / 1.0 = 18.5 -> trending

expected_scores = [64.0, 58.4, 100.0, 18.5]
for i, (actual, expected) in enumerate(zip(result["engagement_score"].to_list(), expected_scores)):
    assert abs(actual - expected) < 0.001, f"Row {i}: {actual} != {expected}"

assert result["trending"].to_list() == [True, True, True, True], \
    f"Unexpected trending: {result['trending'].to_list()}"

print("\n=== All assertions passed ===")
print("engagement_score values:", result["engagement_score"].to_list())
print("trending values:        ", result["trending"].to_list())
