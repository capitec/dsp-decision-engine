"""Content engagement scoring pipeline.

Join a views frame with content_metadata, then compute:
  engagement_score = (views * 0.3 + likes * 0.7) / duration_minutes
  trending        = engagement_score > 2.0
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import polars as pl
from decider.modules.primitives.join import JoinModule
from decider.modules.functional import generate_from_functions

# ── Test data ────────────────────────────────────────────────────────────────

views_df = pl.DataFrame({
    "content_id":   [1,    2,    3,    4],
    "views":        [1000, 500,  800,  200],
    "likes":        [300,  450,  100,  180],
})

content_metadata_df = pl.DataFrame({
    "content_id":       [1,    2,   3,    4],
    "duration_minutes": [5.0,  3.0, 10.0, 2.0],
})

# ── Step 1: Join ──────────────────────────────────────────────────────────────

join_module = JoinModule(
    name="content_join",
    left="views",
    right="content_metadata",
    on="content_id",
    how="inner",
    output_frame="enriched",
)

# ── Step 2 & 3: Compute engagement_score + trending flag ─────────────────────

def engagement_score(views: pl.Expr, likes: pl.Expr, duration_minutes: pl.Expr) -> pl.Expr:
    return (views * 0.3 + likes * 0.7) / duration_minutes

def trending(engagement_score: pl.Expr) -> pl.Expr:
    return engagement_score > 2.0

EngagementModule = generate_from_functions("engagement", engagement_score, trending)
engagement_module = EngagementModule(name="engagement")

# ── Pipeline: join | expressions ──────────────────────────────────────────────

pipeline = join_module | engagement_module

print("Pipeline steps:", [s.name for s in pipeline.steps])

result_lf = pipeline.execute(
    {"views": views_df, "content_metadata": content_metadata_df}
)

result_df = result_lf.collect()

print("\nFinal result:")
print(result_df)

# ── Assertions ───────────────────────────────────────────────────────────────

assert "engagement_score" in result_df.columns, "Missing engagement_score"
assert "trending" in result_df.columns, "Missing trending"
assert len(result_df) == 4, f"Expected 4 rows, got {len(result_df)}"

# content_id=1: (1000*0.3 + 300*0.7) / 5  = (300 + 210) / 5  = 510/5  = 102.0
# content_id=2: (500*0.3  + 450*0.7) / 3  = (150 + 315) / 3  = 465/3  = 155.0
# content_id=3: (800*0.3  + 100*0.7) / 10 = (240 + 70)  / 10 = 310/10 = 31.0
# content_id=4: (200*0.3  + 180*0.7) / 2  = (60  + 126) / 2  = 186/2  = 93.0
# All > 2.0 → all trending=True

scores = result_df["engagement_score"].to_list()
print(f"\nScores: {scores}")
assert all(s > 2.0 for s in scores), f"All scores should be > 2.0: {scores}"
assert all(result_df["trending"].to_list()), "All items should be trending"

print("\nAll assertions passed.")
