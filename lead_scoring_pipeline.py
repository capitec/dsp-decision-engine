"""Lead scoring pipeline — DX evaluation of decider framework.

Tests:
  - generate_from_functions() to define 3 modules
  - Pipeline.chain() composition
  - debug=True execution
  - | operator composition
  - div-by-zero behaviour with emails_opened == 0
"""

import polars as pl
from decider.modules.functional import generate_from_functions
from decider.pipeline import Pipeline

# ── Test data (5 rows; rows 2 and 4 have 0 opens → div-by-zero candidate) ──

leads = pl.DataFrame({
    "lead_id":       [1,    2,    3,    4,    5   ],
    "emails_sent":   [10,   5,    8,    0,    12  ],
    "emails_opened": [4,    0,    6,    0,    9   ],
    "clicks":        [2,    0,    3,    0,    7   ],
})

# ── Module 1: engagement features ───────────────────────────────────────────

def email_open_rate(emails_opened: pl.Expr, emails_sent: pl.Expr) -> pl.Expr:
    return (emails_opened / emails_sent).fill_nan(0.0).fill_null(0.0)

def click_rate(clicks: pl.Expr, emails_opened: pl.Expr) -> pl.Expr:
    return (clicks / emails_opened).fill_nan(0.0).fill_null(0.0)

EngagementFeatures = generate_from_functions(
    "engagement_features",
    email_open_rate,
    click_rate,
)
engagement_features = EngagementFeatures(name="engagement_features")

# ── Module 2: lead temperature ───────────────────────────────────────────────

def hot_score(email_open_rate: pl.Expr, click_rate: pl.Expr) -> pl.Expr:
    return email_open_rate * 0.4 + click_rate * 0.6

LeadTemperature = generate_from_functions("lead_temperature", hot_score)
lead_temperature = LeadTemperature(name="lead_temperature")

# ── Module 3: lead tier ──────────────────────────────────────────────────────

def is_hot_lead(hot_score: pl.Expr) -> pl.Expr:
    return (hot_score > 0.3).cast(pl.Int8)

LeadTier = generate_from_functions("lead_tier", is_hot_lead)
lead_tier = LeadTier(name="lead_tier")

# ── Compose with Pipeline.chain() ───────────────────────────────────────────

print("=" * 60)
print("PIPELINE.CHAIN() — with debug=True")
print("=" * 60)

pipeline_chain = Pipeline.chain(engagement_features, lead_temperature, lead_tier)

result_chain = pipeline_chain.execute({"input": leads}, debug=True)
df_chain = result_chain.collect()
print("\nFinal result (chain):")
print(df_chain)

# ── Compose with | operator ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PIPE (|) OPERATOR — no debug")
print("=" * 60)

pipeline_pipe = engagement_features | lead_temperature | lead_tier
result_pipe = pipeline_pipe.execute({"input": leads})
df_pipe = result_pipe.collect()
print("\nFinal result (pipe):")
print(df_pipe)

# ── Verify equivalence ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EQUIVALENCE CHECK")
print("=" * 60)

are_equal = df_chain.frame_equal(df_pipe)
print(f"chain == pipe: {are_equal}")

# ── Inspect div-by-zero rows explicitly ─────────────────────────────────────

print("\n" + "=" * 60)
print("DIV-BY-ZERO ROWS (lead_id 2 and 4)")
print("=" * 60)
print(df_chain.filter(pl.col("lead_id").is_in([2, 4])))
