"""
HR Analytics: Employee Performance Scoring Pipeline
Uses the decider framework with two parallel branches.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import polars as pl
from decider.modules.functional import generate_from_functions
from decider.pipeline import ForkPipeline
from decider.modules.primitives.join import JoinModule

# ── Test data ─────────────────────────────────────────────────────────────────
employees = pl.DataFrame({
    "employee_id": [1001, 1002, 1003, 1004, 1005],
    "tasks_completed": [45, 30, 50, 20, 38],
    "tasks_assigned": [50, 40, 50, 40, 45],
    "peer_rating": [4.2, 3.8, 5.0, 2.5, 4.0],
})

print("=== Input Data ===")
print(employees)
print()

# ── Branch 1: Output Score = (tasks_completed / tasks_assigned) * 100 ─────────
def output_score(tasks_completed: pl.Expr, tasks_assigned: pl.Expr) -> pl.Expr:
    return (tasks_completed / tasks_assigned) * 100

OutputScoreModule = generate_from_functions("output_scorer", output_score)
output_scorer = OutputScoreModule(name="output_scorer")

# ── Branch 2: Peer Score = peer_rating * 20 ───────────────────────────────────
def peer_score(peer_rating: pl.Expr) -> pl.Expr:
    return peer_rating * 20

PeerScoreModule = generate_from_functions("peer_scorer", peer_score)
peer_scorer = PeerScoreModule(name="peer_scorer")

# ── Attempt 1: Fork via & operator ────────────────────────────────────────────
print("=== Attempt 1: Fork via & operator ===")
try:
    fork = output_scorer & peer_scorer
    print(f"Fork type: {type(fork).__name__}")
    fork_result = fork.execute({"input": employees})
    print(f"Fork result type: {type(fork_result)}")
    print(f"Fork result keys: {list(fork_result.keys())}")
    for key, lf in fork_result.items():
        df = lf.collect()
        print(f"\nBranch '{key}':\n{df}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
print()

# ── Attempt 2: .include() variant ─────────────────────────────────────────────
print("=== Attempt 2: .include() variant ===")
try:
    fork2 = output_scorer.include(peer_scorer)
    print(f"Fork2 type: {type(fork2).__name__}")
    fork2_result = fork2.execute({"input": employees})
    print(f"Fork2 result type: {type(fork2_result)}")
    print(f"Fork2 result keys: {list(fork2_result.keys())}")
    for key, lf in fork2_result.items():
        df = lf.collect()
        print(f"\nBranch '{key}':\n{df}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
print()

# ── Attempt 3: Join the branches ───────────────────────────────────────────────
print("=== Attempt 3: Join branch results via JoinModule ===")
try:
    # Run branches separately to get two named frames with employee_id
    out1 = output_scorer.execute({"input": employees}, lazy=False)
    print(f"Branch 1 columns: {out1.columns}")

    out2 = peer_scorer.execute({"input": employees}, lazy=False)
    print(f"Branch 2 columns: {out2.columns}")

    # Select only employee_id + score columns from each branch to avoid
    # duplicate columns when joining
    out1_slim = out1.select(["employee_id", "output_score"])
    out2_slim = out2.select(["employee_id", "peer_score"])

    join = JoinModule(
        name="score_join",
        left="output_scores",
        right="peer_scores",
        on="employee_id",
        how="inner",
        output_frame="combined_scores"
    )

    join_result = join.execute(
        {"output_scores": out1_slim, "peer_scores": out2_slim},
        output_frames=["combined_scores"]
    )
    combined = join_result["combined_scores"].collect()
    print(f"\nJoined result:\n{combined}")

    # Add composite score
    combined = combined.with_columns(
        ((pl.col("output_score") + pl.col("peer_score")) / 2).alias("composite_score")
    )
    print(f"\nFinal combined scores:\n{combined}")

except Exception as e:
    import traceback
    print(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
print()

# ── Attempt 4: Fork then join inside a Pipeline ───────────────────────────────
print("=== Attempt 4: ForkPipeline | JoinModule ===")
try:
    from decider.pipeline import Pipeline

    fork3 = output_scorer & peer_scorer
    print(f"Fork3 branches: {[type(b).__name__ for b in fork3.branches]}")

    # Use the rejoin operator: ForkPipeline | next_module
    score_join = JoinModule(
        name="score_join2",
        left="output_scorer",
        right="peer_scorer",
        on="employee_id",
        how="inner",
        output_frame="combined"
    )

    pipeline = fork3 | score_join
    print(f"Pipeline type: {type(pipeline).__name__}")
    print(f"Pipeline steps: {[type(s).__name__ for s in pipeline.steps]}")

    pipeline_result = pipeline.execute(
        {"input": employees},
        output_frames=["combined"]
    )
    final = pipeline_result["combined"].collect()
    print(f"\nFull pipeline result:\n{final}")

except Exception as e:
    import traceback
    print(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()
