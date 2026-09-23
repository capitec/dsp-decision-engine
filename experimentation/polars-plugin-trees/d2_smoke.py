import sys, polars as pl
sys.path.insert(0, "/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/decider2/src")
import trees
df = trees.make_frame(20000)
p = trees.decider2_pipeline()
out = p.apply(df, mode="fused")
oracle = trees.run_polars(df, oracle=True)
print(out.select("pts1", "adj", "pts2", "approve").describe())
for col in ("pts1", "adj", "pts2", "approve"):
    print(col, "decider2 == oracle:", (out[col] == oracle[col]).all(), "| approve rate", round(out["approve"].mean(), 3) if col == "approve" else "")
