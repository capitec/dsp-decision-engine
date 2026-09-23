"""Runs ONE hostile call in this process so the parent can see whether it survives.
  categorical  : a pl.Categorical column into a plugin (built WITH dtype-categorical -> error; WITHOUT -> abort)
  panic        : an explicit index-out-of-bounds panic inside the plugin body
  after_error  : a bad tree (error), then a good call in the same process
"""
import sys, polars as pl
from common import log
import decider_trees as dt
mode = sys.argv[1]
df = pl.DataFrame({"s": ["a", "b"], "x": [1.0, 2.0]})
tree = [dt.test(0, "==", "a", 1, 2), dt.leaf(1), dt.leaf(0)]
print(f"[worker {mode}] start", flush=True)
if mode == "categorical":
    print(df.with_columns(pl.col("s").cast(pl.Categorical)).with_columns(dt.walk(pl.col("s"), tree=tree)), flush=True)
elif mode == "panic":
    print(df.with_columns(dt.panic_demo(pl.col("x"))), flush=True)
elif mode == "after_error":
    try:
        df.with_columns(dt.walk(pl.col("x"), tree=tree))
    except Exception as e:
        print("first call raised:", type(e).__name__, str(e)[:160], flush=True)
    print("second call:", df.with_columns(dt.walk(pl.col("s"), tree=tree).alias("leaf"))["leaf"].to_list(), flush=True)
print(f"[worker {mode}] exiting normally", flush=True)
