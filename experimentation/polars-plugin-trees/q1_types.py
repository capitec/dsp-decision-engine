"""Q1: does it work, and is it honest about types?  int64 above 2^53, bool, string."""
from __future__ import annotations
import polars as pl
from common import log
import decider_trees as dt

BIG = 9007199254740992  # 2**53

# --- int64 exactness: [2^53, 2^53+1] == 2^53  must be [True, False]
df = pl.DataFrame({"x": [BIG, BIG + 1]}, schema={"x": pl.Int64})
tree = [dt.test(0, "==", BIG, 1, 2), dt.leaf(1), dt.leaf(0)]
out = df.with_columns(dt.walk(pl.col("x"), tree=tree).alias("leaf"))
plugin_i64 = [bool(v) for v in out["leaf"].to_list()]
print("plugin int64  ==2^53 :", plugin_i64, "(dtype in:", df["x"].dtype, ")")
# the same values pushed through float64 first (what a single-float64 kernel sees)
df_f = df.with_columns(pl.col("x").cast(pl.Float64))
tree_f = [dt.test(0, "==", float(BIG), 1, 2), dt.leaf(1), dt.leaf(0)]
out_f = df_f.with_columns(dt.walk(pl.col("x"), tree=tree_f).alias("leaf"))
via_f64 = [bool(v) for v in out_f["leaf"].to_list()]
print("same via float64     :", via_f64)
# and the honest refusal: an i64 test against a Float64 column
try:
    df_f.with_columns(dt.walk(pl.col("x"), tree=tree).alias("leaf"))
    mismatch_err = None
except Exception as e:
    mismatch_err = f"{type(e).__name__}: {e}"
print("i64 test on f64 col  :", mismatch_err)
log("q1_int64", plugin=plugin_i64, via_float64=via_f64, mismatch_error=mismatch_err)
assert plugin_i64 == [True, False]

# --- bool stays bool
dfb = pl.DataFrame({"flag": [True, False, None]})
treeb = [dt.test(0, "==", True, 1, 2), dt.leaf(1), dt.leaf(0)]
ob = dfb.with_columns(dt.walk(pl.col("flag"), tree=treeb).alias("leaf"))
print("bool                 :", ob["leaf"].to_list(), "in dtype", dfb["flag"].dtype)
try:
    pl.DataFrame({"flag": [1, 0]}).with_columns(dt.walk(pl.col("flag"), tree=treeb))
    bool_err = None
except Exception as e:
    bool_err = f"{type(e).__name__}: {e}"
print("bool test on i64 col :", bool_err)
log("q1_bool", out=ob["leaf"].to_list(), mismatch_error=bool_err)
assert ob["leaf"].to_list() == [1, 0, None]

# --- string: ==, prefix, regex, on a String column, no dictionary anywhere
dfs = pl.DataFrame({"sector": ["retail", "retail-online", "mining", "Mining Ltd", None]})
t_eq = [dt.test(0, "==", "retail", 1, 2), dt.leaf(1), dt.leaf(0)]
t_pre = [dt.test(0, "prefix", "retail", 1, 2), dt.leaf(1), dt.leaf(0)]
t_re = [dt.test(0, "regex", r"(?i)^min", 1, 2), dt.leaf(1), dt.leaf(0)]
os_ = dfs.with_columns(
    dt.walk(pl.col("sector"), tree=t_eq).alias("eq"),
    dt.walk(pl.col("sector"), tree=t_pre).alias("prefix"),
    dt.walk(pl.col("sector"), tree=t_re).alias("regex"),
)
print(os_)
r = {c: os_[c].to_list() for c in ("eq", "prefix", "regex")}
assert r["eq"] == [1, 0, 0, 0, None]
assert r["prefix"] == [1, 1, 0, 0, None]
assert r["regex"] == [0, 0, 1, 1, None]
try:
    dfs.with_columns(pl.col("sector").cast(pl.Categorical)).with_columns(dt.walk(pl.col("sector"), tree=t_eq))
    cat_err = None
except Exception as e:
    cat_err = f"{type(e).__name__}: {e}"
print("str test on Categorical col:", cat_err)
# a bad regex, an unknown op, a dangling node: what the author sees
errs = {}
for name, bad in {
    "bad_regex": [dt.test(0, "regex", "(", 1, 2), dt.leaf(1), dt.leaf(0)],
    "unknown_op": [dt.test(0, "~=", "x", 1, 2), dt.leaf(1), dt.leaf(0)],
    "dangling": [dt.test(0, "==", "x", 1, 9), dt.leaf(1), dt.leaf(0)],
    "col_out_of_range": [dt.test(3, "==", "x", 1, 2), dt.leaf(1), dt.leaf(0)],
}.items():
    try:
        dfs.with_columns(dt.walk(pl.col("sector"), tree=bad)); errs[name] = None
    except Exception as e:
        errs[name] = f"{type(e).__name__}: {e}"
    print(f"{name:18s}: {errs[name]}")
log("q1_string", **r, categorical_error=cat_err, author_errors=errs)
print("Q1 OK")
