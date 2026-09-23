"""What the plugin's per-row cost is made of, 1M rows, single expression each."""
import gc, time, polars as pl
from common import log
import decider_trees as dt, trees
df = trees.make_frame(1_000_000)
c = pl.col
def t(name, e, frame=df):
    frame.with_columns(e); ts = []
    for _ in range(5):
        gc.collect(); t0 = time.perf_counter(); frame.with_columns(e); ts.append(time.perf_counter() - t0)
    ts.sort(); print(f"{name:44s} {ts[2]*1e3:8.1f} ms  {ts[2]*1e3:6.1f} ns/row"); log("q2_isolate_1M", expr=name, ns_per_row_median=ts[2] * 1e3)
one_f = [dt.test(0, "<", 5000.0, 1, 2), dt.leaf(0, 1.0), dt.leaf(1, 2.0)]
one_i = [dt.test(0, "<", 40, 1, 2), dt.leaf(0, 1.0), dt.leaf(1, 2.0)]
one_b = [dt.test(0, "==", True, 1, 2), dt.leaf(0, 1.0), dt.leaf(1, 2.0)]
one_s = [dt.test(0, "==", "mining", 1, 2), dt.leaf(0, 1.0), dt.leaf(1, 2.0)]
one_pre = [dt.test(0, "prefix", "retail", 1, 2), dt.leaf(0, 1.0), dt.leaf(1, 2.0)]
one_re = [dt.test(0, "regex", r"^retail(-online)?$", 1, 2), dt.leaf(0, 1.0), dt.leaf(1, 2.0)]
leaf_only = [dt.leaf(0, 1.0)]
chain4 = [dt.test(0, "<", float(v), 4, i + 1) for i, v in enumerate((1000, 3000, 8000, 20000))][:3] + [dt.test(0, "<", 20000.0, 4, 5), dt.leaf(0, 1.0), dt.leaf(1, 2.0)]
t("leaf only (no test): output + kwargs floor", dt.walk_value(c("income"), tree=leaf_only))
t("1 f64 test", dt.walk_value(c("income"), tree=one_f))
t("1 i64 test", dt.walk_value(c("age"), tree=one_i))
t("1 bool test", dt.walk_value(c("verified"), tree=one_b))
t("1 str == test", dt.walk_value(c("sector"), tree=one_s))
t("1 str prefix test", dt.walk_value(c("sector"), tree=one_pre))
t("1 str regex test", dt.walk_value(c("sector"), tree=one_re))
t("4 f64 tests chained (predictable)", dt.walk_value(c("income"), tree=chain4))
t("polars native: income < 5000", (c("income") < 5000.0).alias("x"))
t("polars native: sector == 'mining'", (c("sector") == "mining").alias("x"))
t("polars native: sector.str.contains regex", c("sector").str.contains(r"^retail(-online)?$").alias("x"))
t("polars native: when(income<5000).then(1).otherwise(2)", pl.when(c("income") < 5000.0).then(1.0).otherwise(2.0).alias("x"))
t("T1 plugin on a 1-chunk frame", dt.walk_value(*[c(f) for f in trees.T1_FEATURES], tree=trees.T1))
t("T1 plugin on a 10-chunk frame", dt.walk_value(*[c(f) for f in trees.T1_FEATURES], tree=trees.T1), pl.concat([df.slice(i * 100_000, 100_000) for i in range(10)], rechunk=False))
