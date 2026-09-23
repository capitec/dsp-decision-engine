"""Single-record cost, decomposed: where do the microseconds go in one plugin evaluation on a 1-row frame?"""
import gc, sys, time, polars as pl
from common import log
import decider_trees as dt, trees
THREADS = pl.thread_pool_size(); c = pl.col
def bench(fn, reps=2000, warm=100):
    for _ in range(warm): fn()
    gc.disable(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns(); fn(); ts.append(time.perf_counter_ns() - t0)
    gc.enable(); ts.sort(); return {"p50_us": ts[len(ts)//2] / 1e3, "min_us": ts[0] / 1e3}
def row(name, r, **kw):
    print(f"  {name:62s} p50 {r['p50_us']:8.1f} us   min {r['min_us']:8.1f}")
    log("q0_parts", threads=THREADS, path=name, loadavg_1m=float(open("/proc/loadavg").read().split()[0]), **r, **kw)
f1 = trees.make_frame(1)
cols4 = [c(f) for f in trees.T1_FEATURES]
from polars.plugins import register_plugin_function
noop_n = lambda cols: register_plugin_function(plugin_path=dt.LIB, function_name="noop", args=cols, is_elementwise=True).alias("n")
print(f"=== 1-row frame, threads={THREADS}")
for k in (1, 2, 4):
    e = noop_n(cols4[:k]); row(f"noop, {k} input column(s), no kwargs", bench(lambda: f1.with_columns(e)), inputs=k)
e = dt.noop_kwargs(*cols4, tree=trees.T1).alias("n"); row("noop_kwargs: 4 inputs + T1 (17 nodes) deserialised, no compile", bench(lambda: f1.with_columns(e)), inputs=4)
e = dt.noop_kwargs(*cols4, tree=trees.T1, compile=True).alias("n"); row("noop_compile: 4 inputs + T1 deserialised + compiled", bench(lambda: f1.with_columns(e)), inputs=4)
e = dt.walk_value(*cols4, tree=trees.T1).alias("n"); row("walk_value: 4 inputs + T1 deserialised + compiled + walked", bench(lambda: f1.with_columns(e)), inputs=4)
e = dt.walk_value(c("income"), tree=[dt.test(0, "<", 5000.0, 1, 2), dt.leaf(0, 1.0), dt.leaf(1, 2.0)]).alias("n"); row("walk_value: 1 input, 3-node tree", bench(lambda: f1.with_columns(e)), inputs=1)
def chain(depth):
    return [dt.test(0, "<", float(-i), depth, i + 1 if i + 1 < depth else depth + 1) for i in range(depth)] + [dt.leaf(0, 1.0), dt.leaf(1, 2.0)]
for depth in (8, 64, 512):
    t = chain(depth)
    e1 = dt.noop_kwargs(c("income"), tree=t).alias("n"); e2 = dt.walk_value(c("income"), tree=t).alias("n")
    r1 = bench(lambda: f1.with_columns(e1), reps=500); r2 = bench(lambda: f1.with_columns(e2), reps=500)
    row(f"  chain depth {depth:4d}: noop_kwargs (deserialise only)", r1, depth=depth, inputs=1)
    row(f"  chain depth {depth:4d}: walk_value (deserialise+compile+walk 1 row)", r2, depth=depth, inputs=1)
if THREADS == 1:
    print("=== streaming re-ships the tree per morsel: 1M rows, 1 thread, depth-512 chain (deserialise ~= 1 ms per call)")
    big = trees.make_frame(1_000_000); t = chain(512); e = dt.walk_value(c("income"), tree=t).alias("v")
    def timeit(fn, reps=5):
        fn(); ts = []
        for _ in range(reps): gc.collect(); t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
        ts.sort(); return ts[len(ts)//2] * 1e3
    a = timeit(lambda: big.with_columns(e)); b = timeit(lambda: big.lazy().with_columns(e).collect(engine="streaming"))
    e8 = dt.walk_value(c("income"), tree=chain(8)).alias("v")
    a8 = timeit(lambda: big.with_columns(e8)); b8 = timeit(lambda: big.lazy().with_columns(e8).collect(engine="streaming"))
    print(f"  depth 512: in-memory {a:7.1f} ms   streaming {b:7.1f} ms   |  depth 8: in-memory {a8:7.1f} ms   streaming {b8:7.1f} ms")
    log("q0_streaming_reship", depth512_inmemory_ms=a, depth512_streaming_ms=b, depth8_inmemory_ms=a8, depth8_streaming_ms=b8)
