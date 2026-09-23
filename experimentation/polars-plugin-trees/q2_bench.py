"""Q2 worker: one engine, one row count, in a fresh process (POLARS_MAX_THREADS is
read at import, so thread counts need separate processes).

  python q2_bench.py <engine> <rows> [reps]
  engine: d2-fused | pl-eager | pl-lazy | pl-streaming | pl-eager-par | pl-oracle-eager
"""
from __future__ import annotations
import gc, os, sys, time, json
import polars as pl
from common import log

sys.path.insert(0, "/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/decider2/src")
import trees

engine, rows = sys.argv[1], int(sys.argv[2])
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 7
threads = pl.thread_pool_size()
df = trees.make_frame(rows)

if engine == "d2-fused":
    p = trees.decider2_pipeline()
    p.precompile()                       # off the request path, as decider2 intends
    run = lambda: p.apply(df, mode="fused")
elif engine.startswith("pl-"):
    how = {"pl-eager": "eager", "pl-lazy": "lazy", "pl-streaming": "streaming",
           "pl-eager-par": "eager", "pl-oracle-eager": "eager"}[engine]
    par = engine == "pl-eager-par"
    oracle = engine == "pl-oracle-eager"
    run = lambda: trees.run_polars(df, how=how, parallel=par, oracle=oracle)
else:
    raise SystemExit(f"unknown engine {engine}")

out = run()                              # warm (numba compile / plugin dlopen / regex)
ref = trees.run_polars(df, oracle=True)
for col in ("pts1", "pts2", "approve"):
    assert (out[col] == ref[col]).all(), f"{engine}: {col} differs from oracle"
assert float((out["adj"] - ref["adj"]).abs().max()) < 1e-9, f"{engine}: adj differs from oracle beyond 1e-9"
times = []
for _ in range(reps):
    gc.collect()
    t0 = time.perf_counter(); run(); times.append(time.perf_counter() - t0)
times.sort()
rec = log("q2_pipeline", engine=engine, rows=rows, polars_threads=threads,
          wall_min_s=times[0], wall_median_s=times[len(times)//2], wall_max_s=times[-1],
          ns_per_row_min=times[0] / rows * 1e9, ns_per_row_median=times[len(times)//2] / rows * 1e9, reps=reps)
print(json.dumps(rec))
