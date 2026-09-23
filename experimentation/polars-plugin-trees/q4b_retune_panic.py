"""Q4 (c) retune-without-recompile and (d) errors/panics, each hostile call in its own process."""
from __future__ import annotations
import signal, subprocess, sys
import polars as pl
from common import log, HERE
import decider_trees as dt
sys.path.insert(0, "/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/decider2/src")
import trees
from decider2.testing.recompile import count_new_compiles

PY = sys.executable
frame_1k = trees.make_frame(1_000)
p = trees.decider2_pipeline(); p.precompile()
# retune: does changing a threshold recompile anything? (there is nothing to compile; it is a new kwargs blob)
from decider2.testing.recompile import count_new_compiles
with count_new_compiles() as ev:
    for thr in (5000.0, 6000.0, 7000.0):
        t = list(trees.T1); t[0] = dt.test(0, "<", thr, 1, 2)
        frame_1k.with_columns(dt.walk_value(*[pl.col(f) for f in trees.T1_FEATURES], tree=t))
print("numba compile events during 3 plugin retunes:", ev.count)
with count_new_compiles() as ev2:
    for thr in (5000.0, 6000.0, 7000.0):
        p.apply(frame_1k, params={"band": {"n0_thr": thr}}, mode="fused")
print("numba compile events during 3 decider2 retunes:", ev2.count)
log("q4_retune", plugin_compile_events=ev.count, decider2_compile_events=ev2.count)

# ---------------------------------------------------------------- (d) errors and panics, each in its own process
res = {}
for mode in ("after_error", "categorical", "panic"):
    pr = subprocess.run([PY, str(HERE / "panic_worker.py"), mode], capture_output=True, text=True, timeout=120, cwd=str(HERE))
    sig = signal.Signals(-pr.returncode).name if pr.returncode < 0 else None
    tail = [l for l in pr.stderr.splitlines() if l.strip()][-2:]
    print(f"\n=== {mode}: rc={pr.returncode}" + (f" KILLED by {sig}" if sig else ""))
    for l in pr.stdout.splitlines(): print("  " + l)
    for l in tail: print("  stderr| " + l[:200])
    res[mode] = {"rc": pr.returncode, "signal": sig, "stdout": pr.stdout.splitlines(), "stderr_tail": tail}
log("q4_panics", **res)
