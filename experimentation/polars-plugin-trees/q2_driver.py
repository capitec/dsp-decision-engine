"""Q2 driver: runs every (engine, rows) cell in its own process; extrapolates before 10M."""
from __future__ import annotations
import json, os, subprocess, sys
from common import log, HERE

PY = "/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/.venv/bin/python"

def free_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024 / 1024

def cell(engine, rows, threads=None, reps=7):
    env = dict(os.environ)
    if threads:
        env["POLARS_MAX_THREADS"] = str(threads)
    p = subprocess.run([PY, str(HERE / "q2_bench.py"), engine, str(rows), str(reps)],
                       capture_output=True, text=True, env=env, cwd=str(HERE))
    if p.returncode != 0:
        print(f"FAILED {engine} rows={rows} threads={threads}\n{p.stderr[-2000:]}")
        log("q2_pipeline_failed", engine=engine, rows=rows, threads=threads, stderr=p.stderr[-2000:])
        return None
    rec = json.loads(p.stdout.strip().splitlines()[-1])
    rec["threads_env"] = threads
    print(f"{engine:16s} rows={rows:>9,} thr={rec['polars_threads']:>2} "
          f"{rec['ns_per_row_median']:8.1f} ns/row  wall {rec['wall_median_s']*1e3:9.2f} ms")
    return rec

ENGINES = ["d2-fused", "pl-eager", "pl-lazy", "pl-streaming", "pl-eager-par", "pl-oracle-eager"]
sizes = [int(x) for x in sys.argv[1:]] or [10_000, 100_000, 1_000_000]
for rows in sizes:
    if rows >= 10_000_000:
        gb = free_gb()
        print(f"MemAvailable before 10M: {gb:.1f} GB")
        if gb < 8:
            print("skipping 10M: under 8 GB available"); log("q2_skipped_10M", mem_available_gb=gb); continue
    reps = 7 if rows <= 1_000_000 else 3
    for eng in ENGINES:
        cell(eng, rows, reps=reps)
        if eng in ("pl-eager", "pl-streaming", "pl-eager-par") and rows >= 100_000:
            cell(eng, rows, threads=1, reps=reps)
