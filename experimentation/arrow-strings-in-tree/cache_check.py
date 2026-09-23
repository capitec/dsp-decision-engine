"""Q4: does the STR-node kernel disk-cache across processes?

Two child processes share a fresh NUMBA_CACHE_DIR with NUMBA_DEBUG_CACHE=1;
we count numba's own "data saved" / "data loaded" lines. The addresses of the
polars buffers are ARGUMENTS (arrays of uint64), never globals -- §W's rule --
so the warm process must save nothing and load everything.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

CHILD = r'''
import numpy as np, polars as pl, sys
sys.path.insert(0, %(here)r)
import arrowc
from kernel import build_tree, run_tree, STR, LEAF, PREFIX, CONTAINS
df = pl.DataFrame({"s": ["dog walker", None, "a much longer merchant descriptor", "hotdog", "dog"] * 1000})
v = arrowc.export(df["s"])
tree = build_tree([(STR, 0, PREFIX, 0, 1, 2, 0), (LEAF, 0, 0, 0, 0, 0, 1), (LEAF, 0, 0, 0, 0, 0, 0)])
out = run_tree(np.empty((5000, 0)), [v], tree, ["dog"])
print("RESULT", out[:5].tolist())
'''


def run(label, cache_dir, here):
    env = dict(os.environ, NUMBA_CACHE_DIR=cache_dir, NUMBA_DEBUG_CACHE="1")
    proc = subprocess.run([sys.executable, "-c", CHILD % {"here": here}], env=env,
                          capture_output=True, text=True, cwd=here)
    text = proc.stdout + proc.stderr
    saved = sum(1 for ln in text.splitlines() if "data saved" in ln)
    loaded = sum(1 for ln in text.splitlines() if "data loaded" in ln)
    result = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
    rec = {"probe": "cache", "run": label, "returncode": proc.returncode, "data_saved": saved,
           "data_loaded": loaded, "result": result[0] if result else None}
    if proc.returncode != 0:
        rec["stderr_tail"] = proc.stderr[-2000:]
    return rec, text


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    cache_dir = tempfile.mkdtemp(prefix="numba-cache-arrow-strings-")
    try:
        with open(os.path.join(here, "results.jsonl"), "a") as out:
            for label in ("cold", "warm"):
                rec, text = run(label, cache_dir, here)
                out.write(json.dumps(rec) + "\n"); out.flush()
                print(json.dumps(rec))
                with open(os.path.join(here, f"cache_{label}.log"), "w") as f:
                    f.write(text)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
