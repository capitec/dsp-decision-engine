"""Disk caching for B and C: two processes share a fresh NUMBA_CACHE_DIR with
NUMBA_DEBUG_CACHE=1; count numba's own "data saved"/"data loaded" lines.
The sm_get_string address and the ArrowArrayView addresses are ARGUMENTS,
so the warm process must save nothing (libnashim.so loads at a different
address in each process -- a captured global would re-specialise)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

from results_io import record

CHILD = r'''
import numpy as np, polars as pl, sys, ctypes
import nashim
from kernel import build_tree, STR, LEAF, PREFIX
from kernel_b import run_tree_b
from kernel_c import run_tree_c
from nashim import NanoView, FULL, lib, GET_STRING_ADDR
df = pl.DataFrame({"s": ["dog walker", None, "a much longer merchant descriptor", "hotdog", "dog"] * 1000})
tree = build_tree([(STR, 0, PREFIX, 0, 1, 2, 0), (LEAF, 0, 0, 0, 0, 0, 1), (LEAF, 0, 0, 0, 0, 0, 0)])
f = np.empty((5000, 0))
nv = NanoView(df["s"])
b = run_tree_b(f, [nv], tree, ["dog"]); bc = run_tree_b(f, [nv], tree, ["dog"], checked=True)
c = run_tree_c(f, [nv], tree, ["dog"], level=FULL); ct = run_tree_c(f, [nv], tree, ["dog"], level=FULL, trusting=True)
print("RESULT", b[:5].tolist(), bc[:5].tolist(), c[:5].tolist(), ct[:5].tolist(), "fnaddr", hex(GET_STRING_ADDR))
'''


def run(label, cache_dir, here):
    env = dict(os.environ, NUMBA_CACHE_DIR=cache_dir, NUMBA_DEBUG_CACHE="1", PYTHONPATH=here)
    proc = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, cwd=here)
    text = proc.stdout + proc.stderr
    saved = sum(1 for ln in text.splitlines() if "data saved" in ln)
    loaded = sum(1 for ln in text.splitlines() if "data loaded" in ln)
    result = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT")]
    rec = record(probe="cache", run=label, returncode=proc.returncode, data_saved=saved, data_loaded=loaded,
                 result=result[0] if result else None)
    if proc.returncode != 0:
        rec["stderr_tail"] = proc.stderr[-2000:]
    return rec, text


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    cache_dir = tempfile.mkdtemp(prefix="numba-cache-nanoarrow-")
    try:
        for label in ("cold", "warm", "warm2"):
            rec, text = run(label, cache_dir, here)
            print(json.dumps(rec))
            with open(os.path.join(here, f"cache_{label}.log"), "w") as f:
                f.write(text)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
