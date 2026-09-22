from __future__ import annotations
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from binding import (StringBuffers, compile_pattern, make_ptr_table, PTR_MATCH_ONE,
                     PTR_MATCH_ONE_UNPROTECTED, PTR_PANIC_PROTECTED, PTR_PANIC_UNPROTECTED)
from data import make_data
import kernels

mode = sys.argv[1]
print(f"[worker pid={os.getpid()}] {mode}", flush=True)
PT = make_ptr_table()
pid = np.int32(compile_pattern("^dog"))
df, _ = make_data(1000, "low", seed=1)
sb = StringBuffers(df["ft3"])
kind, prot = mode.rsplit("_", 1)
slot = PTR_MATCH_ONE if prot == "protected" else PTR_MATCH_ONE_UNPROTECTED

if kind == "oob":
    print("match_one(idx=10**9) from inside njit, n_strings=1000 ...", flush=True)
    r = kernels.match_one_from_kernel(PT, slot, pid, sb.offsets, sb.values, 10**9)
    print(f"returned {r}", flush=True)
elif kind == "short":
    # lie about the values buffer: hand over a 4-byte view while offsets still describe the full column
    short_values = sb.values[:4].copy()
    print(f"match_one(idx=999) with values_len=4 but offsets[999..1000]={sb.offsets[999]}..{sb.offsets[1000]} ...", flush=True)
    r = kernels.match_one_from_kernel(PT, slot, pid, sb.offsets, short_values, 999)
    print(f"returned {r}", flush=True)
elif kind == "synthetic":
    s = PTR_PANIC_PROTECTED if prot == "protected" else PTR_PANIC_UNPROTECTED
    print("deliberately_panic(1) from inside njit ...", flush=True)
    r = kernels.call_one_i32(PT, s, 1)
    print(f"returned {r}", flush=True)

good = kernels.match_one_from_kernel(PT, PTR_MATCH_ONE, pid, sb.offsets, sb.values, 0)
print(f"subsequent good call in the SAME process: match_one(idx=0) -> {good} (string={df['ft3'][0]!r})", flush=True)
print("worker exiting normally", flush=True)
