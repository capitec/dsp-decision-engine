"""One process's worth of work for cache_check.py: import the kernels,
run the lazy typed walk (pointer table as ARGUMENT) and today's f64 walk,
print a checksum. Run twice in two processes with NUMBA_DEBUG_CACHE=1 and
a persistent NUMBA_CACHE_DIR; the second must load, not save.

Also compiles, in the same process, a kernel that captures the ctypes
symbol as a GLOBAL with cache=True -- to show the warning numba emits for
the mechanism §V used, in the same log, for contrast."""
import sys, time, warnings
t0 = time.perf_counter()
import numpy as np
import polars as pl
import strmatch as SM
import kernels as K
from kernels import *
from numba import njit
from numba.core.errors import NumbaWarning

n = 10_000
rng = np.random.default_rng(0)
s = pl.Series("s", np.array(["dog", "cat", "doge", "bird"])[rng.integers(0, 4, n)])
offs, vals = K.string_buffers(s)
h = SM.compile_pattern("^dog")
inp = K.empty_typed_inputs(n)
inp.update(f64s=np.column_stack([rng.uniform(0, 10, n), rng.uniform(0, 20, n)]), thr_f64=np.array([5.0, 10.0]),
           str_tab=K.string_table([(offs, vals)]), handles=np.array([h], np.int64), fn_table=np.array([SM.MATCH_ADDR], np.uint64))
prog = K.owner_shape((STR, STRB, 0, 0, 0, 4, 3, 0))
out = np.zeros(n, np.int32)
t1 = time.perf_counter()
K.walk_typed(*[inp[k] for k in ("f64s","i64s","u8s","i32s","str_tab","thr_f64","thr_i64","handles","fn_table","masks")], *prog.arrays(), out)
t2 = time.perf_counter()
feats = np.column_stack([inp["f64s"], s.str.contains("^dog").to_numpy().astype(np.float64)])
out2 = np.zeros(n, np.int32)
K.walk_f64(feats, np.array([5.0, 10.0]), *K.owner_shape((IS_TRUE, F64, 2, 0, 0, 4, 3, 0)).arrays_f64(), out2)
t3 = time.perf_counter()
assert (out == out2).all()
print(f"RESULT checksum={int(out.sum())} typed_first_call_ms={(t2-t1)*1e3:.1f} f64_first_call_ms={(t3-t2)*1e3:.1f} import_ms={(t1-t0)*1e3:.1f}", flush=True)

if "--with-global-capture" in sys.argv:
    matcher = SM.sm_match_c
    @njit(cache=True)
    def global_capture(handle, offs, vals, out):
        base = vals.ctypes.data
        for i in range(offs.shape[0] - 1):
            out[i] = matcher(handle, base + offs[i], offs[i + 1] - offs[i])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        global_capture(h, offs, vals, out)
        msgs = [str(x.message) for x in w if issubclass(x.category, NumbaWarning)]
    print("GLOBAL_CAPTURE_WARNINGS", msgs, flush=True)
