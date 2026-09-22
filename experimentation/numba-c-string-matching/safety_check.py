"""What does a malformed pattern, a bad index, a NULL handle, a bad offsets
array, or a pathological regex do? Each case runs in its OWN subprocess so
a crash, if any, is a recorded exit code (-11 = SIGSEGV), not an assertion
we could have got wrong. Demonstrated, not asserted."""
import os, subprocess, sys, pathlib, json, time
from results_io import record

HERE = pathlib.Path(__file__).resolve().parent
PY = sys.executable

CASES = {
 "malformed_pattern_at_compile": r'''
import strmatch as SM
try:
    SM.compile_pattern("^(dog")
except SM.PatternError as e:
    print("CAUGHT", e)
''',
 "bad_pattern_slot_caught_at_build": r'''
import kernels as K
from kernels import *
try:
    K.owner_shape((STR, STRB, 0, 0, 7, 4, 3, 0)).validate(2, 0, 0, 0, 1, 2, 0, 1, 0)
except ValueError as e:
    print("CAUGHT", e)
''',
 "bad_pattern_slot_unvalidated_kernel_guard": r'''
import numpy as np, polars as pl, strmatch as SM, kernels as K
from kernels import *
n=1000; rng=np.random.default_rng(0)
s=pl.Series("s", np.array(["dog","cat"])[rng.integers(0,2,n)]); offs,vals=K.string_buffers(s)
inp=K.empty_typed_inputs(n); inp.update(f64s=np.full((n,2),100.0), thr_f64=np.array([5.0,10.0]), str_tab=K.string_table([(offs, vals)]),
   handles=np.array([SM.compile_pattern("^dog")], np.int64), fn_table=np.array([SM.MATCH_ADDR],np.uint64))
prog=K.owner_shape((STR, STRB, 0, 0, 7, 4, 3, 0))   # slot 7 of a 1-entry handle table, NOT validated
out=np.zeros(n,np.int32); K.walk_typed(*[inp[k] for k in ("f64s","i64s","u8s","i32s","str_tab","thr_f64","thr_i64","handles","fn_table","masks")], *prog.arrays(), out)
print("CAUGHT rows flagged ERR_LEAF:", int((out==K.ERR_LEAF).sum()), "of", n)
''',
 "bad_string_column_index_unvalidated_kernel_guard": r'''
import numpy as np, polars as pl, strmatch as SM, kernels as K
from kernels import *
n=1000; rng=np.random.default_rng(0)
s=pl.Series("s", np.array(["dog","cat"])[rng.integers(0,2,n)]); offs,vals=K.string_buffers(s)
inp=K.empty_typed_inputs(n); inp.update(f64s=np.full((n,2),100.0), thr_f64=np.array([5.0,10.0]), str_tab=K.string_table([(offs, vals)]),
   handles=np.array([SM.compile_pattern("^dog")], np.int64), fn_table=np.array([SM.MATCH_ADDR],np.uint64))
prog=K.owner_shape((STR, STRB, 3, 0, 0, 4, 3, 0))   # string column 3; only 1 exists
out=np.zeros(n,np.int32); K.walk_typed(*[inp[k] for k in ("f64s","i64s","u8s","i32s","str_tab","thr_f64","thr_i64","handles","fn_table","masks")], *prog.arrays(), out)
print("CAUGHT rows flagged ERR_LEAF:", int((out==K.ERR_LEAF).sum()), "of", n)
''',
 "corrupt_offsets_past_values_end": r'''
import numpy as np, polars as pl, strmatch as SM, kernels as K
from kernels import *
n=1000; rng=np.random.default_rng(0)
s=pl.Series("s", np.array(["dog","cat"])[rng.integers(0,2,n)]); offs,vals=K.string_buffers(s)
offs=offs.copy(); offs[500:] += 10**9     # a corrupted offsets array pointing far past the buffer
inp=K.empty_typed_inputs(n); inp.update(f64s=np.full((n,2),100.0), thr_f64=np.array([5.0,10.0]), str_tab=K.string_table([(offs, vals)]),
   handles=np.array([SM.compile_pattern("^dog")], np.int64), fn_table=np.array([SM.MATCH_ADDR],np.uint64))
prog=K.owner_shape((STR, STRB, 0, 0, 0, 4, 3, 0))
out=np.zeros(n,np.int32); K.walk_typed(*[inp[k] for k in ("f64s","i64s","u8s","i32s","str_tab","thr_f64","thr_i64","handles","fn_table","masks")], *prog.arrays(), out)
print("CAUGHT rows flagged ERR_LEAF:", int((out==K.ERR_LEAF).sum()), "of", n)
''',
 "null_handle_in_table": r'''
import numpy as np, polars as pl, strmatch as SM, kernels as K
from kernels import *
n=1000; rng=np.random.default_rng(0)
s=pl.Series("s", np.array(["dog","cat"])[rng.integers(0,2,n)]); offs,vals=K.string_buffers(s)
inp=K.empty_typed_inputs(n); inp.update(f64s=np.full((n,2),100.0), thr_f64=np.array([5.0,10.0]), str_tab=K.string_table([(offs, vals)]),
   handles=np.array([0], np.int64), fn_table=np.array([SM.MATCH_ADDR],np.uint64))
prog=K.owner_shape((STR, STRB, 0, 0, 0, 4, 3, 0))
out=np.zeros(n,np.int32); K.walk_typed(*[inp[k] for k in ("f64s","i64s","u8s","i32s","str_tab","thr_f64","thr_i64","handles","fn_table","masks")], *prog.arrays(), out)
print("CAUGHT rows flagged ERR_LEAF:", int((out==K.ERR_LEAF).sum()), "of", n)
''',
 "freed_handle_reused": r'''
import numpy as np, polars as pl, strmatch as SM, kernels as K
from kernels import *
n=1000; rng=np.random.default_rng(0)
s=pl.Series("s", np.array(["dog","cat"])[rng.integers(0,2,n)]); offs,vals=K.string_buffers(s)
h=SM.compile_pattern("^dog"); SM.free_pattern(h)   # use-after-free: magic is zeroed on free
inp=K.empty_typed_inputs(n); inp.update(f64s=np.full((n,2),100.0), thr_f64=np.array([5.0,10.0]), str_tab=K.string_table([(offs, vals)]),
   handles=np.array([h], np.int64), fn_table=np.array([SM.MATCH_ADDR],np.uint64))
prog=K.owner_shape((STR, STRB, 0, 0, 0, 4, 3, 0))
out=np.zeros(n,np.int32); K.walk_typed(*[inp[k] for k in ("f64s","i64s","u8s","i32s","str_tab","thr_f64","thr_i64","handles","fn_table","masks")], *prog.arrays(), out)
print("CAUGHT rows flagged ERR_LEAF:", int((out==K.ERR_LEAF).sum()), "of", n, "(NOTE: this relies on the freed memory not being reused; it is a best-effort guard, not a guarantee)")
''',
 "garbage_handle_id_0xDEADBEEF": r'''
import numpy as np, polars as pl, strmatch as SM, kernels as K
from kernels import *
n=1000; rng=np.random.default_rng(0)
s=pl.Series("s", np.array(["dog","cat"])[rng.integers(0,2,n)]); offs,vals=K.string_buffers(s)
inp=K.empty_typed_inputs(n); inp.update(f64s=np.full((n,2),100.0), thr_f64=np.array([5.0,10.0]), str_tab=K.string_table([(offs, vals)]),
   handles=np.array([0xDEADBEEF], np.int64), fn_table=np.array([SM.MATCH_ADDR],np.uint64))   # a made-up address
prog=K.owner_shape((STR, STRB, 0, 0, 0, 4, 3, 0))
out=np.zeros(n,np.int32); K.walk_typed(*[inp[k] for k in ("f64s","i64s","u8s","i32s","str_tab","thr_f64","thr_i64","handles","fn_table","masks")], *prog.arrays(), out)
print("CAUGHT rows flagged ERR_LEAF:", int((out==K.ERR_LEAF).sum()), "of", n)
''',
 "negative_and_huge_handle_ids": r'''
import numpy as np, polars as pl, strmatch as SM, kernels as K
from kernels import *
n=1000; rng=np.random.default_rng(0)
s=pl.Series("s", np.array(["dog","cat"])[rng.integers(0,2,n)]); offs,vals=K.string_buffers(s)
for bad in (-1, 2**62, 4095):
    inp=K.empty_typed_inputs(n); inp.update(f64s=np.full((n,2),100.0), thr_f64=np.array([5.0,10.0]), str_tab=K.string_table([(offs, vals)]),
       handles=np.array([bad],np.int64), fn_table=np.array([SM.MATCH_ADDR],np.uint64))
    prog=K.owner_shape((STR, STRB, 0, 0, 0, 4, 3, 0))
    out=np.zeros(n,np.int32); K.walk_typed(*[inp[k] for k in ("f64s","i64s","u8s","i32s","str_tab","thr_f64","thr_i64","handles","fn_table","masks")], *prog.arrays(), out)
    print("CAUGHT id", bad, "-> rows flagged ERR_LEAF:", int((out==K.ERR_LEAF).sum()), "of", n)
''',
 "catastrophic_backtracking_pattern": r'''
import time, numpy as np, polars as pl, strmatch as SM, kernels as K
from kernels import *
print("match limit =", SM.get_match_limit())
n=100; s=pl.Series("s", ["a"*40+"b"]*n); offs,vals=K.string_buffers(s)
h=SM.compile_pattern(r"^(a+)+$")
inp=K.empty_typed_inputs(n); inp.update(f64s=np.full((n,2),100.0), thr_f64=np.array([5.0,10.0]), str_tab=K.string_table([(offs, vals)]),
   handles=np.array([h], np.int64), fn_table=np.array([SM.MATCH_ADDR],np.uint64))
prog=K.owner_shape((STR, STRB, 0, 0, 0, 4, 3, 0))
out=np.zeros(n,np.int32)
K.walk_typed(*[inp[k] for k in ("f64s","i64s","u8s","i32s","str_tab","thr_f64","thr_i64","handles","fn_table","masks")], *prog.arrays(), out)
t0=time.perf_counter(); K.walk_typed(*[inp[k] for k in ("f64s","i64s","u8s","i32s","str_tab","thr_f64","thr_i64","handles","fn_table","masks")], *prog.arrays(), out); t=time.perf_counter()-t0
print(f"CAUGHT rows flagged ERR_LEAF (PCRE2 match-limit hit): {int((out==K.ERR_LEAF).sum())} of {n}; {t/n*1e6:.0f} us per row at limit {SM.get_match_limit()} (default PCRE2 limit 10M gave 42 ms/row) -- polars/Rust regex is linear-time, no such row exists")
''',
 "numeric_feat_idx_out_of_range_unvalidated": r'''
import numpy as np, kernels as K
from kernels import *
n=1000
inp=K.empty_typed_inputs(n); inp.update(f64s=np.full((n,2),100.0), thr_f64=np.array([5.0,10.0]))
prog=K.owner_shape((CMP, F64, 9_000_000, GT, 0, 4, 3, 0))   # column 9M of a 2-column array, no validate()
out=np.zeros(n,np.int32); K.walk_typed(*[inp[k] for k in ("f64s","i64s","u8s","i32s","str_tab","thr_f64","thr_i64","handles","fn_table","masks")], *prog.arrays(), out)
print("SURVIVED (read garbage silently; numba boundscheck is off, same as decider2 today) result sum", int(out.sum()))
''',
}

def main():
    for name, code in CASES.items():
        env = dict(os.environ)
        t0 = time.perf_counter()
        p = subprocess.run([PY, "-c", code], capture_output=True, text=True, cwd=HERE, env=env, timeout=300)
        wall = time.perf_counter() - t0
        tail = (p.stdout.strip().splitlines() or [""])[-1][:400]
        err_tail = (p.stderr.strip().splitlines() or [""])[-1][:200]
        outcome = "segfault" if p.returncode == -11 else ("ok" if p.returncode == 0 else f"exit {p.returncode}")
        record("safety", case=name, returncode=p.returncode, outcome=outcome, stdout_tail=tail, stderr_tail=err_tail, wall_s=round(wall, 2))

if __name__ == "__main__":
    main()
