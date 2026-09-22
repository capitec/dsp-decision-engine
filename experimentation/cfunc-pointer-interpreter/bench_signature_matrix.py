"""Item 1, part 2 -- WHY does the raw-pointer mechanism's own baseline
(bench_call_overhead.py: ~7.3 ns/call for a float64(float64) cfunc) not
match §V's cited 2.4-3.84 ns/call, when both are "a C-ABI call from inside
njit, address known at runtime"?

Answer, found empirically below: it is NOT the dispatch mechanism (ctypes-
global vs a raw-address @intrinsic land within noise of each other in
EVERY cell of this matrix -- confirmed again here). It is the (argument
signature, callee origin) pair:

  int32(int32,int32), AOT .so (gcc)      ~1.9 ns   -- cheapest
  int32(int32,int32), JIT cfunc          ~8.6 ns   -- 4.5x the AOT cost
  float64(float64),   AOT .so (gcc)      ~7.2 ns
  float64(float64),   JIT cfunc          ~7.2 ns   -- matches AOT

§V's 2.4-3.84 ns figure was measured on a REAL Rust cdylib's
int32(int32,int32) `trivial`, which sits close to but not exactly at this
box's gcc-AOT-int32 number (~1.9 ns) -- both far cheaper than a JIT-compiled
callee of the same signature. There is no single "C-ABI call overhead"
constant; it depends on the argument width/type and, for narrow integer
args specifically, on whether the callee was numba/LLVM-JIT'd or AOT-
compiled. decider2 steps are float64-typed at the kernel boundary (doc 05
§1.5: only bool/int/float/str are declared, and int/float/bool all cross as
float64/int64/bool -- money as int64, everything else float64), so the
float64 row is the one that matters for the interpreter itself: **~7.2-7.3
ns/call, identical whether the pointer arrives as a numba global or as data
in a runtime-indexed array.**

Same trivial functions as bench_call_overhead.py's variants, cross-checked
against two independently AOT-compiled .so files (gcc -O3, checked in as
`c_stand_ins/`) so this file has no runtime dependency on the sibling
rust-cabi-in-kernel experiment's cargo build.
"""
from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from numba import cfunc, njit, types

from call_ptr import call_f64_f64, call_i32_i32

HERE = Path(__file__).resolve().parent
CSTAND = HERE / "c_stand_ins"
RESULTS_PATH = HERE / "results.jsonl"

N = 5_000_000
REPS = 5


def append_result(record: dict) -> None:
    record = {"experiment": "cfunc-pointer-interpreter", **record}
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _ensure_built(c_name: str, so_name: str) -> Path:
    so_path = CSTAND / so_name
    if not so_path.exists():
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-O3", "-o", str(so_path), str(CSTAND / c_name)],
            check=True,
        )
    return so_path


i32_so = _ensure_built("trivial_i32.c", "libtrivial_i32.so")
f64_so = _ensure_built("trivial_f64.c", "libtrivial_f64.so")

_gcc_i32 = ctypes.CDLL(str(i32_so))
_gcc_i32.trivial.argtypes = [ctypes.c_int32, ctypes.c_int32]
_gcc_i32.trivial.restype = ctypes.c_int32
gcc_i32_fn = _gcc_i32.trivial
gcc_i32_addr = np.uint64(ctypes.cast(gcc_i32_fn, ctypes.c_void_p).value)

_gcc_f64 = ctypes.CDLL(str(f64_so))
_gcc_f64.trivial_add.argtypes = [ctypes.c_double]
_gcc_f64.trivial_add.restype = ctypes.c_double
gcc_f64_fn = _gcc_f64.trivial_add
gcc_f64_addr = np.uint64(ctypes.cast(gcc_f64_fn, ctypes.c_void_p).value)


@cfunc(types.int32(types.int32, types.int32))
def jit_i32(a, b):
    return a + b


@cfunc(types.float64(types.float64))
def jit_f64(x):
    return x + 1.0


jit_i32_fn = jit_i32.ctypes
jit_i32_addr = np.uint64(jit_i32.address)
jit_f64_fn = jit_f64.ctypes
jit_f64_addr = np.uint64(jit_f64.address)


@njit(cache=False)
def global_i32(fn, n):
    t = 0
    for i in range(n):
        t += fn(i, 1)
    return t


@njit(cache=False)
def global_f64(fn, n):
    t = 0.0
    for i in range(n):
        t += fn(float(i))
    return t


@njit(cache=False)
def raw_i32(addr, n):
    t = 0
    for i in range(n):
        t += call_i32_i32(addr, np.int32(i), np.int32(1))
    return t


@njit(cache=False)
def raw_f64(addr, n):
    t = 0.0
    for i in range(n):
        t += call_f64_f64(addr, float(i))
    return t


CASES = [
    ("i32_aot_global", "int32", "aot(gcc)", "ctypes-global", global_i32, (gcc_i32_fn,)),
    ("i32_aot_raw", "int32", "aot(gcc)", "raw-address", raw_i32, (gcc_i32_addr,)),
    ("i32_jit_global", "int32", "jit(cfunc)", "ctypes-global", global_i32, (jit_i32_fn,)),
    ("i32_jit_raw", "int32", "jit(cfunc)", "raw-address", raw_i32, (jit_i32_addr,)),
    ("f64_aot_global", "float64", "aot(gcc)", "ctypes-global", global_f64, (gcc_f64_fn,)),
    ("f64_aot_raw", "float64", "aot(gcc)", "raw-address", raw_f64, (gcc_f64_addr,)),
    ("f64_jit_global", "float64", "jit(cfunc)", "ctypes-global", global_f64, (jit_f64_fn,)),
    ("f64_jit_raw", "float64", "jit(cfunc)", "raw-address", raw_f64, (jit_f64_addr,)),
]


def bench(key, sig, origin, mech, f, fargs):
    f(*fargs, 10)
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        f(*fargs, N)
        times.append(time.perf_counter() - t0)
    ns = [t / N * 1e9 for t in times]
    best, worst = min(ns), max(ns)
    print(f"{sig:8s} {origin:12s} {mech:14s} best={best:6.3f} ns  range={best:.3f}-{worst:.3f}")
    append_result({
        "item": "signature_matrix",
        "key": key,
        "signature": sig,
        "callee_origin": origin,
        "call_mechanism": mech,
        "n_calls": N,
        "reps": REPS,
        "ns_per_call_all": ns,
        "ns_per_call_best": best,
        "ns_per_call_worst": worst,
    })


def main() -> None:
    print(f"N={N:,} calls/rep, {REPS} reps, 2 rounds (interleaved to control for thermal/frequency drift)\n")
    for round_ in range(2):
        print(f"--- round {round_} ---")
        for key, sig, origin, mech, f, fargs in CASES:
            bench(f"{key}_r{round_}", sig, origin, mech, f, fargs)


if __name__ == "__main__":
    main()
