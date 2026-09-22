"""Item 1 -- the decisive measurement, first. Bare call overhead for the
raw-address-through-a-pointer-table mechanism (`call_ptr.call_f64_f64`),
against the three mechanisms EXPERIMENTS.md already measured:

  njit -> njit (inlinable)             §V: 0.00 ns
  njit -> ctypes-global extern "C"     §V: 2.4-3.84 ns
  numba.typed.List[FunctionType]       §S: 28-41x codegen, 14-20x §Q's
                                            interpreter (no bare ns/call
                                            figure recorded standalone --
                                            §S measured it embedded in a
                                            tree walk, not as a microbench;
                                            cited, not re-derived here)

Three variants of the new mechanism are measured, because they are NOT the
same cost:

  (a) address baked in as a compile-time constant, single target -- the
      closest apples-to-apples with §V's ctypes-global number, isolating
      "is a cfunc pointer call, once LLVM knows the address, as cheap as
      an extern "C" call".
  (b) address read from a 1-entry array at a FIXED runtime index (k=0
      every iteration) -- isolates the cost of the array load + indirect
      call, with the branch predictor / BTB able to learn a single target.
  (c) address read from an N-entry array at a VARYING runtime index
      (round-robin) -- the realistic shape for an interpreter walking a
      tree/pipeline of heterogeneous steps, where consecutive calls target
      different functions and the indirect branch predictor cannot settle
      on one target.

Machine discipline: capped at 2GB RSS (`ulimit -v` set by the caller), N
capped at 5e6 to match the sibling C-ABI benchmark's scale, 5 reruns with
the observed range reported (not just the mean) -- §Q found variance is
the finding, not a footnote.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from numba import cfunc, njit, types

from call_ptr import call_f64_f64

HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "results.jsonl"

N = 5_000_000
REPS = 5


def append_result(record: dict) -> None:
    record = {"experiment": "cfunc-pointer-interpreter", **record}
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# The callee -- one trivial cfunc, reused by every variant below so the
# comparison isolates dispatch mechanism, not callee cost.
# ---------------------------------------------------------------------------

@cfunc(types.float64(types.float64))
def trivial_step(x):
    return x + 1.0


ADDR = np.uint64(trivial_step.address)

# A 3-entry pointer table, so variant (c) can round-robin across genuinely
# different callees (not the same address three times, which would let the
# indirect branch predictor cheat).
@cfunc(types.float64(types.float64))
def step_a(x):
    return x + 1.0

@cfunc(types.float64(types.float64))
def step_b(x):
    return x * 2.0 - 1.0

@cfunc(types.float64(types.float64))
def step_c(x):
    return x - 3.0


TABLE3 = np.array([step_a.address, step_b.address, step_c.address], dtype=np.uint64)
TABLE1 = np.array([trivial_step.address], dtype=np.uint64)


# ---------------------------------------------------------------------------
# Baselines, reproduced here (cheap, ~instant) rather than only cited, so
# this file's own numbers are internally comparable on the SAME machine,
# SAME run, SAME numba/llvmlite build as the new mechanism -- §V's numbers
# were measured on 2026-09-21 on this box already, but a fresh baseline
# costs nothing and removes any doubt about run-to-run machine drift.
# ---------------------------------------------------------------------------

@njit(cache=False)
def via_njit_direct(n):
    t = 0.0
    for i in range(n):
        t += _inline_add(float(i), 1.0)
    return t


@njit(cache=False)
def _inline_add(a, b):
    return a + b


# --- (a) compile-time-constant address, single target ---
_ADDR_CONST = int(ADDR)

@njit(cache=False)
def via_const_addr(n):
    t = 0.0
    for i in range(n):
        t += call_f64_f64(np.uint64(_ADDR_CONST), float(i))
    return t


# --- (b) 1-entry table, fixed index ---
@njit(cache=False)
def via_table_fixed(table, n):
    t = 0.0
    for i in range(n):
        t += call_f64_f64(table[0], float(i))
    return t


# --- (c) 3-entry table, round-robin index (varying target) ---
@njit(cache=False)
def via_table_varying(table, n):
    t = 0.0
    k = table.shape[0]
    for i in range(n):
        t += call_f64_f64(table[i % k], float(i))
    return t


VARIANTS = [
    ("njit -> njit (inlinable)", lambda: via_njit_direct(N), "baseline"),
    ("njit -> cfunc, const address", lambda: via_const_addr(N), "cfunc_const_addr"),
    ("njit -> cfunc, table[0] fixed idx", lambda: via_table_fixed(TABLE1, N), "cfunc_table_fixed"),
    ("njit -> cfunc, table[i%3] varying idx", lambda: via_table_varying(TABLE3, N), "cfunc_table_varying"),
]


def time_variant(name, thunk, key):
    thunk()  # warm up / trigger compile, not timed
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        thunk()
        times.append(time.perf_counter() - t0)
    ns_per_call = [t / N * 1e9 for t in times]
    best = min(ns_per_call)
    worst = max(ns_per_call)
    print(f"{name:42s} best={best:6.3f} ns/call  range={best:.3f}-{worst:.3f}  all={['%.3f' % v for v in ns_per_call]}")
    append_result({
        "item": "call_overhead",
        "variant": key,
        "name": name,
        "n_calls": N,
        "reps": REPS,
        "ns_per_call_all": ns_per_call,
        "ns_per_call_best": best,
        "ns_per_call_worst": worst,
    })


def main() -> None:
    print(f"N={N:,} calls/rep, {REPS} reps, best-of shown, full range in results.jsonl\n")
    for name, thunk, key in VARIANTS:
        time_variant(name, thunk, key)
    print()
    print("Reference (EXPERIMENTS.md, measured on this box, same day, not re-derived):")
    print("  §V  njit -> njit                         0.00 ns/call")
    print("  §V  njit -> ctypes-global extern \"C\"      2.4-3.84 ns/call")
    print("  §S  numba.typed.List[FunctionType]        28-41x codegen; 14-20x §Q's array")
    print("      interpreter (measured embedded in a tree walk, no bare ns/call figure)")


if __name__ == "__main__":
    main()
