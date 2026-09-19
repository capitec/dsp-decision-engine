#!/usr/bin/env python3
"""
E-fusion-body-cost
==================

Re-runs doc 01 section 4c's fusion sweep (E5) with the dimension E5 held
constant made explicit: PER-STEP BODY COST.  Also adds a branchiness axis and a
direct check of doc 01 section 4b's mechanism claim ("vector IR values drop to 0
at M>=4") by inspecting .inspect_llvm() and .inspect_asm().

Model
-----
A "module" here is ONE step: one chunk of body source operating on a running
scalar `v` derived from two input columns x, y.

  fused : one njit kernel, one row loop, all M module bodies inlined in sequence.
  split : M njit kernels, each with its own row loop; the driver calls them in
          sequence, ping-ponging between two intermediate float64 buffers
          (this is the "one kernel per module" default of doc 02 section 1.2).

Both sides emit the SAME source text for each module body, with the same
per-module literal constants, so LLVM sees identical arithmetic.  Split kernel k
is compiled once and reused for every M that contains module k.

ratio = t_split / t_fused.   ratio > 1 means FUSION WINS.

Axes
----
  modules      1, 2, 3, 4, 5, 10, 20        (4 added: doc 4b's claimed cliff)
  rows         1, 1_000, 10_000, 100_000, 1_000_000
  body cost    trivial (1 add) | medium (~10 flops) | heavy (8-iter inner loop)
  branchiness  straight (no branches) | branchy (3 branch groups per module)

Run
---
  /path/to/.venv/bin/python fusion_body_cost.py            # full sweep
  ... --quick                                              # small smoke sweep
  ... --modules 1,2,3,4,5,10,20 --rows 1000,1000000        # custom
Results stream to stdout and are written to results.json next to this file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# body source generation
# --------------------------------------------------------------------------

BODY_KINDS = ("trivial", "medium", "heavy")
BRANCH_KINDS = ("straight", "branchy")

INNER_LOOP_N = 8  # iterations in the "heavy" body


def _consts(k: int):
    return dict(
        A=1.0 + 0.0007 * k,
        B=0.5 + 0.0003 * k,
        C=0.25 + 0.0011 * k,
        D=0.75 + 0.0005 * k,
        T1=0.33 + 0.01 * (k % 7),
        T2=0.50 + 0.01 * (k % 5),
        T3=1.00 + 0.02 * (k % 3),
    )


def gen_body(kind: str, branchy: str, k: int, ind: str) -> list[str]:
    """Source lines updating `v` from `xi`, `yi`.  Module index k varies the
    literal constants so the fused body cannot be collapsed by CSE."""
    c = _consts(k)
    L: list[str] = []
    a = lambda s: L.append(ind + s.format(**c))

    if branchy == "straight":
        if kind == "trivial":
            a("v = v + {A!r}")
        elif kind == "medium":
            a("v = v * {A!r} + xi * {B!r}")
            a("v = v + yi * {C!r} - {B!r}")
            a("v = v * {D!r} + (xi + yi) * {A!r}")
            a("v = v - xi * yi * {C!r}")
        elif kind == "heavy":
            a("t = v * {A!r} + {B!r}")
            a(f"for q in range({INNER_LOOP_N}):")
            a("    t = t * {A!r} + {B!r}")
            a("v = t + xi * {C!r}")
        else:
            raise ValueError(kind)
    elif branchy == "branchy":
        # three branch groups per module
        if kind == "trivial":
            a("if xi > {T1!r}:")
            a("    v = v + {A!r}")
            a("elif yi > {T2!r}:")
            a("    v = v - {B!r}")
            a("else:")
            a("    v = v * {D!r}")
            a("if v > {T3!r}:")
            a("    v = v + {C!r}")
            a("else:")
            a("    v = v - {C!r}")
            a("if xi + yi > {T2!r}:")
            a("    v = v * {A!r}")
        elif kind == "medium":
            a("if xi > {T1!r}:")
            a("    v = v * {A!r} + yi * {B!r}")
            a("elif yi > {T2!r}:")
            a("    v = v * {D!r} - xi * {C!r}")
            a("else:")
            a("    v = v + xi * yi * {A!r}")
            a("if v > {T3!r}:")
            a("    v = v * {B!r} + {C!r}")
            a("else:")
            a("    v = v - xi * {D!r}")
            a("if xi + yi > {T2!r}:")
            a("    v = v * {A!r} + {B!r}")
        elif kind == "heavy":
            a("if xi > {T1!r}:")
            a("    t = v")
            a(f"    for q in range({INNER_LOOP_N}):")
            a("        t = t * {A!r} + {B!r}")
            a("    v = t")
            a("elif yi > {T2!r}:")
            a("    t = v")
            a(f"    for q in range({INNER_LOOP_N}):")
            a("        t = t * {D!r} + {C!r}")
            a("    v = t")
            a("else:")
            a("    v = v * {A!r} + {B!r}")
            a("if v > {T3!r}:")
            a("    v = v + {C!r}")
            a("else:")
            a("    v = v - {C!r}")
        else:
            raise ValueError(kind)
    else:
        raise ValueError(branchy)
    return L


def body_lines(kind: str, branchy: str) -> int:
    return len(gen_body(kind, branchy, 0, ""))


# --------------------------------------------------------------------------
# kernel builders
# --------------------------------------------------------------------------

_NS_BASE = {"np": np, "njit": njit}


def _compile(src: str, name: str):
    g = dict(_NS_BASE)
    exec(compile(src, f"<gen:{name}>", "exec"), g)
    return g[name], src


def build_fused(kind: str, branchy: str, M: int):
    name = f"fused_{kind}_{branchy}_{M}"
    lines = [
        "@njit(cache=False, fastmath=False)",
        f"def {name}(x, y, out):",
        "    n = x.shape[0]",
        "    for i in range(n):",
        "        xi = x[i]",
        "        yi = y[i]",
        "        v = xi",
    ]
    for k in range(M):
        lines += gen_body(kind, branchy, k, " " * 8)
    lines.append("        out[i] = v")
    return _compile("\n".join(lines) + "\n", name)


def build_split_kernel(kind: str, branchy: str, k: int):
    name = f"split_{kind}_{branchy}_{k}"
    lines = [
        "@njit(cache=False, fastmath=False)",
        f"def {name}(x, y, inp, out):",
        "    n = x.shape[0]",
        "    for i in range(n):",
        "        xi = x[i]",
        "        yi = y[i]",
        "        v = inp[i]",
    ]
    lines += gen_body(kind, branchy, k, " " * 8)
    lines.append("        out[i] = v")
    return _compile("\n".join(lines) + "\n", name)


def make_split_driver(kernels, bufs):
    """Sequential chain, ping-ponging between two intermediate buffers."""
    ba, bb = bufs

    def driver(x, y, out):
        src = x
        n = len(kernels)
        for idx, kern in enumerate(kernels):
            if idx == n - 1:
                dst = out
            else:
                dst = ba if (idx % 2 == 0) else bb
            kern(x, y, src, dst)
            src = dst

    return driver


# --------------------------------------------------------------------------
# vectorisation inspection
# --------------------------------------------------------------------------

VEC_VAL_RE = re.compile(r"^\s*%\S+ = .*<\d+ x [^>]+>")
LLVM_VALUE_RE = re.compile(r"^\s*%\S+ = ")
LLVM_VEC_FP_RE = re.compile(
    r"^\s*%\S+ = (?:fadd|fsub|fmul|fdiv|fcmp|select|call)\b[^\n]*<\d+ x (?:double|i1)>"
)

_ARITH = r"(?:add|sub|mul|div|max|min|fmadd\d*|fmsub\d*|fnmadd\d*|fnmsub\d*)"
ASM_SD_RE = re.compile(r"^\s*v?" + _ARITH + r"sd\b")
ASM_PD_RE = re.compile(r"^\s*v" + _ARITH + r"pd\b")
WIDE_REG_RE = re.compile(r"\b[yz]mm\d+\b")


def inspect_vec(disp):
    """Vectorisation signals for a compiled dispatcher.

    Raw vector-value counts scale with code size, so the interesting number is
    the *fraction* of FP arithmetic that is packed (SIMD) rather than scalar.
    """
    out = {}
    try:
        ir = "\n".join(disp.inspect_llvm().values())
    except Exception as e:  # pragma: no cover
        return {"err": repr(e)}
    lines = ir.splitlines()
    vals = sum(1 for ln in lines if LLVM_VALUE_RE.match(ln))
    vecvals = sum(1 for ln in lines if VEC_VAL_RE.match(ln))
    vecfp = sum(1 for ln in lines if LLVM_VEC_FP_RE.match(ln))
    out["llvm_vector_values"] = vecvals
    out["llvm_vector_fp_ops"] = vecfp
    out["llvm_value_lines"] = vals
    out["llvm_vec_frac"] = round(vecvals / vals, 4) if vals else None
    try:
        asm = "\n".join(disp.inspect_asm().values())
    except Exception as e:  # pragma: no cover
        out["asm_err"] = repr(e)
        return out
    sd = pd_x = pd_w = 0
    for ln in asm.splitlines():
        if ASM_PD_RE.match(ln):
            if WIDE_REG_RE.search(ln):
                pd_w += 1
            else:
                pd_x += 1
        elif ASM_SD_RE.match(ln):
            sd += 1
    tot = sd + pd_x + pd_w
    out["asm_fp_scalar"] = sd
    out["asm_fp_packed_xmm"] = pd_x
    out["asm_fp_packed_wide"] = pd_w
    out["asm_fp_total"] = tot
    out["asm_simd_frac"] = round((pd_x + pd_w) / tot, 4) if tot else None
    out["asm_wide_frac"] = round(pd_w / tot, 4) if tot else None
    out["asm_lines"] = len(asm.splitlines())
    return out


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------

def timed(fn, args, budget=0.05, min_reps=5, max_reps=41):
    fn(*args)  # warm / compile
    t0 = time.perf_counter()
    fn(*args)
    dt = max(time.perf_counter() - t0, 1e-9)
    reps = int(max(min_reps, min(max_reps, budget / dt)))
    ts = []
    for _ in range(reps):
        a = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - a)
    return statistics.median(ts), reps


# --------------------------------------------------------------------------
# boundary-cost floor: what one extra split kernel costs before any body runs
# --------------------------------------------------------------------------

@njit(cache=False)
def _touch(inp, out):
    for i in range(inp.shape[0]):
        out[i] = inp[i] + 1.0


@njit(cache=False)
def _nothing(inp, out):
    return inp.shape[0]


def boundary_floor(Ns, budget):
    """Per-split-kernel floor = numba dispatch overhead + one load+store per row.
    This is the price doc 01 4c calls 'near-free'."""
    rows = []
    a = np.zeros(max(Ns))
    b = np.zeros(max(Ns))
    for n in Ns:
        ia, ib = a[:n], b[:n]
        t_call, _ = timed(_nothing, (ia, ib), budget, min_reps=21)
        t_touch, _ = timed(_touch, (ia, ib), budget, min_reps=21)
        per_row = (t_touch - t_call) / n
        rows.append({
            "rows": n,
            "dispatch_us": t_call * 1e6,
            "touch_us": t_touch * 1e6,
            "ns_per_row": per_row * 1e9,
            "gb_per_s": (16.0 / per_row / 1e9) if per_row > 0 else None,
        })
        print(f"  n={n:<8d} dispatch={t_call*1e6:7.3f}us  load+store loop="
              f"{t_touch*1e6:10.3f}us  -> {per_row*1e9:6.3f} ns/row  "
              f"({rows[-1]['gb_per_s'] or 0:.1f} GB/s)", flush=True)
    return rows


# --------------------------------------------------------------------------
# main sweep
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", default="1,2,3,4,5,10,20")
    ap.add_argument("--rows", default="1,1000,10000,100000,1000000")
    ap.add_argument("--bodies", default="trivial,medium,heavy")
    ap.add_argument("--branch", default="straight,branchy")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--time-budget", type=float, default=0.05)
    args = ap.parse_args(argv)

    if args.quick:
        args.modules, args.rows = "1,4,20", "1000,1000000"
        args.bodies, args.branch = "trivial,heavy", "straight,branchy"

    Ms = [int(v) for v in args.modules.split(",")]
    Ns = [int(v) for v in args.rows.split(",")]
    bodies = args.bodies.split(",")
    branches = args.branch.split(",")
    maxM = max(Ms)

    rng = np.random.default_rng(20260918)
    maxN = max(Ns)
    X = {n: rng.random(n) for n in Ns}
    Y = {n: rng.random(n) for n in Ns}
    OUT_F = {n: np.zeros(n) for n in Ns}
    OUT_S = {n: np.zeros(n) for n in Ns}
    BUF = {n: (np.zeros(n), np.zeros(n)) for n in Ns}

    results = []
    vecinfo = []
    t_start = time.perf_counter()

    print("--- boundary-cost floor (one extra split kernel, empty body) ---",
          flush=True)
    floor_rows = boundary_floor(Ns, args.time_budget)

    for body in bodies:
        for br in branches:
            tag = f"{body}/{br}"
            print(f"\n=== body={body} branch={br} "
                  f"({body_lines(body, br)} src lines/module) ===", flush=True)

            # compile the split kernels once (kernel k is the same for every M)
            t0 = time.perf_counter()
            split_k = []
            for k in range(maxM):
                fn, _ = build_split_kernel(body, br, k)
                fn(np.zeros(1) + 0.5, np.zeros(1) + 0.5,
                   np.zeros(1), np.zeros(1))  # force compile
                split_k.append(fn)
            t_split_compile = time.perf_counter() - t0
            print(f"  split: compiled {maxM} kernels in {t_split_compile:.2f}s",
                  flush=True)
            vi0 = {"body": body, "branch": br, "shape": "split_module_0",
                   "modules": 1, **inspect_vec(split_k[0])}
            vecinfo.append(vi0)
            print(f"  split module-0 kernel: llvm_vec_vals="
                  f"{vi0.get('llvm_vector_values')} "
                  f"asm scalar/xmm/wide={vi0.get('asm_fp_scalar')}/"
                  f"{vi0.get('asm_fp_packed_xmm')}/{vi0.get('asm_fp_packed_wide')} "
                  f"simd_frac={vi0.get('asm_simd_frac')}", flush=True)

            for M in Ms:
                t0 = time.perf_counter()
                fused, fsrc = build_fused(body, br, M)
                fused(np.zeros(1) + 0.5, np.zeros(1) + 0.5, np.zeros(1))
                t_fused_compile = time.perf_counter() - t0
                vi = {"body": body, "branch": br, "shape": "fused",
                      "modules": M, "src_lines": len(fsrc.splitlines()),
                      "compile_s": round(t_fused_compile, 3),
                      **inspect_vec(fused)}
                vecinfo.append(vi)
                print(f"  M={M:<3d} fused compile {t_fused_compile:6.2f}s  "
                      f"llvm_vec_vals={vi.get('llvm_vector_values'):<5d} "
                      f"llvm_vecfp={vi.get('llvm_vector_fp_ops'):<5d} "
                      f"asm scalar/xmm/wide="
                      f"{vi.get('asm_fp_scalar')}/{vi.get('asm_fp_packed_xmm')}/"
                      f"{vi.get('asm_fp_packed_wide')}  "
                      f"simd_frac={vi.get('asm_simd_frac')}",
                      flush=True)


                for n in Ns:
                    x, y = X[n], Y[n]
                    of, os_ = OUT_F[n], OUT_S[n]
                    ba, bb = BUF[n]
                    drv = make_split_driver(split_k[:M], (ba, bb))

                    fused(x, y, of)
                    drv(x, y, os_)
                    if np.array_equal(of, os_):
                        eq = "exact"
                    elif np.allclose(of, os_, rtol=1e-12, atol=0.0):
                        eq = "close"
                    else:
                        eq = "MISMATCH"

                    tf, rf = timed(fused, (x, y, of), args.time_budget)
                    ts, rs = timed(drv, (x, y, os_), args.time_budget)
                    ratio = ts / tf
                    row = {
                        "body": body, "branch": br, "modules": M, "rows": n,
                        "t_fused_s": tf, "t_split_s": ts, "ratio_split_over_fused": ratio,
                        "fused_ns_per_step_row": tf / (n * M) * 1e9,
                        "split_ns_per_step_row": ts / (n * M) * 1e9,
                        "equiv": eq, "reps_fused": rf, "reps_split": rs,
                    }
                    results.append(row)
                    print(f"      n={n:<8d} fused={tf*1e6:10.2f}us "
                          f"split={ts*1e6:10.2f}us  ratio={ratio:5.2f}  "
                          f"fused_ns/step={row['fused_ns_per_step_row']:6.3f} "
                          f"split_ns/step={row['split_ns_per_step_row']:6.3f} "
                          f"[{eq}]", flush=True)

                    with open(args.out, "w") as fh:
                        json.dump({"results": results, "vec": vecinfo,
                                   "floor": floor_rows}, fh, indent=1)

    elapsed = time.perf_counter() - t_start
    print(f"\ntotal sweep wall time {elapsed:.1f}s", flush=True)
    with open(args.out, "w") as fh:
        json.dump({"results": results, "vec": vecinfo, "floor": floor_rows,
                   "elapsed_s": elapsed,
                   "env": {"python": sys.version, "cpus": os.cpu_count()}},
                  fh, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
