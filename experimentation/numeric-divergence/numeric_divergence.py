#!/usr/bin/env python
"""
Experiment I — where numba and Python disagree numerically, and what that means for money.

Tests the decider2 equivalence ladder (doc 02 §3.1) and the doc 05 §9 acceptance
criteria that demand agreement "exactly" / "matching a numpy reference exactly",
against REVIEW.md §7's counter-claim that exact equality is the wrong test for
floats, the right test for decisions, and that the tolerance policy is unspecified.

Five sections:
  E1  integer overflow          Python int (arbitrary precision) vs numba int64 (wraps)
  E2  rounding                  Python round() vs numba round() vs np.round
  E3  float drift / fastmath    ULP divergence and whether it flips a decision
  E4  money                     Decimal into a kernel; scaled-int64 vs float64 vs Decimal
  E5  division and NaN          does njit raise where Python raises; prange behaviour

Run:
    .venv/bin/python experimentation/numeric-divergence/numeric_divergence.py
    .venv/bin/python experimentation/numeric-divergence/numeric_divergence.py --quick
    .venv/bin/python experimentation/numeric-divergence/numeric_divergence.py --only E3,E4
    .venv/bin/python experimentation/numeric-divergence/numeric_divergence.py --json out.json

Nothing here installs, downloads or mutates anything. It only computes and prints.
Every number printed is measured in-process; no figure is copied from the docs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
import sys
import time
import warnings
from decimal import Decimal, ROUND_HALF_EVEN, ROUND_HALF_UP, getcontext, localcontext

import numpy as np
from numba import njit, prange

RESULTS: dict = {}
CENT = Decimal("0.01")


# ---------------------------------------------------------------- helpers


def hr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def sub(title: str) -> None:
    print()
    print("-- " + title)


def ulps_between(a: float, b: float) -> int:
    """Signed distance in representable float64 steps. NaN-safe-ish."""
    if a == b:
        return 0
    if math.isnan(a) or math.isnan(b):
        return -1
    ia = struct.unpack("<q", struct.pack("<d", a))[0]
    ib = struct.unpack("<q", struct.pack("<d", b))[0]
    if ia < 0:
        ia = -0x8000000000000000 - ia
    if ib < 0:
        ib = -0x8000000000000000 - ib
    return abs(ia - ib)


def ulps_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ia = a.view(np.int64).astype(object)
    ib = b.view(np.int64).astype(object)
    ia = np.where(a.view(np.int64) < 0, -0x8000000000000000 - a.view(np.int64), a.view(np.int64))
    ib = np.where(b.view(np.int64) < 0, -0x8000000000000000 - b.view(np.int64), b.view(np.int64))
    return np.abs(ia.astype(np.int64) - ib.astype(np.int64))


def med_time(fn, *args, warmups: int = 1, repeats: int = 5) -> float:
    for _ in range(warmups):
        fn(*args)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


# ================================================================ E1
# Integer overflow: Python int is arbitrary precision, numba int64 wraps.


@njit(cache=False)
def k_fixedpoint_compound(balance_cents, factor_scaled, scale, months):
    """Fixed-point compounding, the shape a numba-first pricing step would take."""
    b = balance_cents
    for _ in range(months):
        b = (b * factor_scaled) // scale
    return b


def py_fixedpoint_compound(balance_cents, factor_scaled, scale, months):
    b = balance_cents
    for _ in range(months):
        b = (b * factor_scaled) // scale
    return b


@njit(cache=False)
def k_sum_of_squares(x):
    acc = 0
    for i in range(x.shape[0]):
        acc += x[i] * x[i]
    return acc


def py_sum_of_squares(x):
    acc = 0
    for i in range(x.shape[0]):
        acc += int(x[i]) * int(x[i])
    return acc


def run_e1(quick: bool) -> dict:
    hr("E1 — INTEGER OVERFLOW (doc 02 §3.1: 'interpreted != stepped -> numba changed a step')")
    out: dict = {}

    sub("E1.a  fixed-point compounding, rate scale 1e12, 60 months")
    print("A rate held to 12 decimal places (scale 1e12) is not exotic; it is what you get")
    print("if you carry an APR/12 to full precision in fixed point.")
    rows = []
    for scale_pow in (9, 12):
        scale = 10**scale_pow
        factor = int(round((1 + 0.245 / 12) * scale))  # 24.5% p.a. nominal, monthly
        for rand in (1_000, 50_000, 100_000, 500_000):
            cents = rand * 100
            py = py_fixedpoint_compound(cents, factor, scale, 60)
            nb = int(k_fixedpoint_compound(cents, factor, scale, 60))
            rows.append(
                dict(scale=f"1e{scale_pow}", principal_rand=rand, python_cents=py,
                     njit_cents=nb, agree=bool(py == nb),
                     njit_rand=nb / 100.0, python_rand=py / 100.0)
            )
    print(f"{'scale':>6} {'principal R':>12} {'python (R)':>20} {'njit (R)':>24} {'agree':>6}")
    for r in rows:
        print(f"{r['scale']:>6} {r['principal_rand']:>12,} {r['python_rand']:>20,.2f} "
              f"{r['njit_rand']:>24,.2f} {str(r['agree']):>6}")
    out["fixedpoint"] = rows
    bad = [r for r in rows if not r["agree"]]
    print(f"\ndisagreements: {len(bad)} of {len(rows)}")
    if bad:
        thr = min(r["principal_rand"] for r in bad if r["scale"] == "1e12")
        print(f"at scale 1e12 the smallest principal tested that already diverges is R{thr:,}")
        # binary search the exact threshold at scale 1e12
        lo, hi = 1, 500_000
        scale = 10**12
        factor = int(round((1 + 0.245 / 12) * scale))
        while lo < hi:
            mid = (lo + hi) // 2
            if py_fixedpoint_compound(mid * 100, factor, scale, 60) != int(
                k_fixedpoint_compound(mid * 100, factor, scale, 60)
            ):
                hi = mid
            else:
                lo = mid + 1
        print(f"exact first-divergence principal at scale 1e12: R{lo:,}")
        out["fixedpoint_first_divergent_principal_rand"] = lo

    sub("E1.b  sum of squares over loan amounts in cents (variance / scorecard accumulator)")
    rng = np.random.default_rng(20260918)
    n = 1_000 if quick else 10_000
    amounts_cents = rng.integers(50_000_00, 1_000_000_00, size=n, dtype=np.int64)
    py = py_sum_of_squares(amounts_cents)
    nb = int(k_sum_of_squares(amounts_cents))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        np_scalar = np.int64(0)
        for v in amounts_cents[: min(n, 10_000)]:
            np_scalar = np_scalar + v * v
        np_warned = [str(x.message) for x in w][:1]
    print(f"n = {n:,} loans, amounts R50k..R1m held in cents")
    print(f"  python int accumulator : {py}")
    print(f"  njit   int64           : {nb}")
    print(f"  numpy  int64 scalar    : {int(np_scalar)}")
    print(f"  python == njit         : {py == nb}")
    print(f"  numpy  == njit         : {int(np_scalar) == nb}")
    print(f"  int64 max              : {np.iinfo(np.int64).max}")
    print(f"  python value / int64max: {py / np.iinfo(np.int64).max:,.2f}x")
    if np_warned:
        print(f"  numpy warned           : {np_warned[0]}")
    else:
        print("  numpy warned           : (no warning raised)")
    # how many rows before the accumulator wraps?
    big = rng.integers(50_000_00, 1_000_000_00, size=20_000, dtype=np.int64)
    onset = None
    for cut in range(500, 20_001, 500):
        if py_sum_of_squares(big[:cut]) != int(k_sum_of_squares(big[:cut])):
            lo, hi = cut - 500, cut
            while lo < hi:
                mid = (lo + hi) // 2
                if py_sum_of_squares(big[:mid]) != int(k_sum_of_squares(big[:mid])):
                    hi = mid
                else:
                    lo = mid + 1
            onset = lo
            break
    print(f"  batch size at which the int64 accumulator first wraps: "
          f"{onset if onset else '>20,000'} rows")
    out["sum_of_squares"] = dict(
        n=n, python=str(py), njit=str(nb), numpy_int64=str(int(np_scalar)),
        python_eq_njit=bool(py == nb), numpy_eq_njit=bool(int(np_scalar) == nb),
        numpy_warning=np_warned[0] if np_warned else None,
        overshoot_of_int64_max=py / np.iinfo(np.int64).max,
        overflow_onset_rows=onset,
    )

    sub("E1.c  which 'interpreted' do you mean?")
    print("The reference oracle's answer depends on where the value came from:")
    a = 2**40
    arr = np.array([2**40], dtype=np.int64)
    print(f"  python int   2**40 * 2**40        = {a * a}")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        v = arr[0] * arr[0]
        msgs = [str(x.message) for x in w]
    print(f"  numpy int64  arr[0] * arr[0]      = {int(v)}   warnings={msgs}")
    print(f"  njit  int64  2**40 * 2**40        = {int(k_mul(a, a))}")
    print("  => a step reading from a polars/numpy column already wraps in pure Python.")
    out["provenance"] = dict(python=str(a * a), numpy=str(int(v)), njit=str(int(k_mul(a, a))),
                             numpy_warnings=msgs)
    return out


@njit(cache=False)
def k_mul(a, b):
    return a * b


# ================================================================ E2
# Rounding.


@njit(cache=False)
def k_round1(x):
    return round(x)


@njit(cache=False)
def k_round2(x, n):
    return round(x, n)


@njit(cache=False)
def k_npround(x, n):
    return np.round(x, n)


@njit(cache=False)
def k_round2_array(x, n, out):
    for i in range(x.shape[0]):
        out[i] = round(x[i], n)


def run_e2(quick: bool) -> dict:
    hr("E2 — ROUNDING (REVIEW.md §4.4: \"Python's round() is banker's rounding and numba's "
       "is not guaranteed to match\")")
    out: dict = {}

    sub("E2.a  one-argument round() on exact ties")
    ties = [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 0.0, 1.0]
    rows = []
    for x in ties:
        py = round(x)
        nb = k_round1(x)
        npr = float(np.round(x))
        rows.append(dict(x=x, python=py, njit=int(nb), numpy=npr,
                         py_type=type(py).__name__, nb_type=type(nb).__name__,
                         agree=bool(py == nb)))
    print(f"{'x':>8} {'python':>10} {'type':>6} {'njit':>10} {'type':>6} {'np.round':>10} {'agree':>6}")
    for r in rows:
        print(f"{r['x']:>8} {r['python']:>10} {r['py_type']:>6} {r['njit']:>10} "
              f"{r['nb_type']:>6} {r['numpy']:>10} {str(r['agree']):>6}")
    out["ties_1arg"] = rows
    print(f"disagreements: {sum(1 for r in rows if not r['agree'])} of {len(rows)}")

    sub("E2.b  two-argument round(x, 2) — the rand-and-cents case")
    cases = [2.675, 2.665, 1.005, 0.125, 0.135, 2.5, 1.115, 8.835, 1.0049999999999999,
             123.455, 0.145, 2.345, 1234.565]
    rows = []
    for x in cases:
        py = round(x, 2)
        nb = float(k_round2(x, 2))
        npr = float(k_npround(x, 2))
        dec = float(Decimal(repr(x)).quantize(CENT, rounding=ROUND_HALF_UP))
        rows.append(dict(x=x, python=py, njit=nb, njit_npround=npr, decimal_half_up=dec,
                         py_eq_nb=bool(py == nb), cents_delta=round((nb - py) * 100)))
    print(f"{'x':>22} {'python':>12} {'njit':>12} {'np.round':>12} {'Dec HALF_UP':>12} {'same':>6}")
    for r in rows:
        print(f"{r['x']:>22} {r['python']:>12} {r['njit']:>12} {r['njit_npround']:>12} "
              f"{r['decimal_half_up']:>12} {str(r['py_eq_nb']):>6}")
    n_dis = sum(1 for r in rows if not r["py_eq_nb"])
    print(f"disagreements: {n_dis} of {len(rows)}")
    out["two_arg_cases"] = rows
    out["two_arg_disagreements"] = n_dis

    sub("E2.c  population scan: round(x, 2) over realistic rand amounts")
    rng = np.random.default_rng(7)
    n = 100_000 if quick else 1_000_000
    # amounts that land on a 3rd-decimal half tie often: k/1000 for k in a wide range
    k = rng.integers(1_000, 10_000_000, size=n, dtype=np.int64)
    xs = (k.astype(np.float64) * 5.0) / 1000.0  # multiples of 0.005 -> ties at 2dp
    out_nb = np.empty(n, dtype=np.float64)
    k_round2_array(xs[:8], 2, out_nb[:8])  # warm
    t_nb = med_time(lambda: k_round2_array(xs, 2, out_nb), repeats=3)
    t0 = time.perf_counter()
    out_py = np.fromiter((round(float(v), 2) for v in xs), dtype=np.float64, count=n)
    t_py = time.perf_counter() - t0
    diff = out_nb != out_py
    n_diff = int(diff.sum())
    cents_delta = np.round((out_nb - out_py) * 100).astype(np.int64)
    print(f"n = {n:,} amounts, all exact multiples of 0.005 (i.e. tie at 2 decimal places)")
    print(f"  rows where njit round(x,2) != python round(x,2) : {n_diff:,}  ({100*n_diff/n:.2f}%)")
    if n_diff:
        uniq, cnt = np.unique(cents_delta[diff], return_counts=True)
        print(f"  size of disagreement, in cents                   : "
              + ", ".join(f"{int(u):+d}c x{int(c):,}" for u, c in zip(uniq, cnt)))
        idx = np.flatnonzero(diff)[:5]
        for i in idx:
            print(f"    x={xs[i]!r:>22}  python={out_py[i]!r:>12}  njit={out_nb[i]!r:>12}")
    print(f"  total rand difference over the batch             : "
          f"R{(out_nb - out_py).sum():,.2f}")
    print(f"  njit round loop median  : {t_nb*1e3:.2f} ms   ({t_nb/n*1e9:.1f} ns/row)")
    print(f"  python round loop       : {t_py*1e3:.2f} ms   ({t_py/n*1e9:.1f} ns/row)")
    out["population_scan"] = dict(
        n=n, n_disagree=n_diff, pct=100 * n_diff / n,
        total_rand_delta=float((out_nb - out_py).sum()),
        njit_ns_per_row=t_nb / n * 1e9, python_ns_per_row=t_py / n * 1e9,
    )

    sub("E2.d  which rounding mode is each one actually doing?")
    py_ties = [round(v) for v in (0.5, 1.5, 2.5, 3.5)]
    nb_ties = [int(k_round1(v)) for v in (0.5, 1.5, 2.5, 3.5)]
    def classify(vals):
        if vals == [0, 2, 2, 4]:
            return "half-to-even (banker's)"
        if vals == [1, 2, 3, 4]:
            return "half-away-from-zero"
        return f"other: {vals}"
    print(f"  1-arg python round : {classify(py_ties)}")
    print(f"  1-arg njit   round : {classify(nb_ties)}")
    # 2-arg: use a value that is EXACTLY representable as a tie at 2dp: 0.125, 0.375
    exact_ties = [0.125, 0.375, 0.625, 0.875, 2.5e-2 * 1, 0.015625]
    py2 = [round(v, 2) for v in exact_ties]
    nb2 = [float(k_round2(v, 2)) for v in exact_ties]
    dec_he = [float(Decimal(repr(v)).quantize(CENT, rounding=ROUND_HALF_EVEN)) for v in exact_ties]
    dec_hu = [float(Decimal(repr(v)).quantize(CENT, rounding=ROUND_HALF_UP)) for v in exact_ties]
    print(f"{'x (exact 2dp tie)':>20} {'python':>10} {'njit':>10} {'Dec HALF_EVEN':>14} {'Dec HALF_UP':>12}")
    for v, a, b, c, d in zip(exact_ties, py2, nb2, dec_he, dec_hu):
        print(f"{v!r:>20} {a:>10} {b:>10} {c:>14} {d:>12}")
    out["modes"] = dict(python_1arg=classify(py_ties), njit_1arg=classify(nb_ties),
                        exact_ties=[dict(x=v, python=a, njit=b, dec_half_even=c, dec_half_up=d)
                                    for v, a, b, c, d in zip(exact_ties, py2, nb2, dec_he, dec_hu)])
    return out


# ================================================================ E3
# Float drift and fastmath.

NW = 16  # weights in the scorecard sum


def _mk_kernels(fastmath: bool):
    @njit(cache=False, fastmath=fastmath)
    def score_row(income, debt, util, arrears, term, rate_bp, w, bias):
        disp = income - debt
        r = rate_bp / 120000.0
        acc = 0.0
        acc += w[0] * math.log(1.0 + income)
        acc += w[1] * math.log(1.0 + debt)
        acc += w[2] * math.sqrt(util)
        acc += w[3] * (arrears / (1.0 + term))
        acc += w[4] * math.log1p(r)
        acc += w[5] * (debt / (1.0 + income))
        acc += w[6] * math.exp(-util)
        acc += w[7] * (disp / (1.0 + income))
        for j in range(8, NW):
            acc += w[j] * math.log(1.0 + income * (j - 7) / 13.0) / (1.0 + j)
        pd = 1.0 / (1.0 + math.exp(-(acc - bias)))
        # annuity instalment via exp/log (the shape a pricing step takes)
        f = math.exp(-term * math.log1p(r))
        inst = disp * 0.0 + (100000.0 * r) / (1.0 - f)
        return acc, pd, inst

    @njit(cache=False, fastmath=fastmath)
    def score_batch(income, debt, util, arrears, term, rate_bp, w, bias,
                    o_acc, o_pd, o_inst):
        for i in range(income.shape[0]):
            a, p, s = score_row(income[i], debt[i], util[i], arrears[i],
                                term[i], rate_bp[i], w, bias)
            o_acc[i] = a
            o_pd[i] = p
            o_inst[i] = s

    @njit(cache=False, fastmath=fastmath)
    def dti_pass(inst, disp, out):
        for i in range(inst.shape[0]):
            out[i] = (inst[i] / disp[i]) <= 0.35

    return score_row, score_batch, dti_pass


def py_score_row(income, debt, util, arrears, term, rate_bp, w, bias):
    disp = income - debt
    r = rate_bp / 120000.0
    acc = 0.0
    acc += w[0] * math.log(1.0 + income)
    acc += w[1] * math.log(1.0 + debt)
    acc += w[2] * math.sqrt(util)
    acc += w[3] * (arrears / (1.0 + term))
    acc += w[4] * math.log1p(r)
    acc += w[5] * (debt / (1.0 + income))
    acc += w[6] * math.exp(-util)
    acc += w[7] * (disp / (1.0 + income))
    for j in range(8, NW):
        acc += w[j] * math.log(1.0 + income * (j - 7) / 13.0) / (1.0 + j)
    pd = 1.0 / (1.0 + math.exp(-(acc - bias)))
    f = math.exp(-term * math.log1p(r))
    inst = disp * 0.0 + (100000.0 * r) / (1.0 - f)
    return acc, pd, inst


def run_e3(quick: bool) -> dict:
    hr("E3 — FLOAT DRIFT AND FASTMATH (doc 02 §3.1: \"1-ULP drift from log and from "
       "fastmath was observed\"; doc 05 §9.1 \"matching a numpy reference exactly\")")
    out: dict = {}
    rng = np.random.default_rng(31337)
    n = 50_000 if quick else 500_000
    income = rng.uniform(8_000, 120_000, n)
    debt = income * rng.uniform(0.05, 0.6, n)
    util = rng.uniform(0.0, 1.5, n)
    arrears = rng.integers(0, 6, n).astype(np.float64)
    term = rng.choice(np.array([12.0, 24.0, 36.0, 48.0, 60.0, 72.0]), n)
    rate_bp = rng.uniform(1200.0, 3100.0, n)
    w = np.random.default_rng(4242).normal(0.0, 0.4, NW)

    print(f"n = {n:,} rows, a {NW}-term scorecard sum with log/log1p/exp/sqrt/div,")
    print("then a logistic, then an annuity instalment via exp(-n*log1p(r)).")

    t0 = time.perf_counter()
    sr_off, sb_off, dti_off = _mk_kernels(False)
    sr_on, sb_on, dti_on = _mk_kernels(True)
    bufs = [np.empty(n) for _ in range(6)]
    sb_off(income[:4], debt[:4], util[:4], arrears[:4], term[:4], rate_bp[:4], w, 0.0,
           bufs[0][:4], bufs[1][:4], bufs[2][:4])
    sb_on(income[:4], debt[:4], util[:4], arrears[:4], term[:4], rate_bp[:4], w, 0.0,
          bufs[3][:4], bufs[4][:4], bufs[5][:4])
    t_compile = time.perf_counter() - t0
    print(f"compile time for both kernel families: {t_compile:.1f} s")
    # calibrate: put the logistic's steep region in the middle of the score density,
    # which is what a real scorecard does. Without this pd saturates and no cutoff
    # sits anywhere near the data.
    sb_off(income, debt, util, arrears, term, rate_bp, w, 0.0,
           bufs[0], bufs[1], bufs[2])
    bias = float(np.median(bufs[0]))
    print(f"calibration bias (median raw score) = {bias:.6f}")

    t_off = med_time(lambda: sb_off(income, debt, util, arrears, term, rate_bp, w, bias,
                                    bufs[0], bufs[1], bufs[2]), repeats=5)
    t_on = med_time(lambda: sb_on(income, debt, util, arrears, term, rate_bp, w, bias,
                                  bufs[3], bufs[4], bufs[5]), repeats=5)
    acc_off, pd_off, inst_off = bufs[0], bufs[1], bufs[2]
    acc_on, pd_on, inst_on = bufs[3], bufs[4], bufs[5]

    # pure-Python reference over a subsample (it is slow)
    m = min(n, 20_000)
    acc_py = np.empty(m)
    pd_py = np.empty(m)
    inst_py = np.empty(m)
    t0 = time.perf_counter()
    for i in range(m):
        a, p, s = py_score_row(income[i], debt[i], util[i], arrears[i],
                               term[i], rate_bp[i], w, bias)
        acc_py[i], pd_py[i], inst_py[i] = a, p, s
    t_py = time.perf_counter() - t0

    def report(name, a, b, k=None):
        k = k if k is not None else len(a)
        u = ulps_array(np.ascontiguousarray(a[:k]), np.ascontiguousarray(b[:k]))
        exact = int((u == 0).sum())
        rel = np.abs(a[:k] - b[:k]) / np.maximum(np.abs(b[:k]), 1e-300)
        row = dict(name=name, n=int(k), exact_equal=exact, pct_exact=100 * exact / k,
                   max_ulp=int(u.max()), p999_ulp=int(np.percentile(u, 99.9)),
                   max_rel=float(rel.max()))
        print(f"  {name:<34} exact={100*exact/k:6.2f}%  max_ulp={int(u.max()):>6}  "
              f"p99.9_ulp={int(np.percentile(u,99.9)):>5}  max_rel={rel.max():.3e}")
        return row

    sub("E3.a  ULP divergence, per output, vs the pure-Python reference")
    print(f"(python reference computed on the first {m:,} rows; it took {t_py:.2f} s)")
    rows = []
    rows.append(report("acc   python vs njit(fm=off)", acc_py, acc_off, m))
    rows.append(report("acc   python vs njit(fm=on) ", acc_py, acc_on, m))
    rows.append(report("pd    python vs njit(fm=off)", pd_py, pd_off, m))
    rows.append(report("pd    python vs njit(fm=on) ", pd_py, pd_on, m))
    rows.append(report("inst  python vs njit(fm=off)", inst_py, inst_off, m))
    rows.append(report("inst  python vs njit(fm=on) ", inst_py, inst_on, m))
    sub("E3.b  ULP divergence, fastmath off vs on (the same compiled program, one flag)")
    rows.append(report("acc   fm=off vs fm=on", acc_off, acc_on))
    rows.append(report("pd    fm=off vs fm=on", pd_off, pd_on))
    rows.append(report("inst  fm=off vs fm=on", inst_off, inst_on))
    out["ulp"] = rows
    print(f"\n  timing: fm=off {t_off*1e3:.2f} ms ({t_off/n*1e9:.1f} ns/row), "
          f"fm=on {t_on*1e3:.2f} ms ({t_on/n*1e9:.1f} ns/row), "
          f"speedup {t_off/t_on:.2f}x")
    out["timing"] = dict(n=n, fm_off_ms=t_off * 1e3, fm_on_ms=t_on * 1e3,
                          fm_off_ns_row=t_off / n * 1e9, fm_on_ns_row=t_on / n * 1e9,
                          speedup=t_off / t_on, compile_s=t_compile,
                          python_ref_rows=m, python_ref_s=t_py)

    sub("E3.c  does the drift ever flip a decision?")
    # (i) a natural, round cutoff on pd
    flips = {}
    for thr in (0.05, 0.10, 0.125, 0.15, 0.20, 0.25, 0.5):
        d_off = pd_off <= thr
        d_on = pd_on <= thr
        flips[thr] = int((d_off != d_on).sum())
    print("  pd <= threshold, fastmath off vs on:")
    for thr, f in flips.items():
        print(f"    threshold {thr:<6}  rows flipped: {f:,}  of {n:,}")
    # (ii) how many rows are within the drift band of *some* threshold
    differ = int((pd_off != pd_on).sum())
    print(f"  rows where pd differs at all (so SOME cutoff flips them): {differ:,} "
          f"({100*differ/n:.2f}%)")
    # (iii) a cutoff placed on an actually-produced value -> guaranteed flip
    cand = np.flatnonzero(pd_off != pd_on)
    constructed = None
    if cand.size:
        i = int(cand[0])
        t = max(pd_off[i], pd_on[i])
        constructed = dict(row=i, pd_off=float(pd_off[i]), pd_on=float(pd_on[i]),
                            threshold=float(t),
                            decision_off=bool(pd_off[i] < t), decision_on=bool(pd_on[i] < t))
        print(f"  constructed cutoff {t!r}: row {i} pd_off={pd_off[i]!r} pd_on={pd_on[i]!r} "
              f"-> strict '<' gives {pd_off[i] < t} vs {pd_on[i] < t}")
    # (iv) how sensitive is the flip count to WHERE the cutoff is put?
    lo_pd, hi_pd = float(pd_off.min()), float(pd_off.max())
    grid = np.linspace(lo_pd, hi_pd, 2001)
    gflips = np.array([int(((pd_off <= t) != (pd_on <= t)).sum()) for t in grid])
    gap_mass = float(np.abs(pd_off - pd_on).sum())
    exp_flips = gap_mass / (hi_pd - lo_pd)
    print(f"  over 2,001 cutoffs spanning the pd range [{lo_pd:.4g}, {hi_pd:.4g}]:")
    print(f"    cutoffs with >=1 flip: {int((gflips > 0).sum()):,}   max flips at any cutoff: "
          f"{int(gflips.max()):,}")
    print(f"    expected flips for a uniformly random cutoff: {exp_flips:.3e} rows per "
          f"batch of {n:,}")
    if exp_flips > 0:
        print(f"    i.e. about 1 flipped row every {1/exp_flips:.3e} batches of this size")
    out["pd_flips"] = dict(by_threshold={str(k): v for k, v in flips.items()},
                            rows_differing=differ, constructed=constructed,
                            grid_cutoffs_with_flip=int((gflips > 0).sum()),
                            grid_max_flips=int(gflips.max()),
                            expected_flips_random_cutoff=exp_flips)

    sub("E3.d  the affordability rule: instalment / disposable <= 0.35, on exact-boundary data")
    # Build rows that sit EXACTLY on the policy boundary by construction.
    nb = 200_000 if not quick else 20_000
    disp = rng.uniform(3_000.0, 60_000.0, nb)
    inst_b = disp * 0.35          # algebraically exactly on the boundary
    o_off = np.empty(nb, dtype=np.bool_)
    o_on = np.empty(nb, dtype=np.bool_)
    dti_off(inst_b[:4], disp[:4], o_off[:4])
    dti_on(inst_b[:4], disp[:4], o_on[:4])
    dti_off(inst_b, disp, o_off)
    dti_on(inst_b, disp, o_on)
    py_dec = np.fromiter(((inst_b[i] / disp[i]) <= 0.35 for i in range(nb)),
                         dtype=np.bool_, count=nb)
    recip = (inst_b * (1.0 / disp)) <= 0.35   # the reciprocal transform an optimiser may do
    print(f"  n = {nb:,} rows constructed to sit exactly on instalment = 0.35 * disposable")
    print(f"    python   x/y <= 0.35 -> approved: {int(py_dec.sum()):,} ({100*py_dec.mean():.2f}%)")
    print(f"    njit off x/y <= 0.35 -> approved: {int(o_off.sum()):,} ({100*o_off.mean():.2f}%)")
    print(f"    njit on  x/y <= 0.35 -> approved: {int(o_on.sum()):,} ({100*o_on.mean():.2f}%)")
    print(f"    x*(1/y)  <= 0.35     -> approved: {int(recip.sum()):,} ({100*recip.mean():.2f}%)")
    print(f"    python vs njit(off) differ : {int((py_dec != o_off).sum()):,}")
    print(f"    python vs njit(on)  differ : {int((py_dec != o_on).sum()):,}")
    print(f"    x/y vs x*(1/y)      differ : {int((py_dec != recip).sum()):,}")
    out["dti_boundary"] = dict(
        n=nb,
        approved_python=int(py_dec.sum()), approved_njit_off=int(o_off.sum()),
        approved_njit_on=int(o_on.sum()), approved_reciprocal=int(recip.sum()),
        differ_py_vs_off=int((py_dec != o_off).sum()),
        differ_py_vs_on=int((py_dec != o_on).sum()),
        differ_div_vs_recip=int((py_dec != recip).sum()),
    )

    sub("E3.e2  the bridge to money: the SAME drift, after rounding to cents")
    c_py = np.round(inst_py[:m] * 100).astype(np.int64)
    c_off = np.round(inst_off[:m] * 100).astype(np.int64)
    c_on = np.round(inst_on[:m] * 100).astype(np.int64)
    d1 = int((c_py != c_off).sum())
    d2 = int((c_py != c_on).sum())
    d3 = int((c_off != c_on).sum())
    print(f"  instalment rounded to whole cents, over {m:,} rows:")
    print(f"    python vs njit(fm=off) differ by >=1 cent : {d1:,} ({100*d1/m:.4f}%)")
    print(f"    python vs njit(fm=on)  differ by >=1 cent : {d2:,} ({100*d2/m:.4f}%)")
    print(f"    njit off vs njit on    differ by >=1 cent : {d3:,} ({100*d3/m:.4f}%)")
    print(f"    total rand difference (off vs on) over the batch: "
          f"R{(c_off - c_on).sum()/100:,.2f}")
    out["cents_after_drift"] = dict(n=m, py_vs_off=d1, py_vs_on=d2, off_vs_on=d3,
                                     rand_delta=float((c_off - c_on).sum() / 100))

    sub("E3.e  sum reassociation: does fastmath reorder a weighted sum?")
    @njit(cache=False, fastmath=False)
    def s_off(x):
        a = 0.0
        for i in range(x.shape[0]):
            a += x[i]
        return a

    @njit(cache=False, fastmath=True)
    def s_on(x):
        a = 0.0
        for i in range(x.shape[0]):
            a += x[i]
        return a

    ns = 1_000_000
    xs = rng.normal(0, 1e6, ns)
    xs[::2] *= 1e-9   # mixed magnitudes: reassociation shows
    a_off, a_on = float(s_off(xs)), float(s_on(xs))
    a_py = math.fsum(xs)
    a_npsum = float(np.sum(xs))
    print(f"  sequential sum, fastmath off : {a_off!r}")
    print(f"  sequential sum, fastmath on  : {a_on!r}")
    print(f"  numpy  np.sum (pairwise)     : {a_npsum!r}")
    print(f"  exact  math.fsum             : {a_py!r}")
    print(f"  ULP(off, on)      = {ulps_between(a_off, a_on)}")
    print(f"  ULP(off, np.sum)  = {ulps_between(a_off, a_npsum)}")
    print(f"  rel err vs fsum: off={abs(a_off-a_py)/abs(a_py):.3e}  "
          f"on={abs(a_on-a_py)/abs(a_py):.3e}  np.sum={abs(a_npsum-a_py)/abs(a_py):.3e}")
    out["reassociation"] = dict(n=ns, off=a_off, on=a_on, numpy_sum=a_npsum, fsum=a_py,
                                 ulp_off_on=ulps_between(a_off, a_on),
                                 ulp_off_npsum=ulps_between(a_off, a_npsum))

    sub("E3.f  the doc's specific claim: 1-ULP drift from log()")
    @njit(cache=False)
    def k_mathlog(x, o):
        for i in range(x.shape[0]):
            o[i] = math.log(x[i])

    @njit(cache=False, fastmath=True)
    def k_mathlog_fm(x, o):
        for i in range(x.shape[0]):
            o[i] = math.log(x[i])

    @njit(cache=False)
    def k_nplog(x):
        return np.log(x)

    nl = 200_000 if quick else 2_000_000
    xl = rng.uniform(1e-3, 1e7, nl)
    o1 = np.empty(nl); o2 = np.empty(nl)
    k_mathlog(xl[:4], o1[:4]); k_mathlog_fm(xl[:4], o2[:4]); k_nplog(xl[:4])
    k_mathlog(xl, o1); k_mathlog_fm(xl, o2)
    o3 = k_nplog(xl)
    o4 = np.log(xl)
    o5 = np.fromiter((math.log(float(v)) for v in xl[: min(nl, 200_000)]),
                     dtype=np.float64, count=min(nl, 200_000))
    k5 = len(o5)
    pairs = [("njit math.log vs numpy np.log(host)", o1, o4, nl),
             ("njit math.log vs njit np.log     ", o1, o3, nl),
             ("njit math.log vs python math.log ", o1[:k5], o5, k5),
             ("njit math.log vs njit fastmath log", o1, o2, nl)]
    for name, A, B, k in pairs:
        u = ulps_array(np.ascontiguousarray(A), np.ascontiguousarray(B))
        nz = int((u != 0).sum())
        print(f"  {name}  differ: {nz:>9,}/{k:,} ({100*nz/k:6.3f}%)  max_ulp={int(u.max())}")
        out.setdefault("log_probe", []).append(
            dict(pair=name.strip(), n=k, n_differ=nz, pct=100 * nz / k, max_ulp=int(u.max())))
    return out


# ================================================================ E4
# Money.


@njit(cache=False)
def k_schedule_int(balance_c, rate_ppm, inst_c, months, out_int_c, out_bal_c):
    """Amortisation in cents, int64, half-up. rate_ppm = monthly rate * 1e9."""
    b = balance_c
    SC = 1_000_000_000
    for m in range(months):
        interest = (b * rate_ppm + SC // 2) // SC
        principal = inst_c - interest
        b = b - principal
        out_int_c[m] = interest
        out_bal_c[m] = b
    return b


@njit(cache=False)
def k_schedule_int_ratio(balance_c, rate_num, rate_den, inst_c, months,
                         out_int_c, out_bal_c):
    """Same, but the rate is an exact rational (num/den), not a rounded scaled int."""
    b = balance_c
    for m in range(months):
        interest = (b * rate_num * 2 + rate_den) // (2 * rate_den)   # half-up, exact
        principal = inst_c - interest
        b = b - principal
        out_int_c[m] = interest
        out_bal_c[m] = b
    return b


@njit(cache=False)
def k_schedule_float(balance_r, rate, inst_r, months, out_int_r, out_bal_r):
    b = balance_r
    for m in range(months):
        interest = round(b * rate, 2)
        principal = inst_r - interest
        b = round(b - principal, 2)
        out_int_r[m] = interest
        out_bal_r[m] = b
    return b


@njit(cache=False)
def k_annuity_float(p, r, n):
    return (p * r) / (1.0 - math.exp(-n * math.log1p(r)))


@njit(cache=False)
def k_annuity_float_pow(p, r, n):
    return (p * r) / (1.0 - (1.0 + r) ** (-n))


def dec_annuity(p: Decimal, r: Decimal, n: int) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 60
        return (p * r) / (1 - (1 + r) ** (-n))


def dec_schedule(balance: Decimal, rate: Decimal, inst: Decimal, months: int):
    b = balance
    ints = []
    bals = []
    for _ in range(months):
        interest = (b * rate).quantize(CENT, rounding=ROUND_HALF_UP)
        principal = inst - interest
        b = b - principal
        ints.append(interest)
        bals.append(b)
    return b, ints, bals


def run_e4(quick: bool) -> dict:
    hr("E4 — MONEY (REVIEW.md §4.4: \"outputs are instalments, fees and rates that must "
       "reconcile to the cent... Decimal cannot enter a numba kernel\")")
    out: dict = {}

    sub("E4.a  can Decimal enter an njit kernel at all?")
    @njit(cache=False)
    def identity(x):
        return x

    probes = []
    for label, val in [("Decimal scalar", Decimal("1.50")),
                       ("numpy object array of Decimal",
                        np.array([Decimal("1.50"), Decimal("2.25")], dtype=object))]:
        try:
            r = identity(val)
            probes.append(dict(case=label, ok=True, result=str(r), error=None))
            print(f"  {label:<32} ACCEPTED -> {r}")
        except Exception as e:
            first = str(e).strip().splitlines()
            msg = " / ".join(x.strip() for x in first[:3] if x.strip())
            probes.append(dict(case=label, ok=False, result=None,
                                error=f"{type(e).__name__}: {msg[:180]}"))
            print(f"  {label:<32} REJECTED -> {type(e).__name__}: {msg[:150]}")
    out["decimal_entry"] = probes

    sub("E4.b  the annuity instalment: float64 kernel vs 60-digit Decimal, to the cent")
    rng = np.random.default_rng(99)
    ncfg = 2_000 if quick else 20_000
    principals = rng.integers(5_000, 500_000, ncfg) * 100          # cents
    apr_bp = rng.integers(500, 3150, ncfg)                          # basis points p.a.
    terms = rng.choice(np.array([6, 12, 24, 36, 48, 60, 72, 84]), ncfg)
    diffs_exp = []
    diffs_pow = []
    abs_err_cents = []      # |float value - exact value|, in cents
    dist_to_half = []       # distance of the exact value to a half-cent tie boundary
    getcontext().prec = 60
    t0 = time.perf_counter()
    for i in range(ncfg):
        p_c = int(principals[i])
        n = int(terms[i])
        r_dec = Decimal(int(apr_bp[i])) / Decimal(10000) / Decimal(12)
        exact = dec_annuity(Decimal(p_c) / 100, r_dec, n)
        ref = exact.quantize(CENT, rounding=ROUND_HALF_UP)
        r_f = float(r_dec)
        v1 = float(k_annuity_float(p_c / 100.0, r_f, float(n)))
        v2 = float(k_annuity_float_pow(p_c / 100.0, r_f, float(n)))
        exact_c = exact * 100
        abs_err_cents.append(abs(Decimal(repr(v1)) * 100 - exact_c))
        frac = exact_c - exact_c.to_integral_value(rounding="ROUND_FLOOR")
        dist_to_half.append(abs(frac - Decimal("0.5")))
        c_ref = int(ref * 100)
        c1 = int(Decimal(repr(v1)).quantize(CENT, rounding=ROUND_HALF_UP) * 100)
        c2 = int(Decimal(repr(v2)).quantize(CENT, rounding=ROUND_HALF_UP) * 100)
        diffs_exp.append(c1 - c_ref)
        diffs_pow.append(c2 - c_ref)
    t_ann = time.perf_counter() - t0
    de = np.array(diffs_exp)
    dp = np.array(diffs_pow)
    print(f"  {ncfg:,} loan configurations (R5k..R500k, 5.00%..31.50% p.a., 6..84 months)")
    print(f"  reference: Decimal prec=60, quantize to cent ROUND_HALF_UP  ({t_ann:.1f} s)")
    for label, d in (("exp/log1p form", de), ("(1+r)**-n form", dp)):
        nz = int((d != 0).sum())
        print(f"    {label:<16} differs from Decimal in {nz:,}/{ncfg:,} cases "
              f"({100*nz/ncfg:.2f}%), max |delta| = {int(np.abs(d).max())} cent(s), "
              f"delta distribution {dict(zip(*[x.tolist() for x in np.unique(d, return_counts=True)]))}")
    disagree_forms = int((de != dp).sum())
    print(f"    the two algebraically-identical float forms differ from each other in "
          f"{disagree_forms:,}/{ncfg:,} cases")
    max_err = max(abs_err_cents)
    min_gap = min(dist_to_half)
    print(f"    max |float - exact| over all configs : {float(max_err):.3e} cents")
    print(f"    closest any exact value came to a half-cent tie : {float(min_gap):.3e} cents")
    print(f"    margin (gap / error) = {float(min_gap / max_err) if max_err else float('inf'):.3g}x"
          f"  -> {'no tie-break flip is possible on this sample' if min_gap > max_err else 'a flip IS possible on this sample'}")
    out["annuity"] = dict(n=ncfg,
                           max_abs_err_cents=float(max_err),
                           min_dist_to_half_cent=float(min_gap),
                           exp_form_nonzero=int((de != 0).sum()),
                           pow_form_nonzero=int((dp != 0).sum()),
                           exp_max_abs_cents=int(np.abs(de).max()),
                           pow_max_abs_cents=int(np.abs(dp).max()),
                           forms_differ=disagree_forms,
                           seconds=t_ann)

    sub("E4.c  the amortisation schedule: scaled int64 vs float64, both vs Decimal")
    cases = [(250_000_00, 2450, 60), (75_000_00, 3100, 36), (1_000_000_00, 1150, 84),
             (12_500_00, 2875, 12), (500_000_00, 1975, 72)]
    rows = []
    for p_c, apr, n in cases:
        r_dec = Decimal(apr) / Decimal(10000) / Decimal(12)
        inst_dec = dec_annuity(Decimal(p_c) / 100, r_dec, n).quantize(
            CENT, rounding=ROUND_HALF_UP)
        inst_c = int(inst_dec * 100)
        # Decimal reference schedule
        b_ref, ints_ref, bals_ref = dec_schedule(Decimal(p_c) / 100, r_dec, inst_dec, n)
        ref_int_c = [int(x * 100) for x in ints_ref]
        ref_bal_c = [int(x * 100) for x in bals_ref]
        # int64 kernel
        rate_ppm = int((r_dec * Decimal(10**9)).quantize(Decimal(1), rounding=ROUND_HALF_UP))
        oi = np.zeros(n, dtype=np.int64)
        ob = np.zeros(n, dtype=np.int64)
        k_schedule_int(p_c, rate_ppm, inst_c, n, oi, ob)
        # float kernel
        oif = np.zeros(n)
        obf = np.zeros(n)
        k_schedule_float(p_c / 100.0, float(r_dec), float(inst_dec), n, oif, obf)
        # int64 kernel with the rate as an EXACT rational
        num, den = apr, 10000 * 12
        oir = np.zeros(n, dtype=np.int64)
        obr = np.zeros(n, dtype=np.int64)
        k_schedule_int_ratio(p_c, num, den, inst_c, n, oir, obr)

        int_bad = int(sum(1 for m in range(n) if int(oi[m]) != ref_int_c[m]))
        rat_bad = int(sum(1 for m in range(n) if int(oir[m]) != ref_int_c[m]))
        flt_bad = int(sum(1 for m in range(n)
                          if int(round(oif[m] * 100)) != ref_int_c[m]))
        int_tot = int(oi.sum())
        rat_tot = int(oir.sum())
        flt_tot = int(round(oif.sum() * 100))
        ref_tot = sum(ref_int_c)
        row = dict(principal_rand=p_c / 100, apr_bp=apr, months=n,
                   instalment_rand=float(inst_dec),
                   int_months_wrong=int_bad, ratio_months_wrong=rat_bad,
                   float_months_wrong=flt_bad,
                   total_interest_ref_c=ref_tot,
                   total_interest_int_c=int_tot, total_interest_rat_c=rat_tot,
                   total_interest_flt_c=flt_tot,
                   int_delta_c=int_tot - ref_tot, rat_delta_c=rat_tot - ref_tot,
                   flt_delta_c=flt_tot - ref_tot,
                   final_balance_ref_c=ref_bal_c[-1], final_balance_int_c=int(ob[-1]),
                   final_balance_rat_c=int(obr[-1]),
                   final_balance_flt_c=int(round(obf[-1] * 100)),
                   rate_exact_in_ppb=bool(
                       (r_dec * Decimal(10**9)) == (r_dec * Decimal(10**9)).to_integral_value()))
        rows.append(row)
    print(f"{'loan':>12} {'apr':>6} {'n':>4} {'inst R':>11} "
          f"{'int-ppb bad mo':>14} {'int-ratio bad':>14} {'float64 bad mo':>14} "
          f"{'rate exact?':>12}")
    for r in rows:
        print(f"R{r['principal_rand']:>11,.0f} {r['apr_bp']/100:>5.2f}% {r['months']:>4} "
              f"{r['instalment_rand']:>11,.2f} {r['int_months_wrong']:>14} "
              f"{r['ratio_months_wrong']:>14} {r['float_months_wrong']:>14} "
              f"{str(r['rate_exact_in_ppb']):>12}")
    print("  (bad mo = months whose interest charge differs from the Decimal reference)")
    print("  int-ppb   = rate stored as round(rate * 1e9), the obvious scaled-int design")
    print("  int-ratio = rate stored as the exact rational apr_bp / 120000")
    print(f"\n  total interest delta vs Decimal, over the whole loan (cents):")
    for r in rows:
        print(f"    R{r['principal_rand']:>10,.0f}/{r['months']}mo  int-ppb={r['int_delta_c']:+d}  "
              f"int-ratio={r['rat_delta_c']:+d}  float64={r['flt_delta_c']:+d}")
    print(f"\n  final balance after the last instalment (cents; nonzero = residual owed):")
    for r in rows:
        print(f"    R{r['principal_rand']:>10,.0f}/{r['months']}mo  Decimal={r['final_balance_ref_c']:+d}"
              f"  int-ppb={r['final_balance_int_c']:+d}"
              f"  int-ratio={r['final_balance_rat_c']:+d}"
              f"  float64={r['final_balance_flt_c']:+d}")
    out["schedule"] = rows

    sub("E4.d  fee allocation: does the split add back up?")
    fee_c = 1_207_99      # R1,207.99
    for k in (3, 7, 11, 13):
        share_f = (fee_c / 100.0) / k
        parts_f = [round(share_f, 2)] * k
        sum_f = int(round(sum(parts_f) * 100))
        base = fee_c // k
        rem = fee_c - base * k
        parts_i = [base + (1 if j < rem else 0) for j in range(k)]
        sum_i = sum(parts_i)
        print(f"    split R{fee_c/100:,.2f} {k} ways: float64 parts sum to "
              f"R{sum_f/100:,.2f} (residual {sum_f-fee_c:+d}c), "
              f"int64 largest-remainder sums to R{sum_i/100:,.2f} "
              f"(residual {sum_i-fee_c:+d}c)")
        out.setdefault("fee_split", []).append(
            dict(k=k, float_residual_c=sum_f - fee_c, int_residual_c=sum_i - fee_c))

    sub("E4.e  int64 headroom for a cents representation")
    mx = np.iinfo(np.int64).max
    print(f"  int64 max               = {mx:,}")
    print(f"  as cents                = R{mx/100:,.0f}")
    print(f"  safe for a product of two cent values up to R{math.isqrt(mx)/100:,.2f} each")
    print(f"  a cents x rate_ppb (1e9) product overflows above R{mx//10**9/100:,.2f}")
    print(f"  a cents x rate_ppm (1e6) product overflows above R{mx//10**6/100:,.2f}")
    out["headroom"] = dict(int64_max=int(mx), max_rand=mx / 100,
                            safe_product_rand=math.isqrt(mx) / 100,
                            ppb_overflow_rand=(mx // 10**9) / 100,
                            ppm_overflow_rand=(mx // 10**6) / 100)
    return out


# ================================================================ E5
# Division and NaN.


@njit(cache=False)
def k_fdiv(a, b):
    return a / b


@njit(cache=False)
def k_idiv(a, b):
    return a // b


@njit(cache=False)
def k_imod(a, b):
    return a % b


@njit(cache=False, error_model="numpy")
def k_fdiv_numpy_model(a, b):
    return a / b


@njit(cache=False, error_model="numpy")
def k_idiv_numpy_model(a, b):
    return a // b


@njit(cache=False)
def k_arr_div(a, b, out):
    for i in range(a.shape[0]):
        out[i] = a[i] / b[i]


@njit(cache=False, parallel=True)
def k_arr_div_par(a, b, out):
    for i in prange(a.shape[0]):
        out[i] = a[i] / b[i]


def attempt(label, fn, *args):
    try:
        v = fn(*args)
        return dict(case=label, raised=None, value=repr(v))
    except Exception as e:
        return dict(case=label, raised=type(e).__name__, value=str(e)[:90])


def run_e5(quick: bool) -> dict:
    hr("E5 — DIVISION AND NaN (doc 05 §6: \"A real runtime bug (ZeroDivisionError) must "
       "propagate identically in both compiled and fallback paths\")")
    out: dict = {}

    sub("E5.a  scalar division by zero, five ways to write the same 'reference'")
    rows = []
    def py_fdiv(a, b):
        return a / b
    def np_scalar_div(a, b):
        return np.float64(a) / np.float64(b)
    def np_arr_div(a, b):
        return (np.array([a]) / np.array([b]))[0]

    for lab, a, b in [("0.0 / 0.0", 0.0, 0.0), ("1.0 / 0.0", 1.0, 0.0),
                      ("-1.0 / 0.0", -1.0, 0.0)]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = [attempt("python float", py_fdiv, a, b),
                 attempt("numpy scalar", np_scalar_div, a, b),
                 attempt("numpy array ", np_arr_div, a, b),
                 attempt("njit default", k_fdiv, a, b),
                 attempt("njit numpy-model", k_fdiv_numpy_model, a, b)]
        print(f"  {lab}")
        for x in r:
            outcome = x["raised"] or x["value"]
            print(f"      {x['case']:<18} -> {outcome}")
        rows.append(dict(expr=lab, results=r))

    for lab, fn_py, fn_nb, fn_nb_np, a, b in [
        ("1 // 0", lambda a, b: a // b, k_idiv, k_idiv_numpy_model, 1, 0),
        ("1 % 0", lambda a, b: a % b, k_imod, None, 1, 0),
    ]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = [attempt("python int", fn_py, a, b),
                 attempt("numpy int64", fn_py, np.int64(a), np.int64(b)),
                 attempt("njit default", fn_nb, a, b)]
            if fn_nb_np is not None:
                r.append(attempt("njit numpy-model", fn_nb_np, a, b))
        print(f"  {lab}")
        for x in r:
            print(f"      {x['case']:<18} -> {x['raised'] or x['value']}")
        rows.append(dict(expr=lab, results=r))
    out["scalar"] = rows

    sub("E5.b  the same divide, in a row loop over a batch with one bad row")
    n = 100_000
    a = np.ones(n)
    b = np.full(n, 2.0)
    bad = n // 3
    b[bad] = 0.0
    o = np.full(n, -7.0)
    k_arr_div(a[:4], b[:4], o[:4])  # warm
    o[:] = -7.0
    res = attempt("njit serial loop", k_arr_div, a, b, o)
    print(f"  serial njit loop over {n:,} rows, row {bad:,} divides by zero")
    print(f"    -> {res['raised'] or res['value']}   ({res['value'][:60]})")
    written_before = int((o[:bad] != -7.0).sum())
    written_after = int((o[bad + 1:] != -7.0).sum())
    print(f"    output buffer: {written_before:,} of {bad:,} rows before the bad row were "
          f"written, {written_after:,} of {n-bad-1:,} after")
    out["serial_loop"] = dict(n=n, bad_row=bad, raised=res["raised"],
                               written_before=written_before, written_after=written_after)

    print(f"    value at the bad row: {o[bad]!r}")

    o2 = np.full(n, -7.0)
    t0 = time.perf_counter()
    k_arr_div_par(a[:64], b[:64] + 1.0, o2[:64])  # warm, no zero
    t_par_compile = time.perf_counter() - t0
    o2[:] = -7.0
    res_p = attempt("njit prange loop", k_arr_div_par, a, b, o2)
    written_p = int((o2 != -7.0).sum())
    n_inf = int(np.isinf(o2).sum())
    n_nan = int(np.isnan(o2).sum())
    print(f"  prange njit loop (parallel=True, compile {t_par_compile:.1f} s), SAME source,"
          f" same bad row")
    print(f"    -> raised: {res_p['raised']}   returned: {res_p['value'][:40]}")
    print(f"    output buffer: {written_p:,} of {n:,} rows written; "
          f"inf={n_inf}, nan={n_nan}")
    print(f"    value at the bad row: {o2[bad]!r}")
    same = (res_p["raised"] == res["raised"])
    print(f"    serial and prange agree on the error behaviour: {same}")
    print("    NOTE: doc 02 §3.3 says the serial/prange choice is made at warmup by")
    print("    measurement. That is a performance knob silently selecting error semantics.")
    out["prange_loop"] = dict(raised=res_p["raised"], written=written_p, n=n,
                                compile_s=t_par_compile, n_inf=n_inf, n_nan=n_nan,
                                bad_row_value=repr(o2[bad]),
                                agrees_with_serial=same)

    sub("E5.c  is it prange, or is it parallel=True?")
    @njit(cache=False, parallel=True)
    def k_arr_div_par_range(a, b, out):
        for i in range(a.shape[0]):
            out[i] = a[i] / b[i]

    o3 = np.full(n, -7.0)
    k_arr_div_par_range(a[:64], b[:64] + 1.0, o3[:64])
    o3[:] = -7.0
    res_pr = attempt("parallel=True + plain range", k_arr_div_par_range, a, b, o3)
    print(f"  parallel=True with a plain range() loop -> raised: {res_pr['raised']}, "
          f"bad row = {o3[bad]!r}")
    out["parallel_range"] = dict(raised=res_pr["raised"], bad_row_value=repr(o3[bad]))

    sub("E5.d  can a kernel name the offending row? (REVIEW.md §7 says no)")
    variants = {
        "raise ValueError('row ' + str(i))":
            "@njit\ndef f(i):\n    raise ValueError('row ' + str(i))\n",
        "raise ValueError(f'row {i}')":
            "@njit\ndef f(i):\n    raise ValueError(f'row {i}')\n",
        "raise inside a loop, naming the index":
            ("@njit\ndef f(x):\n"
             "    for i in range(x.shape[0]):\n"
             "        if x[i] == 0.0:\n"
             "            raise ValueError('bad row ' + str(i))\n"
             "    return 1\n"),
        "raise ValueError('constant')":
            "@njit\ndef f(i):\n    raise ValueError('constant')\n",
    }
    dr = []
    for label, body in variants.items():
        src = "from numba import njit\nimport numpy as np\n" + body
        ns: dict = {}
        arg = b if "loop" in label else 3
        try:
            exec(compile(src, "<dyn>", "exec"), ns)
            ns["f"](arg)
            res_txt, ok = "compiled, did not raise", True
        except Exception as e:
            first = [x.strip() for x in str(e).splitlines() if x.strip()][:2]
            ok = type(e).__name__ != "TypingError"
            res_txt = f"{type(e).__name__}: {' / '.join(first)[:110]}"
        print(f"  {label:<40} -> {res_txt}")
        dr.append(dict(variant=label, compiled=ok, result=res_txt))
    out["dynamic_raise"] = dr
    return out


# ================================================================ main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller n, same tests")
    ap.add_argument("--only", default="", help="comma list, e.g. E2,E4")
    ap.add_argument("--json", default="", help="write measurements to this path")
    args = ap.parse_args()

    want = {s.strip().upper() for s in args.only.split(",") if s.strip()} or {
        "E1", "E2", "E3", "E4", "E5"}

    print("numeric-divergence harness")
    print(f"python {sys.version.split()[0]}  numpy {np.__version__}", end="")
    try:
        import numba
        import llvmlite
        print(f"  numba {numba.__version__}  llvmlite {llvmlite.__version__}")
    except Exception:
        print()
    print(f"quick={args.quick}  sections={sorted(want)}")

    t_all = time.perf_counter()
    if "E1" in want:
        RESULTS["E1_integer_overflow"] = run_e1(args.quick)
    if "E2" in want:
        RESULTS["E2_rounding"] = run_e2(args.quick)
    if "E3" in want:
        RESULTS["E3_float_drift"] = run_e3(args.quick)
    if "E4" in want:
        RESULTS["E4_money"] = run_e4(args.quick)
    if "E5" in want:
        RESULTS["E5_division"] = run_e5(args.quick)
    RESULTS["_wall_seconds"] = time.perf_counter() - t_all

    hr("TOTAL")
    print(f"wall clock: {RESULTS['_wall_seconds']:.1f} s")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(RESULTS, fh, indent=2, default=str)
        print(f"measurements written to {args.json}")


if __name__ == "__main__":
    main()
