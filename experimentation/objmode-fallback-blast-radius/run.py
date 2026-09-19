"""EXPERIMENT B - can per-node fallback exist inside a fused njit driver?

Tests the claim in decider2 doc 05 s6 / doc 02 s3.2 that there are TWO
independent layers of graceful degradation, layer 1 being "per node ... fall
back to plain Python for that node only. Every other node is unaffected."

Everything printed is measured in-process. Nothing is estimated.

Run:
    .venv/bin/python experimentation/objmode-fallback-blast-radius/run.py
    .venv/bin/python experimentation/objmode-fallback-blast-radius/run.py --rows 500000 --repeats 7
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
import time
import traceback

import numpy as np
import numba
from numba import njit, objmode, prange
from numba.core import errors as nberr

# --------------------------------------------------------------------------
# timing helpers
# --------------------------------------------------------------------------

RESULTS: dict[str, dict] = {}


def bench(label, fn, repeats, warmups=1):
    """Warm up (first call includes compilation), then median of `repeats`."""
    t0 = time.perf_counter()
    out = fn()
    first_call_s = time.perf_counter() - t0
    for _ in range(warmups - 1):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    med = statistics.median(samples)
    RESULTS[label] = {
        "median_s": med,
        "min_s": min(samples),
        "max_s": max(samples),
        "first_call_s": first_call_s,
        "compile_s": max(first_call_s - med, 0.0),
        "out": out,
    }
    print(
        f"  {label:<46} median {med*1e3:9.3f} ms   "
        f"(min {min(samples)*1e3:8.3f}  max {max(samples)*1e3:8.3f})   "
        f"first call {first_call_s*1e3:9.1f} ms  => compile ~{max(first_call_s-med,0.0)*1e3:8.1f} ms"
    )
    return out


def rule(title):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


# --------------------------------------------------------------------------
# the four step functions of a toy record pipeline
# --------------------------------------------------------------------------
# node 1 is the one we will break.  bad_cheap() computes EXACTLY the same
# arithmetic as step_scale(), so (objmode driver) - (all-njit driver) isolates
# the objmode transition cost with no extra node work mixed in.


@njit
def step_scale(x):
    return x * 1.5 + 0.25


def bad_cheap(x):  # plain Python, identical maths to step_scale
    return x * 1.5 + 0.25


def bad_real(code):
    """A genuinely non-compilable node: regex over a formatted string.
    This is the `bad_node` shape used by experimentation/graph_poc/fallback_pipeline.py."""
    return 1.0 if re.match(r"^1", str(code)) else 0.0


@njit
def step_clip(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@njit
def step_logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


@njit
def step_bucket(x):
    if x < 0.25:
        return 0.0
    if x < 0.5:
        return 1.0
    if x < 0.75:
        return 2.0
    return 3.0


# --------------------------------------------------------------------------
# (a) all-njit fused driver
# --------------------------------------------------------------------------


@njit
def driver_alljit(x, out):
    for i in range(x.shape[0]):
        a = step_scale(x[i])
        b = step_clip(a, -5.0, 5.0)
        c = step_logistic(b)
        d = step_bucket(c)
        out[i] = c + d


@njit(parallel=True)
def driver_alljit_prange(x, out):
    for i in prange(x.shape[0]):
        a = step_scale(x[i])
        b = step_clip(a, -5.0, 5.0)
        c = step_logistic(b)
        d = step_bucket(c)
        out[i] = c + d


# --------------------------------------------------------------------------
# (b) njit driver, node 1 escapes to Python per row via objmode
# --------------------------------------------------------------------------


@njit
def driver_objmode_cheap(x, out):
    for i in range(x.shape[0]):
        v = x[i]
        with objmode(a="float64"):
            a = bad_cheap(v)
        b = step_clip(a, -5.0, 5.0)
        c = step_logistic(b)
        d = step_bucket(c)
        out[i] = c + d


@njit
def driver_objmode_real(x, codes, out):
    for i in range(x.shape[0]):
        cv = codes[i]
        with objmode(f="float64"):
            f = bad_real(cv)
        a = step_scale(x[i]) + f
        b = step_clip(a, -5.0, 5.0)
        c = step_logistic(b)
        d = step_bucket(c)
        out[i] = c + d


# --------------------------------------------------------------------------
# (e) njit driver, node 1 hoisted: ONE objmode call for the whole batch
# --------------------------------------------------------------------------


def bad_real_column(codes):
    out = np.empty(codes.shape[0], dtype=np.float64)
    for i in range(codes.shape[0]):
        out[i] = bad_real(codes[i])
    return out


@njit
def driver_objmode_hoisted(x, codes, out):
    with objmode(fcol="float64[:]"):
        fcol = bad_real_column(codes)
    for i in range(x.shape[0]):
        a = step_scale(x[i]) + fcol[i]
        b = step_clip(a, -5.0, 5.0)
        c = step_logistic(b)
        d = step_bucket(c)
        out[i] = c + d


# --------------------------------------------------------------------------
# (d) split: Python pre-pass outside the kernel, then a pure njit driver
# --------------------------------------------------------------------------


@njit
def driver_alljit_with_col(x, fcol, out):
    for i in range(x.shape[0]):
        a = step_scale(x[i]) + fcol[i]
        b = step_clip(a, -5.0, 5.0)
        c = step_logistic(b)
        d = step_bucket(c)
        out[i] = c + d


def split_real(x, codes, out):
    fcol = bad_real_column(codes)
    driver_alljit_with_col(x, fcol, out)


# --------------------------------------------------------------------------
# (c) whole driver in plain Python
# --------------------------------------------------------------------------

py_scale = step_scale.py_func
py_clip = step_clip.py_func
py_logistic = step_logistic.py_func
py_bucket = step_bucket.py_func


def driver_python_cheap(x, out):
    for i in range(x.shape[0]):
        a = bad_cheap(x[i])
        b = py_clip(a, -5.0, 5.0)
        c = py_logistic(b)
        d = py_bucket(c)
        out[i] = c + d


def driver_python_real(x, codes, out):
    for i in range(x.shape[0]):
        a = py_scale(x[i]) + bad_real(codes[i])
        b = py_clip(a, -5.0, 5.0)
        c = py_logistic(b)
        d = py_bucket(c)
        out[i] = c + d


# --------------------------------------------------------------------------
# PART 2 / 5 probes
# --------------------------------------------------------------------------


def probe(name, thunk):
    """Run `thunk`, report the exception class and whether NumbaError catches it."""
    try:
        thunk()
    except BaseException as exc:  # noqa: BLE001 - the point is to see what class it is
        cls = type(exc)
        is_numba = isinstance(exc, nberr.NumbaError)
        msg = str(exc).strip().splitlines()
        head = " | ".join(m.strip() for m in msg[:3] if m.strip())
        print(f"\n  [{name}]")
        print(f"    raised            : {cls.__module__}.{cls.__qualname__}")
        print(f"    isinstance NumbaError : {is_numba}")
        print(f"    mro               : {[c.__name__ for c in cls.__mro__[:5]]}")
        print(f"    message (head)    : {head[:400]}")
        return cls, is_numba, head
    print(f"\n  [{name}]  NO EXCEPTION - compiled/ran fine")
    return None, None, None


class Opaque:
    def __init__(self):
        self.v = 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    N, R = args.rows, args.repeats

    print(f"python  {sys.version.split()[0]}   numba {numba.__version__}   numpy {np.__version__}")
    print(f"rows={N}  repeats={R} (median reported)  threads={numba.config.NUMBA_NUM_THREADS}")

    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 2.0, N).astype(np.float64)
    codes = rng.integers(0, 1000, N).astype(np.int64)
    out = np.empty(N, np.float64)

    # ---------------------------------------------------------------- PART 1
    rule("PART 1 - does a fused njit driver (row loop over 4 njit'd steps) compile and run?")
    driver_alljit(x[:8], np.empty(8))
    print("  driver_alljit compiled in nopython mode:", driver_alljit.nopython_signatures)
    ref = np.empty(N)
    driver_alljit(x, ref)
    chk = np.empty(N)
    driver_python_cheap(x, chk)
    print(f"  njit vs python max abs diff over {N} rows: {np.max(np.abs(ref - chk)):.3e}")

    # ---------------------------------------------------------------- PART 2
    rule("PART 2 - break ONE node. What happens to the driver? (exact errors)")

    @njit
    def driver_calls_python(xx):
        s = 0.0
        for i in range(xx.shape[0]):
            s += bad_real(xx[i])
        return s

    probe("njit driver calls a plain-Python node", lambda: driver_calls_python(x[:4]))

    @njit
    def node_uses_re(c):
        return 1.0 if re.match(r"^1", str(c)) else 0.0

    probe("node itself @njit but uses re/str", lambda: node_uses_re(123))

    @njit
    def driver_calls_bad_njit_node(xx):
        s = 0.0
        for i in range(xx.shape[0]):
            s += node_uses_re(xx[i])
        return s

    probe(
        "njit driver calls an @njit node that cannot compile",
        lambda: driver_calls_bad_njit_node(codes[:4]),
    )

    @njit
    def node_builds_dict(v):
        d = {}
        d["a"] = v
        return d["a"]

    probe("node builds a plain dict", lambda: node_builds_dict(1.0))

    @njit
    def node_takes_object(o):
        return o.v

    probe("unsupported ARGUMENT type (plain Python object)", lambda: node_takes_object(Opaque()))

    probe("unsupported ARGUMENT type (dict of str->float)", lambda: node_takes_object({"v": 1.0}))

    @njit
    def node_str_list(v):
        xs = ["a", "b"]
        return v + len(xs)

    probe("node builds a list of str", lambda: node_str_list(1.0))

    # ---------------------------------------------------------------- PART 3+4
    rule("PART 3+4 - blast radius: all-njit vs one node via objmode vs whole driver in Python")

    print("\n  -- arithmetic-identical node (isolates the objmode transition cost) --")
    a_out = np.empty(N)
    b_out = np.empty(N)
    c_out = np.empty(N)
    bench("(a) all-njit fused driver", lambda: driver_alljit(x, a_out), R)
    bench("(a2) all-njit, prange", lambda: driver_alljit_prange(x, np.empty(N)), R)
    bench("(b) njit driver, 1 node via objmode PER ROW", lambda: driver_objmode_cheap(x, b_out), R)
    bench("(c) whole driver in plain Python", lambda: driver_python_cheap(x, c_out), R)
    print(f"  equality check   max|a-b| = {np.max(np.abs(a_out-b_out)):.3e}   "
          f"max|a-c| = {np.max(np.abs(a_out-c_out)):.3e}")

    print("\n  -- realistic non-compilable node (regex over a formatted string) --")
    r_hoist = np.empty(N)
    r_objm = np.empty(N)
    r_split = np.empty(N)
    r_py = np.empty(N)
    bench("(b2) njit driver, regex node via objmode PER ROW",
          lambda: driver_objmode_real(x, codes, r_objm), R)
    bench("(e) njit driver, regex node HOISTED to 1 objmode call",
          lambda: driver_objmode_hoisted(x, codes, r_hoist), R)
    bench("(d) Python pre-pass column + pure njit driver",
          lambda: split_real(x, codes, r_split), R)
    bench("(c2) whole driver in plain Python",
          lambda: driver_python_real(x, codes, r_py), R)
    print(f"  equality check   max|e-b2| = {np.max(np.abs(r_hoist-r_objm)):.3e}   "
          f"max|e-d| = {np.max(np.abs(r_hoist-r_split)):.3e}   "
          f"max|e-c2| = {np.max(np.abs(r_hoist-r_py)):.3e}")

    # per-row objmode transition cost
    tr_a = RESULTS["(a) all-njit fused driver"]["median_s"]
    tr_b = RESULTS["(b) njit driver, 1 node via objmode PER ROW"]["median_s"]
    tr_c = RESULTS["(c) whole driver in plain Python"]["median_s"]
    print(f"\n  objmode transition cost   = ({tr_b:.6f} - {tr_a:.6f}) / {N} "
          f"= {(tr_b-tr_a)/N*1e9:.0f} ns per row")
    print(f"  (b) vs (a)  slowdown      = {tr_b/tr_a:.1f}x")
    print(f"  (c) vs (a)  slowdown      = {tr_c/tr_a:.1f}x")
    print(f"  (b) vs (c)  objmode-per-row vs just-use-Python = {tr_b/tr_c:.2f}x")

    # objmode inside prange?
    @njit(parallel=True)
    def driver_objmode_prange(xx, o):
        for i in prange(xx.shape[0]):
            v = xx[i]
            with objmode(a="float64"):
                a = bad_cheap(v)
            o[i] = a

    probe("objmode inside prange", lambda: driver_objmode_prange(x[:1024], np.empty(1024)))

    # objmode in prange COMPILED - but is it correct, and does it actually parallelise?
    p_out = np.empty(N)
    bench("(b3) njit prange driver, 1 node via objmode PER ROW",
          lambda: driver_objmode_prange(x, p_out), R)
    expect = bad_cheap(x)
    print(f"  objmode-in-prange correctness: max abs diff vs numpy = "
          f"{np.max(np.abs(p_out - expect)):.3e}")
    tb = RESULTS["(b) njit driver, 1 node via objmode PER ROW"]["median_s"]
    tb3 = RESULTS["(b3) njit prange driver, 1 node via objmode PER ROW"]["median_s"]
    print(f"  objmode serial {tb*1e3:.1f} ms vs objmode prange {tb3*1e3:.1f} ms "
          f"=> prange speedup {tb/tb3:.2f}x on {numba.config.NUMBA_NUM_THREADS} threads "
          f"(all-njit prange speedup was "
          f"{RESULTS['(a) all-njit fused driver']['median_s']/RESULTS['(a2) all-njit, prange']['median_s']:.2f}x)")

    # ---------------------------------------------------------------- PART 5
    rule("PART 5 - does NumbaError catch cleanly, and does ZeroDivisionError propagate identically?")

    print("\n  -- NumbaError as the fallback trigger --")
    print(f"    issubclass(TypingError, NumbaError)      = {issubclass(nberr.TypingError, nberr.NumbaError)}")
    print(f"    issubclass(UnsupportedError, NumbaError) = {issubclass(nberr.UnsupportedError, nberr.NumbaError)}")

    def guarded():
        try:
            return driver_calls_python(x[:4])
        except nberr.NumbaError:
            return "caught by NumbaError"

    print(f"    `except NumbaError` around the bad driver -> {guarded()!r}")

    def guarded_obj():
        try:
            return node_takes_object(Opaque())
        except nberr.NumbaError:
            return "caught by NumbaError"

    try:
        print(f"    `except NumbaError` around bad ARG TYPE   -> {guarded_obj()!r}")
    except BaseException as exc:  # noqa: BLE001
        print(f"    `except NumbaError` around bad ARG TYPE   -> ESCAPED as "
              f"{type(exc).__module__}.{type(exc).__qualname__}")

    print("\n  -- ZeroDivisionError, compiled vs plain --")

    @njit
    def int_div(a, b):
        return a // b

    @njit
    def int_mod(a, b):
        return a % b

    @njit
    def float_div(a, b):
        return a / b

    @njit(fastmath=True)
    def float_div_fast(a, b):
        return a / b

    @njit(error_model="numpy")
    def float_div_numpymodel(a, b):
        return a / b

    @njit
    def float_div_in_loop(xx):
        s = 0.0
        for i in range(xx.shape[0]):
            s += 1.0 / (xx[i] - xx[i])
        return s

    @njit(parallel=True)
    def float_div_in_prange(xx):
        s = 0.0
        for i in prange(xx.shape[0]):
            s += 1.0 / (xx[i] - xx[i])
        return s

    cases = [
        ("int  a // 0", int_div, (7, 0)),
        ("int  a %  0", int_mod, (7, 0)),
        ("float a / 0.0", float_div, (7.0, 0.0)),
        ("float a / 0.0, fastmath=True", float_div_fast, (7.0, 0.0)),
        ("float a / 0.0, error_model='numpy'", float_div_numpymodel, (7.0, 0.0)),
    ]
    for name, fn, argv in cases:
        def run(fn=fn, argv=argv, compiled=True):
            f = fn if compiled else fn.py_func
            return f(*argv)

        for compiled in (True, False):
            tag = "njit " if compiled else "plain"
            try:
                v = run(compiled=compiled)
                print(f"    {name:<38} {tag}  -> returned {v!r}  (NO exception)")
            except BaseException as exc:  # noqa: BLE001
                print(f"    {name:<38} {tag}  -> {type(exc).__qualname__}: {str(exc)[:60]}")

    for name, fn in [("1.0/0.0 inside njit row loop", float_div_in_loop),
                     ("1.0/0.0 inside prange row loop", float_div_in_prange)]:
        for compiled in (True, False):
            tag = "njit " if compiled else "plain"
            f = fn if compiled else fn.py_func
            try:
                v = f(x[:64])
                print(f"    {name:<38} {tag}  -> returned {v!r}  (NO exception)")
            except BaseException as exc:  # noqa: BLE001
                print(f"    {name:<38} {tag}  -> {type(exc).__qualname__}: {str(exc)[:60]}")

    print("\n  -- would `except NumbaError` around a driver swallow a real bug? --")

    @njit
    def driver_with_bug(xx):
        s = 0
        for i in range(xx.shape[0]):
            s += 1 // (int(xx[i]) - int(xx[i]))
        return s

    try:
        driver_with_bug(x[:8])
    except nberr.NumbaError as exc:  # noqa: BLE001
        print(f"    ZeroDivisionError was caught by `except NumbaError`: {type(exc).__qualname__}  <-- BAD")
    except ZeroDivisionError as exc:
        print(f"    ZeroDivisionError escaped `except NumbaError` correctly: {exc}  <-- GOOD")

    rule("PART 6 - the SAME source, three execution modes: do they agree on division by zero?")

    def div_src(a, b):
        return a / b

    def idiv_src(a, b):
        return a // b

    jdiv = njit(div_src)
    jidiv = njit(idiv_src)

    @njit
    def div_serial(num, den):
        s = 0.0
        for i in range(num.shape[0]):
            s += num[i] / den[i]
        return s

    @njit(parallel=True)
    def div_parallel(num, den):
        s = 0.0
        for i in prange(num.shape[0]):
            s += num[i] / den[i]
        return s

    def outcome(fn, *argv):
        try:
            with np.errstate(all="ignore"):
                return f"returned {fn(*argv)!r}"
        except BaseException as exc:  # noqa: BLE001
            return f"{type(exc).__qualname__}: {str(exc)[:40]}"

    print("\n  scalar operands - the type of the operand decides, in the Python path only:")
    rows = [
        ("float(1.0) / float(0.0)", div_src, jdiv, (1.0, 0.0)),
        ("np.float64(1) / np.float64(0)", div_src, jdiv, (np.float64(1.0), np.float64(0.0))),
        ("int(1) // int(0)", idiv_src, jidiv, (1, 0)),
        ("np.int64(1) // np.int64(0)", idiv_src, jidiv, (np.int64(1), np.int64(0))),
    ]
    print(f"    {'expression':<34}{'plain Python':<46}{'njit'}")
    for name, py, jt, argv in rows:
        print(f"    {name:<34}{outcome(py, *argv):<46}{outcome(jt, *argv)}")

    print("\n  a row loop over numpy columns - exactly what a fallback driver does:")
    num = np.ones(64)
    den = np.zeros(64)

    def py_loop(numa, dena):
        s = 0.0
        for i in range(numa.shape[0]):
            s += numa[i] / dena[i]
        return s

    with np.errstate(all="ignore"):
        print(f"    {'plain Python driver':<34}{outcome(py_loop, num, den)}")
        print(f"    {'njit serial driver':<34}{outcome(div_serial, num, den)}")
        print(f"    {'njit prange driver':<34}{outcome(div_parallel, num, den)}")

    rule("SUMMARY TABLE (median ms over the same rows)")
    base = RESULTS["(a) all-njit fused driver"]["median_s"]
    print(f"  {'variant':<52}{'median ms':>12}{'vs (a)':>10}{'ns/row':>10}{'compile ms':>12}")
    for k, v in RESULTS.items():
        print(f"  {k:<52}{v['median_s']*1e3:>12.3f}{v['median_s']/base:>9.1f}x"
              f"{v['median_s']/N*1e9:>10.0f}{v['compile_s']*1e3:>12.0f}")


if __name__ == "__main__":
    main()
