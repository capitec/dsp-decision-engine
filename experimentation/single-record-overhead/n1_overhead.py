"""
EXPERIMENT N1 -- the single-record ("score()") framework overhead budget.

Doc 01 S6.1's reweighted priority: the single-record path is PRIMARY (most
invocations send one record), its budget is 20-100 ms/record, and the
compiled kernel answers one record in ~1 us -- ~0.005% of a 20 ms budget.
"Do not benchmark the kernel." This harness times what decider2 itself adds
AROUND the kernel for one score() call at the verified realistic width
(400 in / 633 out, mixed f8/i8/bool -- doc 01 S4d / experiments J/J2), broken
into six phases:

  1. accept   -- the request arrives (kwargs-unpack vs a plain dict, doc 02
                 S3.5's own two options: "kwargs or a dict of 400 values")
  2. validate -- bind params into the pydantic-validated NamedTuple bundle
                 (doc 03 S4: "one validated bundle per invocation")
  3. marshal  -- assemble a 1-row record array for the kernel, incl. alloc
  4. dispatch -- call the compiled kernel on that 1 row (NOT benchmarked for
                 its own sake -- included only so phases 1-3 and 5-6 can be
                 read against a fixed, known-cheap reference point)
  5. readback -- pull the 633 outputs back out as Python scalars
  6. assemble -- build the response dict

Doc 02 S3.5: "Affordability.score(net_income=42000.0, expenses=18000.0, ...,
params=p) -- bypasses polars entirely." So unlike experiments A/J/J2, there
is NO polars boundary phase here -- that is a deliberate omission, not an
oversight, and is why this harness does not call Series._get_buffers() at
all.

Reused from prior harnesses (not rewritten):
  - shape constants (400 in: 300 f8/80 i8/20 bool; 633 out: 380 f8/158 i8/95
    bool) and the record-in/record-out njit driver codegen (_HEADER,
    gen_record_out_source) -- verbatim from
    experimentation/output-writeback-convention/writeback.py. This is the
    SAME kernel shape E9/J measured (E9: 0.99 us/row; J: 942.2 ns/row @N=1).
  - writing the generated driver to a real .py file and importing via
    spec_from_file_location (never exec(), per doc 05 S4.1) -- pattern
    reused from ruleset-compile-latency/run.py:load_source, reused again in
    writeback.py:load_source. This applies ONLY to the njit driver -- the
    orchestration code below (accept/validate/marshal/readback/assemble) is
    plain Python with no numba caching hazard, so it is defined directly in
    this file, unlike the kernel.
  - elapsed()/log() progress-clock pattern -- prange-crossover/prange_crossover.py
  - peak-RSS-via-resource.getrusage pattern, renamed rss_mb() per the task
    brief -- chunked-writeback-at-scale/measure_variant.py:peak_rss_kb
    (same resource.getrusage(RUSAGE_SELF).ru_maxrss call, converted to MB)

NOT reused: polars extraction/writeback (doc 02 S3.5 bypasses polars, so
there is nothing analogous to experiment J's extract()/writeback_record()
in this harness's timed path).

Run:
    /path/to/.venv/bin/python n1_overhead.py                # full run
    /path/to/.venv/bin/python n1_overhead.py --quick         # smoke test

Writes results.jsonl (one line per measurement, flushed immediately) and
results_summary.json (final rollup) next to itself.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import platform
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import numba
import pydantic
from pydantic import BaseModel, Field
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
T0 = time.perf_counter()


def elapsed() -> float:
    return time.perf_counter() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


def rss_mb() -> float:
    # Reused pattern: chunked-writeback-at-scale/measure_variant.py:peak_rss_kb
    # (resource.getrusage(RUSAGE_SELF).ru_maxrss is a Linux high-water-mark
    # in KB, lifetime-peak for this process), converted to MB here.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def jsonl_append(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()


RESULTS_JSONL = HERE / "results.jsonl"


# ---------------------------------------------------------------------------
# Shape: 400 inputs / 633 outputs, mixed f8/i8/bool -- doc 01 S4d's verified
# realistic width, held IDENTICAL to experiments J/J2 so this is comparable.
# Reused verbatim from output-writeback-convention/writeback.py.
# ---------------------------------------------------------------------------
IN_F8, IN_I8, IN_BOOL = 300, 80, 20
N_IN = IN_F8 + IN_I8 + IN_BOOL
assert N_IN == 400

OUT_F8, OUT_I8, OUT_BOOL = 380, 158, 95
N_OUT = OUT_F8 + OUT_I8 + OUT_BOOL
assert N_OUT == 633

IN_F8_NAMES = [f"i_f8_{k}" for k in range(IN_F8)]
IN_I8_NAMES = [f"i_i8_{k}" for k in range(IN_I8)]
IN_B_NAMES = [f"i_b_{k}" for k in range(IN_BOOL)]
IN_ALL_NAMES = IN_F8_NAMES + IN_I8_NAMES + IN_B_NAMES

OUT_F8_NAMES = [f"o_f8_{k}" for k in range(OUT_F8)]
OUT_I8_NAMES = [f"o_i8_{k}" for k in range(OUT_I8)]
OUT_B_NAMES = [f"o_b_{k}" for k in range(OUT_BOOL)]
OUT_ALL_NAMES = OUT_F8_NAMES + OUT_I8_NAMES + OUT_B_NAMES

IN_DTYPE = np.dtype(
    [(n, "f8") for n in IN_F8_NAMES] + [(n, "i8") for n in IN_I8_NAMES] + [(n, "?") for n in IN_B_NAMES]
)
OUT_REC_DTYPE = np.dtype(
    [(n, "f8") for n in OUT_F8_NAMES] + [(n, "i8") for n in OUT_I8_NAMES] + [(n, "?") for n in OUT_B_NAMES]
)

COEF_F8 = np.array([1.0 + 0.001 * j for j in range(OUT_F8)], dtype=np.float64)


def make_request(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    d = {}
    for name in IN_F8_NAMES:
        d[name] = float(rng.random())
    for name in IN_I8_NAMES:
        d[name] = int(rng.integers(-1000, 1000))
    for name in IN_B_NAMES:
        d[name] = bool(rng.random() < 0.5)
    return d


REQUEST_DICT = make_request()


# ---------------------------------------------------------------------------
# Phase 2 -- params: pydantic-validated bundle -> fixed-type NamedTuple.
# Doc 03 S4: "One validated bundle per invocation. Ordinary pydantic --
# validators are the point." / "Under the hood the model becomes a
# NamedTuple whose type is fixed."
# ---------------------------------------------------------------------------
class AffordabilityParams(BaseModel):
    min_ratio: float = Field(0.3, ge=0, le=1)
    income_cap: float = Field(36.0, ge=6, le=60)
    income_threshold: float = Field(5000.0, ge=0)
    max_term: float = Field(72.0, ge=1, le=360)
    base_rate: float = Field(0.115, ge=0, le=1)
    buffer_pct: float = Field(0.10, ge=0, le=1)
    stress_multiplier: float = Field(1.05, ge=1, le=3)
    sector_cap: float = Field(60.0, ge=0)


class ParamsNT(NamedTuple):
    min_ratio: float
    income_cap: float
    income_threshold: float
    max_term: float
    base_rate: float
    buffer_pct: float
    stress_multiplier: float
    sector_cap: float


PARAMS_RAW = {
    "min_ratio": 0.3,
    "income_cap": 36.0,
    "income_threshold": 5000.0,
    "max_term": 72.0,
    "base_rate": 0.115,
    "buffer_pct": 0.10,
    "stress_multiplier": 1.05,
    "sector_cap": 60.0,
}


def bind_params(raw: dict) -> ParamsNT:
    m = AffordabilityParams(**raw)
    return ParamsNT(**m.model_dump())


# ---------------------------------------------------------------------------
# Phase 1 -- accept. Doc 02 S3.5's score() is called kwargs-style. We test
# that (unpack cost into a generic **kwargs catcher) against a plain-dict
# entry point, since the task brief names both as options.
# NOTE: this measures the unpack/bind cost via a **kwargs catch-all, not a
# literal 400-named-parameter generated function signature (which is what
# doc 02 S3.5's example literally shows). Generating and compiling a real
# 401-argument Python function was dropped for time -- see README "what was
# NOT tested".
# ---------------------------------------------------------------------------
def accept_kwargs(**kwargs) -> dict:
    return kwargs


def accept_dict(request: dict) -> dict:
    return request


# ---------------------------------------------------------------------------
# Phase 3 -- marshal into the kernel's input shape: a 1-row record array,
# allocated fresh (np.empty) each call.
# ---------------------------------------------------------------------------
def marshal(request: dict, dtype=IN_DTYPE, names=IN_ALL_NAMES) -> np.ndarray:
    rec = np.empty(1, dtype=dtype)
    row = rec[0]
    for name in names:
        row[name] = request[name]
    return rec


# ---------------------------------------------------------------------------
# Phase 5 -- read the 633 outputs back into Python objects.
# ---------------------------------------------------------------------------
def readback(out: np.ndarray) -> dict:
    row = out[0]
    return {name: row[name].item() for name in OUT_ALL_NAMES}


def readback_bulk(out: np.ndarray) -> dict:
    """Alternative: numpy void-scalar .item() on the WHOLE row converts
    every field to a Python tuple in one C call, vs. 633 separate per-field
    lookups + per-field .item() calls above. Added because the per-field
    version turned out to dominate total cost in a quick smoke run -- this
    tests whether that is a numpy-usage choice rather than an inherent cost."""
    return dict(zip(OUT_ALL_NAMES, out[0].item()))


# ---------------------------------------------------------------------------
# Phase 6 -- assemble the response dict.
# ---------------------------------------------------------------------------
def assemble(values: dict, params: ParamsNT) -> dict:
    return {
        "status": "ok",
        "module": "affordability",
        "params_echo": params.min_ratio,
        "outputs": values,
    }


# ---------------------------------------------------------------------------
# Phase 4 -- kernel. Reused verbatim (shape + codegen pattern) from
# output-writeback-convention/writeback.py: _HEADER / gen_record_out_source,
# and load_source's "write a real file, import via spec_from_file_location"
# pattern (doc 05 S4.1).
# ---------------------------------------------------------------------------
_HEADER = """\
import numpy as np
from numba import njit


@njit(cache=False, nogil=True)
def driver(inp, coef_f8, out):
    n = inp.shape[0]
    for i in range(n):
        a0 = inp[i].i_f8_0
        a1 = inp[i].i_f8_1
        a2 = inp[i].i_f8_2
        a3 = inp[i].i_f8_3
        a4 = inp[i].i_f8_4
        b0 = inp[i].i_i8_0
"""


def gen_record_out_source() -> str:
    lines = [_HEADER]
    for j, name in enumerate(OUT_F8_NAMES):
        lines.append(f"        out[i].{name} = coef_f8[{j}] * a0 + a1 - a2 * 0.001")
    for j, name in enumerate(OUT_I8_NAMES):
        lines.append(f"        out[i].{name} = b0 + {j}")
    for j, name in enumerate(OUT_B_NAMES):
        lines.append(f"        out[i].{name} = (a3 + {j} * 0.0001) > a4")
    return "\n".join(lines) + "\n"


def load_source(src: str, fname: str = "driver"):
    path = HERE / "_generated_driver.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("n1_driver", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fname), path


log(f"env: python={platform.python_version()} numba={numba.__version__} "
    f"numpy={np.__version__} pydantic={pydantic.VERSION}")
log(f"shape: {N_IN} in ({IN_F8}f8/{IN_I8}i8/{IN_BOOL}bool), "
    f"{N_OUT} out ({OUT_F8}f8/{OUT_I8}i8/{OUT_BOOL}bool)")
log(f"rss before kernel compile: {rss_mb():.1f} MB")
log("compiling record-out kernel (this IS the ~1us path -- not what we're measuring)...")
_t0 = time.perf_counter()
DRIVER, DRIVER_PATH = load_source(gen_record_out_source())
_warm_in = np.empty(1, dtype=IN_DTYPE)
_warm_out = np.empty(1, dtype=OUT_REC_DTYPE)
DRIVER(_warm_in, COEF_F8, _warm_out)  # trigger compile
_compile_s = time.perf_counter() - _t0
log(f"kernel compiled in {_compile_s:.2f}s -> {DRIVER_PATH.name}; rss now {rss_mb():.1f} MB")


def dispatch(rec: np.ndarray) -> np.ndarray:
    out = np.empty(1, dtype=OUT_REC_DTYPE)
    DRIVER(rec, COEF_F8, out)
    return out


def score(request: dict, params_raw: dict) -> dict:
    """The full, realistic single-record path, all six phases -- this is
    what "total framework overhead" times end-to-end."""
    kw = accept_kwargs(**request)
    params = bind_params(params_raw)
    rec = marshal(kw)
    out = dispatch(rec)
    values = readback(out)
    return assemble(values, params)


# sanity: one call, verify it runs and shapes are right.
_sanity = score(REQUEST_DICT, PARAMS_RAW)
assert len(_sanity["outputs"]) == N_OUT
log("sanity call ok")


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------
def timed_ns(fn, n: int, warmup: int = 50) -> np.ndarray:
    for _ in range(warmup):
        fn()
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        out[i] = t1 - t0
    return out


def pctiles(ns: np.ndarray) -> dict:
    return {
        "p50_us": float(np.percentile(ns, 50)) / 1e3,
        "p95_us": float(np.percentile(ns, 95)) / 1e3,
        "p99_us": float(np.percentile(ns, 99)) / 1e3,
        "max_us": float(ns.max()) / 1e3,
        "mean_us": float(ns.mean()) / 1e3,
        "n": int(len(ns)),
    }


BUDGET_MS = 20.0


def pct_of_budget(us: float) -> float:
    return 100.0 * (us / 1e3) / BUDGET_MS


# ===========================================================================
# MEASUREMENT 1 -- phase breakdown at the primary width (400 in / 633 out)
# ===========================================================================
def measure_phases(n: int) -> dict:
    log(f"=== phase breakdown, n={n} calls/phase ===")
    phases = {}

    ns = timed_ns(lambda: accept_kwargs(**REQUEST_DICT), n)
    phases["1_accept_kwargs"] = pctiles(ns)
    ns2 = timed_ns(lambda: accept_dict(REQUEST_DICT), n)
    phases["1b_accept_dict_alt"] = pctiles(ns2)
    log(f"  accept(kwargs)  p50={phases['1_accept_kwargs']['p50_us']:.3f}us  "
        f"accept(dict) p50={phases['1b_accept_dict_alt']['p50_us']:.3f}us")

    ns = timed_ns(lambda: bind_params(PARAMS_RAW), n)
    phases["2_validate_params"] = pctiles(ns)
    log(f"  validate params p50={phases['2_validate_params']['p50_us']:.3f}us")

    ns = timed_ns(lambda: marshal(REQUEST_DICT), n)
    phases["3_marshal"] = pctiles(ns)
    log(f"  marshal (400 fields, alloc incl.) p50={phases['3_marshal']['p50_us']:.3f}us")

    _rec_for_dispatch = marshal(REQUEST_DICT)
    ns = timed_ns(lambda: dispatch(_rec_for_dispatch), n)
    phases["4_kernel_dispatch"] = pctiles(ns)
    log(f"  kernel dispatch (N=1, reference only) p50={phases['4_kernel_dispatch']['p50_us']:.3f}us")

    _out_for_readback = dispatch(_rec_for_dispatch)
    ns = timed_ns(lambda: readback(_out_for_readback), n)
    phases["5_readback"] = pctiles(ns)
    log(f"  readback (633 fields, per-field .item()) p50={phases['5_readback']['p50_us']:.3f}us")

    ns = timed_ns(lambda: readback_bulk(_out_for_readback), n)
    phases["5b_readback_bulk_alt"] = pctiles(ns)
    assert readback(_out_for_readback) == readback_bulk(_out_for_readback), "readback variants disagree"
    log(f"  readback (bulk row.item()) p50={phases['5b_readback_bulk_alt']['p50_us']:.3f}us  "
        f"[{phases['5_readback']['p50_us']/max(phases['5b_readback_bulk_alt']['p50_us'],1e-9):.1f}x vs per-field]")

    _values_for_assemble = readback(_out_for_readback)
    _params_for_assemble = bind_params(PARAMS_RAW)
    ns = timed_ns(lambda: assemble(_values_for_assemble, _params_for_assemble), n)
    phases["6_assemble"] = pctiles(ns)
    log(f"  assemble response dict p50={phases['6_assemble']['p50_us']:.3f}us")

    for name, p in phases.items():
        rec = {"measurement": "phase", "phase": name, **p, "pct_of_20ms_budget_p50": pct_of_budget(p["p50_us"])}
        jsonl_append(RESULTS_JSONL, rec)

    return phases


# ===========================================================================
# MEASUREMENT 2 -- total end-to-end, p50/p95/p99/max, GC on and off
# ===========================================================================
def measure_total(n: int) -> dict:
    log(f"=== total end-to-end, n={n} calls ===")
    results = {}

    assert gc.isenabled()
    counts_before = gc.get_count()
    stats_before = gc.get_stats()
    ns_gc_on = timed_ns(lambda: score(REQUEST_DICT, PARAMS_RAW), n)
    stats_after = gc.get_stats()
    collections_during = [stats_after[i]["collections"] - stats_before[i]["collections"] for i in range(3)]
    p = pctiles(ns_gc_on)
    p["gc_collections_during_run"] = collections_during
    results["gc_enabled"] = p
    log(f"  GC ON : p50={p['p50_us']:.3f}us p95={p['p95_us']:.3f}us p99={p['p99_us']:.3f}us "
        f"max={p['max_us']:.3f}us  gen0/1/2 collections={collections_during}")
    jsonl_append(RESULTS_JSONL, {"measurement": "total", "gc": "enabled", **p,
                                  "pct_of_20ms_budget_p50": pct_of_budget(p["p50_us"]),
                                  "pct_of_20ms_budget_p99": pct_of_budget(p["p99_us"]),
                                  "pct_of_20ms_budget_max": pct_of_budget(p["max_us"])})

    gc.disable()
    try:
        ns_gc_off = timed_ns(lambda: score(REQUEST_DICT, PARAMS_RAW), n)
    finally:
        gc.enable()
    p2 = pctiles(ns_gc_off)
    results["gc_disabled"] = p2
    log(f"  GC OFF: p50={p2['p50_us']:.3f}us p95={p2['p95_us']:.3f}us p99={p2['p99_us']:.3f}us "
        f"max={p2['max_us']:.3f}us")
    jsonl_append(RESULTS_JSONL, {"measurement": "total", "gc": "disabled", **p2,
                                  "pct_of_20ms_budget_p50": pct_of_budget(p2["p50_us"]),
                                  "pct_of_20ms_budget_p99": pct_of_budget(p2["p99_us"]),
                                  "pct_of_20ms_budget_max": pct_of_budget(p2["max_us"])})

    return results


# ===========================================================================
# MEASUREMENT 3 -- allocation: bytes + objects per call (tracemalloc +
# sys.getallocatedblocks())
# ===========================================================================
def measure_allocation(calls: int = 300) -> dict:
    log(f"=== allocation probe, {calls} single-call samples ===")
    score(REQUEST_DICT, PARAMS_RAW)
    score(REQUEST_DICT, PARAMS_RAW)  # warm

    byte_samples = []
    block_samples = []
    for _ in range(calls):
        gc.collect()
        tracemalloc.start()
        b0 = sys.getallocatedblocks()
        cur0, _ = tracemalloc.get_traced_memory()
        score(REQUEST_DICT, PARAMS_RAW)
        cur1, peak1 = tracemalloc.get_traced_memory()
        b1 = sys.getallocatedblocks()
        tracemalloc.stop()
        byte_samples.append(peak1 - cur0)
        block_samples.append(b1 - b0)

    result = {
        "bytes_per_call_median": statistics.median(byte_samples),
        "bytes_per_call_p95": float(np.percentile(byte_samples, 95)),
        "blocks_per_call_median": statistics.median(block_samples),
        "blocks_per_call_p95": float(np.percentile(block_samples, 95)),
        "samples": calls,
    }
    log(f"  ~{result['bytes_per_call_median']:.0f} bytes/call (peak-traced), "
        f"~{result['blocks_per_call_median']:.0f} net allocated-block delta/call")
    jsonl_append(RESULTS_JSONL, {"measurement": "allocation", **result})
    return result


# ===========================================================================
# MEASUREMENT 4 -- width sweep for accept + marshal (kernel/readback/assemble
# held fixed -- NOT recompiled per width, see README "why width sweep skips
# the kernel")
# ===========================================================================
def width_split(w: int) -> tuple[int, int, int]:
    f8 = round(w * 0.75)
    i8 = round(w * 0.20)
    b = w - f8 - i8
    if b < 0:
        b = 0
        i8 = w - f8
    return f8, i8, b


def measure_width_sweep(widths: list[int], n: int) -> list[dict]:
    log(f"=== width sweep {widths}, n={n} calls/phase ===")
    rows = []
    for w in widths:
        f8n, i8n, bn = width_split(w)
        names = [f"i_f8_{k}" for k in range(f8n)] + [f"i_i8_{k}" for k in range(i8n)] + [f"i_b_{k}" for k in range(bn)]
        dtype = np.dtype(
            [(n_, "f8") for n_ in names[:f8n]]
            + [(n_, "i8") for n_ in names[f8n:f8n + i8n]]
            + [(n_, "?") for n_ in names[f8n + i8n:]]
        )
        rng = np.random.default_rng(42 + w)
        req = {}
        for nm in names[:f8n]:
            req[nm] = float(rng.random())
        for nm in names[f8n:f8n + i8n]:
            req[nm] = int(rng.integers(-1000, 1000))
        for nm in names[f8n + i8n:]:
            req[nm] = bool(rng.random() < 0.5)

        ns_accept = timed_ns(lambda: accept_kwargs(**req), n)
        ns_marshal = timed_ns(lambda: marshal(req, dtype=dtype, names=names), n)
        pa, pm = pctiles(ns_accept), pctiles(ns_marshal)
        row = {
            "measurement": "width_sweep", "width": w, "f8": f8n, "i8": i8n, "bool": bn,
            "accept_p50_us": pa["p50_us"], "accept_p99_us": pa["p99_us"],
            "marshal_p50_us": pm["p50_us"], "marshal_p99_us": pm["p99_us"],
            "combined_p50_us": pa["p50_us"] + pm["p50_us"],
            "pct_of_20ms_budget_combined_p50": pct_of_budget(pa["p50_us"] + pm["p50_us"]),
        }
        rows.append(row)
        log(f"  W={w:4d}: accept p50={pa['p50_us']:.3f}us  marshal p50={pm['p50_us']:.3f}us  "
            f"combined={row['combined_p50_us']:.3f}us ({row['pct_of_20ms_budget_combined_p50']:.4f}% of 20ms)")
        jsonl_append(RESULTS_JSONL, row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--phase-n", type=int, default=8000)
    ap.add_argument("--total-n", type=int, default=12000)
    ap.add_argument("--alloc-calls", type=int, default=300)
    ap.add_argument("--widths", default="10,50,100,400")
    args = ap.parse_args()

    if args.quick:
        phase_n, total_n, alloc_calls = 500, 1000, 30
        widths = [10, 400]
    else:
        phase_n, total_n, alloc_calls = args.phase_n, args.total_n, args.alloc_calls
        widths = [int(x) for x in args.widths.split(",")]

    if RESULTS_JSONL.exists():
        RESULTS_JSONL.unlink()

    env = {
        "python": platform.python_version(), "numba": numba.__version__,
        "numpy": np.__version__, "pydantic": pydantic.VERSION,
        "compile_s": _compile_s,
    }
    jsonl_append(RESULTS_JSONL, {"measurement": "env", **env})
    log(f"rss before measurement: {rss_mb():.1f} MB")

    phases = measure_phases(phase_n)
    total = measure_total(total_n)
    alloc = measure_allocation(alloc_calls)
    sweep = measure_width_sweep(widths, min(phase_n, 3000))

    log(f"rss after measurement: {rss_mb():.1f} MB")

    summary = {"env": env, "phases": phases, "total": total, "allocation": alloc, "width_sweep": sweep}
    (HERE / "results_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log("DONE -- wrote results.jsonl and results_summary.json")


if __name__ == "__main__":
    main()
