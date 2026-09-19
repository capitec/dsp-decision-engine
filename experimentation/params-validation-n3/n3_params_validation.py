"""
EXPERIMENT N3 -- params validation on the request path.

Doc 02 S4: "An invocation is 1 row or N rows, which covers realtime payload
params and batch uniformly." Doc 01 S6: "Params may arrive per invocation,
including in a realtime request payload." E0 (doc 01 S5.7 / doc 02 S2.1)
measured 58.5us to validate-and-bind a 63-node STRUCTURE config through the
discriminated union -- a one-time, build/stage-time cost. N3 asks the
adjacent, previously-unmeasured question: a PARAMS document (business-user
tunable values, doc 03 S4 / doc 08 S6.2), which doc 02 S4 says may arrive on
every single realtime request, not just at stage time.

Five things measured, each >=10,000 iterations, p50/p95/p99/max, each also
expressed as a % of a 20ms budget:

  1. Full pydantic validation of a params document per request, at 1/10/50
     module instances (doc 03 S4.1: "mod1 | mod2 | mod3 exposes the composed
     set, namespaced by module instance" -- a realistic pipeline has many).
  2. The pydantic-model -> NamedTuple conversion (doc 03 S4: "the model
     becomes a NamedTuple ... verified in prototype") in isolation from
     validation -- per request (fresh) vs cached (memoized by content, a
     dict-lookup hit).
  3. resolve_params(doc, origin=..., complete=True) per doc 08 S6.2's
     signature, including the completeness check via model_fields_set (doc
     04 S2.1: "completeness is computable from pydantic's model_fields_set").
  4. ParamsCell.get() (doc 08 S4/REVIEW.md S6.3) -- the read cost of the
     validate-once-and-reuse path.
  5. ParamsCell.swap() at genuine N=1, to compare against doc 08 S4's batch
     figure (activate() 0.177us median, measured in a batch/chunked-apply
     context, EXPERIMENTS.md SH).

Design question this feeds (task brief): should a realtime request be
allowed to carry raw params at all, or must it reference a pre-validated
bundle by id? Doc 04 S2.1 already flags payload params as a CODEOWNERS/
governance gap on non-performance grounds; this harness supplies the
performance side of that argument.

Reused from prior harnesses (not rewritten):
  - elapsed()/log() progress-clock pattern, timed_ns()/pctiles()/
    pct_of_budget() timing helpers, jsonl_append(), rss_mb() (peak-RSS via
    resource.getrusage) -- verbatim from
    experimentation/single-record-overhead/n1_overhead.py (itself reused
    from prange-crossover/ and chunked-writeback-at-scale/).
  - AffordabilityParams (8-field pydantic model) and ParamsNT (matching
    NamedTuple) -- verbatim from n1_overhead.py's phase-2 params fixture,
    reused here as the shape of ONE module's params, now composed M times
    to build a realistic multi-module bundle (doc 03 S4.1).
  - bind_params()'s pattern (`Model(**raw)` then `NamedTuple(**m.model_dump())`)
    -- verbatim from n1_overhead.py, decomposed here into its two phases
    (validate, convert) instead of timed as one blob, per the task brief's
    request to isolate conversion from validation.

NOT reused: N2's calling-convention machinery (kwargs/dict/record/
positional) -- out of scope; this harness is entirely about params, which
doc 02 S3.5 always passes as a single `params=` object, never spread across
individual keyword arguments.

Run:
    /path/to/.venv/bin/python n3_params_validation.py            # full run
    /path/to/.venv/bin/python n3_params_validation.py --quick    # smoke test

Writes results.jsonl (flushed per record) and results_summary.json next to
itself.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pydantic
from pydantic import BaseModel, Field, create_model

HERE = Path(__file__).resolve().parent
T0 = time.perf_counter()


def elapsed() -> float:
    return time.perf_counter() - T0


def log(msg: str) -> None:
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


def rss_mb() -> float:
    # Reused: single-record-overhead/n1_overhead.py:rss_mb (itself reused
    # from chunked-writeback-at-scale/measure_variant.py:peak_rss_kb).
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


RESULTS_JSONL = HERE / "results.jsonl"


def jsonl_append(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Timing helpers -- reused verbatim from n1_overhead.py / n2_calling_convention.py.
# ---------------------------------------------------------------------------
def timed_ns(fn, n: int, warmup: int = 200) -> np.ndarray:
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


# ---------------------------------------------------------------------------
# One module's params -- verbatim from n1_overhead.py.
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


MODULE_DOC = {
    "min_ratio": 0.3, "income_cap": 36.0, "income_threshold": 5000.0,
    "max_term": 72.0, "base_rate": 0.115, "buffer_pct": 0.10,
    "stress_multiplier": 1.05, "sector_cap": 60.0,
}


# ---------------------------------------------------------------------------
# Multi-module bundle -- doc 03 S4.1: "mod1 | mod2 | mod3 exposes the
# composed set, namespaced by module instance name." Built with pydantic's
# create_model, ONCE per M (build-time-equivalent), same as doc 02 S2.1's
# "finalisation" phase -- never rebuilt per request.
# ---------------------------------------------------------------------------
def make_bundle_model(m: int):
    fields = {f"mod_{i}": (AffordabilityParams, ...) for i in range(m)}
    return create_model(f"ParamsBundle{m}", **fields)


def make_pipeline_doc(m: int) -> dict:
    return {f"mod_{i}": dict(MODULE_DOC) for i in range(m)}


MODULE_COUNTS = (1, 10, 50)
BUNDLE_MODELS = {m: make_bundle_model(m) for m in MODULE_COUNTS}
PIPELINE_DOCS = {m: make_pipeline_doc(m) for m in MODULE_COUNTS}

# sanity
for m in MODULE_COUNTS:
    inst = BUNDLE_MODELS[m](**PIPELINE_DOCS[m])
    assert len(inst.model_fields_set) == m
log(f"env: python={platform.python_version()} numpy={np.__version__} pydantic={pydantic.VERSION}")
log(f"bundle models built for M={MODULE_COUNTS}, sanity ok")


# ===========================================================================
# MEASUREMENT 1 -- full pydantic validation of a params document per
# request, at 1/10/50 module instances.
# ===========================================================================
def measure_full_validation(n: int) -> dict:
    log(f"=== 1: full validation, M={MODULE_COUNTS}, n={n} ===")
    results = {}
    for m in MODULE_COUNTS:
        Model = BUNDLE_MODELS[m]
        doc = PIPELINE_DOCS[m]
        ns = timed_ns(lambda: Model(**doc), n)
        p = pctiles(ns)
        p.update({"measurement": "1_full_validation", "modules": m})
        p["pct_of_20ms_p50"] = pct_of_budget(p["p50_us"])
        p["pct_of_20ms_p99"] = pct_of_budget(p["p99_us"])
        results[m] = p
        jsonl_append(RESULTS_JSONL, p)
        log(f"  M={m:3d}: p50={p['p50_us']:8.3f}us ({p['pct_of_20ms_p50']:.4f}%)  "
            f"p95={p['p95_us']:8.3f}us  p99={p['p99_us']:8.3f}us  max={p['max_us']:8.3f}us")

    # GC on/off, only for the heaviest case (M=50) -- TIMING RULES: "measure
    # with it both on and off where that could matter." N1/N2 already
    # established no measurable GC effect for record-marshal/kwargs-bind
    # workloads; pydantic validation builds many short-lived objects per
    # call (one submodel instance per module) so it is plausible refcounting
    # alone does not clear cycles the way it did there -- worth one check.
    m = 50
    Model, doc = BUNDLE_MODELS[m], PIPELINE_DOCS[m]
    assert gc.isenabled()
    ns_on = timed_ns(lambda: Model(**doc), n)
    gc.disable()
    try:
        ns_off = timed_ns(lambda: Model(**doc), n)
    finally:
        gc.enable()
    p_on, p_off = pctiles(ns_on), pctiles(ns_off)
    log(f"  M=50 GC on  p50={p_on['p50_us']:.3f}us   GC off p50={p_off['p50_us']:.3f}us  "
        f"(ratio on/off = {p_on['p50_us']/p_off['p50_us']:.3f})")
    gc_check = {"measurement": "1_gc_check", "modules": m,
                "p50_gc_on_us": p_on["p50_us"], "p50_gc_off_us": p_off["p50_us"]}
    jsonl_append(RESULTS_JSONL, gc_check)
    results["gc_check_m50"] = gc_check
    return results


# ===========================================================================
# MEASUREMENT 2 -- pydantic-model -> NamedTuple conversion, isolated from
# validation. Per request (fresh) vs cached (memoized dict-lookup hit).
# Doc 03 S4: "Under the hood the model becomes a NamedTuple whose type is
# fixed, so changing values never recompiles anything."
# ===========================================================================
def convert_fresh(bundle_instance, m: int) -> tuple:
    return tuple(
        ParamsNT(**getattr(bundle_instance, f"mod_{i}").model_dump())
        for i in range(m)
    )


_CONVERT_CACHE: dict = {}


def convert_cached(bundle_instance, m: int, key):
    hit = _CONVERT_CACHE.get(key)
    if hit is not None:
        return hit
    result = convert_fresh(bundle_instance, m)
    _CONVERT_CACHE[key] = result
    return result


def measure_conversion(n: int) -> dict:
    log(f"=== 2: model->NamedTuple conversion, M={MODULE_COUNTS}, n={n} ===")
    results = {}
    for m in MODULE_COUNTS:
        Model, doc = BUNDLE_MODELS[m], PIPELINE_DOCS[m]
        bundle_instance = Model(**doc)  # validated ONCE, outside the timed loop --
        # isolates conversion cost from validation cost (measurement 1 already
        # covers validation).

        ns = timed_ns(lambda: convert_fresh(bundle_instance, m), n)
        p_fresh = pctiles(ns)
        p_fresh.update({"measurement": "2_convert_fresh", "modules": m})
        p_fresh["pct_of_20ms_p50"] = pct_of_budget(p_fresh["p50_us"])
        jsonl_append(RESULTS_JSONL, p_fresh)

        key = ("bundle", m)  # simulates: same params document recurs across requests
        _CONVERT_CACHE.pop(key, None)
        convert_cached(bundle_instance, m, key)  # warm the cache -- every timed call is a hit
        ns2 = timed_ns(lambda: convert_cached(bundle_instance, m, key), n)
        p_cached = pctiles(ns2)
        p_cached.update({"measurement": "2_convert_cached", "modules": m})
        p_cached["pct_of_20ms_p50"] = pct_of_budget(p_cached["p50_us"])
        jsonl_append(RESULTS_JSONL, p_cached)

        results[m] = {"fresh": p_fresh, "cached": p_cached}
        log(f"  M={m:3d}: fresh p50={p_fresh['p50_us']:8.3f}us ({p_fresh['pct_of_20ms_p50']:.4f}%)  "
            f"cached p50={p_cached['p50_us']:8.3f}us ({p_cached['pct_of_20ms_p50']:.4f}%)  "
            f"speedup={p_fresh['p50_us']/max(p_cached['p50_us'],1e-6):.1f}x")
    return results


# ===========================================================================
# MEASUREMENT 3 -- resolve_params(doc, origin=..., complete=True), doc 08
# S6.2's signature, including the completeness check via model_fields_set
# (doc 04 S2.1).
# ===========================================================================
class ParamsBundle:
    __slots__ = ("values", "origin", "defaulted_fields")

    def __init__(self, values, origin, defaulted_fields):
        self.values = values
        self.origin = origin
        self.defaulted_fields = defaulted_fields


def resolve_params_single(doc: dict, *, origin: str, complete: bool = True) -> ParamsBundle:
    """resolve_params for ONE module's params (8 fields) -- the doc 08 S6.2
    signature applied at the smallest realistic scope."""
    if not origin:
        raise ValueError("origin must be a non-empty token")
    m = AffordabilityParams(**doc)
    values = ParamsNT(**m.model_dump())
    defaulted = None
    if complete:
        all_fields = AffordabilityParams.model_fields.keys()
        set_fields = m.model_fields_set
        defaulted = tuple(f for f in all_fields if f not in set_fields)
    return ParamsBundle(values=values, origin=origin, defaulted_fields=defaulted)


def resolve_params_bundle(doc: dict, m: int, *, origin: str, complete: bool = True) -> ParamsBundle:
    """resolve_params for the WHOLE pipeline's params (M module instances)."""
    if not origin:
        raise ValueError("origin must be a non-empty token")
    Model = BUNDLE_MODELS[m]
    inst = Model(**doc)
    values = tuple(
        ParamsNT(**getattr(inst, f"mod_{i}").model_dump())
        for i in range(m)
    )
    defaulted = None
    if complete:
        all_fields = Model.model_fields.keys()
        set_fields = inst.model_fields_set
        defaulted = tuple(f for f in all_fields if f not in set_fields)
    return ParamsBundle(values=values, origin=origin, defaulted_fields=defaulted)


ORIGIN = "file:config/term_loan/production.json@abc1234"


def measure_resolve_params(n: int) -> dict:
    log(f"=== 3: resolve_params(doc, origin=..., complete=True), M={MODULE_COUNTS}, n={n} ===")
    results = {}

    ns = timed_ns(lambda: resolve_params_single(MODULE_DOC, origin=ORIGIN, complete=True), n)
    p = pctiles(ns)
    p.update({"measurement": "3_resolve_params", "modules": 1, "scope": "single_module"})
    p["pct_of_20ms_p50"] = pct_of_budget(p["p50_us"])
    jsonl_append(RESULTS_JSONL, p)
    results["single_module"] = p
    log(f"  single-module: p50={p['p50_us']:8.3f}us ({p['pct_of_20ms_p50']:.4f}%)")

    for m in MODULE_COUNTS:
        doc = PIPELINE_DOCS[m]
        ns = timed_ns(lambda: resolve_params_bundle(doc, m, origin=ORIGIN, complete=True), n)
        p = pctiles(ns)
        p.update({"measurement": "3_resolve_params", "modules": m, "scope": "whole_pipeline"})
        p["pct_of_20ms_p50"] = pct_of_budget(p["p50_us"])
        p["pct_of_20ms_p99"] = pct_of_budget(p["p99_us"])
        jsonl_append(RESULTS_JSONL, p)
        results[f"pipeline_m{m}"] = p
        log(f"  pipeline M={m:3d}: p50={p['p50_us']:8.3f}us ({p['pct_of_20ms_p50']:.4f}%)  "
            f"p99={p['p99_us']:8.3f}us")

    # sanity: completeness check actually reports defaulted fields when a
    # field is omitted.
    partial = {k: v for k, v in MODULE_DOC.items() if k != "sector_cap"}
    b = resolve_params_single(partial, origin=ORIGIN, complete=True)
    assert b.defaulted_fields == ("sector_cap",), b.defaulted_fields
    log("  completeness-check sanity ok (omitted field correctly reported)")
    return results


# ===========================================================================
# MEASUREMENT 4 & 5 -- ParamsCell.get() / .swap(), doc 08 S4 / REVIEW.md
# S6.3. Doc 08 S4 measured activate() at 0.177us median in a BATCH context
# (EXPERIMENTS.md SH: 11,605 swaps across a 3s run of chunked batches). This
# re-measures both operations at genuine N=1 -- one cell, one reader, one
# writer, no batch loop in between.
# ===========================================================================
class ParamsCell:
    """Doc 08 S6.3's ParamsCell -- the only mutation verb. A plain attribute
    swap: under the GIL, a single reference assignment is already atomic,
    which is the mechanism EXPERIMENTS.md SH's 0-straddled-batches result
    relies on (readers see either the old or the new object, never a torn
    write)."""

    def __init__(self, initial):
        self._value = initial

    def get(self):
        return self._value

    def swap(self, new):
        old = self._value
        self._value = new
        return old


def measure_params_cell(n: int) -> dict:
    log(f"=== 4&5: ParamsCell.get()/.swap() at N=1, n={n} ===")
    results = {}

    bundle_a = resolve_params_single(MODULE_DOC, origin=ORIGIN, complete=True)
    bundle_b = resolve_params_single(MODULE_DOC, origin=ORIGIN, complete=True)
    cell = ParamsCell(bundle_a)

    # 4. get() -- the once-per-invocation read a serving call makes.
    ns = timed_ns(lambda: cell.get(), n)
    p_get = pctiles(ns)
    p_get.update({"measurement": "4_cell_get"})
    p_get["pct_of_20ms_p50"] = pct_of_budget(p_get["p50_us"])
    jsonl_append(RESULTS_JSONL, p_get)
    results["get"] = p_get
    log(f"  get():  p50={p_get['p50_us']:.5f}us  p99={p_get['p99_us']:.5f}us  "
        f"({p_get['pct_of_20ms_p50']:.6f}% of 20ms)")

    # 5. swap() at N=1 -- alternating two already-built bundles so each
    # swap does real work (old != new), same as doc 08's activate().
    toggle = [bundle_a, bundle_b]
    state = {"i": 0}

    def do_swap():
        state["i"] ^= 1
        cell.swap(toggle[state["i"]])

    ns2 = timed_ns(do_swap, n)
    p_swap = pctiles(ns2)
    p_swap.update({"measurement": "5_cell_swap_n1"})
    p_swap["pct_of_20ms_p50"] = pct_of_budget(p_swap["p50_us"])
    jsonl_append(RESULTS_JSONL, p_swap)
    results["swap_n1"] = p_swap
    log(f"  swap(): p50={p_swap['p50_us']:.5f}us  p99={p_swap['p99_us']:.5f}us  "
        f"({p_swap['pct_of_20ms_p50']:.6f}% of 20ms)  "
        f"[doc 08 SH batch-context figure: 0.177us median, 0.357us p99]")

    # 6. serving model check: exactly one get() per invocation, no re-read
    # loop -- this is a structural/logic check (matches doc 08 property 1 /
    # EXPERIMENTS.md SH's methodology), not a new timing claim; SH already
    # measured the straddling consequence of re-reading mid-batch and is not
    # re-run here.
    reads = {"count": 0}
    real_get = cell.get

    def counting_get():
        reads["count"] += 1
        return real_get()

    cell.get = counting_get
    _ = cell.get()  # one simulated invocation
    cell.get = real_get
    assert reads["count"] == 1
    log("  serving-read-count sanity ok: exactly 1 cell.get() per simulated invocation")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-calls", type=int, default=12000)
    args = ap.parse_args()

    n = 500 if args.quick else args.n_calls

    if RESULTS_JSONL.exists():
        RESULTS_JSONL.unlink()

    env = {
        "measurement": "env",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pydantic": pydantic.VERSION,
        "n_calls": n,
    }
    jsonl_append(RESULTS_JSONL, env)
    log(f"rss before measurement: {rss_mb():.1f} MB")

    r1 = measure_full_validation(n)
    r2 = measure_conversion(n)
    r3 = measure_resolve_params(n)
    r45 = measure_params_cell(n)

    log(f"rss after measurement: {rss_mb():.1f} MB")

    summary = {
        "env": env,
        "1_full_validation": r1,
        "2_conversion": r2,
        "3_resolve_params": r3,
        "4_5_params_cell": r45,
    }
    (HERE / "results_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log("DONE -- wrote results.jsonl and results_summary.json")


if __name__ == "__main__":
    main()
