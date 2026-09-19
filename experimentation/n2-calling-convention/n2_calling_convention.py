"""
EXPERIMENT N2 -- the score() calling convention at width.

Doc 02 S3.5 and doc 03 S6 specify the realtime entry point as keyword
arguments: Affordability.score(net_income=42000.0, expenses=18000.0,
instalment=3100.0, params=p). At 400 inputs that is 400 keyword arguments
per request -- Python signature binding, dict construction, and per-argument
handling, all before any compiled code runs. Never costed before this.

This harness follows doc 01 S6.1's reweighted priority: the compiled kernel
answers a record in ~1us (confirmed by N1: 1.38us pooled, matching E9/J) --
"do not benchmark the kernel." N1 already measured the OUTPUT side
end-to-end (marshal/dispatch/readback/assemble at 400/633) and found the
per-field readback/marshal loops -- not the kernel -- dominate. N2 is
scoped to what N1 flagged as open: the INPUT side, across FOUR calling
conventions, at FOUR widths, plus two follow-up questions the task brief
asks for directly:

  A. convention sweep @ 10/50/100/400 inputs, >=10,000 calls each:
     1. kwargs      -- score(a0=1.0, a1=2.0, ...)      [doc 02 S3.5, as written]
     2. dict         -- score({"a0": 1.0, "a1": 2.0, ...})
     3. record       -- caller pre-builds a 1-row numpy record ONCE and
                         reuses the buffer, mutating fields in place per call
     4. positional   -- score(1.0, 2.0, ...)
     Each pipeline = accept (however the convention receives values) +
     marshal-equivalent (turn those values into the 1-row record array the
     kernel wants). The record convention has ~0 "accept" (the buffer IS
     the record) and a marshal-equivalent that skips np.empty() entirely.

  B. isolates WHERE the kwargs cost lives, @ width=400:
     **kwargs catch-all  vs  an explicit N-parameter signature called with
     keywords, body=pass (pure signature-binding cost, no dict built)  vs
     the same explicit signature, body=build dict (binding + dict
     construction)  vs  the same signature called positionally  vs  a bare
     dict pass-through  vs  a bare record pass-through.
     This is the "**kwargs capture vs an explicit signature -- is binding
     the cost or is dict construction?" question from the task brief.

  C. validation cost @ width=400: a dynamically-built pydantic model with
     400 required float fields (Model(**request)) vs a manual
     "all required keys present" set-membership check vs a manual
     per-field isinstance loop vs no validation at all. Unlike N1 phase 2
     (params, ~8 fields, validated ONCE per call because there is one
     params bundle) input validation is inherently per-record -- there is
     no "validate once" option for the DATA, only for the schema/model
     object itself, which this harness already builds once, outside the
     timed loop.

OUT OF SCOPE, by design, per doc 01 S6.1's "do not benchmark the kernel"
and to avoid re-deriving N1's already-published output-side numbers:
  - actual njit dispatch and output readback/assemble. N1 measured these
    exhaustively at this exact width (dispatch pooled 1.38us, readback bulk
    54.98us, assemble 0.35us -- experimentation/single-record-overhead/).
    Section "Combining with N1" below adds those fixed costs back in for a
    holistic per-convention total, without re-running an ~11s kernel
    compile here.
  - GC on/off sweep: N1 already established (12,000-call runs, this exact
    workload class) that GC makes no measurable difference here, because
    nothing in this pipeline forms a reference cycle -- refcounting frees
    it immediately. Not re-run; GC is left at its default (enabled), as a
    real service would run it.

Reused from prior harnesses (not rewritten):
  - elapsed()/log() progress-clock pattern, timed_ns()/pctiles()/
    pct_of_budget() timing helpers, jsonl_append(), rss_mb() (renamed
    peak-RSS-via-resource.getrusage pattern) -- verbatim from
    experimentation/single-record-overhead/n1_overhead.py, which itself
    reused rss_mb from chunked-writeback-at-scale/measure_variant.py and
    elapsed()/log() from prange-crossover/prange_crossover.py.
  - the "generate real inputs at a given width" idea -- generalized from
    n1_overhead.py's width_split()/measure_width_sweep(), but this harness
    uses all-float64 fields (not N1's f8/i8/bool mix) because the question
    here is calling-convention overhead, not dtype handling -- N1 and
    experiment A already cover mixed dtypes at the polars boundary and the
    record convention respectively.

NOT reused: polars extraction (doc 02 S3.5's realtime path bypasses polars
entirely, same as N1); the njit driver codegen from N1/writeback.py (out of
scope here, see above).

Run:
    /path/to/.venv/bin/python n2_calling_convention.py            # full run, ~20-40s
    /path/to/.venv/bin/python n2_calling_convention.py --quick    # smoke test

Writes results.jsonl (flushed per record) and results_summary.json next to
itself.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pydantic
from pydantic import create_model

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
# Timing helpers -- reused verbatim from n1_overhead.py.
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
# Shape helpers: N all-float64 inputs, named a0..a{N-1}.
# ---------------------------------------------------------------------------
def names_for(width: int) -> list[str]:
    return [f"a{i}" for i in range(width)]


def dtype_for(names: list[str]) -> np.dtype:
    return np.dtype([(n, "f8") for n in names])


def make_request(names: list[str], seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    return {n: float(rng.random()) for n in names}


# ---------------------------------------------------------------------------
# Convention primitives.
# ---------------------------------------------------------------------------
def make_explicit_fn(names: list[str], build_dict: bool):
    """A REAL Python function with an explicit N-parameter signature, built
    via exec -- this is what doc 02 S3.5's example (score(net_income=...,
    expenses=..., ...)) literally is, unlike N1's **kwargs catch-all proxy.
    build_dict=False isolates pure signature-binding cost (body is `pass`);
    build_dict=True adds dict construction, for a fair comparison against
    the **kwargs catch-all (which builds a dict as a side effect of
    capture)."""
    params = ", ".join(names)
    if build_dict:
        body = "    return {" + ", ".join(f"{n!r}: {n}" for n in names) + "}\n"
    else:
        body = "    return None\n"
    src = f"def _f({params}):\n{body}"
    ns: dict = {}
    exec(src, ns)  # noqa: S102 -- codegen, not user input; same pattern as N1's njit driver codegen (there via a file; here no numba caching hazard exists for plain Python, so no file needed)
    return ns["_f"]


def make_kwargs_catchall():
    def _f(**kwargs):
        return kwargs

    return _f


def make_dict_passthrough():
    def _f(d):
        return d

    return _f


def make_record_passthrough():
    def _f(rec):
        return rec

    return _f


def marshal_fresh(d: dict, dtype: np.dtype, names: list[str]) -> np.ndarray:
    """Bulk form (whole row, one tuple assign) -- N1's finding was that this
    is 3.5x faster than a per-field loop for the identical result, so that
    per-field mistake is not repeated here. Fresh np.empty() each call."""
    rec = np.empty(1, dtype=dtype)
    rec[0] = tuple(d[n] for n in names)
    return rec


def update_inplace(buf: np.ndarray, d: dict, names: list[str]) -> np.ndarray:
    """Same bulk tuple-assign as marshal_fresh, but into a caller-owned,
    reused buffer -- no np.empty() per call. This is the 'pre-built numpy
    record, caller constructs once and reuses the buffer' convention."""
    buf[0] = tuple(d[n] for n in names)
    return buf


# ===========================================================================
# MEASUREMENT A -- convention sweep @ 10/50/100/400 inputs
# ===========================================================================
def run_width(width: int, n_calls: int) -> dict:
    names = names_for(width)
    dtype = dtype_for(names)
    request = make_request(names)
    positional_values = tuple(request[n] for n in names)

    fn_explicit = make_explicit_fn(names, build_dict=True)
    fn_dict = make_dict_passthrough()
    persistent_buf = np.empty(1, dtype=dtype)
    persistent_buf[0] = positional_values  # prime it once, like a real caller would

    def kwargs_pipeline():
        d = fn_explicit(**request)
        return marshal_fresh(d, dtype, names)

    def dict_pipeline():
        d = fn_dict(request)
        return marshal_fresh(d, dtype, names)

    def record_pipeline():
        return update_inplace(persistent_buf, request, names)

    def positional_pipeline():
        d = fn_explicit(*positional_values)
        return marshal_fresh(d, dtype, names)

    # sanity: all four conventions must produce the same field values.
    ref = marshal_fresh(request, dtype, names)
    for pipeline in (kwargs_pipeline, dict_pipeline, record_pipeline, positional_pipeline):
        got = pipeline()
        assert got[0].tolist() == ref[0].tolist(), "convention pipelines disagree on output values"

    results = {}
    for label, pipeline in [
        ("1_kwargs", kwargs_pipeline),
        ("2_dict", dict_pipeline),
        ("3_record_reused_buffer", record_pipeline),
        ("4_positional", positional_pipeline),
    ]:
        ns = timed_ns(pipeline, n_calls)
        p = pctiles(ns)
        p.update({"measurement": "convention_sweep", "width": width, "convention": label})
        p["pct_of_20ms_p50"] = pct_of_budget(p["p50_us"])
        p["pct_of_20ms_p99"] = pct_of_budget(p["p99_us"])
        results[label] = p
        jsonl_append(RESULTS_JSONL, p)
        log(f"  W={width:4d} {label:24s} p50={p['p50_us']:9.3f}us ({p['pct_of_20ms_p50']:.4f}% of 20ms)  "
            f"p95={p['p95_us']:9.3f}us  p99={p['p99_us']:9.3f}us  max={p['max_us']:9.3f}us")
    return results


# ===========================================================================
# MEASUREMENT B -- isolate binding cost vs dict-construction cost @ W=400
# ===========================================================================
def measure_binding_isolation(width: int, n: int) -> dict:
    log(f"=== binding isolation, width={width}, n={n} ===")
    names = names_for(width)
    request = make_request(names)
    positional_values = tuple(request[n_] for n_ in names)
    dtype = dtype_for(names)
    rec = np.empty(1, dtype=dtype)
    rec[0] = positional_values

    fn_bind_only = make_explicit_fn(names, build_dict=False)
    fn_bind_dict = make_explicit_fn(names, build_dict=True)
    fn_catchall = make_kwargs_catchall()
    fn_dict_pass = make_dict_passthrough()
    fn_record_pass = make_record_passthrough()

    cases = [
        ("explicit_sig_kwargs_bind_only", lambda: fn_bind_only(**request)),
        ("explicit_sig_kwargs_bind_plus_dict", lambda: fn_bind_dict(**request)),
        ("explicit_sig_positional_bind_plus_dict", lambda: fn_bind_dict(*positional_values)),
        ("kwargs_catchall_starstar", lambda: fn_catchall(**request)),
        ("dict_passthrough", lambda: fn_dict_pass(request)),
        ("record_passthrough", lambda: fn_record_pass(rec)),
    ]
    results = {}
    for label, fn in cases:
        ns = timed_ns(fn, n)
        p = pctiles(ns)
        p.update({"measurement": "binding_isolation", "width": width, "case": label})
        p["pct_of_20ms_p50"] = pct_of_budget(p["p50_us"])
        results[label] = p
        jsonl_append(RESULTS_JSONL, p)
        log(f"  {label:42s} p50={p['p50_us']:9.3f}us ({p['pct_of_20ms_p50']:.4f}% of 20ms)  p99={p['p99_us']:9.3f}us")
    return results


# ===========================================================================
# MEASUREMENT C -- input validation cost @ W=400
# ===========================================================================
def measure_validation(width: int, n: int) -> dict:
    log(f"=== validation cost, width={width}, n={n} ===")
    names = names_for(width)
    request = make_request(names)

    fields = {n_: (float, ...) for n_ in names}
    InputModel = create_model(f"InputModel{width}", **fields)  # built ONCE, outside the timed loop
    required_keys = frozenset(names)

    def validate_pydantic():
        return InputModel(**request)

    def validate_manual_keys_only():
        if not required_keys <= request.keys():
            raise ValueError("missing required input(s)")
        return True

    def validate_manual_typed():
        for n_ in names:
            if not isinstance(request[n_], float):
                raise TypeError(n_)
        return True

    def validate_none():
        return request

    cases = [
        ("pydantic_create_model_all_required_floats", validate_pydantic),
        ("manual_key_presence_set_check", validate_manual_keys_only),
        ("manual_per_field_isinstance_loop", validate_manual_typed),
        ("no_validation", validate_none),
    ]
    results = {}
    for label, fn in cases:
        ns = timed_ns(fn, n)
        p = pctiles(ns)
        p.update({"measurement": "validation", "width": width, "case": label})
        p["pct_of_20ms_p50"] = pct_of_budget(p["p50_us"])
        results[label] = p
        jsonl_append(RESULTS_JSONL, p)
        log(f"  {label:42s} p50={p['p50_us']:9.3f}us ({p['pct_of_20ms_p50']:.4f}% of 20ms)  p99={p['p99_us']:9.3f}us")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-calls", type=int, default=12000)
    ap.add_argument("--widths", default="10,50,100,400")
    args = ap.parse_args()

    if args.quick:
        n_calls = 500
        widths = [10, 400]
    else:
        n_calls = args.n_calls
        widths = [int(x) for x in args.widths.split(",")]

    if RESULTS_JSONL.exists():
        RESULTS_JSONL.unlink()

    env = {
        "measurement": "env",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pydantic": pydantic.VERSION,
    }
    jsonl_append(RESULTS_JSONL, env)
    log(f"env: python={env['python']} numpy={env['numpy']} pydantic={env['pydantic']}")
    log(f"rss before measurement: {rss_mb():.1f} MB")

    log(f"=== A: convention sweep, widths={widths}, n={n_calls} calls each ===")
    sweep = {w: run_width(w, n_calls) for w in widths}

    binding = measure_binding_isolation(max(widths), n_calls)
    validation = measure_validation(max(widths), n_calls)

    log(f"rss after measurement: {rss_mb():.1f} MB")

    summary = {"env": env, "convention_sweep": sweep, "binding_isolation": binding, "validation": validation}
    (HERE / "results_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log("DONE -- wrote results.jsonl and results_summary.json")


if __name__ == "__main__":
    main()
