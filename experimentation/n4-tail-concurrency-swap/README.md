# N4 — tail latency, concurrency, and config-swap impact on serving

Doc 01 §6.1: the single-record `score()` path is primary (20–100 ms/record
budget); the compiled kernel is ~1 µs and not the problem. Fifteen prior
experiments measured **medians** only. "At a 20 ms SLA the tail IS the SLA" —
this harness measures p50/p95/p99/p99.9/max for four scenarios:

- **M1** — single thread, steady state, n=30,000 `score()` calls. Tail size,
  correlated against `gc.callbacks` GC events.
- **M2** — GC baseline vs `gc.disable()` vs `gc.freeze()` after warmup, n=15,000
  calls each. Does disabling GC fix the tail?
- **M3** — 1/2/4/8/16 threads calling the same compiled kernel concurrently,
  `nogil=True` vs a twin kernel compiled `nogil=False` (same emitted source,
  flag flipped). Throughput *and* tail, serving against serving — not against
  a background compiler, which is what EXPERIMENTS.md §H measured.
- **M4** — continuous single-record traffic through a compiled-kernel
  "generation pointer" (doc 08 §4's `activate()` model) that gets swapped
  every 100 ms between two independently-compiled 10-rule generations. Tail
  during swaps vs steady state, and first-call-after-swap cost specifically.

## Reused (not rewritten) — see the module docstring for exact provenance

- `experimentation/single-record-overhead/n1_overhead.py` — the compiled
  400-in/633-out record kernel (`DRIVER`), `score()` and its phase functions,
  `pctiles()`, `timed_ns()`, `jsonl_append()`, `rss_mb()`
  (`resource.getrusage` pattern), and `gen_record_out_source()` (reused
  verbatim to build the `nogil=False` twin kernel for M3).
- `experimentation/ruleset-compile-latency/emit.py` — `emit_ruleset()`,
  unmodified, to build the two M4 "config generations".
- `experimentation/staged-compile-atomic-swap/run.py` — the `compile_kernel()`
  / `make_args()` pattern for compiling a ruleset kernel and forcing
  compilation with one call.

Both imported modules are safe to import directly: their heavy/sweep code is
behind `if __name__ == "__main__":` guards (confirmed before importing).

## Run

```
/path/to/.venv/bin/python n4_tail_concurrency_swap.py            # full run (~2 min)
/path/to/.venv/bin/python n4_tail_concurrency_swap.py --quick     # ~20s smoke test
```

Writes `results.jsonl` (flushed after every measurement) and
`results_summary.json`. `run.log` is the console transcript from the run this
experiment's numbers are drawn from.

## Caveats

- **The box was not quiet.** Firefox was using ~80% of one CPU and three other
  Claude Code sessions were running concurrently throughout this run (28 cores
  total, so contention is plausible but not certain). All wall-clock numbers
  below should be read with that in mind, especially M1's tail and M3 at high
  thread counts.
- M1/M2 use the *full* `score()` pipeline (accept/validate/marshal/dispatch/
  readback/assemble, per-field readback — the "as naturally implemented"
  shape N1 measured at 971 µs p50). M3 uses dispatch + readback only (no
  pydantic validation), to isolate the concurrency/nogil effect from N1's
  already-measured validation cost. M4 uses kernel dispatch only (no Python
  glue at all), to isolate the swap-pointer-flip cost cleanly — its absolute
  microsecond figures are therefore not comparable to M1/M2's, only its
  *ratio* (first-after-swap vs steady) is the finding.
- M1's 30,000-call run saw exactly **one** GC generation-0 collection, so the
  GC-correlation result (§M1) is suggestive, not a large-N statistical claim —
  said explicitly in the results writeup.
- Sweep sizes were cut from the brief's ">=100,000 calls" to 30,000 (M1) /
  15,000×3 (M2) to fit the ~12-minute experiment budget; thread-count configs
  in M3 are duration-based (1.2 s each) rather than call-count-based, for the
  same reason.
