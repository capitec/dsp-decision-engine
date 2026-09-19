# N3 — params validation on the request path

What this measures, and why: `EXPERIMENTS.md` §N3 (once appended) / doc 01
§6.1 / doc 02 §4 / doc 08 §6.2. See the module docstring in
`n3_params_validation.py` for full detail; summary below.

Doc 02 §4: *"An invocation is 1 row or N rows, which covers realtime payload
params and batch uniformly."* Doc 01 §6: *"Params may arrive per invocation,
including in a realtime request payload."* E0 (doc 01 §5.7 / doc 02 §2.1)
measured 58.5 µs to validate-and-bind a 63-node **structure** config through
the discriminated union — a one-time, build/stage-time cost. This harness
measures the previously-unmeasured adjacent thing: validating a **params**
document (business-user tunable values — no discriminated union; params
don't participate in one, only structure does), which doc 02 §4 says may
legally arrive on *every single realtime request*.

## What it measures

1. **Full pydantic validation** of a multi-module params bundle
   (`ParamsBundleM(**doc)`, `create_model`-generated, M required nested
   `AffordabilityParams` submodels) at M = 1 / 10 / 50 module instances.
2. **The pydantic-model → NamedTuple conversion** (doc 03 §4), isolated from
   validation by validating once outside the timed loop: "fresh" re-derives
   the NamedTuples from the already-validated model every call; "cached"
   memoizes by a content key and hits a dict lookup.
3. **`resolve_params(doc, origin=..., complete=True)`**, doc 08 §6.2's exact
   signature — validation + conversion + the completeness check via
   `model_fields_set` (doc 04 §2.1), at single-module and M=1/10/50 pipeline
   scope.
4. **`ParamsCell.get()`** (doc 08 §4 / REVIEW.md §6.3) — the once-per-invocation
   read of an already-validated bundle.
5. **`ParamsCell.swap()`** at genuine N=1, compared against doc 08 §4's
   batch-context figure (`EXPERIMENTS.md` §H: 0.177 µs median / 0.357 µs
   p99, measured across 11,605 swaps in a 3 s chunked-batch run).

All measurements: ≥10,000 iterations (12,000 used), `time.perf_counter_ns`,
warmup before timing, p50/p95/p99/max reported, each also as a % of a 20 ms
budget. GC was left enabled throughout except one explicit on/off pair at
the heaviest case (M=50 full validation) — N1/N2 already established no
measurable GC effect for marshal/kwargs-style workloads on this interpreter;
pydantic validation allocates differently (many short-lived submodel
instances per call) so it seemed worth one direct check rather than
assuming the same holds.

## Reused (not rewritten)

- `elapsed()`/`log()`, `timed_ns()`/`pctiles()`/`pct_of_budget()`,
  `jsonl_append()`, `rss_mb()` — verbatim from
  `experimentation/single-record-overhead/n1_overhead.py` (itself reused
  from `prange-crossover/` and `chunked-writeback-at-scale/`).
- `AffordabilityParams` (8-field pydantic model) and `ParamsNT` — verbatim
  from `n1_overhead.py`'s phase-2 params fixture, reused here as the shape
  of *one* module's params and composed M times.
- `bind_params()`'s pattern (`Model(**raw)` → `NamedTuple(**m.model_dump())`)
  — reused from `n1_overhead.py`, decomposed here into its two phases
  (validate, convert) instead of timed as one blob.

## Run

```
/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/.venv/bin/python n3_params_validation.py            # full run (~10s)
/home/sholto/Documents/Workspace/capitec/dsp-decision-engine/.venv/bin/python n3_params_validation.py --quick    # smoke test
```

Writes `results.jsonl` (flushed per record) and `results_summary.json` next
to itself. `run.log` in this directory is the captured stdout of the actual
run this experiment reports.

## Headline results (12,000 calls/measurement, GC on unless noted)

| # | measurement | M=1 | M=10 | M=50 |
|---|---|---|---|---|
| 1 | full validation, p50 | 2.99 µs (0.015%) | 17.4 µs (0.087%) | 78.3 µs (0.392%) |
| 2 | conversion, fresh, p50 | 4.47 µs (0.022%) | 34.9 µs (0.174%) | 170.6 µs (0.853%) |
| 2 | conversion, cached, p50 | 0.235 µs | 0.223 µs | 0.235 µs |
| 3 | `resolve_params`, whole pipeline, p50 | 9.68 µs (0.048%) | 56.7 µs (0.283%) | 256.5 µs (1.283%) |
| 3 | `resolve_params`, whole pipeline, p99 | 11.8 µs | 162.9 µs | 278.0 µs |

- **4. `ParamsCell.get()`**: p50 0.159 µs, p99 0.211 µs — 0.0008% of a 20 ms
  budget.
- **5. `ParamsCell.swap()` at N=1**: p50 0.278 µs, p99 0.363 µs — 0.0014% of
  budget. Doc 08 §H's batch-context figure was 0.177 µs median / 0.357 µs
  p99; the p99s essentially agree (0.363 vs 0.357 µs) but N=1's median is
  1.6× the batch-context median. Order of magnitude confirmed, exact number
  not reproduced — see the doc update for the caveat.
- **GC check (M=50 full validation only)**: p50 77.8 µs (GC on) vs 80.4 µs
  (GC off) — ratio 0.97, i.e. no measurable effect, consistent with N1/N2.
- Completeness-check sanity: omitting a field from the raw doc and calling
  `resolve_params(..., complete=True)` correctly reports it in
  `defaulted_fields`.
- Peak RSS: 40.7 MB → 42.8 MB across the whole run (N=1 workload, trivial
  memory as expected — no detached/capped run was needed; `free -g` showed
  19 GB free / 23 GB available before starting).

## Machine state

Single 28-core box, load average ~1.5 throughout (one Firefox tab pinned
near 83% of one core, three other Claude Code sessions each ~8% CPU,
background browser tabs a few % each — noted per the task's timing rules,
since "nothing else running" did not hold). Not re-run isolated because the
measured effect sizes (double-digit-to-triple-digit µs) are far above the
noise floor a few busy background processes would plausibly introduce on an
otherwise-idle core, and the smoke test and full run produced consistent
numbers (M=50 full-validation p50: 80.3 µs quick-run vs 78.3 µs full run).

## What was not tested (dropped for the ~12-minute budget)

- Concurrency / tail-under-load (doc 06's N4) — same scope cut as N1 and N2.
- GC on/off at M=1 and M=10 (only M=50 checked directly).
- `resolve_params(..., complete=False)` — the task didn't ask for it and the
  completeness check is cheap (see finding below), so its absence is
  unlikely to change the verdict.
- The true doc 02 §2.1 discriminated-union machinery for the params
  documents themselves — params don't go through a discriminated union in
  the design (only *structure* does, per doc 02 §2.1/E0), so this harness's
  plain nested-`create_model` bundle is the design-correct shape for a
  params document, not a simplification of E0's mechanism.
