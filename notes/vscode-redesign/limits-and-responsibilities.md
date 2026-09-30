# Supported limits and client-owned responsibilities

Where the redesign ends and the client begins. Numbers are measured, not
extrapolated; each is reproducible from the script it cites. "Supported" means
"tested at this size"; nothing here implies regulatory compliance, equivalence
proof, or unlimited scale beyond what was measured.

## Measured performance

### Tracing overhead (opt-in)

The decision trace is a single int64 per event captured in-kernel, drained and
decoded in the driver, then handed to an adapter. Tracing is **off by default**:
`Engine.run`/`score` capture nothing unless a caller builds a `TraceSink`.

The shipped `TraceSink` end-to-end cost (`benchmarks/trace_overhead.py`, fused,
3-step flow, 1M rows):

| path | no trace | with trace |
|---|---|---|
| batch 1M rows | 6.5 ms (~150M rows/s) | 12.1 s (~82k rows/s) |
| `score()` single record | ~200 µs | ~200 µs |

The batch number is the honest headline: **tracing a full batch is dominated by
the driver's Python decode**, not the in-kernel write. The kernel captures one
int64/event at ~0.8 ns/event (spike: `experimentation/01-trace-capture-findings.md`);
materialising and decoding those events into `TraceEvent`s on the request path
is what costs ~1800× on a large batch. On `score()` (the usual
single-record workload) the overhead is negligible.

This is why batch tracing is opt-in *and bounded*: `TraceSink(max_events=...)`
or `sample_rate=...` keeps a large run from growing unbounded, and loss is
reported in `sink.status()`, never silent. Trace for investigation and
explanation; sample or aggregate for population-scale runs.

The spike established the remaining design bounds the shipped sink inherits:

- one event is **one int64 (8 bytes)**;
- a single shared buffer costs **<3% on `score()`** while a per-step column is
  6× worse;
- the 20-bit step ref, 8-bit kind, 16-bit arm, 20-bit iteration layout is a
  spike choice, versioned in the envelope (`SCHEMA_VERSION`).

**Payload characteristics.** An event resolves to a `TraceEvent` carrying the
event kind, a 1-based step index, the step's durable `Origin` (id, path,
source), branch arm / loop iteration context, and the record's row index; the
fixed-width numeric payload (`value`) rides the int64 for the kinds that carry
one. Event order is guaranteed within one record's stream; cross-record order is
the driver's and unspecified. Batch capture is opt-in and bounded
(`max_events`, `sample_rate`); loss is reported in `sink.status()`, never silent,
and trace-point conservation runs *before* the adapter.

### Local experiment capacity (Polars)

`experimentation/03-experiment-interface-and-local-scale-findings.md`,
`benchmark.py`, best of 3, a representative 3-step flow:

| rows | fused `run()` | interpreted `run()` |
|---|---|---|
| 1 000 | 0.2 ms | 4.2 ms |
| 1 000 000 | 2.3 ms | 4 559 ms |
| 5 000 000 | 32 ms | 24 054 ms |

Scenario routing is the scale lever: a **param-only** scenario routes to fused
`run(frame, params=...)`; a scenario that overrides a value or a single row
routes to the interpreted session/fork path. 5 param-only scenarios over 1M rows
take **42 ms** (fused) vs **31 s** through fork (~750×).

- Param sweeps (fused): 5M rows ≈ 32 ms, so ~10⁸ rows is comfortable; the wall
  clock is seconds and memory is tens of MB per frame. The limit is memory, not
  throughput.
- Override/force scenarios (interpreted fork): ~200 k rows/s, so one scenario
  over 1M rows is ~6 s and each additional scenario adds the same again.
  Interactive ceiling is **~1–10M rows across a handful of scenarios**.
- Nested-column pipelines (Python fallback): ~67 records/s on the real
  affordability flow (52 steps), so anything past ~10⁵ records needs sampling or
  a fused re-expression.

### UI responsiveness (flow graph)

`experimentation/02-flow-scale-and-gesture-findings.md`. The graph is
`@dagrejs/dagre` layout + hand-rolled SVG. Measured dagre layout time:

| calls | layout time |
|---|---|
| 94 | ~330 ms |
| 1 000 | ~5.4 s |
| 2 000 | stack overflow (crash) |

Budgets (from the measured structure + layout, proposed for review): layout
<150 ms target / 500 ms ceiling; ≤500 unfolded call nodes (1 000 ceiling);
≤5 000 total calls per flow (10 000 ceiling); `describe` payload ≤2 MB (5 MB
ceiling). The lever is **fold-by-default**: thousands of *stored* nodes are fine,
thousands of *rendered* nodes are not.

### Large-result aggregation

Experiment results never materialise per-record graph state. `summarise`
reduces each scenario to per-column changed/unchanged/unique counts, the first
divergence, and a sampled drill-down (5 rows) of changed values
(`experiments/summaries.py`). The binding constraint is the same fused-vs-
interpreted split as above: aggregation over a fused output is near-free; the
interpreted fork path is what limits override scenarios to ~1–10M rows.

## Client-owned responsibilities

`decider` emits evidence and metadata; it does not own the policy around them.

- **Trace retention and privacy.** `decider` emits a structured trace but does
  not retain, redact, delete, or deliver it. The client supplies a post-record
  adapter (`engine.trace.Adapter`) that transforms, redacts, enriches, enqueues
  or exports events — potentially on another thread or process. The default
  adapter is a pass-through. An OpenTelemetry export is the client's adapter;
  it is never the required retention or policy layer.
- **CI release policy.** `decider.check.run` reports findings with severities;
  whether a finding blocks a release is the client's policy (e.g. promoting
  `durable_ids` missing-id findings to a failure). The default suite starts
  generic: wall-clock reads, shared-state mutation, numeric risk, missing
  durable ids.
- **Results storage.** Experiments write outputs, the manifest, summary and
  Sankey to a caller-chosen directory. `decider` does not own long-term storage;
  Git LFS or another policy is the client's choice. Experiment *definitions*
  (`experiments/<slug>/experiment.yaml`) are versioned project assets; *results*
  are generated and can be ignored or retained per project policy.
- **Sampling beyond local capacity.** When data exceeds local resources the
  client samples; `decider` reports the frame size and a cost estimate
  (`data.preflight`) but does not silently downsample.
- **Session-memory-only trace/input values.** The extension keeps trace and
  input values in session memory only; nothing is persisted by default.
- **Raw-data MCP capability.** `decider.mcp` returns structural and summarised
  data always, but per-record values and trace events are opt-in behind
  `decider.mcp.rawData` (default off); a disabled raw tool returns a redaction
  notice, never a fabricated value. Confirmation-gated actions (running a flow,
  starting the debugger, running an experiment, generating ids) carry a
  destructive annotation; the client confirms.
- **Reproducibility of ad-hoc forks.** An ad-hoc paused-session fork is a
  debugging convenience, not a reproducible scenario. Converting one to a
  scenario captures the structured changes (params, overrides, revision, input)
  as declared overrides; every live console/state mutation that has no declared
  counterpart is listed as dropped, never silently folded in. A saved scenario
  is reproducible only from a clean committed revision — a dirty-tree run is
  session-only and non-reproducible.

## Supported limits (summary)

- Trace capture: one int64/event; batch capture bounded and sampled; loss
  reported, never silent; conservation checked before the adapter.
- Experiments: param-only sweeps comfortable to ~10⁸ rows (fused); override/
  force scenarios interactive to ~1–10M rows (interpreted); nested-column
  pipelines ~67 records/s — sample beyond ~10⁵ records.
- Flow graph: ≤5 000 total calls, ≤500 unfolded nodes, ≤2 MB describe payload.
- MCP: read access broad; raw record/trace off by default; persistence and code
  execution confirmation-gated.
