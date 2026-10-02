# Experiments: a runner that only runs, metrics that measure through callbacks

Supersedes the execution, comparison and summary parts of
`notes/experiment-module.md` (the "one flat schema, scenarios not experiment
types" decision there still stands). Follows the measurements in the spike
worktree's `notes/experiment-plot-registry-spike.md` and the user's sketch in
`notes/experiment-plot-types-handover.md`.

## Why

`run_experiment` did everything: execution routing (fused, session fork,
subprocess), comparison (`compare.close`, a per-element Python loop), a fixed
summary shape (`summaries.py`), a hard-wired Sankey (`graphs.py`), saving
(`save.py`), resume and job handling, plus a hand-rolled YAML parser
(`yaml.py`) and version-window parser (`revision.parse_window`). Adding a
measurement meant editing the runner, and every measurement saw whole-run
Python lists.

## Chosen design

```
experiments/
  experiments.py   ExperimentDef / Scenario / results / validate / Experiment builder
  runner.py        batches x scenarios -> hooks; nothing else
  callbacks.py     Callback hooks, Batch / NodeBatch / Run, Progress, JobProgress, WriteOutputs
  metrics/         Metric + Summary base; ValueCounts, Describe, Histogram, Divergence, BranchFlows
  plots/           Plot + Figure; Bars, Changes, Distribution, Ranges (altair), Sankey (Vega template)
  revision.py      git resolve / worktree checkout
  _worker.py       subprocess for another revision
```

- **Lightning-style hooks.** `Callback` has `on_run_start`, `on_node_end`,
  `on_batch_end`, `on_progress`, `on_run_end`. The runner checks whether a
  hook is overridden (`callbacks.overrides`, the analogue of Lightning's
  `is_overridden`) to decide whether anything needs node values.
- **Metrics are torchmetrics-shaped.** A `Metric` is a registry model
  (`BaseRegistryModule`), so it loads from `experiment.json` by its `type`
  alias. Every scenario gets a deep copy with `reset()` called first; hooks
  accumulate; `summarise()` returns a typed `Summary`. A metric is global
  (`on_batch_end`, the flow's output) or per node (`nodes="term/*"`, an
  `fnmatch` glob on node paths, fed by `on_node_end`).
- **Plots require a summary type.** `Plot.requires` is checked against the
  named metric's `summary` in `validate`, before anything runs. `render`
  returns an altair chart (Bars, Changes) or, for what Vega-Lite can't
  express, a full Vega spec; the result keeps a `Figure`
  (`{"grammar": "vega-lite" | "vega", "spec": {...}}`) that shows itself in a
  notebook through altair's renderers (`spec_to_html(mode="vega")` for Vega).
  `ExperimentResult.show()` lays a run out with IPython `Markdown`, polars
  tables and the figures. Every figure embeds through one shared loader
  (`plots/base.py`): it loads vega, vega-lite and vega-embed once per page,
  in order, before any figure embeds, through requirejs when the page has it,
  and names the URL when a script doesn't load. altair's `"standard"`
  template drew no Sankey once a notebook inserted it (it embeds before the
  scripts load); its `"universal"` one loads per chart, and in VS Code a
  report of several charts showed vega-embed's "compile is not a function"
  (vega-lite empty), which headless Chrome never reproduced, even with slow
  scripts and requirejs. A slow or filtering proxy in front of the CDN is the
  likely difference; one load per page removes the overlap, and a blocked
  script now says so.
- **Plot options.** Every plot takes `width`/`height`; `Distribution` takes
  `layout` (`overlay`, `stacked`, `side_by_side`, `rows`) and `Bars` takes
  `side_by_side`, `stacked` or `rows`. `Changes` is horizontal, so a couple
  of scenarios don't make a tall thin chart. altair and IPython are the `experiments` extra and
  are imported only when a plot renders or something shows; a missing one
  raises a `DeciderError` naming the extra.
- **Sankey is the one spec we own.** Vega-Lite has no Sankey and there is no
  maintained Python builder for full Vega, so `plots/sankey.vg.json` is David
  Bacci's Vega Sankey (MIT, `plots/sankey.vg.LICENSE`) with its data emptied,
  node labels and record counts in labels and tooltips, the footer dropped
  and the stack domain sorted (upstream relied on its data's order). `Sankey`
  injects one row per node and link; a node's stack is its longest path from
  the input, so nested arms take one stack per branch depth and sequential
  branches don't collapse into one column; outcomes are always the last stack.
- **What a node hands a metric.** A call node hands its outputs; a branch
  node hands what it passes on (its `modifies` names, merged over every
  record); a step inside an arm hands its outputs on only the records that arm
  took (`NodeBatch.rows`). Sequences and loops hand nothing yet.
- **Steps by durable id.** `node_id=` matches a step's `Origin.id` (12 hex,
  from `decider ids`) exactly, beside `nodes=` path globs; `validate` rejects
  an id the flow lacks, a glob matching no step, or both at once.
- **JSON is the asset.** `Experiment.save` / `Experiment.load` write and read
  `experiment.json`; `load` imports `flow.entry` and scans `input.path`
  relative to the file (`example_projects/experiments/premium_sweep/`).
  `Metric.json_schema()` / `Plot.json_schema()` give a UI one form per type.
- **Reports are blocks in the JSON.** `ExperimentDef.report` is a tree of
  registry blocks (`Overview`, `Text`, `Table`, `Chart`, `MetricSection`,
  `Row`, `Section`), so a UI edits it like metrics and plots; `validate`
  checks every metric and plot a block names. `result.show()` displays it as
  one HTML fragment and `result.save_html()` as a page; without one, the
  default is an overview and a `MetricSection` per metric. Inspired by
  Datapane's block tree and Evidently's metrics-to-HTML report; Quarto or
  nbconvert templates were rejected as code-first and not UI-editable.
  Markdown renders with markdown-it-py with raw HTML off, since a report can
  come from a UI. Not built yet: tabs, and inlining Vega's scripts for
  offline pages (altair's `inline` template needs vl-convert).
- **Histogram bins are shared.** `Histogram` takes `edges`, or `range` and
  `bins`; without either it spans the baseline's first batch, which every
  scenario sees in lockstep, so all scenarios count into the same bins.
  Values past the bins count as `below`/`above`, which `Distribution` draws
  as one more bin at either end.
- **`result.show()`** shows a heading, the scenarios with what each changed,
  then one section per metric: what it measures and where, its summary
  table, and its plots, or the built-in plot for its summary type when none is
  attached. An unnamed plot is keyed by the metric it draws.
- **Lockstep batches.** For each batch the baseline runs first, then every
  scenario on the same records; `Batch.baseline` and `NodeBatch.baseline`
  carry the baseline's values for those records, so any metric can compare in
  O(batch) memory. Input may be a `LazyFrame` (`scan_parquet` etc.); progress
  is per batch.
- **Fused unless something needs nodes.** No node selectors and no `at`
  override → fused. Otherwise stepped mode, which pauses after every node but
  is still compiled. `Checkpoint` gained `rows` so a node metric inside a
  branch arm knows which records it saw.
- **Revisions run whole, then slice.** The worker runs the other revision
  over the whole input (written once as parquet) with the same batch
  boundaries, pickling `(output, node batches)` per batch; the parent reads
  batch `i` back when the lockstep reaches it. The subprocess reasoning in
  `notes/experiment-revision-subprocess-isolation.md` still holds.
- **Libraries over our own code.** JSON via pydantic
  (`model_dump_json`/`model_validate_json`) instead of `yaml.py`; float
  comparison via polars `Series.is_close` instead of `compare.close`; version
  windows via `packaging.specifiers.SpecifierSet` (now a declared dependency)
  instead of `parse_window`/`_lt`; streaming via polars lazy frames.

## Measured

`Runner.iterate` with `state.column` read for every call at each `after`
checkpoint, against fused with nothing read; the `ratio → approved → tier`
flow of `benchmarks/trace_overhead.py`:

```
   records    fused         stepped + per-node callback
   100,000    630M rows/s   300M rows/s
 1,000,000    322M rows/s   306M rows/s
10,000,000    392M rows/s   291M rows/s
```

The spike's "live hooks are ~700x slower" holds only for `interpreted` mode;
node-level hooks that take whole columns cost about 25% on stepped.

Whole experiment, baseline + determinism rerun + one scenario
(`uv run python benchmarks/experiment_throughput.py`):

```
     records   global metric (fused)   node metrics (stepped)
     100,000         80 ms                    96 ms
   1,000,000        104 ms                   168 ms
  10,000,000        382 ms                   897 ms
```

## Rejected

- **Metrics only compare summaries** (each scenario run independently).
  Simpler runner, but no row-level "which records changed", which is the most
  asked-for comparison.
- **A streaming IPC worker for revisions** (Arrow batches over stdin/stdout).
  Live progress for revision scenarios too, but a protocol to maintain; the
  run-whole worker is ~30 lines.
- **Live per-row hooks** (the sketch's `handle_before_call_node(record)`).
  Forces interpreted mode; columns per node give the same information.
- **Session replay (`fork`) for overrides.** Debug sessions record events and
  hash step content per session; the runner writes the override into the
  state at the `at` checkpoint directly, using the session's `_cast`.

## Removed

`yaml.py`, `summaries.py`, `compare.py`, `graphs.py`, `save.py`, `model.py`,
`builder.py`, `plots.py`, `result.py`, `_revision_worker.py`;
`ExperimentResult.summary`/`.sankey`, per-row `Finding` divergences,
`ScenarioStatus.RESUMED`/`SKIPPED` and `resume=` (lockstep batches have no
per-scenario resume point); `ComparisonPolicy`/`Tolerance`/`SummarySpec`
(their jobs are `Divergence(rtol=, atol=)` and metrics); `tests`/`graphs` and
`InputSpec.id_column` (never read). `forks.py` and `runs.py` moved to
`decider/debug_bridge/` (`forks.py`, `replay.py`): only the debugger uses
them. `merge` moved to `decider.engine.params`.

## Follow-up: the UI and the VS Code extension

Not ported in this pass:

- `tools/vscode-decider/src/extension.ts` finds `**/experiment.yaml` and sends
  `{yaml: path}`; the bridge command now takes `{definition: path}` and the
  asset is `experiment.json`.
- `tools/decider-ui/src/model/experiment.ts`, `Experiments.tsx`,
  `Sankey.tsx` read `summary` (changed/unchanged/unique/drill_down,
  first_divergence), `sankey`, `ScenarioResult.divergences` and
  `resumed`/`skipped` statuses. The result now has `summaries`
  (`{metric: {scenario: summary}}`) and `plots` (`{plot: {grammar, spec}}`);
  `vega-embed` renders either grammar, so the UI can drop its own Sankey
  drawing. The old nodes/links shape is still `summaries[<branch_flows>]`.

## Open

- Metric state is per process; parallel scenarios or workers would need a
  merge per metric (torchmetrics' `dist_reduce_fx`).
- A lazy CSV input re-scans from the start for every batch slice
  (`runner._batches`); `collect_batches` with re-chunking would fix it.
- `BranchFlows` orders a record's arms by IR pre-order and loop iteration; a
  branch inside a loop whose arm order differs per iteration is approximated.
