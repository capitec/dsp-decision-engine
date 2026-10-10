# Lab and MCP

Read this before changing decision tracing, step ids, datasets, the experiment runner or the MCP server.

## A trace event is one int64 written in the kernel; names are joined in Python

- A traced kernel writes one packed envelope per call into one flat buffer; paths, sources and ids sit in a `StepTable` joined at drain time. Strings never enter a kernel.
- Why: the write costs ~0.8 ns/event; one shared buffer cost under 3% on `run(record)`, a column per step cost 6x more.
- Not: per-step trace columns or live Python callbacks per node (they force the Python fallback).
- See: [experiment](../experiments/trace-capture-cost.md), `docs/architecture/engine/trace.md`

## Tracing is opt-in; batch tracing is bounded and loss is reported

- No `TraceSink`, no trace code in the kernel. `max_events` and `sample_rate` bound a batch; `status()` reports dropped and failed, never silently.
- Why: batch cost is decoding into `TraceEvent`s (1M rows: 4.8 ms untraced, 10.2 s traced), not the kernel write; `run(record)` overhead is negligible.
- Keep decode vectorised or lazy (~53M events/s unpack against ~1.2M/s per-event Python).
- See: [experiment](../experiments/trace-capture-cost.md)

## Each kernel invocation owns its buffer; conservation precedes the adapter

- Order is guaranteed per record only; cross-record order is the driver's. Concurrent `nogil` writers are safe because buffers are never shared.
- Written events are checked against declared calls before any adapter runs, so redaction cannot hide a lost trace point; a raising adapter is counted, never fatal.
- `decikit` emits evidence; retention, redaction and export (including OpenTelemetry) are the client's adapter.
- See: `docs/architecture/engine/trace.md`, `docs/architecture/engine/concurrency.md`

## A row index is not a record identity

- Events carry the run's row index; a durable `RecordRef` is `(dataset, key columns)` from a validated key, and `ordinal` is display-only.
- Why: a bare row index merges records with a duplicate key and changes between loads; duplicate or null keys are rejected.
- See: `decikit/lab/_refs.py`, `decikit/lab/data/_keys.py`

## A step has a derived path and a committed id; neither stands in for the other

- `path`/`source` are derived and change on rename, extraction or reorder; `id` is a committed token on the definition that survives all three. One id per definition, many paths; `(flow_id, step_id)` is the global reference.
- `id=` is optional on every constructor and decorator and on every `ConfigurableStep` document; `id=None` leaves names, paths and execution unchanged. `dag(single)` collapses only when it has no name and no id.
- Params still refer to steps by path; persisted references keep the id plus the name, source and revision at capture time, so a dead id renders a repairable partial result.
- See: [experiment](../experiments/durable-step-id-syntax.md), `decikit/_engine/ir/origin.py`

## An id is random, opaque, 12 hex characters

- `secrets.token_hex(6)`, checked against `[0-9a-f]{12}`; a hand-written id passes the same checks.
- Why: a content hash churns, a counter conflicts when two branches insert at the same place, a name prefix goes stale. Random ids from two branches land on different lines and merge cleanly.
- See: [experiment](../experiments/durable-step-id-syntax.md), `decikit/_tooling/cli/ids.py`

## `decikit ids` is a byte-splice on a clean tree

- It refuses a dirty tree, parses every file before writing any, splices `id="..."` at the offset `ast` gives (comments and formatting untouched), is idempotent, and writes with `os.replace`. `git diff` is the review, `git checkout` the undo.
- A repo-wide duplicate scan catches copy-paste (`--fix` re-ids the later copy); ids from other packages are not scanned.
- It edits Python only; JSON configs accept an `id` but none is generated.
- See: `docs/architecture/tooling.md`

## A dataset provider is one dataset; clients bring their own sources

- The catalogue is `name -> provider`, built from the datasets folder plus `config.json`; an explicit entry replaces an inferred one. `DatasetProvider` is a registry model, so import-path lookup and `json_schema()` come for free.
- Built-ins are `file` and `folder` only (`s3://` works through polars); warehouse sources are client subclasses that read their own credentials, with no `${VAR}` substitution. A folder has one format; mixed formats are an error.
- Providers are read-only (a `save` would add write modes to every provider), and Python is the only source of truth: the extension keeps no datasets setting.
- See: `decikit/lab/data/_providers/`, `docs/architecture/lab/data.md`

## `key` is the record key; the content hash is the fingerprint

- A set `key` replaces the id-column guess, is validated on `load()`, and becomes the `RecordRef` key. A provider may override `fingerprint()` with a cheaper identity (table version, ETag).
- `${PROJECT_ROOT}` (nearest ancestor holding `configs/`, else the cwd) is the only placeholder in `datasets_path`; it cannot go in a `.env`, where python-dotenv expands it to empty.
- See: `decikit/lab/data/_providers/base.py`, `decikit/GUIDE.md`

## One flat experiment definition; scenarios are the only axis of change

- `ExperimentDef` (`experiment.json`) holds flow, input, scenarios, metrics and plots; "diff two commits" and "sweep a param" are scenarios (`revision=`, `params=`), not experiment types. The `Experiment` builder produces the same definition and `run` is the only place execution happens.
- JSON via pydantic is the wire form; pydantic and `packaging.specifiers.SpecifierSet` replaced a YAML subset and a version-window parser. Code is referenced, never embedded.
- Not: Python-code-first experiments (no serialising or validating without `eval`), YAML-only, a second distributed execution path.
- See: [experiment](../experiments/experiment-runner-scale.md), `docs/architecture/lab/experiments.md`

## A scenario override applies only at a declared point

- An input column is settable before anything runs; a produced value needs `at: <step path>` naming the step that produces it. `validate` rejects the rest, so a scenario cannot reach arbitrary state.
- The runner writes the override into state at the `at` checkpoint; comparison reads produced outputs, since an input override is not in the output frame.
- Not: session replay (a debug-session fork) for overrides.
- See: `decikit/lab/experiments/_experiments.py`

## The runner only runs; metrics measure through callbacks

- `Callback` hooks (`on_run_start`, `on_node_end`, `on_batch_end`, `on_progress`, `on_run_end`); a `Metric` is a registry model, deep-copied and `reset()` per scenario, with a typed `Summary`. Per-node metrics select by `nodes=` glob or durable `node_id=`.
- A plot declares the `Summary` it `requires`, checked by `validate` before a run; a report is a tree of registry blocks a UI can edit.
- Not: metrics that only compare summaries (lose "which records changed"), live per-row hooks (force interpreted mode). Open: metric state is per process.
- See: `decikit/lab/experiments/callbacks.py`

## Scenarios run in lockstep batches, fused unless something needs nodes

- Per batch the baseline runs first, then each scenario on the same records, with `baseline` values attached, so any metric compares in O(batch) memory.
- No node selector and no `at` override means a fused kernel; otherwise stepped (compiled, paused per node).
- Why: the old session fork ran ~750x slower than fused; per-node column hooks cost ~25% in stepped mode.
- See: [experiment](../experiments/experiment-runner-scale.md), `decikit/lab/experiments/_runner.py`

## A scenario on another revision runs whole in a fresh subprocess

- The parent checks out the sha into a worktree (cached per run), writes the input once as parquet, and `_worker.py` runs it with the parent's batch boundaries, pickling each batch for the lockstep to read back.
- Why: importing the checked-out module in-process returned the cached live module, so both scenarios ran live code with zero divergences; evicting one name misses what it imports. A fresh process is right at any depth.
- Not: `uv run --project <tree>` for the old engine (23 s cold sync); revisions run under today's `decikit`, and `flow.decikit` rejects an incompatible window.
- See: [experiment](../experiments/experiment-runner-scale.md), `decikit/lab/experiments/_worker.py`

## A run status never reads as a clean completion unless it was one

- `RunStatus`: `completed`, `non_reproducible` (a baseline rerun differed), `partial` (a scenario failed), `failed` (the baseline), `cancelled`, `timed_out`. One failing scenario never stops the others.
- `RunManifest` is immutable and versioned: authored revision kept as intent, resolved to a sha; `decikit` writes small metadata to a caller-chosen directory, not bulk storage. There is no resume.
- See: `decikit/lab/experiments/_manifest.py`

## Plots are registry types; the Sankey is the one vendored spec

- Plots return an altair chart or a full Vega spec, kept as `Figure(grammar, spec)`. Vega-Lite has no Sankey, so `plots/sankey.vg.json` is a vendored MIT Vega spec with its data emptied.
- One shared loader loads vega, vega-lite and vega-embed once per page, in order; altair's per-chart templates drew no Sankey in a notebook and broke multi-chart reports in VS Code.
- altair, IPython and markdown-it-py are the `experiments` extra, imported on render; report markdown renders with raw HTML off.
- See: `decikit/lab/experiments/plots/_base.py`

## The experiment bridge command streams progress and cancels out of band

- `experiment` sends `{"progress": {"done", "total"}}` lines on its request id; `cancel` with a `target` is answered by the reader thread, like `pause`, and stops at the next batch.
- See: `decikit/_tooling/bridge/experiment.py`, `docs/architecture/editor-bridge.md`

## MCP is a stdio server the agent's host owns, stateless per call

- The host spawns `decikit mcp`, giving one process and one cwd as the routing source; the VS Code extension is a peer, never the parent. A call loads what it needs and returns; debugging stays in the editor's debugpy session.
- Tools are thin wrappers over `_tools.py`, which uses the CLI's and bridge's entry points; errors are `decikit.exceptions` as tool errors, not a new vocabulary.
- See: `decikit/_tooling/mcp/_server.py`, `docs/architecture/tooling.md`

## MCP tools are split by where the answer lives and gated by data class

- Headless tools answer from core; editor-bound tools forward to the VS Code window and return its acceptance (`editor_selection` is the one read); code-running or writing tools are confirmation-gated through `destructiveHint`, with no second approval in the server.
- Structural and summarised data are always on; raw (per-record values, trace, `debug_evaluate`) is off, returns a redaction notice rather than a made-up value, and is enabled by `--raw` or `DECIKIT_MCP_RAW=1`. No `decikit.mcp.rawData` editor setting exists, though the notice names it.
- See: `decikit/_tooling/mcp/_policy.py`, `docs/guides/cli-and-mcp.md`

## Config-editing tools build the document first and refuse a write that does not build

- Every write builds the tree, table, scorecard or params document and leaves the file untouched on failure, keeping its JSON layout; `try_config` checks an edit on a worked example.
- Infinite and NaN numbers are spelled `"-Infinity"`, `"Infinity"`, `"NaN"` both ways, since MCP carries plain JSON.
- See: `decikit/_tooling/mcp/_configs.py`, `decikit/_tooling/mcp/_nonfinite.py`

## The editor channel is a per-window socket with a token

- The extension binds `~/.decikit/editor/<hash>.sock` with a fresh `<hash>.token` and `<hash>.folder` in a 0700 directory; the workspace-folder hash routes (nearest ancestor window, else the only window below), and a wrong token is `unauthorized`.
- Why: socket permissions keep out other users, the token other local processes; this is where broad read access meets a live editor. Not: an unauthenticated localhost listener. The Windows pipe is untested.
- See: `ui/vscode/src/agentSocket.ts`, `decikit/_tooling/mcp/_editor.py`
