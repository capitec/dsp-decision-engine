# Design decisions

Why the decider toolkit is built the way it is, for maintainers: read the decision file for an area before changing it, and the experiment behind a rule before arguing with its numbers. Not published on the docs site; how things work is in `docs/architecture/`.

- `decisions/`: the rules each area holds, each with its reason, in a few bullets.
- `experiments/`: one question per file: purpose, method, result, decision. Most name the `benchmarks/` script that reproduces them (`uv run python benchmarks/<name>.py`); "spike" means the code lives outside the repo in `../decider-experimentation/`.
- [`known-limitations.md`](known-limitations.md): open gaps in the current code, with workarounds.
- Proposals for work not yet built live in [`docs/design_specs/`](../design_specs/README.md), one folder per
  rework; when one lands, its rationale moves here.

New rationale goes into the decision file for its area; a new measurement becomes an experiment linked from there.

## Decisions

| Area | File |
|---|---|
| IR, wiring, runners, fusion, compile cache, fallback, `run` on a record or a frame, `each` | [engine](decisions/engine.md) |
| Package layers, public names, checks, diff, thread safety, extension points | [api-layout](decisions/api-layout.md) |
| `Decimal`, `Currency`, rounding, overflow | [money](decisions/money.md) |
| `str`, `Text` spans, `Literal` outputs | [strings](decisions/strings.md) |
| `Columnar[Item]`, `Struct[Item]`, strings inside records | [nested-data](decisions/nested-data.md) |
| Trees, decision tables, flow files | [configurable-steps](decisions/configurable-steps.md) |
| Request path, transports, Arrow boundary | [serving](decisions/serving.md) |
| Debug session, editor bridge, VS Code, flow UI | [debugger-and-editors](decisions/debugger-and-editors.md) |
| Tracing, step ids, datasets, experiments, MCP | [lab-and-mcp](decisions/lab-and-mcp.md) |
| Tree editor saves, live editing, offline merge, `merge-core` | [tree-editor-collaboration](decisions/tree-editor-collaboration.md) |

## Experiments

| Question | Script |
|---|---|
| **Engine** | |
| [What do params bundles cost a kernel; are same-named bundle classes a trap?](experiments/params-bundle-dispatch-cost.md) | spike |
| [What drives the tail of single-record latency?](experiments/single-record-tail-latency.md) | spike |
| [When to fuse, release the GIL, use fastmath, fall back or cache?](experiments/kernel-compilation-choices.md) | spike |
| [How should `score` take a record, and what should it cache?](experiments/single-record-calling-convention.md) | spike |
| [Where does a one-record run spend its time outside the steps?](experiments/per-step-overhead.md) | spike |
| [Can one scatter/gather primitive replace `each`'s two modes?](experiments/each-scatter-gather.md) | `each_unified.py` |
| [Does a loop over candidate bundles pack?](experiments/packed-loop-over-bundles.md) | `bundle_search.py`, `fan_out_reduce.py` |
| [How did the decider toolkit compare with decider2?](experiments/decider-vs-decider2.md) | `modes.py`, `control_flow.py` |
| **Types** | |
| [Can a kernel match prefix, suffix, contains and regex, and in which language?](experiments/string-matching-beyond-equality.md) | spike |
| [What does a fixed-point money type cost; where does rounding disagree?](experiments/money-types-and-rounding.md) | `decimal_steps.py` |
| [How should a compiled step hold a string?](experiments/string-representations.md) | `span_operations.py` |
| [What does a `str` item field cost; should an item be a record?](experiments/item-str-fields-and-record-shape.md) | `nested_item_fields.py` |
| [Can `Columnar[Item]` read Arrow directly and share a kernel?](experiments/columnar-arrow-read-and-shared-kernel.md) | `nested_rows.py` |
| [How should a struct column reach a kernel?](experiments/struct-input-as-record-array.md) | `structs.py` |
| **Configurable steps** | |
| [Should trees run in Rust, a polars plugin, ZEN or generated code?](experiments/alternatives-to-a-numba-tree-walker.md) | spike |
| [Tree and table data: generated literals or runtime arrays?](experiments/runtime-arrays-vs-generated-code.md) | spike |
| [What does the shared tree walker cost to call and compile?](experiments/tree-walker-costs.md) | `tree_walk.py` |
| [Why does a tree of hundreds of rules take minutes to compile?](experiments/wide-output-compile.md) | `tree_wide_outputs.py` |
| [How wide should the parts of a split tree be?](experiments/tree-split-part-size.md) | `tree_split_sweep.py` |
| [Do the shared walker and matcher keep up with generated kernels?](experiments/trees-tables-vs-decider2.md) | `tree_walk.py`, `table_match.py` |
| **Serving** | |
| [How should a kernel read a polars frame?](experiments/reading-a-polars-frame-in-a-kernel.md) | spike |
| [How should outputs reach polars; how big can one batch be?](experiments/output-writeback-and-batch-size.md) | spike |
| [How should the compiled Arrow shim ship?](experiments/shipping-the-arrow-shim.md) | spike |
| [Can a new version compile while the old one serves?](experiments/stage-a-version-while-serving.md) | spike |
| [Where do one request's microseconds go?](experiments/serving-latency-budget.md) | `serving_latency.py` |
| [Would Arrow or a specialised server make requests faster?](experiments/arrow-codec-and-transports.md) | `arrow_codec.py`, `serving_transports.py` |
| [Can kernels read Arrow columns in place?](experiments/zero-copy-arrow-gather-spike.md) | spike |
| **Tooling** | |
| [What does decision-trace capture cost?](experiments/trace-capture-cost.md) | `trace_overhead.py` |
| [How fast do experiments run locally; what do revisions cost?](experiments/experiment-runner-scale.md) | `experiment_throughput.py` |
| [Which durable step-id syntax survives refactors?](experiments/durable-step-id-syntax.md) | spike |
| [How fast can a debug session pick up an edit?](experiments/hot-reload-edit-latency.md) | spike |
| [Which layout keeps a 10,000-call flow interactive?](experiments/flow-layout-scale.md) | `ui/lib` `pnpm bench` |
| [Can screenshot-scored user stories steer the editor UI?](experiments/ui-judge-loop.md) | spike |
| **Tree editor** | |
| [What does live multi-session editing cost?](experiments/tree-editor-live-collab.md) | `tree_editor_live_*.py` |
| [A CRDT library, a JSON-patch crate or a Rust core for the merge logic?](experiments/tree-editor-merge-core.md) | `ui/tree-editor/merge-core-bench/` |
