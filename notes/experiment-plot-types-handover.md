# Handover: pluggable experiment/plot types — design session

Written for: the agent picking up this design session. The user (Sholto) will
be in the loop refining the design live; this document is to get you to the
same starting context, not a decision record of its own. If the session lands
on a design, write *that* up as a `notes/<topic>.md` decision note per the
project's own convention (see "Where this should end up" below) — this file
stays the handover, not the outcome.

## The question on the table

`decider`'s experiment module (`decider/experiments/`) currently has no
pluggable "type" concept: one `ExperimentDef` schema, a closed set of
built-in summaries/graphs. The user sketched a design where experiments and
plots are both user-extensible via a class registry, and asked for a
comparison against what already exists before going further. That comparison
(below, "Where the sketch conflicts") is the state the session starts from —
treat it as a strong prior, not a constraint handed down from outside; push
back on it if the user's refinements change the picture.

## The user's sketch (verbatim)

This is the shape the user wants to explore — read it as a sketch to react
to, not a spec:

```python
class HookTypes(Enum):
    BEFORE_CALL_NODE
    BEFORE_LOOP_CONTINUE
    # ... Just some examples
    @property
    def handler_name(self):
        ...

class BasePlotType(BaseRegistryModule):
    def _register_hook_if_has(self, hooks_list, handler_name):
        hook = getattr(self, handler_name, None)
        if hook is not None: hooks_list.append(hook)
    def register_runner_hooks(self, runner):
        self._register_hook_if_has(runner.hooks, HookTypes.BEFORE_CALL_NODE.handler_name())
    def visualize(...)

class BaseExperiment(BaseRegistryModule):
    plots: t.List[Plots]
    tests: ...
    def load_pipelines() -> t.Dict[pipelines]

class GitCommitExperiment(BaseExperiment):
    base_commit = "HEAD~1"
    ref_commit = "HEAD"

    def load_pipelines(self) -> t.Dict[pipelines]:
        return {
            "ref": load_from_commit()
            "base": load_from_commit()
        }

class FirstDiffPlot(BasePlotType):
    def __init__(self):
        self.execution_orders
    def handle_on_before_call_node(self, el):
```

Intent, as stated: experiment *types* should be pluggable (e.g. "diff two git
commits" as one type, maybe "sweep a parameter" as another), each type
defines how to load the pipeline(s) being compared; plot *types* should also
be pluggable and discoverable, each observing execution via hooks and
rendering a visualisation. Both want to go through a registry so a user can
define their own.

## What already exists (read these before designing anything)

### The experiment module — `decider/experiments/`

One flat, versioned schema, not a type hierarchy:

- `model.py` — `ExperimentDef`: `flow` (one entry point), `revision`,
  `input`, a `scenarios: list[Scenario]`, `comparison: ComparisonPolicy`,
  `summaries: list[SummarySpec]`, and `tests`/`graphs` as bare *file path*
  references (not classes — "reference, not embedded code" is a deliberate
  project rule, see `CLAUDE.md`).
- A `Scenario` can set `params`, `overrides` (+ `at` for a declared override
  point), `row`, and **`revision`** — so "compare current code against
  `HEAD^`" is already just a second scenario on one experiment, not a
  different experiment type. This is exactly what the sketch's
  `GitCommitExperiment` would duplicate.
- `runner.py` — `run_experiment`/`run`: the only place execution happens;
  reuses `forks.fork` (a session replayed to a checkpoint and overridden).
- `graphs.py` — `path_sankey(step, frame, params, baseline, outputs)`: a
  **pure function** that runs the baseline once more with a `TraceSink`,
  decodes `Kind.BRANCH_ARM` events, and returns a plain dict (`nodes`,
  `links`, `outcome`). This is the one built-in "plot" today, and it is
  *not* a class, not a registry entry, and not hook-driven live during a
  run — it consumes an already-captured trace after the fact.
- `result.py` — `ExperimentResult`/`Finding`/`ScenarioResult`/`RunStatus`:
  the states that must never look like a deterministic completed run
  (`non_reproducible`, `partial`, `failed`, `cancelled`, `timed_out`).
- `__init__.py` — the full public surface; useful as a map of the module.
- `notes/experiment-module.md` — the design rationale: chosen interface,
  **three rejected alternatives** (Python-code-first, YAML-only, a second
  execution backend), and the open decisions still left to task 10. Read
  this in full; it's short (139 lines) and answers "why isn't this already
  a type hierarchy" directly.

### The registry mechanism the sketch wants to reuse — `decider/registry/`

- `decider/registry/core.py` — `BaseRegistryModule`: a pydantic base where
  `class Root(BaseRegistryModule, root=True)` opens a registry, every
  subclass registers itself under its import path and an optional short
  alias (`type: Literal["alias"]`), `Root.resolve(tag)` loads it (importing
  lazily if needed), and `ref(Root)` gives a field type that accepts either
  a dict (resolved by `type`) or an instance. `json_schema()` dumps every
  registered subclass's schema, for UI forms.
- `decider/steps/configurable.py` — the existing, working example of this
  pattern: `ConfigurableStep(Step, BaseRegistryModule, root=True)` is how
  trees/tables/scorecards are all pluggable step types today. This is the
  closest precedent for what a `BasePlotType` registry should probably look
  like — read `ConfigurableStep`'s docstring and one concrete subclass
  (`decider/steps/trees.py` or similar) before designing a parallel
  registry for plots.
- `notes/registry-dispatch-by-lookup.md` — why lookup-dispatch was chosen
  over a rebuilt discriminated union (decider_old's `_ext.py` approach was
  rejected: stale unions, O(n^1.39) registration, 1332-character error
  messages). Relevant if the plot/experiment registry design considers
  anything union-shaped instead of lookup-shaped.

### The engine's actual instrumentation — `decider/engine/trace/`

This is the part that most directly contradicts the sketch's `HookTypes` /
`register_runner_hooks` / `handle_on_before_call_node` idea, so read it
before extending or rejecting that idea:

- `decider/engine/trace/envelope.py` — `Kind` (STEP, BRANCH_ARM,
  LOOP_ITERATION, FRAME, TABLE_ROW, TREE_PATH, BAND, ROUNDING, REASON,
  VALUE, OVERRIDE) and `TraceEvent`: **one int64 per event**, emitted inside
  nogil, numba-compiled kernels, decoded by the driver afterward. There is
  no live Python callback per node during a compiled run — that's the
  point of the format (see the module docstring: "Kernels emit one int64
  per event into a per-invocation buffer; the driver drains that buffer").
- `decider/engine/trace/sink.py` — `TraceSink`/`Adapter`: the capture
  surface `path_sankey` already uses.
- `CLAUDE.md` → "Numeric and numba traps": kernels are `nogil=True`;
  compile-time failures fall back to Python per step with a warning. A
  design that calls into Python on every `BEFORE_CALL_NODE` would force the
  Python fallback path for any run with such a plot attached — a direct
  regression against the "millions/billions of records" aggregate-result
  requirement in `notes/vscode-redesign/userstories.md`.

## Where the sketch conflicts (my read — re-litigate with the user, don't take as settled)

1. **Registry-based plot types: good, reuse `BaseRegistryModule` as-is.**
   This part has a direct, working precedent (`ConfigurableStep`) and fixes
   a real gap — today's `graphs: str` is a bare file path, not a typed,
   schema-validated, discoverable plot.
2. **`GitCommitExperiment` as its own experiment type: conflicts with
   `Scenario.revision`.** The existing schema already models "run this
   scenario on another revision" generically; a type hierarchy where each
   comparison axis (git commit, dataset, parameter sweep) is its own class
   re-introduces the "duplicated comparison/aggregation semantics" that
   task 09 (`notes/vscode-redesign/tasks/09-design-the-experiment-module.md`)
   explicitly designed `ExperimentDef`/`Scenario` to avoid.
3. **Live hooks into the runner: conflicts with how the engine is built.**
   `decider/engine/trace/` already gives post-hoc, kernel-safe access to
   per-record branch/loop events. A plot that needs "did this record visit
   this node" should consume `TraceSink`'s decoded events (as `path_sankey`
   does), not get a Python callback fired from inside a compiled kernel.

## Where this should end up

If the design session converges on something, write it as
`notes/<topic>.md` (e.g. `notes/experiment-plot-registry.md`) following the
existing style in `notes/experiment-module.md` and
`notes/registry-dispatch-by-lookup.md`: chosen design, rejected
alternatives with *why*, and open decisions left for implementation. Per
`CLAUDE.md`, design rationale belongs in `notes/`, never as comments in the
code itself.

## Files/folders worth having in context for this session

- `decider/experiments/` (whole folder, ~1250 lines across 12 files) — the
  module being extended.
- `decider/registry/core.py`, `decider/registry/resolve.py` — the registry
  mechanism to reuse.
- `decider/steps/configurable.py` + one concrete subclass under
  `decider/steps/` — the working precedent for a registry-based pluggable
  type.
- `decider/engine/trace/` (`envelope.py`, `sink.py`) — why hooks-into-kernel
  isn't viable, and what a trace-consuming plot should look like instead.
- `notes/experiment-module.md` — the experiment schema's design rationale
  and its three rejected alternatives.
- `notes/registry-dispatch-by-lookup.md` — why the registry is lookup-based,
  not union-based.
- `notes/vscode-redesign/tasks/09-design-the-experiment-module.md` and
  `notes/vscode-redesign/tasks/10-implement-experiments-and-vscode-workflows.md`
  — the original design goals (single deep interface, no duplicated
  comparison/aggregation, aggregate-first at scale) this session's design
  needs to keep satisfying.
- `CLAUDE.md` (repo root) — "Numeric and numba traps" and "Contract"
  sections; both bear directly on this design.
