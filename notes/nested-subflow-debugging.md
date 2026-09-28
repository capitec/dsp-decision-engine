# Debugging a nested subflow: walkable child vs ambient pause

Status: **proposal**. Companion to `nested-data-decisions.md` ("Held: `each`, a
child grain") and `gap-coverage-technical-debt.md` ("break_at / sessions into
the child grain"). This records the two candidate fixes and their trade, with
file:line references throughout.

## What we are trying to achieve

Make a step whose body is another pipeline (`each` today; user code nesting
decider-in-decider later) visible to the debugger the way a `branch` or `loop`
already is: `break_at("order/items/heavy")`, `step_into`, `set` inside the
child, and child nodes in `Session.structure()` / the flow panel.

## Why

Today the child is not visible, and the reason is a *return boundary*.

- `each` lowers to one `CallNode` whose `fn` runs the child as a nested runner
  behind a plain function call: PER_ROW instantiates its own `FusedRunner`
  (`decider/steps/each.py:111`), BATCH its own `Engine().bind(child_ir,
  mode="fused")` (`each.py:153`). The child plan is resolved separately
  (`each.py:54`) and its params are flattened up and re-built by hand
  (`_hoist`/`_child_doc`, `each.py:68-92`), so it is not in the parent's
  `Plan.root`.
- Everything that reports on a run walks the parent plan: `iter_nodes`
  (`decider/engine/ir/nodes.py:102`), `Session.structure()` (`decider/engine/debug/session.py:239-254`),
  `step_map`, `parameters()`, `break_at` prefix matching. A child behind a
  `fn` is invisible to all of them.
- Runners are generators, and checkpoints come from the runner's own `_node`
  recursion (`decider/engine/run/runners/interpreted.py:72-86`), not from
  inside a node's `fn`. `_call` is synchronous (`interpreted.py:92-139`) and
  `_run` invokes a compiled kernel (`decider/engine/run/runners/stepped.py:111-165`);
  neither can yield.

This is already on the record: `notes/gap-coverage-technical-debt.md:32-41` and
`notes/nested-data-decisions.md:220-223` ("the blocker is not `State` but
`Checkpoint`, which is `(origin, when, arm, iteration)` with no coordinate for
which item").

The one structural fact behind everything: **`yield` is threaded control flow**
— it must be `yield from`'d through every frame, and a plain `fn` call or a
compiled kernel is a wall it cannot cross. A thread, by contrast, is **ambient**
— it can stop in place from any depth and let an orchestrator inspect shared
state without being in the call stack.

## How A: a first-class node (remove the boundary)

Make the child a dispatch target in the runner's generator, exactly like
`loop`/`branch` bodies already are (their bodies have no return boundary; the
runner walks them).

- Add `EachNode` (or a general `SubflowNode`) to the closed IR
  (`decider/engine/ir/nodes.py:11-99`), resolve it (`decider/engine/wiring/resolve.py:93-106`,
  a new `each` method beside `loop`/`branch` at `resolve.py:227-278`), and a
  `Each` entry in the `Resolved` union (`decider/engine/wiring/plan.py:105`).
- Teach `_node` to walk it per item (`interpreted.py:72-86`), mirroring
  `_loop`/`_branch` (`interpreted.py:141-189`). Add an `item: int | None`
  coordinate to `Checkpoint` (`decider/engine/run/runners/base.py:14-31`) and
  thread it through `_Scope` and the events (`NodeStarted`/`Paused`,
  `decider/engine/debug/events.py:42-87`), as `arm`/`iteration` are.
- The child joins the parent plan, so `iter_nodes`, `parameters()`,
  `structure()`, `break_at`, `step_map`, and the bridge's `node_json`
  (`tools/decider-bridge/decider_bridge/describing.py:108-131`) all see it for
  free; `_hoist`/`_child_doc` (`each.py:68-92`) disappear, since child params
  then live under their own paths like a loop body's.

Cost: one new node type is one dispatch branch per runner plus the compiler's
structural walks — `_runs`/`_kept` (`decider/engine/compile/units.py:217-239`,
`308-331`), `_constructs`/`_calls` (`decider/engine/compile/packed.py:65-91`),
and `_record_path` classification (`decider/engine/run/engine.py:117`). It is a
one-time cost; the IR is closed by design so node types are added rarely, not
per feature (`Design.md:47`, D3). `optimise` is the proof that expanding to IR
is the house style (`decider/steps/optimise.py:84-131`).

## How B: ambient pause (thread / actor)

Keep the child behind its `fn`, but make "pause" ambient so it works from any
depth without restructuring the runner.

- A `ContextVar[Session]` holding the active session, and a
  `NestedPipeline.__call__` that, inside a `with debugger_context:` block, runs
  the child and calls `ctx.pause()`/`ctx.emit()` at step boundaries. `pause()`
  blocks the runner thread on a condition variable; an orchestrator thread sends
  "continue" and reads shared state.
- Nesting becomes ordinary: user code calling decider inside decider is
  debuggable because `pause()` stops in place, whatever the call stack.

Cost: two, both real.

1. **Thread-safe engine state.** The orchestrator reading `state`/`Executable`
   while the runner thread is mid-write is a data race, already documented:
   `notes/child-grain-batch.md:246-261` (`Executable.report`, `Executable._checked`,
   `Executable.cache` are shared mutable state — "a plain data race today").
   "See things through nested runs" is the feature; doing it without corruption
   is the work.
2. **A second driver.** The debug path becomes blocking/actor while `run`/`score`
   stay synchronous, which breaks D12 — "running and debugging share one code
   path" (`Design.md:56`) — the load-bearing invariant of the runner design.

A cheaper subset of B exists: passive tracing only — run the child to
completion, append its events into `parent.events` via the context. That gives
the UI "this step ran a subflow" and a log, but no `break_at`/`set`/resume into
the child, because the parent generator is parked inside `_call` and cannot
suspend mid-child.

## Trade

| | How A: node type | How B: thread / actor |
|---|---|---|
| Mechanics | remove the return boundary (`_node` walks the child) | make pause ambient (`pause()` blocks the thread) |
| Interactive stepping | yes, one process | yes, any depth |
| `break_at` / `set` / `step_into` into child | yes | yes |
| Child visible to `structure()`/`parameters()`/params doc | yes, free | no (still behind `fn`; needs separate wiring) |
| Concurrency | none | thread-safe `Executable`/`State` required |
| Drivers | one (D12 holds) | two (debug path diverges) |
| Generality | exceptional nesting (one construct) | ambient nesting (arbitrary user re-entry) |
| Cost | one dispatch branch × 3 runners + compiler walks, once | rewrite runner as actor + thread-safety |

`yield` cannot cross a `fn`/kernel boundary, so the two fixes are the only two:
either delete the boundary (A) or stop relying on `yield` (B). A is cheaper and
preserves the pull model; B is the general answer but costs concurrency and a
second driver.

## Recommendation

Build A now: it matches the closed-IR design, D12, and the one construct that
exists. Revisit B only when nested execution stops being exceptional — i.e.
when arbitrary user code nests decider runs and wants them debuggable — because
that is the only case where "ambient pause from any depth" earns back the
thread-safety and second-driver cost.
