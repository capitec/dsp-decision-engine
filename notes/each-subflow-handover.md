# Handover: make `each`'s child pipeline steppable, remove its private engine

Status: **agreed plan, not started**. This is a task list for whoever
implements it next. Read this whole document before touching code — later
tasks depend on earlier ones and assume the vocabulary defined here.

This work touches three layers, in order: the Python engine
(`decider/engine/`, `decider/steps/each.py` — Tasks 1-6), the Python bridge
that describes a running pipeline as JSON for external tools
(`tools/decider-bridge/` — Task 7), and the two UI clients that render that
JSON (`tools/decider-ui/`, a React library; `tools/vscode-decider/`, the VS
Code extension that uses it — Task 8). Do all engine tasks first; the bridge
and UI tasks read `SubflowNode`/`iter_with_subflows` and cannot be usefully
started before Task 2 exists.

Do not read `notes/nested-subflow-debugging.md` as instructions — it records
an earlier, rejected design (a `Checkpoint.item` coordinate threaded through
every runner via `yield from`). This document supersedes it. If you want
background on why that approach was rejected, it's in that file, but nothing
in this document depends on it.

## The problem, in one paragraph

`decider/steps/each.py` runs a child pipeline (the `item` argument to
`each()`) once per row (`EachMode.PER_ROW`) or once over an exploded list
(`EachMode.BATCH`). To do this, it currently builds its own private,
throwaway execution engine inside a closure — see `_per_row` and `_batch` in
that file. This makes the child pipeline invisible to everything that
inspects a running pipeline (`Session.structure()`, `Session.break_at()`,
`parameters()`), and it duplicates work the engine already does properly
(routing params to the right step, running with the right runner).

## Vocabulary you need before starting

- **IR node**: an instance of one of the four frozen dataclasses in
  `decider/engine/ir/nodes.py` (`CallNode`, `SequenceNode`, `BranchNode`,
  `LoopNode`). A whole pipeline is a tree of these. `to_ir()` on a `Step`
  builds this tree.
- **`Plan`**: what `resolve()` (`decider/engine/wiring/resolve.py`) turns an
  IR tree into — every name bound to the version that produced it, ready to
  run. `Plan.calls`, `Plan.versions` etc. are what `Session` and the compiler
  actually walk, not the raw IR tree.
- **Runner**: `InterpretedRunner`, `SteppedRunner`, `FusedRunner` in
  `decider/engine/run/runners/`. Each is a generator: `runner.iterate(plan,
  state, params)` yields one `Checkpoint` per node reached (`before`/`after`).
  Draining the generator runs the whole plan.
- **`Session`** (`decider/engine/debug/session.py`): wraps a runner's
  generator and lets you pause at breakpoints, inspect and override values,
  one checkpoint at a time. This is "the debugger" — there is no separate
  debugger class.
- **`Executable`** (`decider/engine/run/engine.py`): a bound pipeline
  (`plan` + `runner`); `.run(df)` and `.score(record)` drain the runner's
  generator to completion without stopping.

## Ground rules that apply to every task below

- **Do not change what `run()`/`score()` return for any existing test.**
  Every task must keep `uv run pytest tests/run/test_each.py -q` and
  `uv run pytest tests/test_guide.py -q` green before moving to the next
  task.
- **No locks on per-execution state.** If you find yourself wanting a
  `threading.Lock` around something that changes on every call (a `State`, a
  `RunParams`, a report), the fix is to give each execution its own private
  copy of that object, not to lock a shared one. Locks are only appropriate
  around genuinely shared, content-addressed caches (e.g. `ParamsCache`,
  compiled-kernel caches) where redundant recomputation is wasteful but
  never wrong.
- **Follow the repo's `# ponytail:` comment convention**: if you notice a
  simplification you can't make yet because a test pins down the current
  behaviour, leave a one-line `# ponytail: <what could be simpler>` comment
  instead of changing the behaviour silently.
- Run `uv run pytest tests/test_conventions.py -q` after any file you touch
  changes size — it enforces a 500-line-per-file limit and checks for
  banned references (no "doc 03", "§4.2", "the agent", etc. in code).

---

## Task 1 — Add `SubflowNode`, an IR node that carries a child tree for inspection only

**Outcome**: a new IR node type exists that behaves *exactly* like `CallNode`
to everything that resolves, compiles, or runs a plan, but additionally
carries the child IR tree so a new opt-in walker can find it.

**File to touch**: `decider/engine/ir/nodes.py`

**What to do**:

1. Open `decider/engine/ir/nodes.py`. Look at `CallNode` (currently around
   lines 24–56): it's a frozen, slotted dataclass with fields `origin`,
   `kind`, `fn`, `inputs`, `outputs`, `params`, `reference`, `nogil`,
   `consts`, and a `children()` method that returns `()` (a `CallNode` has no
   children — it's a leaf).

2. Add a new class right after `CallNode`:

```python
@dataclass(frozen=True, slots=True, eq=False)
class SubflowNode(CallNode):
    """A `CallNode` whose `fn` drives another IR tree behind the call (`each` today).

    Wiring, compiling and running still see a plain `CallNode`: `fn` is
    called exactly as `CallNode.fn` is, and `subflow` is not resolved into
    this node's own reads, writes or params. `subflow` only makes the child
    visible to introspection that opts in, via `iter_with_subflows`; every
    other walk (`iter_nodes`, wiring, compiling) is unaware of it.
    """

    subflow: IRNode | None = None
```

   Important: **do not override `children()`**. It must stay inherited from
   `CallNode` (returns `()`). This is deliberate — every existing walk that
   calls `.children()` (the compiler's `_runs`/`_constructs`/`_calls` in
   `decider/engine/compile/units.py` and `decider/engine/compile/packed.py`,
   `resolve()`'s walk in `decider/engine/wiring/resolve.py`) must keep
   treating a `SubflowNode` as one opaque call, exactly like today. If you
   make `children()` return `(self.subflow,)`, those walks will try to
   resolve/compile the child a second time as part of the parent plan, which
   will break in confusing ways. This is the single most important
   constraint in this whole document.

3. Add a second, free function in the same file, after `iter_nodes`:

```python
def iter_with_subflows(node: IRNode) -> Iterator[IRNode]:
    """Every node of an IR tree, parents before children, also unfolding a `SubflowNode`'s child tree.

    `iter_nodes` stays opaque to a `SubflowNode`'s `subflow` (wiring, params
    and compiling all treat it as one plain call); this is only for
    introspection that wants to show the child too, such as
    `Session.structure()`.

    Example::

        paths = [n.origin.path for n in iter_with_subflows(ir)]
    """
    yield node
    for child in node.children():
        yield from iter_with_subflows(child)
    if isinstance(node, SubflowNode) and node.subflow is not None:
        yield from iter_with_subflows(node.subflow)
```

   Note this does **not** replace `iter_nodes` — both functions must exist.
   `iter_nodes` is used by `resolve()` and everything that must stay opaque
   to subflows; `iter_with_subflows` is only for the introspection paths
   this task adds later (Task 2).

**Verify**: `uv run pytest tests/ -k "nodes or ir" -q` and
`uv run pytest tests/test_conventions.py -q` both pass. Nothing else should
be affected yet — this task only adds new code, it doesn't wire anything up.

---

## Task 2 — Wire `each` to use `SubflowNode`, make the child's structure and params visible at its real path

**Outcome**: `pipeline.session(...).structure()` lists the child's steps
nested under the each node's own path (e.g. an `each("items", ..., name="items")`
whose child has a step called `heavy` shows up as `"items/heavy"`, not just
opaque `"items"`). The child's params are tunable at their real nested path
(`{"order": {"items": {"heavy": {"heavy_kg": 30.0}}}}`), and the hand-rolled
param-forwarding code (`_hoist`/`_child_doc`) is deleted.

**Files to touch**: `decider/steps/each.py`, `decider/engine/debug/session.py`,
`decider/engine/params/schema.py`, `tests/run/test_each.py`.

**What to do**:

### 2a. `each.py`: build the child under its own path, return a `SubflowNode`

Current code (`decider/steps/each.py`, `EachStep.to_ir`, roughly lines 46–65):

```python
    def to_ir(self, ctx: IRContext) -> CallNode:
        from decider.engine.ir.context import IRContext as _Ctx
        from decider.engine.wiring.resolve import resolve

        # The child is its own plan: its inputs are the item's fields, its params read from a
        # relative document the each node forwards. Built under a fresh root so its paths stay
        # relative to itself.
        child_ir = _Ctx().build(self.item)
        child_plan: Plan = resolve(child_ir)
        new_fields = tuple(n for n, v in child_plan.outputs.items() if v.producer is not None)
        params, paths = _hoist(child_plan, self.name)
        out_name = self.output if self.output is not None else self.column
        fn = _per_row(child_plan, new_fields, paths) if self.mode is EachMode.PER_ROW \
            else _batch(child_ir, self.column, out_name, paths)
        inp = Input(self.column, list[dict], NullPolicy.MISSING_AS, [], arg="items")
        out = Output(out_name, list[dict])
        kind = "scalar" if self.mode is EachMode.PER_ROW else "frame"
        return CallNode(ctx.origin(self), kind, fn, (inp,), (out,), params)
```

Change it to build the child under `ctx.child(self.name)` (the each node's
own nested context) instead of a fresh, detached `_Ctx()` root, and return a
`SubflowNode` carrying it:

```python
    def to_ir(self, ctx: IRContext) -> SubflowNode:
        from decider.engine.wiring.resolve import resolve

        # The child is built under this node's own path, so its steps' origins nest under it
        # (e.g. "items/heavy"), which is what `Session.structure()` and `parameters()` show, and
        # what a debug session steps into.
        child_ir = ctx.child(self.name).build(self.item)
        child_plan: Plan = resolve(child_ir)
        new_fields = tuple(n for n, v in child_plan.outputs.items() if v.producer is not None)
        out_name = self.output if self.output is not None else self.column
        fn = _per_row(child_plan, new_fields) if self.mode is EachMode.PER_ROW \
            else _batch(child_ir, self.column, out_name)
        inp = Input(self.column, list[dict], NullPolicy.MISSING_AS, [], arg="items")
        out = Output(out_name, list[dict])
        kind = "scalar" if self.mode is EachMode.PER_ROW else "frame"
        # `subflow` is only for introspection (`Session.structure()`, `parameters()`); wiring and
        # running still see this as a plain call to `fn`, exactly as before.
        params = tuple(d for c in child_plan.calls for d in c.node.params)
        return SubflowNode(ctx.origin(self), kind, fn, (inp,), (out,), params, subflow=child_ir)
```

Notes on this change:
- `params` used to come from `_hoist(child_plan, self.name)`, which built a
  *flattened, renamed* list of `ParamDecl`s (because the child's real params
  weren't visible at their own path, so `each` had to fake a document shape
  for them). Now that the child is built at its real nested path, you do
  **not** need to rename or flatten anything — the child's own `ParamDecl`s,
  taken straight from `child_plan.calls`, already carry the right names.
  However: check whether `SubflowNode.params` (used by `resolve()`/params
  validation for *this one node*) needs to be empty instead, with the
  child's params only reachable via `iter_with_subflows` — read
  `decider/engine/params/schema.py`'s `parameters()` function (see step 2c)
  and `decider/engine/wiring/resolve.py`'s `call()` method (search for how
  `node.params` is used — it feeds `NodeParams`/param declarations for
  exactly this one node) before deciding. If a param declared on the each
  node itself would end up **double-counted** (once via the each node's own
  `.params`, once via `iter_with_subflows` walking into `subflow`), remove
  the `params = tuple(...)` line above and pass `()` for this node's own
  params instead — the child's params should only be discoverable by walking
  into `subflow`, not duplicated onto the parent node too. Whichever way you
  go, add a one-line comment explaining the choice.
- Update the function signatures of `_per_row` and `_batch` (later in the
  same file) to drop the now-unused `paths` parameter, and delete the
  `_hoist` and `_child_doc` functions entirely (roughly lines 68–92 in the
  current file). Their only job was building/consuming the flattened
  document shape you no longer need. Inside `_per_row`'s and `_batch`'s
  closures, delete the lines that call `_child_doc(child_params, paths)` and
  replace them with however `child_params` should be shaped, given your
  choice above (likely: `child_params` becomes the real nested params
  document directly, e.g. `{"heavy": {"heavy_kg": 30.0}}`, and you pass it to
  `RunParams`/`Engine.run(..., params=...)` unmodified — check how
  `RunParams` is constructed elsewhere in the file to match).

### 2b. Delete the now-unused `CallNode` import if applicable

`each.py` currently imports `CallNode` from `decider.engine.ir.nodes`. After
this change the file returns `SubflowNode` instead. Update the import line
accordingly:

```python
from decider.engine.ir.nodes import SubflowNode
```

(Search the file for any other use of the name `CallNode` before removing
the import — if none remain, remove it; if `CallNode` is still referenced
elsewhere in the file for a type check, keep both imported.)

### 2c. `parameters()`: walk with `iter_with_subflows`

Open `decider/engine/params/schema.py`, find the `parameters()` function
(around line 92):

```python
def parameters(root: IRNode) -> ParamsSchema:
    """Collect the params of every `CallNode` in an IR tree.
    ...
    """
    shared: dict[str, ParamDecl] = {}
    used_by: dict[str, list[str]] = {}
    local: dict[str, dict[str, ParamDecl]] = {}
    for node in iter_nodes(root):
        if not isinstance(node, CallNode):
            continue
        ...
```

Change `iter_nodes` to `iter_with_subflows` (update the import at the top of
the file accordingly, from `decider.engine.ir.nodes import CallNode, IRNode,
iter_nodes` to also import `iter_with_subflows`). This makes `parameters()`
walk into a `SubflowNode`'s child tree and pick up its `CallNode`s (the
child's real steps) too. If you decided in 2a to give the each node itself
empty `params`, this is the mechanism that surfaces the child's real params
instead.

### 2d. `Session.structure()`: walk with `iter_with_subflows`

Open `decider/engine/debug/session.py`, find `structure()` (search for `def
structure`). It currently does:

```python
        return [{"path": n.origin.path, "kind": n.kind if isinstance(n, CallNode) else
                 type(n).__name__.removesuffix("Node").lower(), "source": n.origin.source}
                for n in iter_nodes(self.executable.plan.root.node)]
```

Change `iter_nodes(...)` to `iter_with_subflows(...)`, and update the file's
import line for `decider.engine.ir.nodes` to include `iter_with_subflows`
alongside the existing `CallNode, IRNode, SequenceNode, iter_nodes` (keep
`iter_nodes`, it's still used elsewhere in the same file — search for other
call sites, e.g. `self._sequences = {n.origin.path for n in
iter_nodes(plan.root.node) ...}`, which should stay as `iter_nodes`, not
change).

One detail: `SubflowNode` is a subclass of `CallNode`, so `isinstance(n,
CallNode)` on line above still matches it correctly (it'll report `n.kind`,
e.g. `"scalar"` or `"frame"`, same as before) — no change needed to that
branch.

### 2e. Tests

Open `tests/run/test_each.py`. Find
`test_the_childs_params_are_tunable_through_the_parent_document` (search for
that name). It currently asserts:

```python
    params = {"order": {"items": {"heavy_kg": 30.0}}}
```

Update this to the new, real nested shape — given the each node named
`"items"` and its child step named `"heavy"` (from `flow(heavy, name="item")`
— check the actual child step name used in this test's `pipeline()` helper
at the top of the file, it may be `"item"` not `"heavy"`; use whatever name
the child step actually resolves to, which you can check by printing
`exe.plan.root.node` or by running `structure()` in a scratch script). The
params document should become something like:

```python
    params = {"order": {"items": {"item": {"heavy_kg": 30.0}}}}
```

(adjust the nesting to match whatever `ctx.child(self.name)` actually
produces — verify with a quick script before hardcoding the assertion).

Add a new test in the same file asserting `structure()` shows the child's
step at its nested path, e.g.:

```python
def test_structure_shows_the_childs_steps_at_their_nested_path():
    session = pipeline().session(FRAME)
    paths = {n["path"] for n in session.structure()}
    assert "items" in paths
    assert any(p.startswith("items/") for p in paths)
```

**Verify**: `uv run pytest tests/run/test_each.py tests/debug -q` passes.
`uv run pytest tests/test_guide.py -q` passes (guide examples use `each`;
check `GUIDE.md` for any params-document example using the old flattened
shape and update it to match, since `tests/test_guide.py` executes every
Python block in `GUIDE.md` verbatim).

---

## Task 3 — Remove `each`'s private nested engine; run the child through a runner the each node owns, not a closure

**Outcome**: `_per_row` and `_batch` no longer construct their own
`FusedRunner()`/`Engine().bind()` inside a closure that nothing outside can
see or replace. Instead, each `SubflowNode` exposes a small object that runs
its child (fast path, same behaviour and performance as today) but can be
substituted by a `Session` later (Task 4) with a different runner when
stepping into it.

**Files to touch**: `decider/steps/each.py`, possibly a new small module
under `decider/engine/run/` if the helper is generic enough to be shared
(your call — start by keeping it in `each.py` and only extract it if a
second nesting construct needs the same helper later; don't build a
speculative abstraction for a need that doesn't exist yet).

**What to do**:

1. Read `_per_row` and `_batch` in `decider/steps/each.py` fully (they're
   both fairly long closures — `_per_row` builds a `FusedRunner()` once,
   then for every item builds a fresh `State`, loads its fields, drains
   `runner.iterate(...)`, and reads the results back out; `_batch` builds an
   `Engine().bind(child_ir, mode="fused")` once and calls `.run(exploded_df)`
   on it).

2. The runner (`FusedRunner()` in `_per_row`) and the bound `Engine` in
   `_batch` are currently created **once**, lazily, and reused for every
   subsequent call (look for the `nonlocal exe` / module-level `runner =
   FusedRunner()` pattern). This reuse is fine and should stay — it's what
   makes `each` fast (the child's kernels compile once, not per row). Don't
   remove the caching; you're changing *what* is cached and *how it's
   reached*, not adding per-call construction.

3. Give each `SubflowNode`'s `fn` closure a way to be asked "run the child
   with runner X instead of your default" without changing its default,
   fast, no-debugger behaviour at all. A minimal approach: instead of the
   closure hard-coding `FusedRunner()`, make the runner a mutable attribute
   the closure reads each call, e.g. (sketch, adapt to the actual code you
   find):

```python
class _EachRunner:
    """Runs `plan`'s child once per item; `runner` is swappable so a `Session` can step into it."""

    def __init__(self, plan, new_fields):
        self.plan = plan
        self.new_fields = new_fields
        self.runner = FusedRunner()   # default: fast, same as today

    def __call__(self, items, **child_params):
        # same body _per_row's inner `run(...)` function has today, but
        # using `self.runner` instead of a fixed local `runner` variable.
        ...
```

   and have `_per_row` return an instance of this instead of a bare closure.
   `SubflowNode.fn` still gets called exactly the same way (Python calls
   `instance(items, **child_params)` the same as it calls a function) — no
   change needed anywhere else that calls `node.fn(...)`.

4. Do the equivalent for `_batch`'s `Engine().bind(child_ir, mode="fused")`
   — make the bound mode swappable (e.g. keep the `Engine` object around but
   allow rebinding to a different `mode` before a `.run()` call), or, if
   that's awkward, wrap it in a similar small class holding a swappable
   `Executable`.

5. **Do not add any new public API yet.** This task's job is only to make
   the *default* behaviour identical to today (same runner, same
   performance) while making the runner a named, reachable attribute instead
   of a closure-local. Task 4 is what actually uses this swap point.

**Verify**: `uv run pytest tests/run/test_each.py -q` passes unchanged — this
task must be behaviourally invisible from the outside. If you have a way to
benchmark (check `benchmarks/` for an existing `each` benchmark script), run
it before and after to confirm no performance regression; if none exists,
skip this — don't add a new benchmark file for this task.

---

## Task 4 — Ambient attach point: let a `Session` step into a running `each` node

**Outcome**: `session.break_at("items/heavy")` (or whatever the child step's
real nested path turned out to be in Task 2) pauses the session just before
that step runs *inside* an item, and `session.step_into()`, `session.set(...)`,
`session.value(...)` all work on it, the same way they already work for a
step inside a `branch` arm or `loop` body — just one level deeper, into an
`each`'s child.

**Why a thread, and how it's scoped**: the child currently runs behind a
plain Python function call (`SubflowNode.fn`), one call per parent row or
item. A runner's `iterate()` is a generator — `yield` cannot cross a plain
function call boundary; the generator would need to be recursively driven
*through* that function call for `Session`'s single `next(self._iterator)`
loop (`decider/engine/debug/session.py`, method `_next`) to see the child's
checkpoints too. Rather than doing that (which requires the child to become
part of the parent's own resolved `Plan` — a much larger change explicitly
avoided in this plan, see the "superseded" note at the top of this
document), this task starts a **worker thread only when a `Session` actually
asks to step into a specific each node**, and only for the duration of that
one step. The worker thread runs the child's runner to completion in the
background and blocks at each checkpoint on a condition variable; the
`Session`'s own thread (the one the user is calling `step()`/`resume()`
from) wakes it up to advance. Outside of this, `run()`/`score()` and any
`Session` that never steps into an each node are completely unaffected —
there is no thread, no synchronization overhead, nothing changes.

**Files to touch**: `decider/engine/debug/session.py`, and the `_EachRunner`
(or equivalent) class you built in Task 3, in `decider/steps/each.py`.

**What to do**:

1. In `decider/engine/debug/session.py`, read `step_into()` and `_go()`
   carefully (search for `def step_into` and `def _go`). Understand how
   `Session` currently walks a `Sequence`/`Branch`/`Loop`'s bodies one
   checkpoint at a time via `self._iterator` (a generator from
   `runner.iterate(...)`).

2. You need a way for `Session` to recognise "the checkpoint I'm at right
   now is a `SubflowNode`, and the user wants to go deeper" — this is
   exactly the same shape of decision `Session.step_into()` already makes
   for a `Branch`/`Loop` (it just calls `next()` on the same iterator one
   more time, because those constructs *are* walked by the same generator).
   For a `SubflowNode`, there is no such next checkpoint from the parent's
   own generator, because the child runs entirely inside one `fn()` call.
   You will need a new, explicit case: when `Session.current` is a `before`
   checkpoint at a `SubflowNode`'s path (check `isinstance` against the
   resolved node behind it — look at how `Session` gets from a `Checkpoint`
   back to its IR node today, likely via `origin.path` lookups into
   `self.executable.plan`, or by checking `Plan.calls`/`Call.node`) and
   `step_into()` is called, `Session` should:
   - Create the swappable child runner from Task 3 (reach it via the
     `SubflowNode`'s `fn`, which is now the `_EachRunner`/equivalent
     instance you built — you may need `SubflowNode.fn` to be discoverable
     from `Plan.calls` via the matching `Call.node.fn`).
   - Start a worker thread that calls into the child's execution exactly
     once (one item, or one row, depending on `PER_ROW`/`BATCH` — start with
     `PER_ROW` only; leave `BATCH` for a follow-up, see the note at the
     bottom of this task), using an interpreted or stepped runner (not the
     default fused one) so it pauses at every node, substituted via the
     swap point from Task 3.
   - Block the calling thread until the worker thread reaches its first
     checkpoint (a condition variable, or a `queue.Queue(maxsize=1)` used as
     a simple handoff channel — a `Queue` is likely simpler to reason about
     correctly than a raw condition variable; prefer it unless you have a
     concrete reason not to).
   - Return that checkpoint to the caller of `step_into()`, wrapped/adapted
     however `Session`'s existing `Checkpoint`/`Paused` event shapes expect
     (check `decider/engine/debug/events.py` for the `Paused` event shape
     Session already emits, and match it).

3. Subsequent `step()`/`step_into()`/`set()`/`value()` calls, while "inside"
   an each node this way, should be routed to the worker thread's checkpoint
   (send a "continue" message, block for the next checkpoint) rather than to
   `self._iterator` (the parent's own generator), until the child run
   finishes, at which point control returns to the parent's own generator
   exactly where it left off (the parent's `SubflowNode.fn` call returns
   normally, and the parent's `_iterator` continues from its `after`
   checkpoint as always).

4. Keep this scoped to the **one item currently being stepped into**. Don't
   try to let a user step through every item of a `PER_ROW` each node's
   entire column in one go across parent rows — that's out of scope. Getting
   `break_at`/`step_into`/`set`/`value` working for the *current* item, the
   same way they work inside a `branch` arm today, is the goal.

**Verify**: write a new test in `tests/debug/` (look at the existing tests
in that directory for the `Session` testing pattern, e.g.
`tests/debug/test_session_edit.py` or similar for setup style) that does
roughly:

```python
def test_step_into_reaches_the_childs_steps():
    session = pipeline().session(FRAME)   # pipeline() from tests/run/test_each.py's helpers
    session.break_at("items")             # or whatever the each node's own path is
    session.resume()                       # paused just before the each node runs
    cp = session.step_into()               # should now be inside the child, at its first step
    assert cp.origin.path.startswith("items/")
```

Adjust exact assertions once you've confirmed the real path shape from
Task 2. If this proves substantially harder than expected for `BATCH` mode,
it's fine to ship `PER_ROW` support first and leave a `# ponytail:` comment
noting `BATCH` isn't steppable yet, with a one-line reason.

---

## Task 5 — Fix the pre-existing shared-mutable-state bug (unrelated to `each`, but touched by this work)

**Outcome**: `Executable` no longer stores "the latest call's report" on
itself as shared mutable state; each call gets its own report back instead.
This is a correctness bug today under concurrent serving (two threads
calling `.score()` on the same bound `Executable` race on this field), not
something this task introduces — but it's directly relevant because Task 4
introduces the first place in this codebase where two runs of the *same*
child plan genuinely happen concurrently (the worker thread and, if the
parent pipeline is itself concurrent, another caller).

**Background**: see `notes/child-grain-batch.md`, lines 246–261, for the
existing writeup of this class of bug (search that file for "plain data
race today").

**File to touch**: `decider/engine/run/engine.py`.

1. Open `decider/engine/run/engine.py`. Find `class Executable`'s
   `__init__` (search for `self.report = RunReport()`) and `_params` (search
   for `self.report = run.report`). There's already a comment there:
   `# ponytail: the latest call's report only; return it per call if
   concurrent callers need their own.` — this task is doing exactly what
   that comment says.

2. Find every caller of `.report` on an `Executable` instance (search the
   whole repo for `.report` — check `decider/engine/debug/session.py` around
   `_log_params`, which reads `self._params.report` — note this reads it off
   the `RunParams` object returned by `_params()`, **not** off `Executable`
   directly; that's already the correct, non-shared pattern. The problem is
   only the `self.report = run.report` line inside `Executable._params`,
   which additionally stashes it on `self` for anyone who reads
   `exe.report` directly after calling `.run()`/`.score()`).

3. Check `decider/serving/handler.py` and any other file for `.report`
   reads on an `Executable` (search `exe.report` / `executable.report`
   across the repo, including tests). If nothing reads `Executable.report`
   except through the pattern in step 2, you can likely just delete `self.report
   = run.report` from `_params()` and `self.report = RunReport()` from
   `__init__`, and remove the `report` attribute entirely — callers who want
   a report should get it from the `RunParams` returned by `.prepare()`, not
   from the `Executable` itself. If something *does* rely on reading
   `exe.report` after a plain `.run()`/`.score()` call (no `Session`
   involved), you'll need to keep returning it some other way — check
   whether `.run()`/`.score()` could return it as part of a small result
   wrapper, or whether it's acceptable to say "read `RunParams.report` via
   `.prepare()` yourself if you need it outside a `Session`". Don't guess;
   grep first, and if it's ambiguous, leave a `# ponytail:` comment and stop
   short of a bigger API change than this task calls for.

4. Separately, check `SteppedRunner._converted` and `_alive` (in
   `decider/engine/run/runners/stepped.py`, search for `self._converted =
   {}` and `self._alive = {}`, and their only writer, `_bundle`). These are
   keyed by `(params.key, call_id)`. Confirm: can two different `Session`
   instances (or a `Session` and a plain `.run()` call) end up calling
   `_bundle` with the **same** key concurrently, on the **same**
   `SteppedRunner` instance? Read how `Session.__init__` gets its runner
   (search for `self._runner = copy.copy(executable.runner)` in
   `session.py`) — `copy.copy` is a **shallow** copy, so check whether
   `_converted`/`_alive` (plain dicts) are shared between the copy and the
   original after a shallow copy (they are, unless `SteppedRunner` defines
   `__copy__` to deep-copy those two dicts specifically). If they're shared
   across a `Session`'s runner copy and the `Executable`'s own original
   runner, and Task 4's worker thread reuses the *same* `SteppedRunner`
   instance the parent `Session` is using (rather than a fresh one), you
   have a real race here. The fix, per the plan's "no locks" rule, is to
   give the worker thread its own runner instance (which Task 4 already
   does, if you followed step 2's guidance to use an interpreted/stepped
   runner "substituted via the swap point from Task 3" — confirm that swap
   point creates a **new** runner instance, not a shared one, before closing
   this task).

**Verify**: `uv run pytest tests/ -q` (full suite) passes. If you have time,
add a concurrency smoke test: bind one `Executable`, call `.score()` on it
from two threads simultaneously in a loop (a few hundred iterations), assert
both threads get correct, uncorrupted results. Put this in
`tests/run/test_each.py` or a new `tests/run/test_concurrency.py` if it
doesn't fit naturally elsewhere.

---

## Task 7 — `decider-bridge`: describe the child's steps in the JSON tree external tools consume

**Context**: `tools/decider-bridge/decider_bridge/bridge.py`'s `Bridge.describe()`
builds the entire "flow panel" tree via `node_json(self.ir, steps, located)`
(`describing.py`), which is the JSON both `decider-ui` and the VS Code
extension render. This is a **separate** tree walk from `Session.structure()`
(Task 2) — same underlying IR, different code that walks it — so Task 2 does
not automatically fix this; it must be done explicitly.

**Outcome**: the JSON `describe()` returns for a pipeline containing an
`each()` includes the child's steps, in a shape `Controls`/`Timeline` (which
already walk generically, see below) can consume without changes, and in a
shape the UI tasks (Task 8) can build on.

**Files to touch**: `tools/decider-bridge/decider_bridge/describing.py`,
possibly `tools/decider-bridge/tests/test_bridge.py`.

**What to do**:

1. Open `tools/decider-bridge/decider_bridge/describing.py`. Find `node_json`
   (search for `def node_json`). Its `CallNode` branch (the `if
   isinstance(node, CallNode):` block) currently returns a dict with no
   `"children"` key — a call is always a leaf in this JSON today. Its
   `else` branch (sequence/branch/loop) recurses via `node.children()` and
   includes a `"children"` key.

2. `SubflowNode` (from Task 1) is a `CallNode` subclass whose `children()`
   deliberately stays `()` (that's the whole point — wiring/compiling must
   not see it). So `node_json`'s `isinstance(node, CallNode)` branch will
   currently produce a normal, childless call dict for it, same as any other
   call. Add a check: if `isinstance(node, SubflowNode) and node.subflow is
   not None`, add a `"children": [node_json(node.subflow, steps, located)]`
   key to the returned dict (reuse the same recursive call the group branch
   already uses). Import `SubflowNode` from `decider.engine.ir.nodes`
   alongside the existing `CallNode`, `BranchNode`, `LoopNode` import at the
   top of the file.

3. Do **not** change `field_metadata` in the same file unless you find it
   misses the child's params — check by hand: it walks `node.children()` in
   its `else` branch already (search `def field_metadata`), which, same as
   `node_json`, does not reach `SubflowNode.subflow` because `children()`
   stays `()`. Add the equivalent one-line special case there if the flow
   panel needs the child's field metadata (units, display hints) shown —
   check `tools/decider-ui`'s use of `fields` in `DescribeResult` (search
   `protocol.ts` for `fields:`) to decide if this matters for a first pass;
   if unsure, do it anyway, it's a two-line change matching the one in step 2.

4. Check `bridge.py`'s `_index` method (search `def _index`) and
   `controls.py`'s `_groups`/`_writers`, `timeline.py`'s `_calls` — all three
   already walk `node.get("children", ())` generically (verified: none of
   them special-case "a call never has children"), so once `node_json` emits
   a `"children"` key for a `SubflowNode`, these should pick up the child's
   steps automatically, with no code change needed. Do not skip verifying
   this — write a quick throwaway script that calls `Bridge().describe(...)`
   on a pipeline with an `each()` step and print `self.calls` (from
   `_index`) to confirm the child's step paths appear in it.

5. Check `tools/decider-bridge/tests/test_bridge.py` for any assertion
   counting `ir["children"]` or exact tree shape on a pipeline that happens
   to use `each()` (search the file for `each`) — `test_each_edit_compares_on_its_own`
   is one such test; read it and update any shape assertion that now
   includes the child's steps.

**Verify**: `uv run pytest tools/decider-bridge/tests -q` passes (check
`tools/decider-bridge`'s own test runner instructions — it likely also uses
`uv run pytest`, confirm by checking for a `pyproject.toml`/`tox.ini` in that
directory; if it has its own venv, `cd tools/decider-bridge && uv run pytest`
instead of running from the repo root).

---

## Task 8 — UI clients: decide how a subflow's steps are shown (`decider-ui`, `vscode-decider`)

**Read this whole task before writing any code — it contains a design
decision you must make explicitly, not improvise silently.**

**Context**: `tools/decider-ui` is a shared React library consumed by
`tools/vscode-decider` (and possibly other hosts — check
`tools/jupyterlab-decider` too, search it for `@decider/ui` imports). Its
`src/model/protocol.ts` defines the TypeScript shape of the JSON Task 7
produces, and `src/layout.ts` computes the visual graph (node positions,
edges) from that JSON.

**The problem this task must solve**: `layout.ts` currently treats
`kind === "call"` as an unconditional, hard-coded leaf. Concretely (all in
`tools/decider-ui/src/layout.ts`):

- `isLeaf = (n: IRNodeJson) => n.kind === "call" || n.folded !== undefined;`
- `armKeys()`'s walk: `if (n.kind === "call") return;` (does not descend)
- `fold()`: `if (ir.kind === "call") return ir;` (never folds a call, because
  it assumes a call has nothing inside it to fold)
- `dataEdges()`, `firstLeaf()`, `lastLeaves()` all use `isLeaf`/`kind ===
  "call"` the same way.

And in `protocol.ts`: the `IRNodeJson` union is exactly two shapes —
`CallNodeJson` (`kind: "call"`, no `children` field at all in its TS type)
and `GroupNodeJson` (`kind: "sequence" | "branch" | "loop"`, has `children:
IRNodeJson[]`). The shared `walk()` function's whole recursion is gated on
`node.kind !== "call"`.

This means: simply adding a `"children"` key to a call-shaped JSON dict (as
Task 7 does) will be **silently ignored** by every part of the UI today —
`walk()` won't descend into it, `layout.ts` won't draw it, nothing will
break, but nothing will show it either. This is safe (no regression) but
means Task 7 alone delivers nothing user-visible in the UI. This task is
what makes it visible.

**Design decision you must make** (do not guess; if you can, ask; if you
can't ask, write down which you chose and why in a comment where the type is
defined):

- **Option A — treat a subflow like a foldable group.** Give `SubflowNode`'s
  JSON a *new* `kind`, e.g. `"subflow"`, that is a `GroupNodeJson`-shaped
  entry (has `children`) but also carries the call-shaped fields
  (`callKind`, `inputs`, `outputs`, `params`, `python`, etc.) that
  `NodePanel.tsx` needs for its details pane. This lets it reuse `fold()`'s
  existing open/close-by-path mechanism (so a subflow starts folded, like a
  closed branch/loop, and the user expands it) with minimal new layout code,
  at the cost of `isLeaf`/`armKeys`/`dataEdges`/`walk` all needing a new
  `n.kind === "subflow"` branch alongside their existing `"call"` checks.
  This is the more complete option and matches how branches/loops already
  behave, so it's likely the least surprising to a user — probably the
  right choice if you have time to test it against the graph layout
  visually (open the dev harness/storybook if `tools/decider-ui` has one —
  check its `README`/`package.json` scripts for something like `npm run
  dev`).
- **Option B — metadata only, no inline graph change.** Add an optional
  `subflowSteps?: CallNodeJson[]` (flat list, not a nested tree) field to
  `CallNodeJson`, populate it from Task 7's JSON, and only use it in
  non-graph views: `FindStep.tsx` (so searching for a step name finds one
  inside a subflow too — check `findSteps()`'s `text()`/`name()` helpers),
  and `NodePanel.tsx`'s details pane (show "runs N steps: ..." as a plain
  list, not a nested graph). The graph itself keeps drawing an `each()` node
  as one box, exactly as today. This is smaller, safer, and ships something
  useful (search + inspection) without touching `layout.ts`'s core
  assumptions at all.

**If you are not confident you can validate Option A visually** (i.e., you
cannot run the UI and look at it), **do Option B**. Shipping a correct,
smaller thing beats a graph layout change you can't verify.

**Files to touch** (either option): `tools/decider-ui/src/model/protocol.ts`,
`tools/decider-ui/src/layout.ts` (Option A only), `tools/decider-ui/src/FindStep.tsx`,
`tools/decider-ui/src/NodePanel.tsx`. Do not need to touch
`tools/decider-ui/src/Explain.tsx` or `StateTable.tsx` unless you find they
break — both consume `CallNodeJson[]` lists (already-flattened, via
`callNodes()`), which Option B's flat list slots into directly, and Option
A's `callNodes()` (in `protocol.ts`, uses `walk()`) already flattens
recursively once `walk()` knows about the new kind.

**A note on `vscode-decider`'s Python debugger integration** — do not
change this, but be aware of it so you don't accidentally break it:
`tools/vscode-decider/src/adapter.ts` (search `node?.kind === "call"` around
line 277 and 289) and `extension.ts` (search `info.node?.kind === "call"`
around line 377) use a paused call node's `python.file`/`python.line` to
attach VS Code's own Python debugger (`debugpy`) directly at that line —
this is a **separate, already-shipped** mechanism (see the debug-tools
worktree commit "a 'Step into the Python' button, and the Python debugger in
the test profile") for stepping into a step's *Python source*, distinct from
the engine-level `Session` stepping this whole document is about. Whichever
option you choose above, keep `SubflowNode`'s JSON satisfying
`node.python?.file`/`node.python?.line` as it does today (pointing at the
each node's own `fn`) so this existing feature keeps working unchanged —
don't remove or repurpose those fields.

**Verify**: `tools/decider-ui`'s own test suite, if any — check
`tools/decider-ui/package.json` for a `test` script and run it (likely `npm
test` or `pnpm test`, run from inside `tools/decider-ui`). If you touched
`vscode-decider`, check that project's `package.json` too. If neither has
automated tests covering this, at minimum verify TypeScript compiles cleanly
(`npm run build` or `tsc --noEmit` from inside each touched package) — a
discriminated union change in `protocol.ts` will surface any missed call
site as a compile error, which is the main safety net here.

---

## Task 9 — Final sweep: update `GUIDE.md` and docstrings if needed

**Outcome**: `GUIDE.md`'s `each()` example (if any) matches the new params
document shape from Task 2; `each()`'s own docstring in `decider/steps/each.py`
(the `each(...)` factory function at the bottom of the file, and
`EachStep`'s class docstring) still accurately describes behaviour if
anything user-visible changed (it shouldn't have — the public
`each(column, item, ...)` signature stays the same; only internal params
routing and `SubflowNode` are new).

**Files to touch**: `GUIDE.md`, `decider/steps/each.py` docstrings.

1. Search `GUIDE.md` for `each(` and `EachMode` to find any example. Since
   `tests/test_guide.py` executes every Python code block in `GUIDE.md`
   verbatim as a test, if you changed the params document shape in Task 2,
   any guide example showing a params document for an `each()` child must be
   updated to match, or `test_guide.py` will already have caught this in
   Task 2's verification step — if it's still failing now, fix it here.

2. Re-read `EachStep`'s docstring and the `each()` factory function's
   docstring in `decider/steps/each.py` for anything that describes the old
   flattening behaviour (e.g. anything implying params are "hoisted" or
   forwarded specially) and correct it to describe the new, simpler
   behaviour (child params are just nested at their real path, like any
   other nested step's).

**Verify**: `uv run pytest tests/test_guide.py -q` and
`uv run pytest tests/ -q` (full suite) both pass.

---

## Order of work

Tasks must be done in order: 1 → 2 → 3 → 4, with 5 and 6 done after 4 (5
depends on 3 and 4 existing so you can check what they actually share; 6 is
a cleanup pass after the engine work). Task 7 (`decider-bridge`) only
depends on Task 1 (`SubflowNode`) and Task 2 (real nested paths/params) —
it can start once those two are done, in parallel with Tasks 3-6 if you have
the capacity, since it doesn't touch the runner/threading work at all. Task
8 (UI) depends on Task 7's JSON shape existing. Don't start a task until the
previous one's "Verify" step passes.
