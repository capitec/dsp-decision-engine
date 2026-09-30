# Experiment 05 follow-up findings — Durable ID integration proof

**Run before:** task 02 freezes the public ID interface
**Builds on:** `05-step-id-findings.md` (syntax choice), `05-step-id-follow-up.md` (spec)

## Question and answer in one line

The `id=` design integrates cleanly with the real engine: every declaration
shape lowers with the ID on its `Origin`, name/path/`step_map`/execution are
byte-for-byte unchanged, and the source rewrite is a comment- and
format-preserving, idempotent, atomic insertion — the gate is **passed**, and
task 02 may adopt the syntax and build the generator on the engine changes in
this worktree.

## What changed in `decider/` (the integration proof)

Minimal, additive threading of an optional `id` from every constructor/decorator
to the IR `Origin`:

- `decider/engine/ir/origin.py` — `Origin` gains `id: str | None = None`;
  new `check_id()` validates `[0-9a-f]{12}`.
- `decider/engine/ir/context.py` — `IRContext.origin()` reads `step.id` and
  validates it, so every node's origin carries the id (or `None`).
- `decider/engine/debug/session.py` — the locator re-emission carries `o.id`.
- Each step dataclass gains `id: str | None = None` and its factory accepts
  `id=`: `FunctionStep`/`step()`, `FrameStep`/`frame_step()`, `SequentialStep`/
  `flow()`, `DagStep`/`dag()`, `BranchStep`/`branch()`, `LoopStep`/`loop()`,
  `EachStep`/`each()`, `OptimiseStep`/`optimise()`.
- `decider/steps/configurable.py` — `ConfigurableStep` gains a regular
  `id: str | None = None` field (dumped and reloaded, so JSON configs carry it).

No name, path, `step_map`, `Origin.source`, or execution logic was touched: the
id is purely additive and defaults to `None` everywhere, so existing code and
tests are unchanged.

## 1. Every declaration shape lowers with the id on its `Origin`

**What I did:** wrote `05-step-id/integration.py`, which constructs each shape
with `id=` and lowers it with `engine.to_ir`, asserting `origin.id` on the node
and (separately) that the id-less baseline has identical paths, `step_map` keys
and output frame.

**What I found:** all ten shapes lower cleanly and carry the id:

| shape | reaches `Origin.id` |
|---|---|
| `@step(output=…, id=…)` | yes |
| `@frame_step(…, id=…)` | yes |
| `step(fn, id=…)` (wrapped import/reuse) | yes |
| `flow(…, id=…)` | yes |
| `dag(…, id=…)` | yes |
| `branch(…, id=…)` | yes |
| `loop(…, id=…)` | yes |
| `each(…, id=…)` | yes |
| `optimise(…, id=…)` | yes |
| JSON `TreeConfig`/`DecisionTableConfig`/`ScorecardConfig` with `"id"` | yes |

`source` is unchanged (`__main__:…`, `decider.steps:SequentialStep`, `spike_lib:…`
as before); the id rides on the existing `Origin`, not a new mapping.

**One caveat:** `dag(single)` with no `name` collapses to its member (by
design), so its `id=` would be lost; I made the collapse conditional on
`name is None and id is None` so an id forces the wrapper. Task 02 should keep
that rule (an id always gets a home).

**Decision implication for task 02:** the interface is exactly `id="0123abcdef45"`
on every constructor/decorator plus an `"id"` JSON field; no special-casing by
shape. The generator emits one spelling per shape and nothing else changes.

## 2. Flow IDs: `flow ID + step ID` is a global reference

**What I did:** nested `flow(flow(step, name="inner", id=…), name="root", id=…)`
and asserted each node's `origin.id`.

**What I found:** flow ids and step ids land on their own nodes
(`root=cccccccccccc`, `root/inner=bbbbbbbbbbbb`, `root/inner/one=aaaaaaaaaaaa`),
so `(flow id, step id)` is unambiguous. The root flow is just the outermost
named flow; no special "pipeline" node exists in the engine.

**Decision implication for task 02:** the generator does not need to mark a
"root" specially — it gives every *named* flow an id (the outermost is the
root by construction). It only needs to locate the root-flow declaration so it
can be sure one exists (e.g. the module-level `pipeline = flow(…)` that
`.run()`/`build()` consumes, or the name in the serving/CLI config). Anonymous
flows stay id-less and transparent, which is correct.

## 3. Reuse / composition

**What I did:** one decorated step in two flows; a sub-flow under a parent; a
copied JSON config; an imported step from a package with no ids
(`05-step-id/spike_lib.py`).

**What I found:**
- A reused step keeps its single id at both paths (`fa/common`, `fb/common`) —
  id is an identity of the *definition*, path is an identity of the *placement*.
- A sub-flow id survives under a parent id; ids nest exactly like paths.
- A copied config's `id` round-trips through `model_dump_json`/`load`.
- An imported no-id step lowers with `id is None` and its real `source`
  (`spike_lib:helper`) — derived-only identity, as documented.

**Decision implication for task 02:** ids are one-per-definition, many-per-path;
the repo-wide duplicate scan must key on the *id literal* (definition), not the
path. Imported no-id steps are fine to leave derived-only; external packages
opt in by adding ids later.

## 4. Actual source transformation (git fixture)

**What I did:** `05-step-id/transform.py` rewrites a fixture in a temp `git
init` repo by splicing `id="…"` into each missing declaration (single-line and
multi-line calls, trailing-comma form, and bare `@step` decorators), changing
no other byte.

**What I found:**
- **Preserves** module docstring, comments, imports, existing decorators,
  spacing and the untouched `flow(…)` call sites — byte-for-byte.
- **Parse-before-write** — the whole file is `ast.parse`d first; the edits are
  offset splices applied right-to-left.
- **Idempotent** — a second pass is a no-op (declarations that already have an
  `id` are skipped).
- **Duplicate detection** — a repo-wide regex scan over all `id="…"` literals
  flags an exact copy; two independent generations are disjoint.
- **Clean-tree gate + failure atomicity** — an untracked file makes the tree
  dirty and blocks the run; a syntax error in a target file aborts before any
  write (verified by `git checkout`).

**Caveat for task 02:** the splice handles single-line, multi-line with the
closing paren on its own line, and bare `@step`. A multi-line call whose closing
paren trails the last argument on the same line is the one form it does not
special-case (it would emit `, id="…"`); task 02's generator should either
handle that case or reformat to the paren-on-its-own-line style, which is the
dominant form.

**Decision implication for task 02:** a byte-splice transform (ast to locate,
offsets to insert) is sufficient — no AST round-trip/pretty-printer is needed,
so comments and formatting are preserved by construction. Build the generator
on this technique.

## 5. Generated-ID collision and uniqueness scope

**What I did:** generated 100k tokens (no collision) and ran the repo-wide scan
from item 4; also demonstrated the import-path case in item 3.

**What I found:** 48-bit random tokens are collision-free at any realistic repo
scale. The repo-wide scan is intentionally **stricter** than the flow-local
reference scope: two flows would never share a step reference, yet a global
scan still flags a literal duplicate id. That strictness is the right call — it
is what catches copy/paste and two-branch-copy-the-same-step, and it keeps the
id *globally* usable as a reference, not just within one flow. External-package
ids participate in the scan only for files the generator can see (the current
repo); an imported step keeps whatever id its own package committed, and a
cross-package collision is both astronomically unlikely and out of scope for a
single-repo generator.

**Decision implication for task 02:** keep the repo-wide duplicate scan as the
uniqueness rule (document it as stricter-than-necessary-by-design), and do not
attempt to de-duplicate against external packages.

## Verdict

The established invariants hold against the real engine and a real rewrite:
explicit additive `id=`, readable one-line-per-step source, and durable identity
independent of name/path/order. Task 02 can proceed to build the generator on
the `decider/` changes committed in this worktree.

Spike scripts (new, in `05-step-id/`): `integration.py`, `transform.py`,
`spike_lib.py`.
