# Experiment 05 findings — Durable step-ID source syntax

**Run during:** task 02, before the ID generator is implemented
**Feeds:** tasks 02, 04, 08, 09, 10, 11

## Question and answer in one line

The durable ID is a short **opaque token attached to the step's own
declaration** — `id="…"` as a keyword on every step constructor, `@step(id=…)`
as the spelling for the dominant plain-function form — never derived from the
step's name, path, or position, so it survives rename, extraction, and
reordering without changing the text of the flow it sits in.

## What exists today (the mapping a durable ID must not break)

Steps are declared as plain functions, `@step(output=…)`, `@frame_step(…)`,
`flow`, `dag`, `branch`, `loop`, `each`, `optimise`, and JSON configs
(`TreeConfig` / `DecisionTableConfig` / `ScorecardConfig`). Every step has a
`name`; the engine builds a **path** by joining names with `/`
(`term/cap_by_income`) and records a **source** import path
(`credit.rules:cap_by_income`) in `Origin` (`decider/engine/ir/origin.py`).
`step_map`/`to_ir` expose the step object behind each path
(`decider/engine/ir/context.py`).

The path is a fine *discovery* ID but is not durable: renaming a step, lifting
it into a helper function, or reordering a `flow`/`dag` changes the path. A
durable ID must therefore be an explicit committed value, independent of name
and position, that the engine carries alongside — never instead of — the path.

## Chosen syntax

A single uniform optional `id` keyword on every constructor and decorator, the
same shape as the existing `name=`. Concrete spellings:

```python
@step(id="3f9a7c2e1b5d")                                   # plain function, on the definition
@step(output="term_cap", id="3f9a7c2e1b5d")                # existing decorator, id appended
@frame_step(reads=["client_id"], writes=["bureau_score"], id="3f9a7c2e1b5d")
term = flow(term_cap, cap_by_income, name="term", id="3f9a7c2e1b5d")
p03  = dag(a, b, c, name="p03", id="3f9a7c2e1b5d")
by_sector = branch(cond, arm_a, arm_b, modifies=["cap"], name="by_sector", id="3f9a7c2e1b5d")
repay = loop(unpaid, body, carries=["balance"], max_iterations=360, name="repay", id="3f9a7c2e1b5d")
items = each("items", flow(heavy, name="item"), name="items", id="3f9a7c2e1b5d")
best  = optimise(count, evaluate, max_candidates=1024, name="best", id="3f9a7c2e1b5d")
pipeline = flow(step(deductions.statutory_deductions, id="3f9a7c2e1b5d"), offer)  # imported fn
```

For JSON configs (trees, tables, scorecards) the `id` is a field in the JSON
document next to `name`, because the document is the committed source of truth
for that step and `ConfigurableStep` forbids extra fields today
(`ConfigDict(frozen=True, extra="forbid")`).

Two spellings, one meaning: `@step(id=…)` on a `def` is exactly `step(fn,
id=…)` applied at the definition. The generator uses the decorator form for
locally-defined functions (so the `flow`/`dag` call sites stay untouched) and
the `step(fn, id=…)` wrap only for an imported or reused bare function, where
there is no local definition to decorate.

## Chosen ID format

`secrets.token_hex(6)` — **12 hex characters, 48 bits, fully opaque**. It is:

- **Random, not derived** — a content hash changes when the step changes, a
  counter conflicts when two branches insert at the same position, a name
  prefix goes stale on rename. Opaque random has none of these.
- **Short enough to ignore** — the readability rule (below) is "not noticed",
  and 12 chars is the shortest length with a negligible collision chance even
  when two branches add thousands of steps and merge.
- **Uniform and checkable** — validated as `[0-9a-f]{12}`; the generator
  regenerates on a repo-wide collision and rejects a malformed manual ID.
- **Stdlib only** — `secrets.token_hex(6)`, no new dependency.

A manually supplied ID is an **override** (e.g. to match an existing trace),
not the only path to stability; it is validated against the same charset and
the same uniqueness scan.

## Readability rule

An ID must be **ignorable**: it must never carry information a reader has to
decode. The rules that follow from that:

1. It is opaque — it encodes no name, path, or position, so it can never go
   stale or mislead when the human name changes.
2. It is short — one `id="…"` fragment; no line gains more than that.
3. It lands on the step's *definition* for the dominant plain-function form, so
   the `flow(...)`/`dag(...)` sites a reader actually skims stay unchanged.
4. If an ID draws attention, it is doing it wrong — there is nothing to
   remember or correlate by eye.

## Generator safety and error behaviour

- **Clean-tree gate** — runs only when every affected file is git-tracked and
  the working tree is clean. Otherwise it aborts *before editing anything*,
  naming the dirty/untracked files and the remediation (commit or stash).
- **Idempotent and additive** — existing valid IDs are left alone; only missing
  ones are filled. It inserts the minimal `id=` fragment and never reformats,
  reorders, or otherwise rewrites source.
- **Parse before write** — the whole tree is parsed first; any syntax error
  aborts with no partial edits. Writes are applied only after a complete
  successful pass.
- **Duplicate scan** — all IDs are collected across the repo before writing; a
  duplicate aborts (or, in `--fix`, a fresh ID is assigned to the later copy).
- **Validation** — IDs match `[0-9a-f]{12}`; step names still go through
  `check_name`; a malformed manual ID is reported by name and location, not
  silently rewritten.

## Duplicate and merge handling

- **Rename / extract / reorder** never touch the ID literal, so their diffs are
  minimal and their merges are clean.
- **Copy/paste** produces a literal duplicate ID; the duplicate scan flags it
  and re-IDs the copy on the next generator run (or under `--fix`).
- **Independent branch edits** are disjoint by construction: random tokens mean
  two branches that each add steps to the same flow touch *different* literals
  on *different* lines, so git merges them with no conflict — the one property a
  content hash (churns) or counter (conflicts) cannot offer.
- **Two branches each copy the same step** is the only realistic collision;
  the same duplicate scan catches it at the next generation.

## Migration guidance

- The first generator run over a clean tree adds IDs to every tracked step and
  is its own reviewable, bisectable commit, so it can be reverted or re-applied
  independently of any feature work.
- The existing `name`/`path` semantics are untouched: the ID is purely additive.
  Downstream consumers keep `path`/`name` for human-facing display and prefer
  `id` for durable references, capturing both as metadata at trace time so an
  unresolved ID still renders a repairable partial result after a refactor.
- Imported steps from another package carry no ID until that package adds one;
  such steps are derived-only (no durable identity) — acceptable, and matches
  "explicit IDs are optional during development but required where traces,
  comparisons, or experiments need durable links."
- JSON config assets gain an `id` field in the same pass; params and tables that
  reference a step by path keep working, and only durable references switch to
  the `id`.

## How each scenario was tested

Run `uv run python notes/vscode-redesign/experimentation/05-step-id/prototype.py`
(a throwaway script; it does not modify `decider`). It renders the ID form for
all ten declaration shapes and asserts each parses via `ast`, then simulates the
evolution operations over a minimal `(id, name, module)` model:

| Scenario | Check | Result |
|---|---|---|
| rename | id unchanged, name changed | pass |
| extraction | step moves module, id unchanged | pass |
| reordering | order changes, ids stay unique | pass |
| copy/paste | literal duplicate id detected | pass |
| independent branch edits | 2×1000 random ids disjoint | pass |
| duplicate detection | duplicate scan returns the shared id | pass |
| source formatting | every id form is plain `id="…"`, `ast.parse`-clean (black/ruff not installed; a plain string literal is formatter-neutral exactly like the existing `name="…"`) | pass |
| source mapping | ids are additive to `name`/`path`/`source` in `Origin`; nothing in the path-join or `step_map` logic is touched | by construction |
| clean-tree failure | documented gate; not scripted (needs a git fixture) | by design |

The source-mapping and clean-tree cases are "by construction/design" rather
than scripted: the former is a no-op change to `Origin`, the latter is a git
precondition the generator enforces before writing, both cheap to pin down in
task 02's tests.

## Readability comparison vs the ID-free baseline

Baseline (no IDs):

```python
def debt_ratio(income: float, debt: float) -> float:
    return debt / income

def approved(debt_ratio: float, limit: float = param(0.4)) -> bool:
    return debt_ratio <= limit

pipeline = flow(debt_ratio, approved, name="credit")
```

With IDs (decorator form on the definitions):

```python
@step(id="3f9a7c2e1b5d")
def debt_ratio(income: float, debt: float) -> float:
    return debt / income

@step(id="8b1d4f00a5c3")
def approved(debt_ratio: float, limit: float = param(0.4)) -> bool:
    return debt_ratio <= limit

pipeline = flow(debt_ratio, approved, name="credit")
```

The `flow(...)` line is byte-for-byte identical; the cost is one short decorator
line per locally-defined function. On the real
`example_projects/10-retail-credit-e2e/sonnet/pipeline.py` the `dag(...)`/`flow(...)`
assemblies read exactly as before — only the member functions and the named
sub-units (`_p03_unit`, `_p12_unit`, …) each pick up one `id=` fragment, and the
`build()` body is unchanged. That is the acceptable trade: a fixed, ignorable
one-line-per-step cost buys a stable identity no derived scheme can provide.
