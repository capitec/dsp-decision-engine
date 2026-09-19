# 08 — Configuration and the compilation lifecycle

What config is allowed to change, what that costs, and how a change reaches a
running process.

This document exists because docs 02–07 left a contradiction. Doc 07 §4 makes
pipeline structure first-class in JSON; doc 04 §2 classifies structure as "new
graph, new compile, new review"; doc 02 §3.4 moves all compilation into the image
build. Those three cannot all hold. Doc 04 §1 also lists "business users add
rules, AI-assisted" as a persona requirement that none of them can serve.

Most of this is **Proposed**. §1 and §2 are the load-bearing decisions.

---

## 1. The rule

> **Config may change structure. Compilation must never happen on a request
> path.**

That is the whole constraint on *latency*. There is a second, on *content*:

> **Code is a superset of config.** Anything expressible in config is also
> expressible in code. Config goes as far as it can and then stops: at the point
> where you would be writing code in config, you write code, register it, and
> reference it by id.

Together these say: config is a *composition and parameterisation* language over
a registry of code, never a programming language. Everything below follows from
the two.

### 1.1 References, not pointers

The second rule needs a test, because "config references code" describes both the
mechanism that works and the one doc 01 §5.4 records as a misfeature. The
difference is registration:

| | example | resolves via | validated? |
|---|---|---|---|
| **reference** ✅ | `{"type": "credit_scorer", "dti_weight": 200.0}` | the discriminated union — one native pydantic pass | yes: id checked with a did-you-mean, params typed and bounded, `extra="forbid"` |
| **pointer** ❌ | `{"module_name": "x", "function_name": "y"}` | `import_module` + `getattr` at runtime | no: any importable attribute, no declared interface, no schema |

`decider`'s extension mechanism is the first shape — `register_graph_module(CreditScorer)`
in an extension package, imported at startup by `initialize_decider`, addressed
from config as `{"type": "credit_scorer"}`. `DefinedFunction`
(`decider/serializable/function.py`) is the second, and it is how `output_fn`
forfeits config-as-inspectable-data.

> **The rule: config may reference code by registered id. It may not contain
> code, and it may not carry an unregistered pointer.**

Every reference therefore resolves to a definition with a declared interface and
a schema, which is what keeps the whole config tree renderable, diffable and
checkable without executing anything. `decider2` keeps the extension shape and
drops `DefinedFunction`.

### 1.2 Why "config never changes structure" was never true

The current implementation already puts an expression language in config.
`_ComputedFeature` (`decider/modules/rules/common/feature.py:59`) holds

```json
{"type": "computed", "expression": "monthly_income - monthly_expenses"}
```

which is parsed with `ast.parse`, checked against a five-entry whitelist
(`ALLOWED_POLARS_FUNCTIONS`), and turned into executable code at module
construction via `simpleeval`. That is codegen from config, in production, today.

So the question was never *whether* config compiles into code. It is only that
the target changed: a polars expression costs ~0 to build, a numba kernel costs
200 ms–2 s. The design problem is **latency placement**, not representation.

### 1.3 Why an interpreter is the wrong answer

An earlier suggestion was to avoid recompilation with a generic compiled walker
over a flattened rule structure — the approach `experimentation/jittree/test.py`
benchmarks as "Approach B", where "the walker compiles exactly once, ever; new
configs are just new arrays."

**That generalises from a toy.** `jittree` walks a binary tree of
`if x[idx] op thresh`. The real vocabulary in `decider/modules/rules/` is:

- 13 unary operators (`<=`, `<`, `==`, `>`, `>=`, `!=`, `between`, `is_in`,
  `string_match`, `is_null`, `is_not_null`, `is_true`, `is_false`)
- three `cases` variants (ranges, string-match, is-in), each with N branches
- `CompositeRule` — arbitrary AND/OR/NOT trees over conditions
- variable-length `is_in` sets and string matching
- `_ComputedFeature` — an open expression language (removed by §3.2, but it is
  what a faithful port would have to carry)
- `FlatRuleTree` with `first_match` and `all` prioritisation, and struct outputs

A generic numba walker over that is an interpreter with a string path, a
variable-length set path, and a sub-expression evaluator. It would be the most
complex code in the framework, and it would exist solely to avoid a compile.

> **Settled: do not build a general interpreter.** Recompiling is simpler than
> interpreting, and the cost is affordable once it is off the request path.
> Generic kernels are used only where the artefact is *already* tabular (§3.4) —
> as a modelling choice, never as a performance dodge.

---

## 2. Three change classes

Doc 04 §2's params/structure binary is too coarse: it has no row for a table, no
row for a rule, and it puts "add a rule" in the same bucket as "write new Python".
Three classes replace it.

| class | what it is | artefact | cost of a change | who |
|---|---|---|---|---|
| **values** | params; table contents | params document | **free** — swap a bundle, no compile | business user |
| **interiors — values** | a rule's thresholds and its enabled flag | params document | **free** — they are arguments, not literals (§2.2) | business user |
| **interiors — shape** | rules added, removed or restructured; operators; nesting | interior document | **one background compile + swap** (§4) | business user, reviewed |
| **skeleton** | which modules exist, how they wire, `Branch`/`Loop` composition, step logic | Python | **rebuild and redeploy** | engineer |

Three properties make this hold together:

1. **Values never recompile.** A params bundle's *type* is fixed (doc 03 §4), so
   changing a value is a pointer swap. This is unchanged from doc 02 §4.
2. **Interiors recompile, but the interface does not change.** A module's
   declared `reads`/`writes` are code. So an interior change cannot alter the
   graph, cannot alter lineage, and cannot break wiring — it can only change what
   happens *inside* one declared box. That is what makes it safe to let a UI do.
3. **The skeleton is code.** A config document may not add a module, rewire two
   modules, or introduce a `Branch`. See §7 for why, and for how that could be
   widened later without rework.

### 2.2 A rule's thresholds are arguments, not emitted literals

**Measured — [EXPERIMENTS.md](EXPERIMENTS.md) §L.** A rule set can be emitted with its thresholds baked
into the source, or with them passed as an array argument. The argument form:

| | literal | argument |
|---|---|---|
| 8 threshold retunes | **8 full recompiles**, 343 ms each *at 5 rules* | **0 compile events**, `signatures` 1→1 |
| runtime | baseline | **+1–4.5 ns/row** (2.5–11.2%) |
| emitted lines | 11 / 25 / 65 | **identical** |

**The runtime price is ~4.5 nanoseconds against a 20–100 ms single-record budget**
(doc 01 §6.1) — free on the primary path, 3–11% on the daily batch. Adopt it.

**Rule enablement should be a mask array too.** §G measured that a *disabled* rule
still costs its full compile time; a mask costs −2.0% at 10 rules (noise) and
+5.4% at 30, and break-even against a single recompile is ~427 million rows.

> **One claim withdrawn.** An earlier draft implied hoisting constants would also
> reduce compile time by shrinking emitted code. It does not — line counts are
> **identical** at every rule count and the compile-time delta has no consistent
> sign. The case for arguments is *no recompile on retune*, and nothing else.

This is also what makes the stale-code hazard (doc 05 §4.2) structurally
impossible on the path a business user edits: a threshold change writes no source
at all.

### 2.1 The params document rejects composition

`resolve_params` **rejects any document carrying composition keys** (`use`,
`type`, `steps`, `arms`, `modifies`). This is what makes doc 04 §2's structural-
invariance guarantee true by construction rather than by convention, and it is
what lets CODEOWNERS attach different rules to different documents.

Completeness is computable without the framework owning storage: pydantic's
`model_fields_set` distinguishes a field the document supplied from one that fell
back to a code default, so `resolve_params(doc, complete=True)` can name every
offending field. Doc 07 §2's "explicit and complete" requirement is enforced here,
not by `decider build --verify` (which does a different job — doc 05 §8).

---

## 3. Bounded module interiors

**Proposed.** A *data-shaped module* declares its interface in Python and takes
its body from a validated document.

```python
# modules/policy_rules/__init__.py  —  code. Reviewed, rarely changes.
from decider2 import ruleset

PolicyRules = ruleset(
    name="policy_rules",
    reads=["term_cap", "min_net_salary", "employer_sector_code"],
    writes=["term_cap", "decline_code"],
    params=PolicyRuleParams,
)
```

```json
// config — a UI writes this. Cannot reach anything not declared above.
{"policy_rules": {"rules": [
  {"id": "AFF01", "description": "cap term below the income floor",
   "when": {"op": "lt", "feature": "min_net_salary", "value": {"param": "income_floor"}},
   "then": {"term_cap": {"min_with": {"param": "low_income_cap"}}}},

  {"id": "SEC03", "description": "decline unlisted employer sectors",
   "when": {"op": "composite", "logic": "and", "conditions": [
     {"op": "is_in", "feature": "employer_sector_code", "values": [7, 9, 14]},
     {"op": "lt", "feature": "term_cap", "value": 12}]},
   "then": {"decline_code": "SEC03"}}
]}}
```

Four properties, each load-bearing:

1. **The interface is code.** `reads`/`writes` are declared in Python, so static
   lineage (doc 02 §5), schema propagation and wiring validation all survive
   without executing or even reading the interior. A UI edit cannot make
   `lineage("final_score")` return a different answer.
2. **The vocabulary is closed.** Every node kind has an emitter, so codegen is
   *total* — there is no interior document that validates and then fails to
   compile. This is the property an open expression language would destroy, which
   is why §3.2 removes the one that exists. A rule's leaves are declared features
   or **registered** feature ids (§1.1) — never expressions.
3. **A rule carries an id and a description.** These are the join key and the
   prose that doc 04 §6's reviewable artefact needs, and the reason-code taxonomy
   that doc 04 §4.1 warns was dropped in a previous port. Making them required
   fields of the rule schema is how they stop being optional.
3b. **`enabled: false` means NOT EMITTED.** [EXPERIMENTS.md](EXPERIMENTS.md) §G measured that a disabled
   rule still costs its full compile time — LLVM proves the body dead and
   execution collapses to 0.031 ms, but compile is unchanged. A UI that lets a
   user "turn off" fifty rules would otherwise charge them fifty rules of compile
   on every activation, forever, for code that never runs.
4. **Values inside a rule may reference params.** `{"param": "income_floor"}`
   keeps the tunable in one canonical location (doc 01 §5.3) instead of inlining
   a literal into the rule document — the same failure the 546 `pl.lit()` literals
   record.

### 3.1 `flat_rules` is already this shape

`RuleRoot` / `LeafRule` / `UnaryRule` / `CasesRule` / `CompositeRule`
(`decider/modules/rules/flat_rules/nodes.py`) are a closed algebra, discriminated
on `type`, and every node already implements `build_condition` /
`build_expression` against a `BuilderConfig`. Porting it means **swapping the
emitter target from polars expressions to numba source** — the taxonomy, the
validation and the recursion structure carry over unchanged.

Two things to fix in the port, both recorded as defects in doc 01 §5.4:

- **Node ids must be deterministic**, content-derived or declared. `CompositeCondition`
  currently assigns `str(uuid.uuid4())` when no id is given
  (`common/nodes/conditions.py`), which makes path codes incomparable across
  versions and — in decider2 — would make generated source non-deterministic and
  break the numba cache (doc 05 §4.2).
- **`output_fn` must not return.** It is a `{module_name, function_name}` code
  pointer stored in config — the ❌ row in §1.1 — which forfeits
  config-as-inspectable-data. The output shape is declared by `writes`. Where a
  project genuinely needs custom output assembly, that is a registered module
  referenced by id, not a dotted path resolved with `getattr`.

### 3.2 Computed features are registered steps, not expression strings

**Settled by §1.** `_ComputedFeature` — an expression string in config, parsed and
evaluated through `simpleeval` — is the one place the current design writes code
in config. It goes.

A derived value is a **step in code**, registered like any other, and referenced
from a rule by id:

```python
# modules/features/affordability.py — code, in an extension package
@step(description="income remaining after committed expenses")
def disposable_income(monthly_income: float, monthly_expenses: float) -> float:
    return monthly_income - monthly_expenses
```

```json
{"when": {"op": "lt", "feature": "credit:disposable_income",
          "value": {"param": "floor"}}, ...}
```

Four things this buys over an expression string, and it costs less work, not
more — there is no expression compiler to write:

1. **It is a real node in the graph**, so it appears in `lineage()`, in
   `render()` and in the reviewable artefact instead of hiding as a string inside
   a rule. Doc 04 §6 is the top-ranked risk; this is free ground.
2. **It has a declared signature**, so its inputs and dtype are known statically,
   the null policy is declared where doc 03 §1 puts it, and it type-checks against
   the rule that uses it.
3. **It is testable in isolation** and reusable across rules and rulesets, rather
   than being copied between rule documents as a string.
4. **There is no expression language to specify, secure, version or explain** —
   and no second way to write arithmetic.

The cost is real and worth naming: adding a derived value now requires a code
change, where today it is a config edit. That is the rule working as intended —
an arithmetic expression *is* code, so it lives in code. What a UI can still do
without a deploy is compose registered features into new rules, which is the
common case.

> **Consequence for §3's lineage guarantee.** A rule referencing a registered
> feature means a `ruleset`'s interior can pull in inputs its own steps do not
> name. So the module's declared `reads` becomes an **upper bound**, not a
> description: the transitive closure of every referenced feature's inputs must
> fit inside it, checked at config-validation time, before any compile. That
> keeps "a UI edit cannot change what `lineage()` returns" a guarantee rather
> than a hope, and it gives a good error — *"rule AFF01 references
> `credit:bureau_delta`, which reads `bureau_score`; `policy_rules` does not
> declare it. Add it to `reads`."*

### 3.3 What a UI can edit, end to end

| surface | document | validation | recompile |
|---|---|---|---|
| params | params document | pydantic model, bounds, `extra="forbid"` | no |
| table contents | params document | table schema (O4) | no (§3.4) |
| rules in a `ruleset` | interior document | closed node vocabulary + declared `reads`/`writes` | yes, staged |
| anything else | — | — | code change |

All three editable surfaces export as JSON Schema (§6.2), so a UI needs no Python
and no framework release to render a form.

### 3.4 Codegen or generic kernel, per module kind

**Proposed.** The choice is made by whether the artefact is *naturally* tabular,
not by how much a recompile costs.

| kind | shape | strategy | interior change |
|---|---|---|---|
| `ruleset` | heterogeneous predicates, nested boolean logic, per-rule outputs | **codegen** | staged compile |
| `decision_table` | N rows × M condition columns, uniform operators | **generic kernel** | free |
| `scorecard` | bins → points, uniform | **generic kernel** | free |

`decision_table` is already a parameters frame scanned against one expression
(`decider/modules/credit/decision_table/impl.py`), so a generic kernel there is a
table scan over a 2D array, not an interpreter — twenty lines, and the zero-
recompile property falls out for free. `ruleset` is not tabular, and pretending it
is would produce exactly the complexity §1.3 rejects.

> The test to apply when adding a kind: *can one compiled loop evaluate every
> instance of this kind, with the instance supplied as arrays?* If yes, generic
> kernel. If it needs a `switch` over node types, codegen.

---

## 4. The compilation lifecycle

**Proposed.** Compilation is a state machine over immutable pipeline generations.
A process holds one active generation and at most one staging.

```
                  stage(structure)
   ACTIVE ─────────────────────────▶ COMPILING ──────▶ STAGED
      ▲                                   │  (background)  │
      │                                   ▼                │ activate()
      │                                FAILED              │
      └────────────────────────────────────────────────────┘
                     previous generation keeps serving throughout
```

```python
rt = decider2.Runtime(compiled)          # one active generation

# values — synchronous, microseconds, no compile
rt.params.swap(new_params)               # returns the previous bundle

# interiors or a new structure — compiles off the request path
handle = rt.stage(pipeline)              # returns immediately; compiles in a worker
handle.state                             # 'compiling' | 'staged' | 'failed'
handle.wait(timeout=...)                 # optional
rt.activate(handle)                      # atomic pointer swap
rt.rollback()                            # back to the previous generation, no compile
```

Five properties:

1. **`apply` and `score` read the generation pointer exactly once per
   invocation.** ✅ **Measured — [EXPERIMENTS.md](EXPERIMENTS.md) §H: 0 straddled batches** of 1564,
   across 11,605 swaps in 3 s. `activate()` costs 0.177 µs.

   The regression test is the *deliberately wrong* version: re-reading the pointer
   inside a chunk loop straddled **99.87%** of batches. Note what that models — a
   polars-style chunked `map_batches` apply re-reads per chunk by construction, so
   any route through it fails this property automatically. This is the failure the current
   implementation has: `BaseConfig.reload()` (`decider/config/base.py:105-116`)
   is lazy and per-object, so two config objects touched at different moments in
   one batch can be on different versions — half a run decided under one
   threshold, half under another, with nothing recording where the boundary fell.
2. **The free path and the expensive path are different calls.** `swap` cannot
   compile; `stage` always might. No single call is sometimes fast.
3. **Compilation happens in a worker, never in a request — and the worker is a
   SUBPROCESS, not a thread.** ⚠ **Measured, and a thread does not work —
   [EXPERIMENTS.md](EXPERIMENTS.md) §H.**

   | serving kernel | alone | during a background compile | retained |
   |---|---|---|---|
   | `nogil=True` | 1.183 ms | 6.406 ms | **26%** |
   | `nogil=False` | 1.163 ms | 1.242 ms | **55%** |

   **Releasing the GIL makes serving worse**, which inverts the obvious reasoning:
   a `nogil` kernel re-acquires the GIL between dispatches and the compiler thread
   is Python-level and GIL-greedy, so every call waits a full switch interval. The
   penalty is a **fixed +5.2 ms per invocation** — unchanged at 10× the batch size
   — so exposure is set by call granularity, not data volume. It tracks
   `sys.setswitchinterval` (0.05 ms → +0.15 ms), which is a mitigation but not a fix.

   **And numba's compiler does not parallelise:** two compiles in separate threads
   run 1.036× faster than sequential, and a small compile started during a large
   one took **10.5×** longer.

   ✅ **The subprocess path is now measured end to end — [EXPERIMENTS.md](EXPERIMENTS.md) §K.** Serving
   retains **97.9%** throughput during a child compile (against 26–55% for a
   thread), the parent load triggers **zero** numba compile events in 9.3 ms, and
   child and parent checksums are bit-identical.

   | config change | compile | total, change-to-serving |
   |---|---|---|
   | 10 rules | 1.91 s | **2.56 s** |
   | 30 rules | 6.66 s | **7.34 s** |

   That is the number a configuration UI can honestly promise, and it corroborates
   §G's "≤10 s up to ~35 rules".

   Two contract requirements the handover added, both in doc 05 §4:
   the generated module must be imported **by module name**, not via
   `spec_from_file_location` (which loses the cache across processes entirely),
   and the **`sys.modules` registration name** must be derived identically by the
   compiling child and every later loader — a seventh cache condition whose
   violation surfaces as a cryptic `ModuleNotFoundError('<dynamic>')`.

   A failed compile leaves the active generation untouched and surfaces on the
   handle — never as a swallowed exception. (Contrast `subscribe_version_updates`,
   which hot-swaps credit logic inside `except Exception: pass` on a 10 s poll.)
4. **Activation is explicit.** Staging does not activate. That is the hook an
   approval workflow attaches to; the framework does not implement the workflow.
5. **Rollback is free**, because the previous generation is still compiled and in
   memory. ✅ **Measured: 3.36 µs, zero numba compile events**, with the previous
   generation serving the next batch. Holding three generations resident costs
   **2.4 MB** more than holding one (296 MB vs 252 MB), so retention is not a
   reason to drop them. Rollback across a *restart* is the caller's problem, and the
   discontinuity is real: after a deploy, an old document may no longer validate
   because the code moved.

### 4.1 Two deployment modes

| mode | structure source | `--verify` guarantees | use |
|---|---|---|---|
| **sealed** | fixed at image build | zero compilations, ever | highest assurance; batch |
| **live** | may arrive from a config document | zero compilations **for the baked-in baseline** | UI-driven rule edits |

Both build the baseline at image build. `live` additionally carries a compiler
and a writable cache directory. A deployment states which mode it is in; it is
not inferred.

This resolves the contradiction in doc 02 §3.4 / doc 05 §8: "a runtime load
triggers zero compilations" is a statement about **the baseline**, not a
prohibition on ever compiling.

### 4.2 What forces a recompile

Enumerated, because "type fixed" is not a specification:

**Free — no compile:**
- any params *value*, of a type already compiled
- table contents, for a generic-kernel kind (§3.4)
- `.bind()`-ing a different value (a bound value stays a runtime value — doc 03 §4.3)

**Compiles:**
- any interior change to a codegen kind
- any skeleton change
- a params field whose *type* changes — including the cases the docs never
  enumerate: a `float | None` field toggling between set and unset flips
  `float64` ↔ `Optional(float64)`; an `int` field given a float value; a
  container field changing length where length is part of the type
- an input column's dtype or **nullability** changing (which is why `build` needs
  an input schema — doc 05 §8, O11)
- a fusion re-grouping, or a pinned boundary moving
- a different CPU target

**Sharp edges, from doc 01 §4c and doc 05:**
- Two distinct params NamedTuple classes sharing a `__name__` *and* field names
  blow per-call dispatch from ~1 µs to 15–24 µs **permanently**. Derive the class
  name from the module id and memoise generation on the pydantic model.
- Numba's file-backed cache stamps entries with `(st_mtime, st_size)` of the
  generated `.py` and indexes by path. Byte-identical regeneration at a different
  path or time misses **100%** of entries. So: write generated sources to a
  stable, pinned directory; do not regenerate at startup in `sealed` mode.
- A topological sort is not unique, and the order chosen *is* the generated
  source, which *is* the cache key. **The tie-break must be an explicitly stable
  rule** (declaration order, then name) or a cosmetic reorder of `module(...)`'s
  arguments silently invalidates the whole cache.

### 4.3 Where generated code lives

`sealed`: generated sources and the numba cache are baked into the image and the
filesystem may be read-only.

`live`: both need a writable directory that survives for the process lifetime.
It is configured explicitly, not defaulted to a temp directory — a cache that
silently relocates is a cache that silently stops working. `decider build --verify`
asserts the baseline loads from it with zero compilations.

### 4.4 The interactive loop

`stage`/`activate` is also the notebook story: edit a step, re-stage, keep the
session. Two known traps, both from doc 01 §4c:

- Redefining a pydantic params model in a cell yields a *new* bundle type, so the
  next stage recompiles. Correct, and worth a visible warning rather than silent
  latency.
- If two redefinitions produce classes with the same `__name__` and fields, the
  dispatch penalty above applies permanently to that session. The regression test
  in doc 05 §9.8 catches it in CI; interactive use needs it surfaced at the point
  it happens.

---

## 5. Impact review

Doc 04 §2 can no longer claim a param change is safe because the machine code is
identical. What replaces it:

```python
decider2.impact(active, candidate, sample) -> ImpactReport
```

Run both generations over a representative sample and report what moved: the
fraction of records whose declared decision outputs changed, the distribution of
each change, and which rules newly fired or stopped firing (from `branch_path` —
doc 04 §4.1). Cheap, because the sample is small and both generations are already
compiled.

This is the mechanism that makes a params edit reviewable *in the terms a credit
risk reviewer cares about* — "1.8% of applications change decision, all in the
declining direction" — rather than in terms of validator bounds. It applies
identically to an interior change, which is what makes §3 governable at all.

**Not enforced:** that anyone runs it. Like everything else in doc 04 §2.1, the
framework makes the line visible and policy attaches to it.

---

## 6. Where config comes from

**Settled: not the framework's business.**

The previous implementation owned this and it went badly in a specific way worth
recording. `decider/config/` is 553 lines — `CoreConfigManager` with four
abstract storage primitives, backends registered through a second discriminated
union (`decider/config/_ext.py`), semver `create`/`save`/`pull`, a stale-version
guard, and a 10-second poll that hot-swaps credit logic inside
`except Exception: pass`. It also carries a live bug: `pull_version`
(`decider/config/core.py:131`) evaluates `target.version` where `target` is a
`str` or a `Version`, neither of which has that attribute — masked only because
the poll always passes `force=True`.

None of that is decision-engine work. Banks have change-management systems, and
four-eyes approval, RBAC, environment promotion and change tickets appear nowhere
in those 553 lines anyway.

### 6.1 The principle

> A config document belongs to the framework's runtime surface only if the
> framework can say something about it that nobody else can. Where it came from
> is the caller's, always.

### 6.2 The seam

```python
# --- build time: CI and the Dockerfile only -------------------------------
pipeline = decider2.from_config(structure_document, admit=Admit.INTERIORS)
pipeline.fingerprint() -> str                 # content-derived; the compile key
compiled = pipeline.build(require_cached=True)  # cache miss RAISES, never a surprise compile

# --- runtime: pure. No json/os/pathlib/socket import below this line ------
pipeline.params_schema(flat=False) -> dict     # JSON Schema — the UI contract
pipeline.interior_schema("policy_rules") -> dict
pipeline.export_params() -> dict               # complete, from code defaults
pipeline.resolve_params(document, *, origin: str, complete=True) -> Params

decider2.diff(old, new) -> list[Change]
decider2.impact(active, candidate, sample) -> ImpactReport
```

`origin` is a **required, non-empty, opaque token** supplied by whoever loaded the
document. The framework stores it verbatim, never parses it, and refuses to run
without one. That is how provenance survives without the framework owning
storage — and a lint rule (`origin=` is never a string literal at a serving entry
point) is what stops it degrading to `origin="prod"`.

**Deliberately absent:** no `ConfigSource` protocol, no `ConfigManager`, no
version type or ordering, no polling, no subscription. A one-method Protocol is
ceremony; a user's loader is a function that returns a dict. Adding a Protocol
later is non-breaking, removing one is not.

Doc 02 §6's `config/` package is **renamed `binding/`** — it holds `register.py`,
`finalise.py` and `errors.py`, which are module-type registration machinery, not
config. There must be no `decider2/config/` package and no symbol named
`ConfigManager`. A directory called `config/` accreted 553 lines once already.

### 6.3 A file backend, complete

```python
def load(path):                                   # user code, not framework
    return json.loads(Path(path).read_text()), f"file:{path}@{git_sha()}"

doc, origin = load("config/term_loan/production.json")
rt.params.swap(pipeline.resolve_params(doc, origin=origin, complete=True))
```

A database-backed store with promotion and rollback is the same three lines with
a different `load`. That is the whole extension story.

---

## 7. Admission policy — how to widen this later

**Settled: bounded interiors now (`Admit.INTERIORS`); full composition in config
is not foreclosed.**

The graph is a validated pydantic instance either way (doc 02 §2). So the
difference between "interiors only" and "arbitrary structure in config" is **not
representational — it is a policy about what a document may contain**:

```python
decider2.from_config(doc, admit=Admit.INTERIORS)    # ships now
decider2.from_config(doc, admit=Admit.COMPOSITION)  # later, same code path
```

The reason this stays cheap is that **interiors already require every mechanism
composition would need.** Adding a rule is a structural change that must
validate, recompile, stage and swap, and must produce a new fingerprint for the
audit record. `Admit.COMPOSITION` adds no machinery — only a wider schema.

Four things must be true now or the option closes:

1. **Every module has a stable id, including on the Python-only path.** Doc 02 §2.2
   says Python composition "never touches any of this machinery"; that must not
   mean a Python-composed module has no id. Doc 05 §4.2 needs ids for
   deterministic generated names and doc 04 §5.1 for comparable path codes, so
   this is required anyway.
2. **Combinators serialise from day one**, even though nothing reads them under
   `INTERIORS`. A round-trip test — `from_config(pipeline.to_dict()) == pipeline` —
   is what stops the data model drifting into something composition cannot
   express.
3. **`reads`/`writes` are declared on every module**, not only data-shaped ones.
4. **The lifecycle in §4 is generic over "a new pipeline object appeared"** and
   does not special-case "an interior changed".

### 7.1 Extensions widen capability without widening admission

The answer to "config cannot express what I need" is almost never a wider
admission policy. It is an **extension**: write the thing in code, register it,
reference it by id (§1.1). That is how `decider` works today —
`initialize_decider` imports extension packages at startup, each calling
`register_graph_module`, and config addresses them by `type`.

This is the mechanism that makes "code is a superset of config" operational
rather than aspirational. It also means the registry, not the document schema, is
where a project's vocabulary grows — so a client extension adds capability to
every config document without the framework shipping a release.

The operational constraint from doc 02 §2.2 applies with full force: **all
registration must complete before the first config is validated**, and
finalisation must seal the union so a late registration raises rather than
silently leaving a stale nested union.

### 7.2 Prefer generating code over widening admission

If a pipeline-building UI is wanted, **have it emit Python** and commit it. The
graph stays in git, gets reviewed, gets a commit hash, and the audit story needs
no new concept — a generated pipeline is indistinguishable from a hand-written
one. `Admit.COMPOSITION` is for the narrower case where a structural change must
land *without* a commit, and it should be adopted only if that case turns out to
be real.

---

## 8. What lands in the audit record

Replacing doc 04 §5.2's "pipeline identity = hash of the exported JSON", which
identifies the config but not the code that ran:

| field | source |
|---|---|
| skeleton identity | the framework + module distribution versions that were imported |
| structure fingerprint | `pipeline.fingerprint()` — covers skeleton *and* interiors |
| compiled artefact id | hash of generated source + signatures + variant flags + CPU target |
| params digest | `Params.digest` over the canonical resolved values |
| params origin | the caller's token, verbatim, never parsed |
| generation | which generation served this invocation (§4) |
| declared variants | `fuse(...)` groups, `parallel(...)` regions and `fastmath` per kernel — authored, so they are part of the structure fingerprint rather than a runtime choice to record separately |
| fallback set | which nodes ran as Python (doc 05 §6) — **empty in production** |
| inputs, outputs, tapped values | as doc 04 §5.2 |

The last three are new and each closes a reproducibility hole: doc 04 §5.2 as
written cannot answer "re-run this decision" because the variant choice is
measured at warmup, the fallback set is decided lazily at first call, and neither
is recorded.

---

## 9. Open questions

- **O15 — the interior document schema.** §3's `when`/`then` sketch is
  illustrative. The real schema is the `flat_rules` algebra plus an id,
  description and param references. Needs writing before `ruleset` is built.
- **O16 — does a `ruleset` compile fast enough to stage?** Doc 01 §4b's
  ~15 ms/emitted-line and the `arms^depth` fan-out wall both apply. Thirty rules
  with composite conditions may be seconds or may be minutes. Measure before
  promising a UI.
- **O17 — approval granularity.** Does activation need per-rule approval, or is
  document-level enough? Framework-neutral, but it determines whether a rule
  carries an approval field.
- **O18 — interiors and the `sealed` mode.** Can a `sealed` deployment take an
  interior document at build time only? Probably yes and probably useful — it
  gives UI-authored rules with no runtime compiler.
- **O4 (tables) is now partly answered:** table contents are values for a
  generic-kernel kind (§3.4). The authoring and validation surface is still open.

### E10 — configuration lifecycle

**Two to three days, on top of the E1+E2+E3 vertical slice.** Settles §4, and is
the smallest thing that would falsify this document.

1. One `ruleset` with three rules over the vertical slice's waterfall.
2. Swap a params bundle mid-batch; assert no record straddles two bundles and
   `driver.signatures` stays at length 1.
3. Add a fourth rule via an interior document; assert `stage` compiles in a
   worker, the active generation keeps serving throughout, `activate` is atomic,
   and `rollback` needs no compile.
4. Measure stage-to-active latency at 3, 10 and 30 rules — this answers O16 and
   decides whether a UI can promise "live in seconds".
5. Assert a failed compile leaves the active generation untouched and surfaces on
   the handle.
6. Negative control: a params document carrying a `steps` key is rejected by
   `resolve_params`.

**Would invalidate:** §3, if declaring `reads`/`writes` in code turns out to
constrain real rule sets unacceptably; or §4, if staged compilation of a
realistic `ruleset` is slow enough that a UI cannot promise a bounded wait — in
which case the generic-kernel boundary in §3.4 has to move, and §1.3 gets
revisited with a measurement instead of an argument.
