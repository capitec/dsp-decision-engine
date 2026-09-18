# 03 — Authoring API

How people actually write logic. **Most of this document is Proposed** — it
exists in concrete form so the interfaces can be argued with before any code is
written.

---

## 1. A step

A step is a plain, pure Python function over scalars. No base class, no `self`,
no framework imports required in the simple case.

```python
def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses
```

- The **function name** is the output name by default.
- **Parameter names** declare inputs (§2).
- Steps must be **pure**: same inputs → same outputs, no side effects, no I/O.
  This is what makes four execution modes, caching and what-if override possible.
- Keep them small. A step you can read in ten seconds is one a credit-risk
  reviewer can check against policy.

`@step` is only needed when you want to override defaults:

```python
from decider2 import step

@step(output="term_cap", description="caps term by income band")
def apply_income_cap(term_cap: float, min_net_salary: float, params) -> float:
    if min_net_salary < params.income_threshold:
        return min(term_cap, params.income_cap)
    return term_cap
```

> **Proposed:** the decorator is optional. Plain functions keep the readable case
> clean; the decorator appears only where metadata is genuinely needed. The
> tradeoff is that tooling must handle both forms.

### Null policy is declared in the signature

Three tiers, all ordinary Python. **Most steps use the first two and never
mention nulls at all** — handling missing data is a property of the *input*, so it
belongs in the signature rather than in every function body.

```python
# 1. required (default) — any null fails fast at the boundary, naming the column
def affordability(instalment: float) -> float:
    return instalment * 0.35

# 2. declared fill — the framework substitutes at extraction, so the step sees a
#    plain number and there is nothing to forget
def affordability(bureau_score: float = missing_as(0.0)) -> float:
    return bureau_score * 0.01

# 3. the step genuinely needs to distinguish
def affordability(bureau_score: float | None) -> float:
    if bureau_score is None:
        return 0.0
    return bureau_score * 0.01
```

Why this shape:

- **Tier 2 is both the cheapest and the most auditable.** The step body sees a
  plain float — no check, no wrapper, at the 256 µs/1 M floor — and
  `missing_as(0.0)` is a declaration a credit-risk reviewer can read in the
  signature, rather than an `if` buried in a body.
- **Tier 3 is plain `Optional`.** `if x is None` is what a Python developer
  already expects. It costs ~1.5 ns/row more than the fastest available mechanism
  (1907 µs vs 413 µs per 1 M — doc 01 §4), which is a few percent on a realistic
  scorecard. That is the right price for introducing no new concepts.
- **`Optional` is also *safer*.** It holds `None` when invalid, so the underlying
  memory is unreachable. That matters more than it sounds: **the slot where a null
  sits contains leftover garbage, not zero** — a measured left join produced `9.0`
  at a position that was genuinely null. Any design where a step can read the raw
  value without checking is a silent wrong answer waiting to happen, so we don't
  offer one.

> **Rejected:** a NaN sentinel (~100 µs/1 M faster) — NaN cannot represent
> `Int64` or `Boolean` nulls, and `.to_numpy()` silently degrades nullable
> `Int64→float64` (lossy above 2⁵³) and `Boolean→object`. Also rejected: a
> `.value`/`.valid` wrapper — fastest of the explicit options, but it exposes an
> unchecked accessor and adds a concept Python developers don't already have.

---

## 2. Four wiring rules

All wiring is by name. There are exactly four rules:

| parameter name matches… | meaning |
|---|---|
| another step's output | wired to that step's value |
| nothing in the module | a **leaf input** — a column (batch) or argument (realtime) |
| the reserved name `params` | this module's per-invocation params bundle |
| the reserved name `shared` | the pipeline-level shared params bundle (§4.2) |

```python
def disposable_income(net_income: float, expenses: float) -> float:   # two leaf inputs
    return net_income - expenses

def ratio(disposable_income: float, instalment: float) -> float:      # wired + leaf
    return disposable_income / instalment
```

Ordering is **derived** by topological sort — you never declare execution order,
and it is never stored in config (doc 01 §5.2).

---

## 3. Scopes and value identity

The whole design rests on one invariant:

> **Every scope has a declared interface. Names are distinct within a scope.
> Versioning happens only at scope boundaries.**

There are four scopes:

| scope | names distinct inside? | may overwrite at its boundary? |
|---|---|---|
| **step** | n/a — leaf, one output | — |
| **module** | yes | yes — a declared output may shadow an input |
| **branch arm** | yes | yes — the branch node's declared `modifies` |
| **loop body** | yes | yes — declared `carries`, once per iteration |

### 3.1 Inside a module: no overwrite

Within one module, every step produces a **distinct** name. Two steps declaring
the same output is a hard error, suggesting either distinct names or composition
into separate modules.

The payoff is that a module's interior is a **pure DAG** — fully topologically
sorted, order-independent, with no declaration order to reason about. There is no
"ordering is load-bearing" tax anywhere in the authoring model.

### 3.2 At a module boundary: overwrite is the normal case

A module may produce a value it also consumes. That is how a waterfall is
expressed, and each policy rule becomes its own auditable unit:

```python
ApplyIncomeCap = module(cap_by_income_band, name="income_cap")   # term_cap -> term_cap
ApplySectorCap = module(cap_by_sector,      name="sector_cap")   # term_cap -> term_cap

pipeline = SeedTermCap | ApplyIncomeCap | ApplySectorCap
```

Ordering lives in `|`, where sequence is explicit and expected — **not** in
source declaration order, and not in `module()`'s argument list (an argument list
whose order changes behaviour breaks under innocuous reformatting).

**This style is cheap — measured — but not for the reason you might expect.**
An earlier draft claimed adjacent modules fuse into one kernel so small modules
cost nothing. E5 and E7 disproved that: fusion is non-monotone and past ~5
modules a single fused kernel is *slower* than one per module, because register
pressure stops it vectorising (doc 01 §4b–4c).

What actually makes the style safe is that **boundary stores are near-free** —
even with a full polars round-trip at every boundary, split beats fused at 1 M
rows / 10 modules. So:

> **Keep writing small modules.** They are the unit of audit, reuse and
> readability, and nothing about that changed. The compiler decides fusion
> groups; the authoring unit is deliberately decoupled from the compilation unit
> (doc 02 §1.1).

Use `pipeline.explain_kernels()` if performance matters, and pin a boundary if
the heuristic chooses badly.

It is also a better governance unit than the alternative. "A single policy rule" as a
module with signature `term_cap → term_cap` is independently testable and
**visible in the pipeline definition** — compare `_stage_00` →
`_stage_09`, function 37 of a 900-line module (doc 01 §5.1).

### 3.3 Versions and the audit trail

The engine versions values internally at each boundary crossing. **You never type
a version.** Every version has exactly one producer, so "which unit is
responsible for this value" stays exactly answerable:

```
term_cap:  60.0  (SeedTermCap)
        →  48.0  (ApplyIncomeCap)
        →  36.0  (ApplySectorCap)
```

A version chain is therefore just a list of scope-boundary crossings, which makes
the audit trail uniform across sequence, branch and loop:
*"entered TermCapRules, took arm 2, term_cap 60 → 48."*

> For contrast, overwrite is impossible in `decider` today — not by design, but as
> a side effect. Name matching rebinds a step's input to the matching node
> (`decider/modules/functional.py:181-185`), so a self-referential parameter
> points at its own node and `topological_sort` reports a cycle
> (`decider/graphutil.py:38`). The workaround cost is doc 01 §5.1.

---

## 4. Params

One validated bundle per invocation. Ordinary pydantic — validators are the point.

```python
from pydantic import BaseModel, Field

class AffordabilityParams(BaseModel):
    min_ratio: float = Field(0.3, ge=0, le=1, description="Minimum affordability ratio")
    income_cap: float = Field(36.0, ge=6, le=60, description="Term cap in months")
    income_threshold: float = Field(5000.0, ge=0)
```

Steps read named fields (`params.min_ratio`). Under the hood the model becomes a
`NamedTuple` whose *type* is fixed, so changing *values* never recompiles
anything — verified in prototype.

Rules:

- **Per invocation.** An invocation is 1 row or N rows, which covers realtime
  payload params and batch uniformly.
- **Exactly one canonical location per parameter.** A value must not exist both
  in params and as a literal in a step (doc 01 §5.3).
- Business users tune params; they cannot change structure. Validators bound the
  blast radius.

### 4.1 Composition: params are namespaced per module instance

`mod1 | mod2 | mod3` exposes the **composed** set, namespaced by module instance
name:

```python
pipeline = Affordability | ApplyIncomeCap | ApplySectorCap | Scoring

params = {
    "affordability": {"min_ratio": 0.3},
    "income_cap":          {"cap": 48.0},
    "sector_cap":          {"cap": 60.0},
    "scoring":       {"weight": 100.0},
}
```

A step sees **only its own module's params**. Three reasons this beats a flat
merge:

1. **No collisions.** Two policy modules can both have a field named `cap`. A
   flat merge would force global uniqueness, destroying reusability.
2. **The same module can appear twice** — `ApplyIncomeCap(name="income_cap_a")` and
   `ApplyIncomeCap(name="income_cap_b")`, separately tuned. Impossible when flat. This is
   where `name=` earns its keep.
3. **Encapsulation is a governance property.** A step in `income_cap` cannot read
   `sector_cap`'s params, so a param change's blast radius is bounded to one module by
   construction.

Compilation is unaffected: each module's NamedTuple type is fixed independently,
so a fused driver simply takes one per constituent module —
`_driver(cols…, params_income_cap, params_sector_cap, …, outs…)` — and retuning any of them
still never recompiles.

**On browsing:** namespacing deepens the model, which cuts against the
business-user requirement. But since the reviewable artefact is generated anyway
(doc 06 O3), it can present a **flat view over a namespaced model** — `income_cap.cap`,
`sector_cap.cap` in one sorted table. Correctness and browsing ergonomics don't trade
off.

### 4.2 Shared params

Some tunables are genuinely global — `base_rate`, `fee_multiplier`. Namespacing alone
would force setting them once per module.

`shared` is a **second reserved parameter name**, working exactly like `params`.
A step that needs global values asks for them:

```python
def cap_by_income_band(term_cap: float, min_net_salary: float,
                       params, shared) -> float:
    return min(term_cap, params.cap * shared.base_rate)
```

```json
{"shared": {"base_rate": 5.0},
 "income_cap":   {"cap": 48.0},
 "sector_cap":   {"cap": 60.0}}
```

The fused driver takes **one** shared bundle and passes it to whichever steps
asked for it — `_driver(cols…, shared, params_income_cap, params_sector_cap, …, outs…)`.

Four properties:

- **No duplication.** One bundle, passed by reference to whoever needs it. A
  shared value is never copied into per-module bundles.
- **Origin is visible at the point of use.** `shared.base_rate` reads as global;
  a module-scoped field read would be `params.cap`. No marker needed at the
  declaration site, because the parameter name does the marking — consistent with
  how `params` already works (§2) rather than a mechanism bolted on beside it.
- **One definition site for validators.** A single `SharedParams` model per
  pipeline/app declares types and constraints once. Modules that use `shared`
  declare a **required-fields contract** (names + types), checked at composition.
  That keeps a module testable in isolation — pass a stub `shared` — while
  ensuring any constraint violation has exactly one declaration to point at in
  the error message.
- **No local override.** Deliberately excluded: a lookup chain ("module value if
  present, else shared") makes "which value actually applied" ambiguous, which
  defeats the audit requirement.

> **Why not a contextvar.** Ambient supply via `ContextVar` is the natural Python
> instinct here, and it's how this was done previously. It cannot work at
> execution time: a njit'd step has no access to Python runtime state, so
> anything a compiled step reads must arrive as an argument. Making
> `interpreted` mode read a contextvar while `fused` mode reads an argument would
> also mean the modes differ in *mechanism*, undermining the equivalence ladder
> that makes debugging in `stepped`/`interpreted` trustworthy (doc 02 §3.1).
> An explicit argument is additionally safer under concurrency — two in-flight
> realtime requests with different shared values are independent as arguments,
> whereas a contextvar must be set correctly per request.
>
> A contextvar *would* be legitimate for supplying `shared` during manual
> **construction**, to avoid threading it through nested instantiation and test
> setup. Not added yet: `pipeline.apply(frame, params=…, shared=…)` covers the
> normal path and config-driven construction covers the rest. Revisit if manual
> assembly proves painful.

### 4.3 Binding at composition

Engineers freeze most knobs; business users tune what remains exposed:

```python
CreditFlow = Affordability | ApplyIncomeCap.bind(cap=48.0) | Scoring
```

`cap` leaves the caller-facing params interface. A bound value stays a **runtime
value with a fixed default**, not a compile-time constant — so binding never
triggers recompilation and remains cheap.

> **Open:** whether a number can be promoted to a param *in place* (e.g. an
> inline `p("income_cap", 48.0)` marker harvested into the model) rather than by
> editing a separate model. The human-factors argument is strong — whichever path
> is cheaper is the path people take — but harvesting means the params model is
> derived from source, which must stay stable and diffable. Tracked in doc 06.

### Tables (keyed lookups) — provisional

Segment-varying cutoffs are **not** params. Sketch only; lowest-confidence part
of this document:

```python
class TermTable(Table):
    key: int                 # term, 6..60
    max_loan: float
    quote_rate: float

def max_loan(term: float, tables) -> float:
    return tables.term.max_loan[term]     # dense array + present mask underneath
```

---

## 5. Assembling a module

```python
from decider2 import module

Affordability = module(
    disposable_income,
    ratio,
    apply_income_cap,
    apply_sector_cap,
    final_score,
    name="affordability",
    params=AffordabilityParams,
    taps=["term_cap", "branch_path"],
)
```

`Affordability` is a **pydantic instance describing the graph** — not a generated
class. It can be printed, rendered, diffed, serialised and validated. See doc 02
§2.

Two steps declaring the same output is a **build-time error** (§3.1), reported
with both remedies: give them distinct names, or split them into modules composed
with `|`.

---

## 6. Running it

### Batch

```python
frame = Affordability.apply(frame, params=p)
```

Per-column zero-copy in, batched write-back out, `prange`/serial chosen by row
count.

### Realtime, single record

```python
score = Affordability.score(net_income=42000.0, expenses=18000.0,
                            instalment=3100.0, params=p)
```

Bypasses polars entirely (doc 02 §3.4).

### Debugging one record

```python
with Affordability.debug(net_income=42000.0, …, params=p) as dbg:
    dbg.step()                       # advance one step, using COMPILED step code
    print(dbg.current)               # which step just ran
    print(dbg.values)                # all values, with version chains
    dbg.set("term_cap", 36.0)        # what-if: override mid-flow
    dbg.run()                        # finish
    print(dbg.trace)                 # full ordered record
```

Defaults to `stepped` (real compiled step code — production numerics, no drift).
`Affordability.debug(..., mode="interpreted")` drops to Python steps when you need
to see *inside* a step. Requires no taps and no redeploy.

---

## 7. Taps — production diagnostics

Declared as data, never a code pointer (doc 01 §5.4):

```python
module(..., taps=["term_cap", "branch_path"])
```

Each tap becomes an extra output column, and E5 confirms taps are cheap and
**do not split the kernel** — +0.11 ns/row/tap at 1 M rows (linear to at least 4
taps), against +2.0 ns/row for an actual kernel split, so ~17× cheaper.
`branch_path` is the special case: which branch fired, encoded as one `int64`
compile-time immediate per branch, effectively free at any batch size.

### Tapping a value that gets rewritten

In a waterfall the same name holds several values in turn — a term cap starting
at 60, cut to 48 by one rule and 36 by another. So a tap has to say *which*:

```python
taps=["term_cap"]              # the FINAL version — the default, and what you
                               # almost always want
taps=["term_cap@sector_cap"]         # the version produced by module `sector_cap`
taps=["term_cap@*"]            # every version, one column each — "which rule bit?"
```

Qualification is by **producing module name**, not by position. A positional form
(`term_cap@5`) would silently repoint at a different rule's output the moment
someone inserts a rule earlier in the pipeline — the same fragility class as the
unstable auto-generated node ids noted in doc 01 §5.4. A module name is stable
under insertion, and "term_cap after the sector-cap rule" means something in an audit report
where "version 5" does not.

Defaulting to *first* rather than final is what E5 found the naive
implementation does, and it is wrong in a way that fails quietly: you would get
60 in an audit report that should read 36.

---

## 8. Composition: sequence, branch, loop

There is **one type** — Module — and three combinators over it. Branches and
loops are not new node kinds; they are modules built from modules, which is what
keeps the §3 scope invariant uniform.

### 8.1 Sequence — `|`

```python
from decider2.frame import Join, Aggregate, Filter

pipeline = (
    Join(bureau, on="client_id", how="left")
    | Affordability
    | Underwriting                                   # fuses with Affordability
    | Filter(pl.col("decision") != "declined")
    | Aggregate(by="branch_id", metrics={"mean_score": pl.mean("final_score")})
)
```

- `|` builds a **dependency-resolved graph**, not a sequential stage list
  (doc 01 §5.2).
- **Adjacent record-tier modules fuse into one kernel** — one boundary crossing,
  intermediates in registers.
- Frame operations declare schema transforms, so lineage and wiring validation
  survive.

### 8.2 Branch

```python
TermCapRules = Branch(
    is_private_sector,                 # condition step -> bool
    CapForPrivate,                     # arm for True
    CapForPublic,                      # arm for False
    modifies=["term_cap"],
)
```

Rules:

- **`modifies` declares what the node changes**, not its whole output surface.
  Everything else **passes through untouched** — so an arm that doesn't care
  about `term_cap` simply doesn't mention it. This is the "skip" semantic that
  an internal module needed a hand-written mirror module 
  to fake.
- **Every arm must produce every declared `modifies` value, with agreeing
  types** — validated at build time. This is deliberately stricter than
  `decider`, whose `BranchModule` reconciles a cross-arm dtype mismatch with
  `pl.concat(..., how="diagonal_relaxed")`
  (`decider/modules/primitives/branching.py:96`), silently upcasting the whole
  column to string and having corrupted genuinely-computed booleans.
- Each arm is its **own scope** — distinct names inside, one version of each
  `modifies` value emitted at the node boundary.

**Routing** is the same combinator with a router returning an index:

```python
PriceByBand = Branch(risk_band_index, [BandA, BandB, BandC, BandD],
                     modifies=["price_category"])
```

In the record tier a branch compiles to a **real branch** — only the taken arm
executes, preserving short-circuiting, which is the entire source of the 7.8×
advantage at depth 50 (doc 01 §2). A columnar engine cannot do this; it must
evaluate every arm and select.

### 8.3 Loop

```python
BestOffer = Loop(
    should_continue,                   # (carried…, loop_idx) -> bool
    OfferStep,                         # body module
    carries=["best_offer", "best_score"],
    max_iterations=511,                # REQUIRED
)
```

A loop is the one place overwrite is genuinely unavoidable — an accumulator at
iteration 3 *is* a new version of the one at iteration 2 — so it is made
explicit rather than implicit. `carries` values are the body's inputs at
iteration start and must be produced by the body at iteration end. Inside one
iteration the scope invariant holds normally.

**`max_iterations` is required, not optional.** An unbounded loop inside compiled
code cannot be interrupted. Every real loop encountered so far is bounded with no
convergence iteration (a large internal workload), and `steptree_poc` already modelled the bound.

Loops get real `break`/`continue` early exit in the record tier — precisely what
the polars port destroyed when it replaced a descending `while` with
`filter → group_by(max) → join-back` (an internal file path), losing early exit
and evaluating every arm eagerly.

### 8.4 Why combinators rather than node kinds

A module is a scope with an interface; a branch is a scope with an interface. They
are structurally the same thing, so modelling them as one type plus combinators
means the graph model, compiler, lineage, trace and audit machinery each handle
**one** concept. Nested compilation is already proven feasible —
`experimentation/steptree_poc/jit_codegen.py` recurses its emitter into
`true_body`/`false_body`/`body` to generate nested compiled control flow.

### Escape hatch

```python
from decider2.frame import breaks_lineage

@breaks_lineage(outputs=["risk_flag"])
def custom_thing(frame: pl.LazyFrame) -> pl.LazyFrame: ...
```

Works, and is deliberately conspicuous. `grep -r "@breaks_lineage"` lists every
lineage gap in a codebase. Outputs are still declared so schema propagation
survives.

---

## 9. Inspection

```python
Affordability.lineage("final_score")   # static: inputs + steps that can affect it
Affordability.render()                 # diagram — no execution required
Affordability.schema()                 # required inputs, produced outputs
diff(v1, v2)                           # "step x changed y from 1 to 2"
```

All static. None require running data.

### What a non-programmer sees

Both current representations fail readability (doc 01 §6) — the Python *and* the
995-line JSON AST. **Proposed:** the reviewable artefact is generated from the
module data — an ordered table of steps with declared inputs/outputs, the
`description` from `@step`, the params each reads with their validated bounds,
and the value-version chain. Not raw code, not raw JSON.

> This is the requirement I am least confident we have solved. It deserves a
> prototype of the rendered view early, judged by an actual reviewer.

---

## 10. Config

```json
{"use": "credit:flow", "name": "cc_flow",
 "params": {
   "shared":        {"base_rate": 5.0},
   "affordability": {"min_ratio": 0.35},
   "income_cap":          {"cap": 48.0},
   "sector_cap":          {"cap": 60.0}
 }}
```

Params are namespaced by module instance name, with `shared` holding globals
(§4.1–4.2).

Config supplies **params and composition**, never structure and never code
pointers.

Resolution is the hybrid design settled in doc 02 §2.1–2.2: a discriminated union
of thin generated models validates the whole config tree in one native pydantic
pass, then each validated node's graph is attached via its back-reference. There
is no runtime registry, and the Python composition path never touches this
machinery at all. Practical consequences for authors:

- A misspelled param is a **hard error** (`extra="forbid"`), not silence.
- An unknown `use` id produces suggestions: *"unknown module id
  'credit:affordabilty' (51 registered). Did you mean: credit:affordability?"*
- Errors in deeply nested pipelines name the **exact step index**, not just the
  nesting path.

---

## 11. Testing

```python
def test_income_cap_applies_below_threshold():
    assert Affordability.score(min_net_salary=4000.0, term_cap=60.0, params=p) == 48.0
```

Steps are pure and individually callable, so rule-level assertions are cheap —
which is the point, given the observed state of one test file for 66 modules
(doc 01 §5.6).

Framework-provided:

```python
assert_modes_agree(Affordability, corpus)   # interpreted ≡ stepped ≡ fused
golden.record(Affordability, corpus)        # optional regression baseline
```

---

## 12. Worked example

```python
from pydantic import BaseModel, Field
from decider2 import module, step

class SharedParams(BaseModel):          # declared once for the pipeline
    base_rate: float = Field(5.0, ge=0, le=30)

class AffordabilityParams(BaseModel):
    min_ratio: float = Field(0.3, ge=0, le=1)
    weight: float = Field(100.0, gt=0)

class IncomeCapParams(BaseModel):
    cap: float = Field(36.0, ge=6, le=60)
    income_threshold: float = Field(5000.0, ge=0)

# --- module 1: affordability. Interior is a pure DAG, distinct names. ---

def disposable_income(net_income: float, expenses: float) -> float:
    return net_income - expenses

def ratio(disposable_income: float, instalment: float) -> float:
    return disposable_income / instalment

def term_cap(requested_term: float) -> float:
    return requested_term

Affordability = module(
    disposable_income, ratio, term_cap,
    name="affordability",
    params=AffordabilityParams,
)

# --- module 2: one policy rule. term_cap -> term_cap, its own auditable unit. ---

@step(output="term_cap", description="cap term by income band")
def cap_by_income_band(term_cap: float, min_net_salary: float,
                       params, shared) -> float:
    if min_net_salary < params.income_threshold * shared.base_rate:
        return min(term_cap, params.cap)
    return term_cap

ApplyIncomeCap = module(cap_by_income_band, name="income_cap", params=IncomeCapParams)

# --- module 3: sector-dependent cap, as a branch ---

def is_private_sector(employer_sector_code: float) -> bool:
    return employer_sector_code == 1.0

TermCapBySector = Branch(
    is_private_sector, CapForPrivate, CapForPublic, modifies=["term_cap"],
)

# --- module 4: scoring ---

def final_score(ratio: float, term_cap: float, params) -> float:
    if ratio < params.min_ratio:
        return 0.0
    return ratio * params.weight + term_cap

Scoring = module(final_score, name="scoring", params=AffordabilityParams,
                 taps=["term_cap"])

# --- the pipeline: ordering is visible here, and only here ---

pipeline = Affordability | ApplyIncomeCap | TermCapBySector | Scoring
```

Invoked as:

```python
pipeline.apply(frame, params={"affordability": {...}, "income_cap": {...}, "scoring": {...}},
               shared={"base_rate": 5.0})
```

Every element on show: pure functions, name-based wiring, a pure DAG inside each
module, `term_cap` overwritten at module *boundaries* only, a branch with declared
`modifies`, per-module params, one shared bundle read at its point of use, a tap —
and the waterfall order legible in one line.

All four record-tier modules **fuse into a single kernel**, so the small-module
style costs nothing at runtime.

Absent: registration boilerplate, UUIDs, ordering declarations, type
discriminators, identity-passthrough functions, and `name_override`.
