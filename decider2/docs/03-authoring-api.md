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
- The **docstring** is the description — the one a reviewer reads in the
  generated artefact (doc 04 §6). It is not an optional decorator argument,
  because the artefact that the entire governance story depends on must not be
  populated from a field the authoring guidance tells you to omit.
- Steps must be **pure**: same inputs → same outputs, no side effects, no I/O.
  This is what makes four execution modes, caching and what-if override possible.
- Keep them small. A step you can read in ten seconds is one a credit-risk
  reviewer can check against policy.

`@step` is only needed when you want to override defaults:

```python
from decider2 import step

@step(output="term_cap")
def apply_income_cap(term_cap: float, min_net_salary: float, params) -> float:
    """Cap the term where net salary falls below the income floor."""
    if min_net_salary < params.income_threshold:
        return min(term_cap, params.income_cap)
    return term_cap
```

> **Proposed:** the decorator is optional. Plain functions keep the readable case
> clean; the decorator appears only where metadata is genuinely needed. The
> tradeoff is that tooling must handle both forms.

### 1.1 One rule should cost one artefact

**Settled — this is the design's own law applied to itself.** Doc 01 §5.3: *"the
tunable form must be cheaper to write than the literal, or it will not be
adopted. Whichever path is cheaper is the path people take."* The evidence is 546
inline `pl.lit()` literals against zero uses of the config mechanism.

An earlier draft of this document broke that law. Writing one policy rule cost a
function, a `module(...)` call, an instance `name=`, a pydantic params model, an
entry in the pipeline expression and an entry in a config file — **six artefacts
across four files for one `if`**, against a rule set of roughly thirty
(doc 04 §6). That is the shape that produced the 546 literals.

The target is one artefact, in one file:

```python
# pipelines/term_loan.py — the entire rule
def cap_by_income_band(
    term_cap: float,
    min_net_salary: float,
    cap: float = param(48.0, ge=6, le=60),
    income_threshold: float = param(5000.0, ge=0),
) -> float:
    """Cap term at 48 months below the income floor."""
    if min_net_salary < income_threshold:
        return min(term_cap, cap)
    return term_cap

pipeline = Affordability | cap_by_income_band | Scoring
```

Three mechanisms get there, none of them new machinery:

1. **A bare function is a valid pipeline element** (§5.3). Used in a pipeline
   expression it becomes a single-step module: name from the function, interface
   inferred (§5.1), params namespace its own name. Collapses four artefacts into
   the two lines above.
2. **`param()` in the signature generates the params model** (§4.4). No separate
   model file for the common case.
3. **The config entry is generated, not written** — `decider export --params`
   materialises it complete from the declared defaults (doc 07 §5).

The module-plus-model form (§5) does not go away. It earns its place when several
steps share params, when a validator is cross-field, or when a rule is big enough
to want its own directory. The point is that the *simple* case is not charged for
the complex one.

### 1.2 Money, rounding and overflow

**New, and measured — [EXPERIMENTS.md](EXPERIMENTS.md) §I.** The doc set did not
mention money, rounding or integer range anywhere in 3,187 lines, while the
engine's outputs are instalments, fees and offer amounts that must reconcile to
the cent with a downstream ledger.

Two divergences were measured, and both are wrong-answer bugs rather than
performance issues.

**1. `round()` does not mean the same thing in a step and in a test.**

| x | Python `round(x,2)` | njit | `Decimal` HALF_UP |
|---|---|---|---|
| 2.675 | 2.67 | **2.68** | 2.68 |
| 2.665 | 2.67 | **2.66** | 2.67 |
| 1234.565 | 1234.57 | **1234.56** | 1234.57 |

njit follows numpy; CPython uses banker's rounding; neither matches `Decimal`
consistently. Every row above is one cent on an instalment, and the
`interpreted`/`fused` disagreement is exactly what the equivalence ladder would
report — correctly, but only if a test exercises a `.xx5` value.

**2. int64 overflows at realistic loan sizes.** Fixed-point compound interest at
rate scale 1e12 diverges from R27,431 upward; at R100,000 principal, CPython gives
R336,241.93 and njit gives **−R90,971.75**. A sum-of-squares accumulator over loan
amounts in cents **wraps at 2,667 rows**. CPython ints are arbitrary precision;
numba int64 wraps silently, as does numpy.

**Rules:**

- **Money is a scaled `int64` of cents, never `float` and never `Decimal`.**
  `Decimal` cannot cross the boundary at all — it raises a Rust panic (doc 05 §1.5).
- **Never call bare `round()` in a step.** Use the framework's `round_half_up`,
  which is defined to give the same answer in every execution mode. A lint
  enforces it (doc 07 §6).
- **Accumulate in float64, not int64**, where a running total can grow: float64 is
  exact to 2⁵³, which in cents is about R90 trillion. An int64 cent accumulator is
  not safe at batch scale.
- **The corpus must include boundary values.** The overflow above was found by
  binary search, not by sampling — random draws would never surface it. This is
  the first concrete requirement on `corpus`, which §11 and doc 05 §9 depend on
  and which no document has yet defined.

---

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

## 2. Wiring rules

Wiring is by name, with two declared exceptions:

| parameter… | meaning |
|---|---|
| name matches another step's output | wired to that step's value |
| name matches nothing in the module | a **leaf input** — a column (batch) or argument (realtime) |
| is named `params` | this module's per-invocation params bundle |
| is named `shared` | the pipeline-level shared params bundle (§4.2) |
| **has a `param()` default** | a **params field**, not an input (§4.4) |
| **has a `missing_as()` default** | an input with a declared fill (§1) |

The last two are keyed on the *default* rather than the name. That is not a new
axis — `missing_as()` already worked this way, and both read in the place a
reviewer looks for an interface rather than being buried in a body.

```python
def disposable_income(net_income: float, expenses: float) -> float:   # two leaf inputs
    return net_income - expenses

def ratio(disposable_income: float, instalment: float) -> float:      # wired + leaf
    return disposable_income / instalment
```

**Inside a module**, ordering is derived by topological sort — you never declare
execution order, and it is never stored in config (doc 01 §5.2). **Between
modules**, `|` is a sequence and written order is execution order (§8.1).

### 2.1 Precedence, and the one error that matters

Across a pipeline a name can be available from more than one place. The rule:

| situation | resolution |
|---|---|
| a name is written by several modules in sequence | **most recent wins.** This is the waterfall, it is the designed semantic (§3.2), and the version chain records every step of it |
| a name is available only from the input frame | the frame column |
| a name is available from **both** an upstream module output and the input frame | **build error.** Qualify it |

The third row is the whole point. Without it, this happens silently:

```python
A | B | Affordability              # net_income comes from the input frame
A | B | NewThing | Affordability   # NewThing outputs net_income —
                                   # Affordability now reads that instead
```

No error, different decisions. The same thing happens on an upgrade: a shared
library adds a step that happens to be named `net_income`, and every consumer
downstream rebinds. That is a wrong-answer bug with no signal, and it is the
worst failure mode the design can have — for a person and much more so for an
agent, whose only feedback is the error it did not get.

So it is an error, and the message says what to do:

```
'affordability' input 'net_income' is ambiguous: produced by module 'new_thing'
(added in decider2_credit 1.1.0) and present as an input column.
Qualify it:  Affordability.at(inputs={"net_income": "frame:net_income"})
        or:  Affordability.at(inputs={"net_income": "new_thing.net_income"})
```

Overwrite between modules stays silent because it is *intended* and *auditable*;
shadowing between a module and the frame is never intended.

### 2.2 An unbound input is a typo, and is treated as one

Writing `disposible_income` does not fail — it quietly becomes a demand for a new
input column, and surfaces much later as a missing-column error naming the wrong
thing. It gets the same treatment as a misspelled param (§10):

```
step 'ratio' input 'disposible_income' is not produced by any step in scope and
is not a declared input column. Did you mean 'disposable_income' (produced by
step 'disposable_income' in this module)?
```

`pipeline.schema()` lists every unbound input at once, so a project adapting a
library sees all of them in one place rather than discovering them one at a time.

---

## 3. Scopes and value identity

The whole design rests on one invariant:

> **Every scope has a declared interface. Names are distinct within a scope.
> Versioning happens only at scope boundaries.**

There are **five** scopes:

| scope | names distinct inside? | may overwrite at its boundary? |
|---|---|---|
| **step** | n/a — leaf, one output | — |
| **module** | yes | yes — a declared output may shadow an input |
| **branch arm** | yes | yes — the branch node's declared `modifies` |
| **loop body** | yes | yes — declared `carries`, once per iteration |
| **pipeline** | no — overwrite is the waterfall | yes — at each `\|` |

> **Correction.** An earlier draft listed four scopes and omitted the pipeline,
> which made `A \| B \| C` one unpoliced flat pool of names — hundreds of them
> across sixty modules, with no collision rule, while duplicate outputs *inside*
> one module were a hard error. The strictness was inverted relative to the risk.
> The pipeline is a scope; §2.1 gives its rules.

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
> readability, and nothing about that changed. Kernel boundaries are a separate,
> explicit concern — the authoring unit is deliberately decoupled from the
> compilation unit (doc 02 §1.1).

By default `apply()` emits one kernel per module and nothing is fused implicitly.
If a group is hot, say so — `fuse(A | B | C)` is a combinator like any other, and
it is guaranteed not to change the answer (doc 02 §1.2). `explain_kernels()`
reports emitted lines and whether each kernel vectorised, so the question is
answered by observation rather than by a heuristic.

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

✅ **Measured — [EXPERIMENTS.md](EXPERIMENTS.md) §N3.** "Under the hood" undersells
this conversion: at 50 module instances it costs *more* than validating the raw
document in the first place (170.6 µs conversion vs 78.3 µs validation) — both
scale the same way with module count, but conversion pays a second full pass over
the same data. Combined, validate-and-convert is still cheap in absolute terms
(256.5 µs, 1.28% of a 20 ms budget at 50 modules), but if a params document
recurs across requests, caching the *converted* `NamedTuple` bundle by content is
worth far more than caching the validation result alone — a memoized lookup
measured 726–765× cheaper at the same scale.

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

### 4.4 Declaring a param in the signature

**Proposed, and it settles O2.** A model in its own file is the right shape for a
module with many params or a cross-field validator. It is the wrong shape for one
rule with one knob, and doc 01 §5.3's law says the wrong shape does not get used.

```python
def cap_by_income_band(
    term_cap: float,
    min_net_salary: float,
    cap: float = param(48.0, ge=6, le=60, description="Term cap in months"),
) -> float:
    """Cap term at 48 months below the income floor."""
    return min(term_cap, cap) if min_net_salary < 5000 else term_cap
```

`param()` takes exactly what `Field()` takes. At import the signature is
harvested into a generated pydantic model — the same object a hand-written model
produces, with the same namespace (`{"cap_by_income_band": {"cap": 48.0}}`), the
same validators, the same fixed NamedTuple type, and therefore the same guarantee
that retuning never recompiles.

**Why the signature and not the body.** O2 asked whether a literal could become a
tunable *where it sits* — `p("income_cap", 48.0)` inline. Four reasons the
signature is the better slot:

1. **No AST rewriting.** A `p()` call inside a body would have to be rewritten to
   `params.cap` at codegen, so step bodies would stop being plain Python and
   `interpreted` mode would have to replicate the substitution — which is exactly
   the kind of divergence between modes that the equivalence ladder exists to
   prevent. In the signature the value arrives as an ordinary argument, which is
   what the compiled calling convention already does.
2. **It answers O2's own objection.** O2 records that it is *"unclear how a
   business user browses knobs that live inside function bodies."* In the
   signature they do not live in bodies, and `params_schema()` (doc 08 §6.2)
   finds them by inspection.
3. **A signature is a declaration; a body is not.** O2's other concern was that a
   harvested model must stay stable across edits and remain diffable for audit.
   Harvesting a declaration is stable in a way harvesting statements is not.
4. **It reuses an idiom already in the design.** `missing_as(0.0)` occupies the
   same slot and already changes what a parameter means (§1).

**Three costs, stated plainly:**

- **Two ways to declare a param.** Inline and explicit model must produce the
  same object, and a lint forbids both in one module (doc 07 §6).
- **Direct calls need care.** `cap_by_income_band(term_cap=60.0,
  min_net_salary=4000.0)` would otherwise receive the `param()` sentinel. `@step`
  substitutes declared defaults so a plain call works, and a test passes
  `params=` to override — doc 03 §11's testing story depends on steps staying
  directly callable.
- **numba and default arguments.** ✅ **Confirmed, [EXPERIMENTS.md](EXPERIMENTS.md) §E11.** The emitted step
  does *not* need its default stripped: emitted with the sentinel default, with it
  stripped, and called through a driver all produce an identical numba signature
  `(float64, float64, float64)` and identical results. Codegen may leave the
  default in place. The only failure is calling the dispatcher *without* the
  param — `ValueError: Cannot determine Numba type of <class 'ParamSpec'>` — which
  is correct and loud.
- **The dispatch risk did not materialise.** 20 steps each declaring a param named
  `cap`, each generating its own bundle, left per-call dispatch unchanged.

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

### 5.1 The interface is inferred, and then it is real

A module's interface is what it needs and what it produces. **You do not declare
it** — it is inferred from the steps, exactly as today — but it is *materialised
into the module data*:

```python
Affordability.interface
# inputs   net_income          float   required   (leaf)
#          expenses            float   required   (leaf)
#          instalment          float   required   (leaf)
# outputs  disposable_income   float
#          ratio               float
#          final_score         float
# params   AffordabilityParams
# shared   base_rate: float
# taps     term_cap, branch_path
```

Nobody typed that, so authoring cost is unchanged — which matters, because doc 01
§5.3's law is that the expensive path does not get used. But because it now
exists as data rather than as an accident of function names:

- **it renders, diffs and serialises**, so a breaking change is visible;
- **a tool can read it without executing anything**, which is what makes an edit
  checkable rather than runnable;
- **`.at()` has something to rebind** (§5.2);
- **a library can freeze it**, so CI catches an interface change.

```python
Affordability = module(..., contract="contracts/affordability.json")
```

`contract=` snapshots the interface to a checked-in file. Changing the module
without updating it fails the build, naming the field that moved. Opt-in per
module — a shared library freezes its interfaces, a project's inline modules
don't. Semver over that file is then a mechanical question: removing or renaming
anything, or tightening a validator, is major; adding an optional param is minor.

### 5.2 Adapting a module to different names — and keeping it rare

Name matching does the work. A relabel is only for the names that genuinely
differ, and the design's job is to make that set as small as possible, because a
project with hundreds of values cannot afford a mapping at every call site.

**Three layers, cheapest first. Most projects never reach the third.**

**1. Name matching.** No ceremony. This covers the overwhelming majority.

**2. A project vocabulary map** — one declaration for systematic differences,
rather than one per module:

```python
# vocabulary.py — declared once for the project
vocabulary = Vocabulary(
    {"net_income": "monthly_net_salary",          # explicit pairs
     "instalment": "monthly_instalment"},
    prefixes={"bureau_": "cb_"},                  # systematic families
)

pipeline = (Affordability | Underwriting | Scoring).with_vocabulary(vocabulary)
```

This is the layer that handles the hundreds-of-variables case. One place to read,
one place to change, and it covers values produced *mid-pipeline* as well as
input columns — which a rename-at-the-frame-boundary approach cannot.

**3. Instance relabel**, for the genuinely local case — the same module used
twice against different sources:

```python
AffordCurrent = Affordability.at(inputs={"net_income": "current_net_income"})
AffordProposed = Affordability.at(inputs={"net_income": "proposed_net_income"})
```

`.at()` is declared data on the instance, exactly like `.bind()` (§4.3): applied
at the scope boundary, so the module's interior is untouched, and it renders,
diffs and serialises.

**Three properties keep this from becoming mess:**

- **A relabel is a diff, not a mapping.** You name only what differs. A module
  with 15 inputs and one mismatch carries one entry.
- **The rendered artefact shows resolved names.** A reviewer reading the
  generated view sees `monthly_net_salary → afford_score`, never the indirection.
  The relabel exists in the composition source only.
- **The framework tells you when one is needed**, with a suggestion. You never
  hunt for them:

```
'affordability' needs input 'net_income'; nothing in scope produces it and it is
not a declared input column. Closest available: 'monthly_net_salary'.
Add it to the project vocabulary, or:
    Affordability.at(inputs={"net_income": "monthly_net_salary"})
```

> **Why not just rename at the frame boundary?** It is simpler, and it is the
> right answer when one naming scheme can win. It breaks when two libraries
> disagree about a name, and it cannot address a value produced mid-pipeline —
> a waterfall output feeding a second library. The vocabulary map is the same
> idea without either limit.

> **Why not no rename at all?** Because the workaround is a passthrough function
> per mismatched name, and doc 01 §5.1 counted **79** of those in one project,
> plus five naming conventions invented to cope with the same pressure. That is
> the outcome this mechanism exists to prevent.

### 5.3 A bare function is a pipeline element

**Proposed.** A function used directly in a pipeline expression *is* a module —
a single-step one, with its name from the function, its interface inferred
(§5.1), and its params namespace its own name:

```python
pipeline = Affordability | cap_by_income_band | cap_by_sector | Scoring
```

Nothing is special-cased. The engine cannot tell this from
`module(cap_by_income_band, name="cap_by_income_band")`, which is what it desugars
to — the same statement doc 07 §3 already makes about inline modules. Everything
downstream is unchanged: it has an audit identity, a params namespace, a declared
interface, a place in the version chain, and a tap surface.

**The growth path stays a pure move, with no semantic change at any step:**

```
bare function in pipelines/x.py
  →  module(fn, name=...)                 when it needs several steps or a model
  →  modules/x.py                         when it needs its own file
  →  modules/x/{steps,params,__init__}.py when it needs its own directory
  →  registered                           when it must be addressable from config
```

Each move is mechanical and none of them changes behaviour. That matters more
than it sounds: doc 01 §5.3's law is about the cost of the *first* step, and a
framework whose first step is a four-file directory gets 546 inline literals
instead.

The lint from doc 07 §3 still holds, with one word changed: **every `def` in a
pipeline file must appear in the pipeline expression or in a `module(...)` call
in that file.** Steps may not float.

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
score = Affordability.score(
    {"net_income": 42000.0, "expenses": 18000.0, "instalment": 3100.0, …},
    params=p,
)
```

Bypasses polars entirely (doc 02 §3.4). Takes a **dict**, not 400 individual
keyword arguments — measured, not a style choice: at this document's realistic
width (400 inputs, doc 01 §4d), a literal keyword-argument call costs 5.95% of a
20 ms budget on the calling convention alone (CPython's keyword-binding cost
scales close to quadratically with parameter count), against 0.30% for a dict
carrying the same data. See doc 02 §3.5 and EXPERIMENTS.md §N2. kwargs syntax
remains fine for a small, hand-written call — a low-arity module, a test — where
the width is small enough that the cost doesn't matter either way.

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
    | Underwriting                                   # may fuse with Affordability
    | Filter(pl.col("decision") != "declined")
    | Aggregate(by="branch_id", metrics={"mean_score": pl.mean("final_score")})
)
```

- **`|` is a sequence: written order is execution order.** Reordering two modules
  changes behaviour, visibly and intentionally; inserting one affects only what
  comes after it. Within a module the DAG is topologically sorted and order is
  irrelevant (§2), so the two levels have one rule each and neither is implicit.

  > **Correction.** An earlier draft said `|` builds "a dependency-resolved graph,
  > not a sequential stage list", while §3.2 said "ordering lives in `|`". Those
  > are different execution models and the doc set asserted both. The hybrid was
  > the dangerous reading: it left undefined what happens when `A | B` has B
  > producing a value A consumes, so an author could not tell whether written
  > order was load-bearing. It is. What doc 01 §5.2 rejects is *hand-maintained*
  > order in JSON or in a comment — order lives in the pipeline expression, where
  > it is visible, diffable and derived from nothing.
- **Each record-tier module is its own kernel by default** — nothing fuses
  implicitly, because fusion is non-monotone and loses past ~3 modules with cheap
  arms (doc 01 §4b–4c). Wrap a hot group in `fuse(...)` to ask for one kernel;
  it changes codegen only, never the answer (doc 02 §1.2).
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
convergence iteration in the large internal workload surveyed in doc 01, and an
in-repo prototype already modelled the bound.

Loops get real `break`/`continue` early exit in the record tier — precisely what
the polars port destroyed when it replaced a descending `while` with
`filter → group_by(max) → join-back`, losing early exit
and evaluating every arm eagerly.

### 8.4 Why combinators rather than node kinds

A module is a scope with an interface; a branch is a scope with an interface. They
are structurally the same thing, so modelling them as one type plus combinators
means the graph model, compiler, lineage, trace and audit machinery each handle
**one** concept. Nested compilation has a working precedent: a prototype emitter recurses into
`true_body`/`false_body`/`body` to generate nested compiled control flow. It lives
on an unpushed branch — see the citation note in the README — so treat it as a
feasibility signal rather than as evidence, and re-establish it in E2.

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

**There are two config documents, with two lifetimes** (doc 08 §2):

| document | holds | changing it costs |
|---|---|---|
| **params** | values behind fixed types, and table contents | nothing — a bundle swap |
| **interior** | the body of a data-shaped module: rules, bins, table rows | one background compile and a staged swap |

Neither may carry **composition**. `resolve_params` rejects a document containing
`use`, `type`, `steps`, `arms` or `modifies`, which is what makes the structural
guarantee in doc 04 §2 true by construction rather than by convention. The
pipeline skeleton — which modules exist and how they wire — is Python. Doc 08 §7
records how that could be widened later without rework, and why generating Python
is the better answer for a pipeline-building UI.

Neither document may **contain code**, and neither may carry an **unregistered
pointer**. The governing rule is doc 08 §1: *code is a superset of config* — you
go as far as config takes you, and at the point where you would be writing code
in config, you write code, register it, and reference it by id.

| | | |
|---|---|---|
| ✅ **reference** | `{"type": "credit_scorer", "dti_weight": 200.0}` | resolves through the union — id checked, params typed and bounded |
| ❌ **pointer** | `{"module_name": "x", "function_name": "y"}` | `getattr` at runtime — no interface, no schema (`flat_rules`' `output_fn`, doc 01 §5.4) |
| ❌ **code** | `{"expression": "a - b"}` | a second, unspecified way to write arithmetic (`_ComputedFeature`) |

A derived value is therefore a **step**, written and registered like any other,
and referenced from a rule by id. Extensions are how a project's config
vocabulary grows without the framework shipping a release — doc 08 §7.1.

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

The four record-tier modules compile to four kernels — nothing is fused unless
asked for. The style still costs almost nothing at runtime, **not because small
modules fuse, but because boundary stores are near-free** (doc 01 §4c). E5 and E7
falsified the fusion premise; the guidance survived it. If profiling later showed
this waterfall was hot, the change would be one line —
`fuse(ApplyIncomeCap | TermCapBySector)` — and it could not alter a decision.

Absent: registration boilerplate, UUIDs, ordering declarations, type
discriminators, identity-passthrough functions, and `name_override`.
