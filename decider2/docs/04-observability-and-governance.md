# 04 — Observability and governance

Who can change what, how a decision is explained, and what survives an audit.

Much of the mechanism is specified elsewhere — this document is the consolidated
view, and adds the parts that aren't a natural fit anywhere else (§2.1 on what is
actually *enforced*, §5.2 on the audit record's contents, §5.3 on the fact that
per-record diagnostics are PII by construction).

Starting point for calibration: the current framework has **no** observability.
`grep -rE "def (debug|explain|inspect)|DEBUG" decider/` returns zero hits, there
is a field literally named `debug` that does nothing observable
(`decider/executor.py:61`, while `collect` at `:62` is what actually gates
anything), and all 37 debug scripts live outside the framework reaching into
`benchmark.py` internals. Diagnostics were never lifted to a framework concept —
three inconsistent per-module hooks exist and **none is used in production**
(doc 01 §5.4).

---

## 1. The three audiences

| audience | writes | needs |
|---|---|---|
| **data scientists** | most logic, plus its tests | a surface with little ceremony; to test one rule in isolation |
| **software engineers** | harder custom logic, extensions | escape hatches that don't fight the framework |
| **business users** | parameter values; new rules, AI-assisted | to change a value and be *confident* nothing else moved |

A fourth audience never writes anything but must be able to *read*: **credit risk
and compliance**, checking rules against a policy document. That constraint is
the hardest one in the design and it is not yet met (§6).

---

## 2. The governance boundary is the params/structure boundary

The split derived for performance reasons turns out to be the right permissions
split too, which means **one mechanism serves both**:

| | changes | consequence |
|---|---|---|
| **params** | values behind a fixed type | no recompile; graph unchanged; bounded by validators |
| **structure** | which steps exist and how they wire | new graph, new compile, new review |

So: business users move params, engineers move structure.

**Be precise about what this guarantees, because the obvious phrasing is false.**
A param change cannot alter the **graph** — which steps exist, how they wire,
what can affect what. Static lineage, the rendered diagram and the schema are all
invariant under it, and the compiled machine code is identical.

It emphatically **can** alter which arm a given record takes. The branch
instruction is in the code; the constant it compares against is the param. The
worked example in doc 03 §12 is exactly this:

```python
if min_net_salary < params.income_threshold * shared.base_rate:
```

Moving `income_threshold` changes the outcome for every applicant near the
boundary. So the guarantee is *structural invariance*, not *behavioural
invariance*, and the honest consequence is that a param change needs **impact
review** — what fraction of a representative sample changes decision — precisely
because validators alone cannot bound that. Doc 08 §5 specifies the mechanism.

Three mechanisms bound the blast radius:

- **Validators.** `Field(36.0, ge=6, le=60)` — an out-of-range value fails before
  anything executes.
- **Namespacing.** Params are scoped per module instance (doc 03 §4.1), so a step
  in `income_cap` cannot read `sector_cap`'s params. Blast radius is one module by
  construction.
- **`.bind()`.** Engineers freeze knobs at composition time, removing them from
  the caller-facing interface entirely (doc 03 §4.3).

### 2.1 What is actually enforced, and what isn't

Worth being straight about this, because it's easy to over-claim.

**Enforced by the framework:** value ranges (validators); that a step cannot see
another module's params; that a misspelled param is an error rather than silence
(`extra="forbid"`); that a **params document cannot carry composition**, because
`resolve_params` rejects composition keys (doc 08 §2); that a params document
resolves complete when asked, naming every field that fell back to a code
default; and that a resolved bundle carries a non-empty `origin` token.

**Not enforced by the framework:**

- That a business user edits a config document and not `modules/policy_rules/params.py`.
  Nothing technically stops them.
- **Where a params document came from**, or who was allowed to supply it. The
  framework records the `origin` token it was handed; it does not parse or verify
  it (doc 08 §2).
- **Who may activate a staged pipeline.** Approval is the caller's policy (doc 08 §4).

What the design provides is that the boundary is **visible and separable** —
params and module interiors are data documents, the graph skeleton is Python — so
repository policy can differ between the two paths (review rules, CODEOWNERS,
separate approval). Enforcement is CI and review policy; the framework's job is
making the line unambiguous enough that policy can attach to it.

> **Correction to an earlier draft.** This section previously credited
> `decider build --verify` with enforcing config completeness. It does not —
> docs 02 §3.4 and 05 §8 define that flag as a numba-cache assertion and nothing
> else. Completeness is computable from pydantic's `model_fields_set` and is
> enforced by `resolve_params(..., complete=True)`; see doc 08 §2.

---

## 3. Static lineage — answers without running anything

"Which inputs and steps can affect output *z*" is a property of the graph, so it
is answerable with **no data and no execution**:

```python
pipeline.lineage("final_score")   # inputs + steps that can affect it
pipeline.render()                 # diagram
pipeline.schema()                 # required inputs, produced outputs
```

This is the form a compliance reviewer actually wants, because it is a question
about the *rules*, not about one applicant.

It comes free from the record tier but **does not survive arbitrary polars**,
which is why frame operations must declare schema transforms (doc 02 §5). The
escape hatch is deliberately conspicuous:

```python
@breaks_lineage(outputs=["risk_flag"])
def custom_thing(frame: pl.LazyFrame) -> pl.LazyFrame: ...
```

`grep -r "@breaks_lineage"` yields every lineage gap in a codebase — a governance
feature, not just a warning. Queries crossing one report `unknown` rather than
guessing. Outputs are still declared so **schema** propagation survives even
where lineage doesn't.

---

## 4. Explaining one decision

Three tiers, by cost (doc 02 §3.1, §7):

### 4.1 Taps — always-on, in production

```python
module(..., taps=["term_cap", "branch_path"])
```

Each tap is an extra output column. Measured **+0.11 ns/row/tap** at 1 M rows,
linear to at least 4 taps, and it **does not split the kernel** (a real split
costs +2.0 ns/row — so a tap is ~17× cheaper).

`branch_path` is the cheapest and most useful: which branch fired, as one
`int64`. It isn't *computed* — it's a compile-time immediate stored on whichever
branch executes, so it is effectively free at any batch size.

Version qualification matters in a waterfall (doc 03 §7): `term_cap` gives the
**final** value, `term_cap@sector_cap` a specific module's, `term_cap@*` every version
as its own column. Qualify by producing module, never by position — inserting a
rule renumbers positions and silently repoints the tap.

**Reason codes need no new machinery.** A reason code is a step output; "which
rule fired" is `branch_path`. Worth stating explicitly because a legacy system
being ported carried a substantial decline-reason taxonomy and **every one of
those codes was dropped** in translation — the capability should be obvious
enough that it doesn't get dropped again.

### 4.2 Trace and intervention — one record at a time

```python
with pipeline.debug(net_income=42000.0, params=p) as dbg:
    dbg.step()                      # advance one step, using COMPILED step code
    print(dbg.values)               # all values, with version chains
    dbg.set("term_cap", 36.0)       # what-if: override mid-flow
    dbg.run()
    print(dbg.trace)
```

Defaults to `stepped` — real compiled step code, so numerics match production.
`mode="interpreted"` drops to Python steps when you need to see *inside* a step.
Requires no taps and no redeploy.

What makes this trustworthy is the **equivalence ladder**: `interpreted ≡ stepped
≡ fused` is an automated test, and a disagreement localises to a layer
(`interpreted ≠ stepped` means numba changed a step's semantics; `stepped ≠
fused` means fusion did). Without that, debugging in a different mode from
production proves nothing.

> **Open (O9):** what "step" means at a `Branch` — step over, or into the taken
> arm? — and at a `Loop` — one iteration, or one step within one? This shapes the
> trace data model, so it needs settling before `observe/trace.py`.

### 4.3 OTel — the pipeline, not the records

Spans wrap **module and kernel boundaries**: a handful per invocation,
negligible cost. Per-record spans would be millions per batch and are never
emitted.

> **OTel measures the pipeline; taps and traces explain the records.** Per-record
> diagnostics travel as columns.

---

## 5. The audit trail

### 5.1 Version chains give exact attribution

Each value version has exactly one producer, so the chain *is* the record:

```
term_cap:  60.0  (SeedTermCap)
        →  48.0  (ApplyIncomeCap)
        →  36.0  (ApplySectorCap)
```

This is why overwrite is modelled with internal versioning rather than forbidden
(doc 03 §3.3): the fix for the wiring problem and the implementation of "step x
changed y from 1 to 2" are the same mechanism.

**Node identity must be deterministic** — content-derived or explicitly
declared, never auto-generated — or path codes aren't comparable between
versions and the trail is worthless across a release. The current framework's
flat-rule leaf ids are auto-generated and unstable, which is exactly this bug
(doc 01 §5.4).

### 5.2 What an audit record should contain

Structure and params are data, so a complete record is assemblable rather than
reconstructed:

| field | source |
|---|---|
| pipeline identity | hash of the exported JSON (doc 07 §5) |
| resolved params | the explicit, complete config — *not* code defaults |
| framework + module versions | build metadata |
| inputs | the record as received |
| outputs | final values |
| tapped values | declared diagnostics, incl. `branch_path` |

"What were the parameters on 3 March" answers from one artefact. That is the
whole reason production config must be explicit rather than inheriting defaults
(doc 07 §2) — otherwise the answer requires cross-referencing a config file
against a code version at that commit.

**A config diff is an audit record.** Because structure is a pydantic object and
params are data, `diff(v1, v2)` produces a readable change record with no extra
machinery.

### 5.3 Per-record diagnostics are PII by construction

Not a framework feature, but it follows directly from the design and belongs on
the record: a full trace of a declined application contains applicant financials
and bureau data. Tap columns are applicant data. So:

- trace output inherits the sensitivity of its input and must not be logged
  casually or shipped to a general observability backend;
- this is another reason OTel carries **stage** spans only — per-record spans
  would put applicant data in a tracing system by default;
- `stepped`/`interpreted` sessions operate on real records and should be treated
  as production data access.

---

## 6. The reviewable artefact — the weakest part of the design

**Requirement:** a credit-risk or compliance reviewer, who does not read code,
can verify a rule against a policy document.

Both existing representations fail this (doc 01 §6), and it's worth registering
*how* comprehensively:

- **The Python fails.** To check one policy rule, a reviewer must locate a
  function whose name encodes both the rule identifier and its position in a
  waterfall (`_stage_07_...`), understand two single-letter helper
  functions, work out that one input silently re-derives a gate computed
  elsewhere, and trace where the value arrived from. The rule's *name* is in a
  function name, its *effect* is a `min_horizontal` call, and its *position in
  the sequence* is implicit in a parameter name.
- **The declarative form fails worse.** The equivalent JSON rule format runs to
  ~1000 lines of serialised AST with UUID node ids and `result_idx: -1`
  indirection to express roughly thirty rules.

So "make it config-driven" is not the answer. The proposal is to generate the
reviewable view from the module data — ordered steps with their `description`,
declared inputs and outputs, the params each reads with validated bounds, and the
value-version chain — but **this is untested** (O3), and E4 exists to put it in
front of an actual reviewer with a policy document.

Two reasons it's ranked as the top risk despite looking like a documentation
task:

1. If it can't be met, the governance story collapses and several design choices
   made in its service (structure-as-data, `description` on steps, version
   chains) lose their justification.
2. It is **people-blocked, not code-blocked**, so it is the one thing that should
   start before implementation rather than after.

---

## 7. Correctness

There is no migration and no permanent oracle, so correctness is
**specification-based**: rule-level assertions expressing intent, which double as
something a reviewer can read.

```python
def test_income_cap_applies_below_threshold():
    assert ApplyIncomeCap.score(min_net_salary=4000.0, term_cap=60.0, params=p) == 48.0
```

Cheap because steps are pure and individually callable. Framework-provided:

```python
assert_modes_agree(pipeline, corpus)    # the equivalence ladder
golden.record(pipeline, corpus)         # optional regression baseline
```

Golden-trace comparison ships as a **capability** for regression baselines, not
as the theory of correctness — "85% per-column agreement with an oracle" can tell
you a rule is *unchanged*, never that it is *right*.

Calibration again: the current project has one pytest-style test file for 66
module directories, six assertions total, and `testpaths` configured such that it
doesn't run (doc 01 §5.6).
