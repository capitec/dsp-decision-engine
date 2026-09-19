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
- **That a realtime request's raw payload params ever touched a reviewed
  document at all.** Doc 02 §4 permits params "in a realtime request payload,"
  which is a params document that never passed through a file a CODEOWNERS rule
  could watch — the framework validates it (`resolve_params`) but has no
  opinion on where the caller got the values. ✅ **Measured —
  [EXPERIMENTS.md](EXPERIMENTS.md) §N3: this is not a performance restriction in
  disguise.** Validating a realtime payload costs at most 1.28% of a 20 ms
  budget even at 50 module instances, so restricting payload params to a
  reference-by-id (rather than raw values) cannot be justified as "too
  expensive to validate per request" — it would have to be justified as a
  governance decision on its own terms, which is what this list already says.

What the design provides is that the boundary is **visible and separable** —
params and module interiors are data documents, the graph skeleton is Python — so
repository policy can differ between the two paths (review rules, CODEOWNERS,
separate approval). Enforcement is CI and review policy; the framework's job is
making the line unambiguous enough that policy can attach to it.

> **Correction to an earlier draft.** This section previously credited
> `decider2 build --verify` with enforcing config completeness. It does not —
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

### 4.1 Emitted intermediates — always-on, in production

```python
pipeline.emit("term_cap", "TermCapBySector_path")
```

Diagnostics are not a separate mechanism: an intermediate you want to see in
production is a **column you asked for** (doc 03 §7). Measured **+0.11 ns/row per
value** at 1 M rows, linear to at least 4, and it **does not split the kernel** (a
real split costs +2.0 ns/row — ~17× cheaper).

A branch's `_path` value is the cheapest and most useful: which arm fired, as one
`int64`. It isn't *computed* — it's a compile-time immediate stored on whichever
arm executes, so it is effectively free at any batch size.

Version qualification matters in a waterfall (doc 03 §7): `term_cap` gives the
**final** value, `term_cap@sector_cap` a specific module's, `term_cap@*` every version
as its own column. Qualify by producing module, never by position — inserting a
rule renumbers positions and silently repoints it.

**Reason codes need no new machinery.** A reason code is a step output; "which
rule fired" is the branch's `_path`. Worth stating explicitly because a legacy system
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
Requires no emitted columns and no redeploy.

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

> **OTel measures the pipeline; emitted columns and traces explain the records.** Per-record
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
| pipeline identity | structure fingerprint + compiled artefact id (doc 08 §8) — **not** a hash of the params document, which identifies the values rather than the thing that ran |
| resolved params | the explicit, complete config — *not* code defaults |
| framework + module versions | build metadata |
| inputs | the record as received |
| outputs | final values |
| emitted intermediates | declared diagnostics, incl. a branch's `_path` |

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

## 6. The reviewable artefact — TESTED, AND IT FAILED

**Result first, because it is the design's top-ranked risk and it now has an
answer.** A proxy reviewer was given a policy extract, a generated rule sheet and
one decision record, and asked three questions (`decider2/reviewer-test/`).

The documents contained **two deliberate compliance breaches**:

1. The rule read `monthly_income` (`gross_salary + other_income_declared` =
   R9 200) where policy §7.4.2 says **verified net monthly income** — which for
   this applicant was R7 480, below the R8 000 floor. They should have been capped
   at 48 months. They received 60.
2. The adjustment ADJ-0117 raised the cap from 48 to 60, which §7.4.5 explicitly
   forbids — *"no term reduction may be waived, overridden or extended by a
   pricing, campaign or affordability adjustment"*. The sheet rendered a
   non-compliant adjustment as a routine approved change.

**Neither was found.** The reviewer instead reported a false positive on an
unrelated rule, because the sheet's "In force now" column contained `60`, `60`
and `unchanged` — three rows, two kinds of value, no explanation. Task A was
completed but reported as "hard to follow".

Three conclusions, and the third is the design change:

**6.1 Displaying a value is not the same as making it checkable.** The sheet
showed both the authored and in-force values and the reason they differed —
which was the best idea in the eleven mock projects — and the reviewer still
could not tell that the in-force value violated policy. **The sheet rendered the
adjustment; it did not render the policy constraint the adjustment broke.**
Anything a rule must not do has to appear on the sheet as a stated limit, not be
absent because nothing violated it at authoring time.

**6.2 The provenance section is where the first breach lived and it was not
readable as provenance.** "Where `monthly_income` comes from" gave the definition
correctly — `gross_salary + other_income_declared` — and the reviewer did not
connect it to policy's "verified net". A definition renders as a formula; a
reviewer needs the *policy term* it claims to implement, and a flag when a value
named in policy exists in the pipeline and is **not** the one the rule reads.
`verified_net_income` was right there in the decision record, unused.

**6.3 The failure mode was cross-referencing, not comprehension.** Asked
afterwards what made it hard, the reviewer said: *"I wasn't sure what the ask was
and I found myself trying to keep context between different md files."* The sheet
could not be read alone — answering "was this decision correct?" required holding
a policy extract, a rule sheet and a decision record simultaneously and joining
them by hand.

That is the whole problem, and it is a stronger result than "the sheet was
confusing". **An artefact that requires the reviewer to join three documents has
already failed, however good each document is.** The join is the work, and the
join is exactly what the framework can do and a human cannot do reliably.

So the unit is not "a sheet per rule". The unit is **one question, answerable from
one screen, with everything needed inlined** — the policy term, the value the rule
actually read, its provenance, and the limit it must respect, all rendered at the
point of use rather than referenced.

### 6.3b A reviewer cannot adjudicate a mismatch they are shown

**Settled by a second, smaller test, and this supersedes the rendering approach.**

The first test was rejected as too much cross-referencing (§6.3), so it was rerun
with everything on one screen: the policy clause, the value the rule read, its
definition, the policy term it claims to implement, and a flagged near-miss value
that existed but was not read. Roughly fifteen lines, one question.

**It failed again**, and the reviewer's response is the result:

> *"the income is 9200 so the rule shouldn't fire — why is there 2 sources of
> income, where is the 7480 coming in, it's hard to know how that should interact.
> Are you saying the 7480 is a better source than gross? Why do we have
> contradicting sources?"*

Every one of those is the right question. **Asking them is the finding.** A real
application genuinely carries several income figures — gross, declared, verified
net after statutory deductions — and they are not contradictory, they are
different measurements. Policy names one. The rule read another. But deciding
*which one policy meant* is precisely what the reviewer came to find out, so
showing both and expecting adjudication asks them to supply the answer they are
seeking.

> **A reviewer cannot adjudicate a mismatch they are shown. They can only confirm
> a check the system already made.** No rendering fixes this, because it is not a
> presentation problem.

**A mechanism was proposed and then withdrawn.** The first response to this was a
declared policy-term binding — `PolicyTerm("§7.4.2", reads="verified_net_income")`
— with a step declaring `implements="§7.4.2"` and the framework checking the
signature against the binding at build time. It was rejected, correctly:

> *"if we have to for every rule say what it reads that would be way too tedious…
> it's already obvious from the function definition that the system only reads
> `monthly_income`. I think this might be overengineering."*

The objection is right about the part that matters. The step's signature is
already the complete statement of what it reads, and `reads=` puts a value name in
a **second place** that must be kept in sync and is only as good as whoever wrote
it. That is a real cost for a mapping the framework cannot verify.

### 6.3c What the framework can and cannot do here

The distinction this exercise actually established:

| | |
|---|---|
| **The framework can** | show what a rule reads, and the resolved definition of each value it reads, at the point of use |
| **The framework cannot** | know that policy's phrase *"verified net monthly income"* means `verified_net_income` and not `monthly_income` — both are plausible readings of a real domain term, and both values legitimately exist |

Closing that second gap requires a human to write the mapping down. **Writing it
down is the registry that was just rejected as overengineering, and that rejection
is reasonable** — it is one line per policy clause, permanently maintained, to
catch a bug class that a single rule-level test also catches:

```python
def test_low_net_income_caps_term():
    assert term_cap(verified_net_income=7_480.0, gross_salary=9_200.0) == 48
```

That test fails the moment the rule reads the wrong income, costs one function,
and is already the design's stated correctness mechanism (§7, doc 03 §11).

> **Settled: the framework does not attempt to verify that a rule implements a
> policy clause.** It surfaces what a rule reads and what those values mean;
> confirming that against policy is a human judgement supported by tests, not a
> build-time check. Two attempts to make a rendered artefact carry that judgement
> both failed, and the mechanism that would have worked costs more than the bug.

**What survives, and it is the cold-read study's own finding independently:** a
rule needs a **join key to the policy clause it implements** — `implements="§7.4.2"`
as metadata, with no `reads=` and no registry. That is one string, it duplicates
nothing, and it makes "which rules implement §7.4" answerable mechanically instead
of by reading prose. The cold-read study reached the same conclusion from the
opposite direction: *"the top-ranked risk is 'verify a rule against a policy
document' and a rule carries no key to join to a policy document on."*

**6.4 There is no single reviewable artefact, and there should not be.** A
single prescribed format was the wrong shape — see §6.5.

### 6.5 Trace is data; every rendering is replaceable

**Settled, and it corrects this section's original framing.** Doc 04 previously
proposed *the* reviewable artefact, as though one format could serve a credit-risk
reviewer, an operations agent under time pressure, a regulator doing an inventory,
and a developer debugging. It cannot, and different teams will want different
things from the same decision.

So the framework owes three things, and a fourth is explicitly not its business:

1. **A structured decision record**, emitted as data — every value, its producer,
   its authored and in-force form, which rules were evaluated and which fired,
   what each read, and the provenance of every threshold. This is the *only*
   thing the framework guarantees, and it is what everything else is built from.
2. **Renderers over that data**, shipped as defaults and **replaceable without
   forking**. The rule sheet in `decider2/reviewer-test/` becomes one renderer,
   not the artefact. A team that wants a different shape writes one.
3. **Configurable verbosity**, because trace detail trades against speed. A
   production realtime path may emit only fired-rule ids and the values that
   moved; a dispute investigation may emit everything. The level is a parameter,
   and what it costs is measurable (doc 02 §7: an emitted value is ~0.11 ns/row, a full
   trace ~4× materialisation).
4. **Not the presentation.** Which columns, what wording, what a team's reviewers
   are used to — that is theirs. Same principle as doc 08 §6: the framework owns
   the data and its guarantees, never the surface.

> This is the same correction as config sourcing. Prescribing one format is how a
> framework ends up with an artefact that serves nobody exactly, and it is why
> nine of eleven mock projects produced no reviewer artefact at all — the ones
> that tried were designing for an imagined single reader.

### 6.6 What the original section got right, and the evidence for it

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
