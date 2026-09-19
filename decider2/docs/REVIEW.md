# Review of the decider2 doc set

Reviewed 2026-09-18 against commit `367b575`, doc set at 3,187 lines across eight files.

Method: ten reviewers over distinct dimensions, each finding adversarially checked twice
(once against the doc text, once against whether it matters for the stated next step), plus
independent verification of every `decider/<path>:<line>` citation and every referenced
prototype against the working tree and full git history.

164 findings survived verification. This document keeps the ones worth acting on before
implementation starts, and says plainly which are cheap doc edits and which are design gaps.

**Read §1 first.** It is the only finding that invalidates other work rather than adding to it.

## Status — actioned 2026-09-18

| finding | status |
|---|---|
| §1 citations | **noted.** The cited paths exist on an unpushed branch. A note in the README records which claims resolve only there, and doc 02 §3.2's "not new" de-risking claim is restated — `decider2` is a ground-up build, and the prototypes are a feasibility signal rather than a pedigree. Re-run the citation pass against a pushed commit and pin it. |
| §1b scrub artefacts | **fixed.** Four ungrammatical substitutions repaired; an internal file path removed. `AliasCombineModule` and `_stage_*` remain — decide whether to keep them or relax the README claim. |
| §2.1 fusion (3 sites) | **fixed** in doc 03 §8.1, §12 and the inline comment. |
| §2.2 E8 vs O10 | **fixed** in doc 06. |
| §2.3 fourth null tier | **fixed** — doc 05 §2 is three tiers, with the rejection recorded. |
| §2.4 doc 02 §3.3 | **fixed** — no fixed `prange` threshold, `fastmath` selective, `parallel=True` 1.2–2.6×, and "warmup" disambiguated as measurement-not-compilation. |
| §2.5 fusion cap | **resolved by removing it.** The cap is withdrawn, not restated: it named a range its own table shows as 10–28% worse, and no constant spans 0.11× to 1071×. Fusion is now explicit — one kernel per module for `apply()`, maximal for `score()`, `fuse(...)` for anything else, guaranteed semantically transparent. Doc 02 §1.2; withdrawal recorded in doc 01 §4c and doc 06 O7/O12. |
| §4.1 no input-side rename | **designed.** Doc 03 §5.2 — three layers (name matching → project `Vocabulary` → instance `.at()`), with the framework naming the relabel you need. Falsification test added to the first milestone (O19). |
| §4.2 pipeline not a scope | **fixed.** Doc 03 §3 lists five scopes; §2.1 gives the pipeline's precedence rules and makes module-output-shadows-frame-column a build error — the silent-rebind bug. Rate-of-firing is O20. |
| §4.7 `\|` sequence or DAG | **decided: sequence.** Written order is execution order between modules; topological sort within one. Doc 03 §8.1, with the contradiction recorded. |
| §4.8 wiring typo has no error | **fixed.** Doc 03 §2.2 — did-you-mean against in-scope names and columns, and `pipeline.schema()` lists all unbound inputs at once. |
| §5 no module contract | **designed.** Doc 03 §5.1 — the interface is inferred, materialised into the module data, and freezable with `contract=`. Semver becomes mechanical over that file. Scope of "breaking" is O19. |
| §2.6 "nine experiments" | **fixed** in the README, with the outstanding five named. |
| §3 control-flow over-claim | **fixed** — doc 04 §2 now claims structural invariance only, and requires impact review (doc 08 §5). |
| §3.1 `--verify` / payload params | **fixed** — doc 04 §2.1 rewritten, with the correction recorded in place. |
| §4.5 business users add rules | **answered** by doc 08 (three change classes, bounded interiors, staged compile). |
| §6 config seam | **answered** by doc 08 §6–§7, and doc 02 §6's `config/` renamed `binding/`. |
| §7 compile gaps | **largely measured — see EXPERIMENTS.md.** Doc 05 §9 now has criteria 10–15. Confirmed: per-node fallback cannot exist inside a fused driver (§B), and byte-identical is insufficient for cache survival (§C). Both now specified. |
| §4.4 dtypes | **resolved.** Doc 05 §1.5 is the admissible-dtype contract, measured across 26 dtype/nullability combinations. Utf8/Categorical/Enum enter as codes; Decimal and List are inadmissible; money is scaled int64 (doc 03 §1.2). |
| §4.3 cost of one rule | **designed, 6 artefacts → 1.** Doc 03 §1.1 — a bare function is a pipeline element (§5.3), `param()` in the signature generates the model (§4.4, settling O2), the docstring is the description, and the config entry is generated. Gated on E11 confirming numba accepts the generated signature. |
| §4.6 one output per step, §4.9 doc errors | **open.** |

---

## 1. The evidence base cites code that does not exist

The README stakes the doc set's credibility on verifiability:

> Where these docs describe how `decider` behaves today, the claim is cited as
> `decider/<path>:<line>` so a reader can check it rather than take the doc's word.

That convention does not hold. Verified against the working tree and against
`git log --all -S<symbol>` over the entire history:

| cited | status |
|---|---|
| `decider/modules/record.py` (cited ~8×, incl. `:18-20` quoted verbatim) | **does not exist, never has** |
| `_build_jit_driver` (`record.py:108-183`) | **no such symbol anywhere in history** |
| `decider/modules/primitives/branching.py:96` (`how="diagonal_relaxed"`) | **does not exist, never has** |
| `decider/plan.py:32-37` (`ExecutionPlan.execute(audit=True)`) | **does not exist, never has** |
| `name_override` (`functional.py:61`, `:169`, applied `:190-206`) | **no such symbol anywhere in the repo** |
| `experimentation/steptree_poc/jit_codegen.py` (cited 3×) | **does not exist**; `experimentation/` holds only `graph_poc/` and `jittree/` |
| `grep -rn numba decider/` | **zero hits** — numba appears only under `experimentation/` |
| `_detect_module_kind` (`functional.py:21`) | exists at `:18`, but returns `Literal["expr", "frame"]` — **two kinds, not three** |
| `functional.py:397`, `:246` | file is **201 lines** |
| `primitives/sequential.py:135` | file is **105 lines** |
| `executor.py:61-62` (`debug`/`collect`) | correct claim, actual lines **54-55** |
| `graphutil.py:14`, `:38` | exact |

Three separate reviewers reached this independently. The pattern is not uniform drift: a
cluster of citations is exactly right, another cluster is off by a few lines, and a specific
set of four files/symbols is absent from history entirely.

**Why this matters more than a broken link.** The fabricated cluster carries some of the doc
set's most load-bearing arguments:

1. **Doc 02 §3.2's de-risking claim.** *"The compile mechanism itself is **not new** — `decider`
   already njits each function and codegens a fused row loop."* If nothing in `decider` has
   ever imported numba, decider2 is not generalising a proven in-house mechanism; it is
   building one from scratch. That changes the risk profile of the whole project.
2. **The `diagonal_relaxed` data-corruption bug**, cited in docs 01, 02 and 03 as the
   justification for declared output schemas — *"silently upcasting the whole column to string
   and having corrupted genuinely-computed booleans."* The conclusion may still be right;
   the evidence for it is not checkable.
3. **`name_override`**, listed as one of six measured consequences of the self-reference ban
   and as a misfeature decider2 removes. Removing something that does not exist also means
   its *legitimate* use case — adapting a reusable module to a different naming context — was
   discarded without being considered. See §4.1.
4. **`steptree_poc/jit_codegen.py`**, the sole "already proven feasible" citation for nested
   `Branch`/`Loop` codegen, which doc 05 §4.3 depends on. The nearest real prototype,
   `experimentation/jittree/test.py`, emits a binary decision tree via `exec` — the form
   doc 05 §4.1 forbids — and has no `Branch`/`Loop` recursion at all.

**Resolved:** these are committed to a local branch that was never pushed, so the claims are
true and the citations point at a tree a reader of this repo cannot reach. Doc 01's stated
purpose is to let later readers *challenge* conclusions rather than re-derive them, so the
citations need to resolve against something reachable before the doc set is used as a
specification.

**Action taken:** the README now records which claims resolve only on that branch. Doc 02 §3.2
no longer reads as de-risking — `decider2` is a ground-up build and nothing from the old
implementation is carried forward, so an unreleased prototype is a feasibility signal, not a
pedigree.

**Still to do:** push the branch (or a squashed reference commit) and pin its SHA in the
README, then re-run the line numbers. Roughly half the citations against files that *do* exist
have drifted a few lines — `debug`/`collect` are at `executor.py:54-55`, not `:61-62`;
`_detect_module_kind` at `:18`, not `:21`; `class TModule` at `:142`, not `:167` — and two are
past end-of-file on the pushed tree (`functional.py:397`, `sequential.py:135`).

### 1b. Related: the confidentiality claim does not hold either

README: *"no domain logic, field names, thresholds, rule identifiers or file paths from it
appear here."* But doc 01 §5.2 names `AliasCombineModule`, doc 03 §8.3
cites an internal file path, and `_stage_00`–`_09` appear in docs 01, 03 and 04.

A scrub also left four sentences ungrammatical, where a project name was replaced with the
phrase "a large internal workload": doc 02:262, doc 03:550, doc 06:38, doc 06:42. Example —
*"(the a large internal workload port does exactly that)"*.

**Action:** finish the scrub, or relax the README claim to what is actually true.

---

## 2. Stale claims from the two reversals

The doc set's best quality is that it let two experiments overturn written claims. The
propagation of those reversals was incomplete, and it failed in the sections people copy from.

### 2.1 The falsified fusion claim survives verbatim in the worked example

Doc 01 §4b is explicit about what it corrected:

> Doc 02 §1 and doc 03 §3.2 previously claimed the small-module style "costs nothing at
> runtime" because adjacent modules fuse. **That is false** past ~4 branch groups with cheap
> arms, where it is 6–9× *slower*.

The fix was applied to doc 02 §1.1 and doc 03 §3.2 — the two sites named — and to nowhere
else. Three occurrences remain:

- `03-authoring-api.md:738` — the **last line of the canonical worked example**:
  *"All four record-tier modules **fuse into a single kernel**, so the small-module style
  costs nothing at runtime."* Word for word the sentence §3.2 says was disproved.
- `03-authoring-api.md:487` — *"**Adjacent record-tier modules fuse into one kernel**"*
- `03-authoring-api.md:479` — the inline comment `# fuses with Affordability`

The worked example is what an implementer copies. **Action:** grep-and-fix all three.

### 2.2 Doc 06's E8 entry contradicts doc 06's O10, ~140 lines apart

O10 (settled): plain `Optional[float]`, chosen **against** the fastest measurement, with the
reasoning recorded. E8's summary, in the same file: *"All five candidates compile; the
NamedTuple form wins on both axes at once."* That is precisely the claim the reversal
overturned — doc 01 §4 says the NamedTuple is faster but *less safe*.

### 2.3 Doc 05 §2 offers a fourth null tier that doc 03 §1 explicitly rejects

Doc 03 §1: three tiers, and *"Also rejected: a `.value`/`.valid` wrapper."*
Doc 05 §2: a four-row table whose fourth row is `x: Maybe[float]` — `.value`/`.valid` — cited
to doc 03 §1 as its source. Doc 01 §4 takes a third position ("measured and available if a hot
path ever needs it").

Doc 05 is labelled **"the first thing to build."** Three positions on the public authoring
surface, in the implementation spec. **Action:** pick one and say so in all three.

### 2.4 Doc 02 §3.3 carries three retracted numbers and contradicts §3.4 one paragraph later

- *"`prange` is a pessimisation below ~50k rows"* — doc 01 §4, doc 05 §5.1 and doc 06 O6 all
  say **there is no fixed row threshold** and *"Do not hardcode a constant."*
- *"`fastmath` off by default — measured as noise"* — doc 01 §4c corrects this to
  **2–2.5× on arithmetic-heavy drivers**; doc 05 §5.2 has the corrected wording.
- *"triples compile time"* — doc 01 §4b corrects to **1.2–2.6×, shape-dependent**.
- §3.3 says variants are *"built eagerly at startup"*; §3.4, immediately below, requires
  a runtime load to trigger **zero** compilations.

The last one is not just staleness — "warmup" is doing two jobs across the doc set
(compile-time vs measurement-time) and the two readings are mutually exclusive.

### 2.5 The ~6–9 step fusion cap sits inside the region E5 measured as slower

E5's table (doc 01 §4c) gives step counts in its second column: 3 modules = 6 steps,
5 modules = 9 steps. Split/fused ratio at 6 steps is 0.87–0.90; at 9 steps, 0.72–0.76. So the
cap in doc 02 §1.1, doc 05 §7 and doc 06 O7 licenses groups the same table shows are **11–28%
slower than not fusing**, above the 10k-row break-even.

The narrative ("harmful past ~5 modules") and the number ("~6–9 steps") were derived from
different columns. **Action:** state the cap in the same unit as the measurement, and pick a
number the table supports (~3 steps at >10k rows, with the arm-cost exception).

### 2.6 Minor factual drift

- README: *"Nine experiments were run."* Doc 06 defines **ten** (E0–E9) and marks **five**
  DONE. E1 — which doc 06 calls "the one to build first" — and E4, the top risk, are both
  still planned.
- Doc 01 §4's E8 table gives the NamedTuple at 749/358 µs; the prose beneath it, doc 03 §1,
  doc 05 §2 and doc 06 O10 all quote **413 µs**, which appears in no table.
- Doc 06 E5 claims *"Every composition mechanism in doc 03 §4 verified"*. E5's own results
  cover five things, none of which is `.bind()` or the shared required-fields contract —
  both of which E6 exists to test. The Sequencing section then retires most of E6 on that
  basis.

---

## 3. The governance boundary rests on a claim that is false

Doc 04 §2, and the same reasoning in doc 02 §4:

> A param change **provably cannot alter control flow**, because the compiled code is
> identical — the same machine code runs with different constants.

Identical machine code is exactly the situation in which changing a constant **does** alter
control flow. The branch instruction is in the code; the constant it compares against is the
param. The doc set's own worked example proves it — doc 03 §12:

```python
if min_net_salary < params.income_threshold * shared.base_rate:
    return min(term_cap, params.cap)
```

Changing `income_threshold` changes which arm executes for every applicant. That is the only
sense of "control flow" a credit-risk reviewer cares about.

This is the load-bearing safety claim for letting business users edit params without
engineering review, and it appears in the one section whose stated purpose is
*"Worth being straight about this, because it's easy to over-claim."*

**The true and still-useful claim:** a param change cannot alter the *graph* — which steps
exist, how they wire, what can affect what. Static lineage is invariant; per-record outcomes
are not. **Action:** restate it that way, and add the honest consequence — a param change
needs impact review (what fraction of a sample changes decision) precisely because it *can*
change every decision.

### 3.1 Two related over-claims in the same section

- **`decider build --verify` is credited with enforcing config completeness** (doc 04 §2.1,
  under "Enforced by the framework"). Doc 02 §3.4, doc 05 §8 and doc 05 §9.7 all define that
  flag as a numba-cache assertion and nothing else. Doc 07 §6 lists completeness as a *lint
  rule* — i.e. the CI-and-review category doc 04 §2.1 puts on the "not enforced" side of its
  own line.
- **"Params live in config files, so repository policy can attach to them."** Doc 02 §4 and
  doc 01 §6 both say params may arrive in a realtime request payload. Both paths write the
  same bundle and are indistinguishable at the step, so an upstream caller can supply
  governed policy values with no config file, no review and no provenance recorded.

---

## 4. Usability at realistic scale

Nearly every finding below is a rule that is obviously right at three modules and inverts
somewhere between ten and fifty. The design has been reasoned about at sixty and validated at
three.

### 4.1 A module is only reusable in a project that already speaks its vocabulary — **blocker**

Found independently by four reviewers.

Params got a full reuse mechanism: per-instance namespacing, three paragraphs of
justification, `.bind()`, the same-module-twice case. **Values got nothing.**

A step's input names are its parameters; its output names are its function names. Neither is
adaptable by a consumer. So a shared `credit:affordability` reading `net_income` cannot be
used by a project whose column is `monthly_net_salary` without editing the shared module.
Doc 01 §5.8 removes `name_override` as *"Unnecessary — output name is declared directly"* —
which answers only the output side, and only for a module you author.

The workaround is a rename step per mismatched name, which is exactly the
79-identity-passthrough-functions pathology doc 01 §5.1 exists to kill.

**Interaction the reviewers found (worse than either part alone):** because the pipeline is a
single flat value namespace (§4.2) and a name can hold several versions, a step's input
resolves to *"whatever the most recent producer upstream of me wrote."* So **a module's
semantics are a function of where it sits in the `|` chain.** Move it, or insert a rule before
it, and it silently reads a different value. Combined with no rename, a purely additive
library change — v1.1 adds a step named `risk_band` — silently re-binds a consumer's input
that previously came from the frame.

**Action, and it is one mechanism not three:** a declared, data-shaped relabel at the instance
boundary, alongside `.bind()`:

```python
Affordability.at(inputs={"net_income": "monthly_net_salary"},
                 outputs={"score": "afford_score"})
```

Declared data on the module instance, so it renders, diffs and serialises; applied at the
scope boundary, so the interior is untouched. This is also the natural place for a UI to
expose wiring. Also needed: a stated precedence rule between an upstream output and a frame
column, and a way for a step's input to address a *version* (`term_cap@seed`) — taps can
already do this; wiring cannot, so the waterfall is write-only and reporting "requested term
alongside granted term" requires inventing a new name.

### 4.2 The pipeline is not a scope, and a module has no declared interface

Doc 03 §3's four scopes are step, module, branch arm, loop body. `A | B | C` therefore shares
one flat value namespace, and the "declared interface" the invariant rests on is never
actually declared — `module(...)` takes only `name=`, `params=`, `taps=`. There is no
`inputs=`/`outputs=`.

The strictness is inverted relative to the risk: duplicate outputs *inside* a module are a
hard error, while the pipeline-wide namespace — hundreds of names across sixty modules — has
no collision rule at all.

This is also what blocks §4.1's fix, §5 (a public contract), and doc 04 §5.2's "module
versions" row, since there is no declared surface to version.

### 4.3 The cost of one rule violates the doc set's own law

Doc 01 §5.3 states the law that explains the previous adoption failure:

> the tunable form must be *cheaper to write* than the literal, or it will not be adopted.
> Whichever path is cheaper is the path people take.

Evidence: 546 inline `pl.lit(<number>)` against zero config uses, zero validators.

Under the recommended waterfall style, one policy rule costs: one function + one `module(...)`
call + one instance `name=` + one pydantic params model + one entry in the pipeline expression
+ one namespaced entry in the config JSON. Six artefacts in four files for one `if`.

Doc 04 §6 puts the ported rule set at "roughly thirty rules."

O2 (in-place param promotion, `p("income_cap", 48.0)`) is the mechanism that would fix this,
and doc 06 lists it under "Open and unscheduled." Given that authoring cost is now the *only*
remaining defence against the 546-literal outcome, that is the wrong priority. **Action:**
promote O2/E6-part-one to scheduled, ahead of most of the compile work.

### 4.4 Half a credit record has no representation

`grep -rn 'Utf8|categorical|Enum|date|datetime|decimal|round|money' decider2/docs/` returns
nothing relevant. Doc 01 §4's dtype table covers Float64/Int64/Int32/UInt8/Boolean. Doc 05 §1's
extraction spec — "the first thing to build" — is a two-buffer Arrow unpack, which is valid
only for fixed-width primitives:

- **Utf8/String** is `[validity, offsets, data]` — three buffers, so the documented tuple
  unpack raises `ValueError`
- **Categorical/Enum** is dictionary-encoded (indices + dictionary child)
- **Decimal128** is 16-byte items with no numpy scalar
- **List** is `[validity, offsets]` plus a child
- **Date/Datetime/Duration** are primitives but need a unit and an epoch convention

Meanwhile the docs' own examples assume non-numeric data: sector codes, product codes, decline
reason taxonomies, `pl.col("decision") != "declined"`. And `dtype` in the extraction snippet
at doc 05 §1.3 is **unbound** — nothing says where it comes from. If it comes from the step's
Python annotation and the column is `Int64`, `np.frombuffer` reinterprets the bits.

Money is its own gap: outputs are instalments, fees and rates that must reconcile to the cent.
`Decimal` cannot enter a numba kernel, Python's `round()` is banker's rounding and numba's is
not guaranteed to match, and the equivalence ladder's list of numba-vs-Python divergences
(float ULP, integer overflow) omits rounding entirely.

**Action:** doc 05 needs an admissible-dtype contract before anything is built — which dtypes
cross the boundary, which are rejected with what message, and what the money representation is.

### 4.5 "Business users add rules" is impossible as designed

Doc 04 §1 lists it as a persona requirement. Doc 04 §2 classifies adding a rule as structure —
*"new graph, new compile, new review."* Doc 02 §3.4 moves compilation into the Docker image
build. So the promised path resolves to: edit Python, rebuild the image, redeploy.

Doc 02 §2 lists "AI-assisted rule editing → mutate validated data, not generate code" as a
thing structure-as-data buys, so the *editing* mechanism exists; the *deployment* mechanism
does not.

Two sections of the same document contradict each other and neither acknowledges the other.

**There is a way out, and it is already prototyped in this repo.**
`experimentation/jittree/test.py` benchmarks four strategies and its Approach B is exactly the
missing tier: flatten the rule structure to arrays and run **one generic njit walker that
compiles exactly once, ever — new configs are just new arrays, no recompilation.**
`experimentation/graph_poc/prototype.py` states the same design question in its docstring.
Neither is mentioned in the doc set, which considers only codegen.

That gives a three-class model instead of the current binary:

| class | changes | cost |
|---|---|---|
| **values** | params, table contents | free, no recompile |
| **shape** | rules added/removed within a data-driven module (tree, rule table, scorecard) | free, if that module is table-driven rather than codegen'd |
| **code** | new step logic, new combinator wiring | rebuild |

The middle class is what the business-user persona needs, and what the "certain modules
configurable with config" goal wants. It costs some speed in exchange for zero recompile, and
the tradeoff should be per-module and explicit.

### 4.6 One output per step

Credit calculations are routinely multi-valued over shared work: an amortisation produces
instalment + total interest + fees from one rate/term solve; a scorecard segment produces
score + band + reason code. Under one-output-per-step the author writes N functions that each
redo the solve. A tuple return is free inside the kernel — a register unpack.

### 4.7 `|` is a sequence in one doc and a dependency-resolved graph in another

Doc 03 §3.2: *"Ordering lives in `|`"*, and §12: *"the waterfall order legible in one line."*
Doc 03 §8.1: *"`|` builds a **dependency-resolved graph**, not a sequential stage list."*

The reconciliation exists only as a table cell in doc 01 §5.8 (*"order matters only where a
value is overwritten"*). The undecided case is the one that determines the authoring model:
`A | B` where B produces a value A consumes and nothing is overwritten. Silently reordering
makes `|` cosmetic and falsifies "ordering is visible here, and only here." Erroring makes it
a sequence. Neither is stated where authors read.

### 4.8 The most common authoring error has no error message

The docs give first-class error design to config typos (difflib suggestions, exact step index,
`extra="forbid"`) and to null violations (column, count, example rows). The most frequent
mistake in a name-wired system gets the opposite treatment: writing `disposible_income`
silently becomes a demand for a new input column, and surfaces much later as a missing-column
error naming the wrong thing.

**Action:** wiring resolution should fail with a did-you-mean against in-scope value names and
frame columns — the same treatment the config path already gets.

### 4.9 Two other things nobody can build against yet

- **Doc 03 §5's own module-assembly example is a build error** under doc 03 §3.1, stated three
  paragraphs below it: `apply_income_cap` and `apply_sector_cap` both output `term_cap`.
  Doc 03 §11's canonical test calls the income-cap assertion on `Affordability`, which has no
  `min_net_salary` input and produces `term_cap` as an output; doc 04 §7 has the same test on
  the right module. Neither passes the `shared` bundle its step requires.
- **`corpus` is undefined.** It appears five times — the equivalence ladder's input, the
  golden baseline, E3, E4 — and nothing in eight documents says where it comes from. There is
  no permanent oracle by design, so fixture generation is load-bearing and unowned.

---

## 5. Cross-project reuse, versioning and distribution

This is one of the four stated forward goals and receives one sentence in doc 07's preamble,
one line in doc 02 §6's layout block, and one row of doc 04 §5.2's audit table.

- **A module has no version.** Doc 04 §5.2 wants "framework + module versions" in the audit
  record; nothing produces one. Identity is a bare string baked into the union tag
  (`Literal["credit:affordability"]`), config references it as `{"use": "credit:affordability"}`
  with no version field, and **the union design makes two versions structurally unable to
  coexist in one process.** Upgrade the library, change a step, and every stored config
  referencing that id now binds to different behaviour with no diff and no error.
- **No public contract, so no breaking-change detection.** `Module.schema()` returns "required
  inputs, produced outputs" — not enough. A contract needs external vs wired inputs, per-input
  null policy, params schema, required `shared` fields, and tap names. Those are the same
  fields §4.1 and §4.2 need, so this is one data-model decision, not a release-process task.
  The CLI (`contract snapshot` / `contract check`) and the semver rule can come later; the
  *data model* cannot.
- **`shared` is an unnamespaced, un-overridable global a library imposes on its consumers.**
  Doc 03 §4.1 rejects flat params because *"a flat merge would force global uniqueness,
  destroying reusability"* — and `shared` then reintroduces exactly that, binding on bare field
  names in a model the *consuming* pipeline owns. Two library modules both wanting `rate` have
  no resolution path. Keep "one bundle at runtime, passed by reference" — E5 measured that as
  free — and make only the *binding* declared and renameable, sharing §4.1's mechanism.
  Separately: `shared` has two documented homes (`config/shared.json` per doc 07 §1 vs inline
  in the pipeline config per doc 03 §10) with no merge rule, and as a pipeline-level model it
  **can never be a union member**, so it sits outside the config-validation machinery entirely.
- **Discovery and finalisation are unowned.** Doc 02 §2.2 states a non-optional constraint —
  *"all registration must complete before the first config is validated"* — and names silent
  mis-validation as the failure mode. No document says how decider2 discovers what to register
  across several installed distributions, who calls `finalise()`, or whether a late
  registration raises. Multi-project reuse is precisely the case that reproduces the bug.
- **"Explicit and complete" config makes every library param addition a simultaneous hard
  break** in every consuming project, with no migration tool. Doc 07 §2 promises "a tool that
  materialises one from the other (§5)"; §5 only offers `export` from a *Python* pipeline,
  which a config-driven project no longer has. Needed: `decider config fill`, writing new
  params at their schema default and leaving existing values untouched, so an upgrade lands as
  a reviewable diff.
- **`name=` is doing three jobs** — params namespace key, tap qualifier (`term_cap@sector_cap`)
  and audit identity — and no document says how `module(..., name="affordability")` acquires
  the id `credit:affordability`, who owns the `credit:` prefix, or whether instance names must
  be unique. Two imported scorecards both defaulting to `name="scoring"` collide on day one.
  Doc 05 §4.2 also makes ids load-bearing for *codegen determinism*, while doc 02 §2.2 says the
  Python path "never touches any of this machinery" — so a Python-composed module plausibly has
  no id at all.

---

## 6. Config, UI, and where the seam goes

This section answers the question raised during review: *should config management be in the
framework at all?*

**Short answer: the instinct is right, but "config" is the wrong unit to draw the line on.**
There are two things called config in this doc set and they fall on opposite sides of the only
line that matters.

### 6.1 The two documents

| | structure document | params document |
|---|---|---|
| holds | `use`/`type`, `steps`, `arms`, `modifies` — composition | values behind fixed types |
| changing it | **changes the machine code** | cannot change the machine code |
| lifetime | **build input** — like a Dockerfile | runtime input |
| who fetches it | CI, once | the user's code, whenever |

Doc 03 §10 currently says *"Config supplies **params and composition**, never structure"* —
but doc 04 §2 defines structure as "which steps exist and how they wire", and composition *is*
wiring. So **one document currently spans the governance boundary the whole design rests on**,
and doc 07 §5 explicitly merges them into one production artefact while doc 04 §2.1 rests its
entire enforcement argument on them being separable files.

Split them and most of the difficulty dissolves. A params document then **cannot trigger a
recompile at all**, because types come from `params.py`, which is code. The runtime config
surface shrinks to validation plus a swap.

### 6.2 The principle

> A config document belongs to the framework's *runtime* surface only if changing it cannot
> change the machine code. Anything that can change the machine code is a build input. Where
> either document comes from is the user's, always.

### 6.3 What the framework keeps — and it is small

Only what it alone can know, all of it pure functions over data it was handed:

1. what a params bundle is (pydantic model → fixed-type NamedTuple)
2. validation: `extra="forbid"`, exact step index, difflib suggestions, declared bounds
3. namespacing per module instance; `shared` routed by reference
4. **rejecting composition keys (`use`, `type`, `arms`, `steps`) in a params document** — this
   is what would make doc 04 §2's safety claim true by construction rather than false
5. per-field provenance via pydantic's `model_fields_set`, which makes completeness *computable*
   from a plain dict (so the framework never needs to own the bytes to enforce doc 07 §2)
6. **JSON Schema export of the params surface** — the UI contract
7. `structure_fingerprint`, from the canonicalisation doc 05 §4.2 already mandates for the
   numba cache
8. one atomic cell for the live params, read **exactly once per invocation**
9. storing the caller's `origin` token verbatim, and refusing to run without one

```python
# build time — CI and the Dockerfile only
pipeline = decider2.from_dict(structure_document)
pipeline.structure_fingerprint() -> str
compiled = pipeline.build(require_cached=True)   # cache miss RAISES, never a 92 s surprise

# runtime — no json/os/pathlib/socket import anywhere below this line
pipeline.params_schema(flat=False) -> dict        # the UI contract
pipeline.export_params() -> dict                  # complete, from defaults
pipeline.resolve_params(document, *, origin: str, complete=True) -> Params
decider2.diff(old: Params, new: Params) -> list[Change]

class ParamsCell:                                 # the only mutation verb
    def get(self) -> Params: ...
    def swap(self, new: Params) -> Params: ...    # returns the previous — rollback is free
```

Deliberately absent: no `ConfigSource` protocol, no `ConfigManager`, no version type or
ordering, no polling, no `decider2/config/` package at all (rename doc 02 §6's `config/` to
`binding/` — it is module-type registration machinery, not config, and a directory called
`config/` accreted 553 lines once already).

A one-method Protocol is ceremony. The user's loader is a function returning a dict. Adding a
Protocol later is non-breaking; removing one is not.

### 6.4 The thing that genuinely cannot be delegated

Binding a params document to the specific compiled artefact that will consume it. Three
reasons:

1. **Only the compiler knows the types.** Safety is a question about the NamedTuple types baked
   into the generated driver and the numba cache keyed on its source.
2. **Every failure here is silent** — a mismatch produces a cache miss and a compile in the
   request path, or a stale kernel running new constants. There is no exception for a user's
   loader to catch.
3. **It races the framework's own in-flight calls.** A swap landing mid-`apply` must not split
   one record across two bundles.

The current implementation demonstrates exactly this failure mode: `BaseConfig.reload()`
(`decider/config/base.py:105-116`) is lazy and per-object, so two config objects touched at
different moments in one batch can be on different versions — half a run decided under one
affordability floor, half under another, with nothing recording where the boundary fell.

### 6.5 What the doc set does not know about `decider/config/`

Worth flagging because it changes what "removing config from the framework" means.
`decider/config/` is **553 lines already implementing much of the stated goal**, and no
decider2 document mentions it:

- `CoreConfigManager` with four abstract storage primitives (`_load_version`, `_write_version`,
  `_version_exists`, `_latest_version`) — that *is* a backend seam
- `register_config_manager` (`decider/config/_ext.py`) — backends registered through the same
  discriminated-union mechanism as modules
- semver create/save/pull, a stale-version guard, `subscribe_version_updates`
- `RequestHandler.module_fn` (`decider/serving/handler.py:32-41`) rebuilding the whole module
  graph when the polled version changes — i.e. **runtime structure hot-swap, in production use**

So decider2 as documented is a *regression* against the current implementation on the author's
stated goal, not a gap. Two things to carry forward from that code as cautionary evidence:

- `subscribe_version_updates` polls every 10 s and hot-swaps credit-granting logic inside
  `except Exception: pass` — a silent failure surface on the path that changes decisions
- `pull_version` has a **live bug** at `decider/config/core.py:131`: `target` is either a `str`
  (the parameter type) or a `Version`; neither has `.version`, so any non-forced pull with a
  config already loaded raises `AttributeError`. It has not surfaced because
  `subscribe_version_updates` calls it with `force=True`.

The reference project shows the conflation at file level:
`projects/loan_scoring/configs/0.0.0/main.json` is
`{"dti_weight": 200.0, "utilization_weight": 100.0, "score_base": 800.0, "type": "credit_scorer", "name": "credit_scorer"}`
— params and structure in one object.

### 6.6 What a UI needs that is not there yet

- **A param path address grammar.** Doc 03 §4.1 says a flat view over a namespaced model is
  fine "since the reviewable artefact is generated anyway" — true for *browsing*, unstated for
  *editing*. Two modules with a field named `cap` need an unambiguous write-back address.
- **Metadata beyond type/bounds/description**: label, unit (months vs ratio vs currency),
  grouping, ordering, choices, conditionality, approval level. All of it fits in pydantic's
  `json_schema_extra`, so this is additive.
- **A four-level provenance chain nobody has named**: `Field()` default → `.bind()` at
  composition → config file → per-invocation payload. Doc 03 §4.2 rejects a *two*-level chain
  for shared params on audit grounds — *"makes 'which value actually applied' ambiguous"* —
  and then builds a four-level one without acknowledging it. A UI must show which layer won.
  `.bind()` in particular is a code default in every sense doc 07 §2 means, and it is unstated
  whether `export` materialises bound values (in which case `.bind()` doesn't hide them) or
  omits them (in which case the config is not complete).
- **Cross-module constraints** (`income_cap.cap <= sector_cap.cap`) have nowhere to live, given
  the encapsulation rule.
- **Tables are the artefact a business user most wants to edit** and are the least designed
  part — doc 03 §4 calls them provisional, O4 leaves build-vs-invocation open, and they appear
  in neither doc 04 §2's governance table (which has exactly two rows) nor doc 04 §5.2's audit
  record nor doc 07's project layout. The whole lifecycle apparatus is built on a binary that
  tables do not fit.
- **A `Branch` condition is a bare step with no module**, so no params namespace, no id, no
  registration path — yet doc 07 §4's JSON addresses it as `"credit:is_private_sector"`, a
  registered id. The branch cutoff is the single most tunable value in a credit pipeline.
- **Frame ops take no `name=` and no `params=`**, so every constant inside a `Filter` or
  `Aggregate` is a literal in a Python file — which under doc 04 §2's own definition is
  structure. The 546-literals lesson was applied to the record tier only.

---

## 7. Compile and boundary (doc 05) — what an implementer cannot build from

- **Per-node fallback cannot exist inside an njit driver.** A nopython driver cannot call a
  Python function. The only in-kernel escape is `objmode`, which no document mentions and which
  would defeat the row loop anyway. So doc 05 §6's layer 1 is meaningful only when the driver
  is *already* Python; inside the fused path one un-njit-able node fails layer 2 and the whole
  group drops to a Python loop. The README's "degrades to plain Python per-node rather than
  failing" overstates the blast radius by a lot.
- **Fallback is silent, lazy and unauditable.** It is decided at first call for given argument
  types — in a design whose invariant is that a runtime load triggers zero compilations. A
  doomed compile *is* a compile attempt at runtime. Nothing reports the fallback set; nothing
  can forbid it at build; it appears in no audit record; and a fallen-back node changes
  numerics. Add a rung to doc 05 §9: *the fallback set is reported, and empty in production
  builds.*
- **The equivalence ladder agrees by construction exactly where compilation didn't happen.**
  When node *k* falls back, `fused` executes the same Python object `interpreted` does, so
  agreement on that node is tautological — and the fallback set is type-dependent, so CI can
  certify a different program than production runs.
- **`≡` is never defined**, and doc 05 §9.1 says "matching a numpy reference exactly" while
  §5.2 enables `fastmath` selectively and doc 02 §3.1 records observed 1-ULP drift. Exact
  equality is also the wrong test for float outputs and the *right* test for decisions —
  the tolerance policy needs to be per-output-kind.
- **`decider build` takes no input schema**, yet column dtypes and nullability determine the
  record dtype, the numba signature, the `float64`-vs-`Optional(float64)` choice per step, and
  the calling convention (the ~64-column threshold). There is no declaration site for the input
  frame's schema anywhere in eight documents. This is also O11, filed as a lineage question
  when it is a *compilation* prerequisite.
- **"Byte-identical" is necessary but not sufficient.** Numba's file-backed cache stamps
  entries with `(st_mtime, st_size)` of the generated `.py` and indexes by path. Regenerating a
  byte-identical file at startup changes mtime and misses 100% of entries. Also unstated: where
  the generated files and cache live in a read-only container.
- **A topological sort is not unique and the tie-break rule is never stated** — yet the chosen
  order *is* the generated source, which *is* the cache key. A cosmetic reorder of
  `module(...)`'s arguments (which doc 03 §3.2 explicitly says must not change behaviour) would
  silently invalidate the entire cache.
- **A runtime error inside a kernel cannot name the row.** In nopython mode `raise` takes only
  compile-time-constant arguments, so `raise ValueError(f"row {i}")` does not compile. Under
  `prange` an exception aborts the parallel region with no defined iteration and the output
  arrays already hold partial results.
- **The kernel-to-kernel boundary is unspecified.** Doc 05 §7 makes the compiler split a
  pipeline into several groups; §3 specifies only the *polars* boundary. Whether intermediates
  between two kernels round-trip through polars or stay in numpy differs by roughly 200×.
- **Frame ops inside a `Branch` arm or `Loop` body are type-legal and uncompilable** under the
  "one type — Module" claim, and no legality rule is stated.
- **"Reason codes need no new machinery" is false at realistic scale.** A decline taxonomy is
  N rules, any number of which fire, producing an *ordered variable-length list*, usually
  reported top-k. A step gives one scalar of fixed type. That is O5 (ragged collections), which
  doc 06 defers. The doc notes a legacy taxonomy was entirely dropped in a previous port —
  this is the mechanism by which it would be dropped again.
- **Taps are "effectively free" under the calling convention E9 replaced.** The +0.11 ns/row
  figure is E5's, measured on the in-kernel store with positional outputs. E9 changed the
  output side to a record array and found strided write-back is now 54.7% of total time. Every
  tap adds a field to that array.
- **The `required` null policy is all-or-nothing** — one bad row raises for the whole batch.
  No row-level rejection, quarantine or route-to-manual-review, which every production batch
  needs.
- **Determinism is specified for codegen, never for results.** Nothing says whether the same
  inputs and config produce bit-identical outputs on a different machine — and the CPU-target
  tradeoff, selective `fastmath`, and the runtime-measured serial/`prange` choice all argue
  they do not. The audit record cannot answer "reproduce this decision."
- **`Join(bureau, …)` captures a live Python object** in the canonical pipeline — no id, no
  declared schema, no serialised form, no config address — so doc 03 §8.1's example has no JSON
  form at all, while doc 07 §4 asserts JSON can compose modules. `export`'s only stated failure
  mode is an inline module.

---

## 8. What to protect

Eleven reviewers went looking for problems; these are the things that should survive any fix.

1. **The evidence discipline.** Measuring before writing code, recording negative results in
   place, and in O10 reversing a decision *against its own fastest measurement* with the
   reasoning written down ("The data did not change; the weighting did"). The propagation bugs
   in §2 are a reason to grep harder, not a reason to be less willing to record a reversal.
2. **Decoupling the authoring unit from the compilation unit** (doc 02 §1.1). The best
   structural idea in the set. It survived its own justification being falsified. Every finding
   about fusion caps is about the *number*, not this separation.
3. **Structure-as-data** — a pydantic instance, never a generated class, plus "the union is the
   index". It is the precondition for render, diff, export, config UIs and AI-assisted editing.
   If implementation pressure pushes back toward generated classes, everything downstream
   becomes impossible.
4. **Real `.py` files for generated drivers, never `exec`** (doc 05 §4.1). 29.5 s cold → 0.80 s
   warm. Both prototypes in this repo violate it, which is exactly why it is written down.
5. **The middle rung of the equivalence ladder.** `stepped` looks redundant and is not.
   E3's explicit falsification criterion is good science; keep it as written.
6. **Null policy declared in the signature.** The concept is right — how missing data is
   handled is a property of the input. Fix the `missing_as()` default-argument syntax, not the
   idea.
7. **Taps as declared data with version qualification by producing module name, not position**
   (O13). The reasoning generalises directly to input-side version addressing (§4.1).
8. **Refusing ambient `ContextVar` supply**, for the stated reasons. It also rules out the
   `reload()` straddle bug in the current implementation (§6.4).
9. **`.bind()`'s shape** — it distinguishes "frozen by an engineer" from "tunable by a business
   user" without becoming a compile-time constant. Any adaptation mechanism added for reuse
   (§4.1) should follow the same rule: declared data, runtime values.

---

## 9. Suggested sequencing

**Before writing any code** (days, not weeks):

1. Resolve §1 — say which tree the citations refer to, and re-derive or drop what cannot be
   checked. Nothing else should be specified on top of unverifiable evidence.
2. Apply §2 (stale claims) and §3 (the control-flow over-claim). Pure doc edits.
3. Decide §6.1 — split the structure document from the params document. This is the cheapest
   decision in the list and it unblocks the config, UI, versioning and recompilation goals at
   once.
4. Write the §4.4 admissible-dtype contract and the input-frame schema declaration site.
   Doc 05 cannot be implemented without them.

**Then, and these are design work, not doc edits:**

5. The module interface (§4.2) + boundary relabel (§4.1) + public contract (§5) — one data-model
   decision serving three goals. This is the reuse blocker and it is cheapest now, before
   anything serialises.
6. Promote O2 (in-place param promotion) to scheduled (§4.3). Authoring cost is now the only
   defence against repeating the 546-literal outcome.
7. Add the third compilation class (§4.5) — a table-driven generic kernel for rule tables,
   trees and scorecards, so "shape" changes are free. `experimentation/jittree/test.py`
   Approach B already measures it; promote it from prototype to design.
8. Start E4 (the reviewable artefact) now. It is people-blocked, it is the top risk, and two
   further problems compound it: the review view is populated from `description`, which only
   exists if the author used `@step` — and doc 03 §1 recommends omitting the decorator; and
   a rule carries no key to join to a policy clause on, so verification is a join where neither
   side has a key.

**The first implementation milestone in doc 06 is right** — one realistic waterfall, declared
once, compiled three ways, batch and single record, param retune proving no recompile, a tap
proving cheap diagnostics. Add to it: a second project reusing the same module under a
different column vocabulary. That is the smallest thing that would falsify §4.1, and §4.1 is
the finding most expensive to fix later.
