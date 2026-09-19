# 02 — Affordability assessment: an ideal-world sketch

An answer to [`02-affordability-assessment.md`](../../02-affordability-assessment.md),
written as the shape it would take in a codebase if the authoring surface could
be anything. Nothing here runs. Bodies are `pass` with a one-line comment; the
value is in the signatures, the composition expressions, the table files and the
config documents.

Doc 03 is treated as a proposal and departed from in ten places. Every departure
is numbered in [`FRAMEWORK-DEMANDS.md`](FRAMEWORK-DEMANDS.md) with the spec
section that forced it. Read that file second and this one first.

---

## 1. The shape, in one page

```
02-affordability-assessment/
  pipelines/
    affordability.py          THE pipeline. 127 lines. Seven stages, one cut,
                              four declared properties.
    consumers.py              Five consumers, five call sites, no composition.
  policy/
    modes.py                  The four assessment modes, as profiles.
    overlays.py               The overlay operator registry and the
                              conservative-only proof.
    ladder.py                 The evidence ladder's ten sections — the artefact
                              Compliance signs.
  modules/
    framing/                  Stage 1  household, dependants, scope
    income/     waterfall.py  Stage 2  the six-tier evidence contest
                haircuts.py            the 36-cell matrix and its modifiers
                variable_pay.py        month counting, where it goes wrong
    deductions/ tax.py        Stage 3  PAYE, effective-dated
                statutory.py           UIF per employer, retirement, court orders
    expenses/   norms.py      Stage 4  the statutory and internal norm tables
                bases.py               four bases, one contest
    obligations/ treatments.py Stage 5 the 45-row behaviour matrix
                 assemble.py            two lists into one, with dedup
                 annotate.py            one `over()`, scalar and annotation
    capacity/                 Stage 6  the ladder, the buffer, the overlay targets
    verdict/                  Stage 7  the Verdict domain, and five narrowings
  tables/
    statutory/                Compliance and Credit Systems. Dated directories.
    policy/                   Credit Risk Policy. Dated directories.
  config/affordability/
    policy.json               what Credit Risk Policy may set — and only that
    product/10-flex-loan.json what a product team may set — six fields
    overlays/register.json    what the Credit Committee approves
  contracts/affordability.json
  schemas/affordability_input.json
  tests/
    test_properties.py        the twelve acceptance criteria, as properties
    compliance/expense_norms_cases.csv   a spreadsheet that runs as a test
  vocabulary.py               names and — the collision — units
  sample-evidence-ladder.md   what the ombud adjudicator actually reads
```

### Why this layout

Doc 07 §1 gives `modules/`, `pipelines/`, `config/`, `contracts/`, `schemas/`,
`tests/`. Three things are added and one is deliberately absent.

**`policy/` is added**, holding the four modes, the overlay operators and the
ladder sections. These are not modules — they compute nothing — and they are not
config — they are code, reviewed in git. They are the *governance surface*: the
three artefacts a non-engineer signs off. Putting them in `modules/` would bury
them; putting them in `config/` would make them editable by whoever can write a
JSON file. They get their own directory because three different named
committees own the three files, and a directory is the cheapest way to make that
visible.

**`tables/` is added and is split by owner, not by subject.** `tables/statutory/`
and `tables/policy/` are the ownership boundary as a filesystem path, which is
what CODEOWNERS attaches to. A subject-based split (`tables/income/`,
`tables/expenses/`) reads better and puts the PAYE table and the haircut matrix
in adjacent directories with different owners and different approval paths,
which is how a quarterly policy retune eventually lands a change in a statutory
file.

**Each table version is a separate dated file**, never a `valid_from` column in
one file. A column means every historical version is rewritable by a single
edit and a replay silently changes; a file means a version is immutable, content
addressed, and a mutation is detectable (`test_pinned_and_dated_resolvers_agree`).

**Absent: a `modes/` directory, and any file named after a consumer.** The
temptation is `modules/affordability_arrangement/` and it is the whole failure
mode the spec names. There is exactly one `Assessment` object in this tree.

---

## 2. Walkthrough: one applicant, inputs to verdict

Read [`sample-evidence-ladder.md`](sample-evidence-ladder.md) alongside this.

**Stage 1, framing** (`modules/framing/`). A joint application becomes one
household: `household_size`, `dependants_count` as the higher of two
declarations with the discrepancy recorded, and `principal_debtor_set` — the
identities whose accounts are in scope. Note what framing does *not* do: it does
not touch the account list. It cannot, because it sits upstream of the evidence
cut (§5), and the framework checks that from static lineage at build time. So
"an account on which either applicant is a principal debtor counts once" is
implemented as an identity set here and a dedup in stage 5.

**Stage 2, income** (`modules/income/`). Two levels of `over()`: 0..6 sources per
applicant, 0..12 component months per source. Per source, six tier candidates
compete in a `contest(select=STRONGEST)`; the winner binds and **the four losers
survive**, because an adjudicator's first question is what else you had and why
you did not use it. The haircut comes from a `behaviour_table` whose `X` cells
are a `NOT_PERMITTED` behaviour rather than a sentinel float, so an impermissible
combination *removes the candidate from the contest* rather than scoring it at
100%. Three additive modifiers, capped at 55%.

**Stage 3, deductions** (`modules/deductions/`). PAYE from a `dated_table`
resolved on `decision_date`; the rebate class from age *at* `decision_date`;
unemployment insurance as a reduction **over the employer list**, because the
ceiling is per employer and a single multiplication against household gross is
wrong by up to one ceiling. Retirement observed or imputed, and which one is an
emitted value. Court orders summed, and their credit-agreement match keys
emitted for stage 5 to consume.

**Stage 4, expenses** (`modules/expenses/`). Four bases computed, `contest(HIGHEST)`
with a declared tie-break to statutory. `living_expenses_cents` is declared an
`overlay_target` with `floor="statutory_norm_cents"` — a floor that is a
*computed value*, not a constant, which is how "an overlay may never reduce it
below basis C" becomes a definition-time check.

**The cut.** Everything above is the expensive part and does not vary across
project 06's scenarios. `cut("evidence", carries=[...], resumes=[...])` declares
the boundary once.

**Stage 5, obligations** (`modules/obligations/`). Two lists into one via a
record-tier `over(..., group=dedup_key, collapse=...)`. Then a single `over()`
across 0..80 accounts producing the per-account annotation, with every scalar
defined as a reduction over a *column of that annotation*. The treatment comes
from a 45-row `behaviour_table` whose cell selects one of ten registered
behaviours plus its coefficients.

**Stage 6, capacity** (`modules/capacity/`). The prescribed ladder as one step
with one `@rung`, then `contest(LOWEST)` over the proportional buffer and the
residual floor. `max_affordable_instalment_unadjusted_cents` is the result; the
overlay stack then produces `max_affordable_instalment_cents`, and both survive.

**Stage 7, verdict** (`modules/verdict/`). `Verdict` is a closed domain with no
boolean context. Five declared narrowings, each with a required
`treat_indeterminate_as=`.

---

## 3. The hard parts, and how each is expressed

### 3.1 One calculation, five consumers, four modes

This is the sketch's central problem and it gets the sketch's central invention:
**a mode is a `profile`, not a `Branch`.**

```python
ARRANGEMENT = profile(
    "arrangement", code=3,
    switches={"income.on_stale_evidence": StaleEvidence.PERMIT_WEAKER_TIER,
              "verdict.tolerance_basis": ToleranceBasis.SUSTAINABILITY},
    params={"capacity.buffer_grid_name": "arrangement"},
    emits=["*always", "discretionary_income_after_cents"],
)
```

A profile binds three surfaces and nothing else: params, `policy()` switches, and
which outputs are materialised. It is validated against the pipeline at build
time, so **there is no spelling of a mode that introduces a step, removes a step,
or rewires two steps.** That is the structural enforcement spec question 2 asks
for, as against the documentation of it.

The part that makes it more than a naming convention is that a switch is a
*closed enum selecting an already-compiled arm*. All four modes' behaviours are
in the same kernel; a mode supplies a compile-time immediate. Which gives a test:

```python
assert len({kernel_fingerprint(Assessment.under(m)) for m in ALL_MODES}) == 1
```

Four modes, one machine code. A mode that needed different machine code fails
here, and — per the spec — "would be a different calculation and would have to be
justified as such". Acceptance criterion 2, "a change to the tax calculation
reaches all four at once, demonstrably", is discharged by that assertion rather
than by a code review.

Doc 03 would reach for `Branch(mode_index, [NewApp, LimitIncrease, Arrangement,
Scenario], modifies=[...])`. Four arms over seven stages is four near-copies with
a shared entry point, and doc 03 §8.2's rule that "every arm must produce every
declared `modifies` value" makes the `modifies` list the union of everything all
four modes touch, which is most of the pipeline. It would work and it would drift.

The five *consumers* are a thinner problem than the four modes, and the answer is
mostly subtractive: `pipelines/consumers.py` is five one-line expressions and
imports nothing from `modules/`. The consumer-facing config surface is **six
fields** (`config/affordability/product/10-flex-loan.json`). Spec question 1 asks
how one calculation serves five consumers "without either a fork or a parameter
surface so wide that every consumer configures a different calculation"; six
fields is the answer, and it is enforced by `--owner=` filtering rather than by
discipline.

### 3.2 Three shapes of answer

Shapes (a) and (b) differ only in whether `proposed_instalment_cents` is
supplied. That is one nullable input, and the null is load-bearing: with no
instalment the verdict is `PASS` or `INDETERMINATE` only and `MARGINAL` is
unreachable — asserted, not assumed. The `LIMIT_INCREASE` profile carries
`forbids=["proposed_instalment_cents"]`, so shape (b)'s caller supplying one is
an error rather than a silently ignored argument.

Shape (c) — the largest *amount* — is not here, and the boundary is expressed as
three declared properties on the pipeline rather than as a docstring:

```python
Assessment = monotone_in(Assessment, "proposed_instalment_cents",
                         direction=NON_INCREASING, of="affordability_verdict_code",
                         edges_from=["statutory_expense_norms", "buffer_grid",
                                     "residual_floor", "verdict_tolerance_pct"])
Assessment = Assessment.assert_cut_equivalent(evidence)
Assessment = Assessment.assert_materialisation_neutral("accounts_annotated")
```

Those three lines *are* the contract with project 03. They appear in
`contracts/affordability.json` under `declared_properties`, so a consumer can
read what it is entitled to rely on without reading this project's code.

### 3.3 Monotonicity, and where the corpus comes from

Spec acceptance 3: monotonicity "is tested across band edges, buffer boundaries
and the residual floor — not asserted". The interesting word is *where*. Doc 03
§11 offers `assert_modes_agree(pipeline, corpus)` and never says where a corpus
comes from; doc 03 §1.2 then says the one concrete thing known about a corpus is
that it must contain boundary values, because the int64 overflow it records "was
found by binary search, not by sampling".

`edges_from=` closes that. Every `dated_table` declares `declares_edges=(...)`,
so the framework can enumerate 72 statutory band floors, 96 buffer grid cells, 6
residual floor cells and the tolerance band, cross them with the declared param
bounds, and generate a corpus at every edge and one cent either side of it. The
non-monotone step the spec warns about — the residual floor and the buffer
interacting — lives exactly at a cell boundary, which is a measure-zero set under
random draws and an everyday occurrence in the applicant population.

### 3.4 A ragged list, a scalar and an annotation that cannot disagree

The invention is `over()`, and the property is that **the scalar is defined as a
reduction over the annotation**:

```python
Accounts = over(
    "accounts",
    steps=[obligation_cents, obligation_basis, exclusion_reason, ...],
    aggregate={
        "obligations_from_accounts_cents": SUM("obligation_cents"),
        "obligations_internal_cents":      SUM("obligation_cents", where="is_internal"),
        "worst_arrears_months":            MAX("account_arrears_months"),
    },
    annotate=["treatment_code", "obligation_cents", "obligation_basis_code",
              "dedup_source_won", "exclusion_reason_code", ...],
    max_elements=80,
    materialise_by_default=False,
)
```

There is no expression anywhere in this tree that computes
`existing_obligations_cents` from anything other than the `obligation_cents`
column. It is therefore not merely *true* that the two agree — there is no
arrangement of this code in which they could differ, which is what acceptance
criterion 10 asks for.

The caller's choice is whether the annotation is **materialised**, not whether it
is computed. Project 06 orders its search on it; projects 03 and 07 would be
handed 14 million × 80 rows they never wanted. `assert_materialisation_neutral`
is the test that asking for it does not change any scalar.

This has a cost and §3.9 below and demand #7 state it honestly.

Spec change scenario 6 — project 06 wants the three most expensive obligations,
which nobody else wants — is then a *consumer-declared reduction over a published
annotation*:

```python
top3 = Obligations.reduce("accounts_annotated", TOP_N("obligation_cents", n=3))
```

Not a new output on the shared module. That seam is what stops "one more output
for one more consumer" widening the interface once a quarter.

### 3.5 A matrix whose cells select behaviour plus coefficients

45 account types, ten behaviours, five coefficient columns, edited quarterly by
non-engineers. Spec question 5 asks what this is structurally, and the answer is
that doc 08 §3.4's two kinds do not cover it: every row applies the same *shape*
of rule so it is not a `ruleset`, but the behaviour column is a discriminant over
code rather than a value, so it is not a `decision_table` either.

```python
TREATMENTS = behaviour_table(
    "obligation_treatment",
    key=("account_type_code",),
    behaviours={"STATED": stated, "PCT_LIMIT": pct_limit, "GREATER_OF": greater_of,
                "MIN_PAYMENT": min_payment, "CONTINGENT": contingent,
                "TERM_AWARE": term_aware, "REFER": refer, ...},
    coefficients=("rate", "floor_cents", "contingent_pct", "min_pct", "min_floor_cents"),
    owner=policy,
    versions="tables/policy/obligation_treatment/",
    unknown_key="REFER",
    change_class={"behaviour": "interior", "*": "value"},
)
```

`unknown_key="REFER"` is spec change scenario 4 in one line: a new account type
appears at the bureau and produces `indeterminate` until it has a treatment,
rather than silently scoring zero. It is the *default*, not something an author
remembers.

`behaviour_table` is used three times in this tree, at 14, 36 and 45 rows — the
expense category classification (`SHARED`/`PERSONAL`, which is a *combining
function* in a cell), the haircut matrix (`PCT`/`NOT_PERMITTED`, where the `X`
cells are a behaviour rather than a sentinel), and the treatment matrix. Three
different subjects, one shape; that is what makes it a component and not a
special case.

The compile behaviour is worth stating because it is not what one would guess.
All ten behaviours are compiled into a switch, so moving a cell from `STATED` to
`GREATER_OF` is **free** — a data change, no recompile. Adding an eleventh
behaviour is a code change. But spec §5.5.2 says behaviour changes "are approved
differently", and they are: `change_class` is declared **per column**, so the
behaviour column routes through review and the coefficient columns do not. The
compile cost and the approval class split at different places, which is demand #5.

### 3.6 Four hundred calls, evidence computed once

```python
held = Scenario.hold(application, upto=evidence, params=params)
for candidate in candidate_sets:
    yield held.resume(accounts=candidate.remaining_accounts,
                      settlement_quotes=candidate.quotes,
                      proposed_instalment_cents=candidate.proposed_instalment_cents,
                      materialise=["accounts_annotated"])
```

Spec question 3's real demand is the second clause: "without the separation
becoming a second entry point that can drift from the first". Three things
deliver it.

1. **One declaration.** `cut()` is declared in the pipeline; `hold()` and
   `resume()` are *derived* from it. There is no second composition to maintain.
2. **The same kernels.** `resume` enters at the cut in the compiled record
   layout. `held` is not a cached dict of Python values.
3. **A build-time soundness check.** The framework verifies from static lineage
   that no step upstream of the cut reads anything in `resumes`. If one did, a
   held prefix would be stale across all 400 scenarios and each would be
   plausibly wrong. That check is why framing derives `principal_debtor_set` from
   applicant identity rather than from the account list, and the build error
   names the step and the offending read.

Plus a fourth rung on doc 02 §3.1's equivalence ladder:
`interpreted ≡ stepped ≡ fused ≡ held-and-resumed`.

### 3.7 Effective-dated statutory tables over a seven-year replay window

The design goal is that **there is no spelling of the wrong thing**.
`dated_table(...).asof` is the only accessor. `.latest` and `.today` do not exist.
`decision_date` is a reserved, frame-only pipeline input that no step may produce,
so nothing can overwrite it mid-flow. Omitting it from the input schema fails the
build with all nine dated artefacts named and their source lines.

Two further properties, both of which doc 03 leaves to the author:

- **A dated-table read emits its resolved version automatically.** Acceptance
  criterion 12 — every table version resolved is recorded on every assessment —
  is not something a `taps=` list can be missing. `taps=` in this tree never
  mentions a version.
- **Replay pins the resolver, not the call site.** `replay(snapshot,
  resolver="pinned", assert_resolver_agreement=True)` runs the same pipeline with
  the persisted version ids, *and* asserts that resolving `decision_date` produces
  exactly those ids. A disagreement means a version file was edited in place,
  which is otherwise undetectable and is the one failure effective dating cannot
  defend itself against.

Spec change scenario 1 — a new norms table effective next month, with both live
simultaneously during the transition — needs no mechanism at all: two files in
`tables/statutory/expense_norms/`, and rows C11–C13 of
`tests/compliance/expense_norms_cases.csv` are the proof.

### 3.8 Conservative-only, enforced at definition time

Spec §5.6.2(1) and acceptance 8 are unusually precise: an overlay that would
increase capacity is invalid **when defined**, not rejected when run, and not
bypassable by a negative magnitude. The design carries the asymmetry in three
places, all checked when `register.json` is validated — months before any
applicant meets the overlay, at the same moment the Credit Committee approves it.

1. **The operator declares its direction, and no operator has a free-signed
   magnitude.** There is `scale_down` (magnitude `gt=0, le=1`) and `scale_up`
   (`ge=1`). An author who writes `scale_down(magnitude=1.4)` gets a *bounds*
   error on the field they typed — a better error than a policy error on a
   composed effect.
2. **The target declares which direction tightens it.**
   `tightens_when(DECREASES)` on capacity, `tightens_when(INCREASES)` on the
   buffer, the residual floor and expenses. Composing (1) with (2) gives
   admissibility, and the rejection names the admissible alternatives.
3. **Targets may declare a computed floor.** `living_expenses_cents` carries
   `floor="statutory_norm_cents"`, and only operators annotated
   `preserves_floor=True` may name it.

`config/affordability/overlays/register.json` carries a `_rejected` block with
both real error messages: the direction rejection, and the negative-magnitude
attempt, which fails as a bounds error and never reaches the direction rule at
all.

Running with the stack disabled (spec §5.6.2(5), acceptance 9) is not a mode and
not a flag. It is an empty register. Every overlay target's `base` is computed
unconditionally and emitted always, so a disabled run and a live run differ only
in which of two already-present numbers the caller reads.

### 3.9 `indeterminate` kept distinct from `fail`

Spec question 12 names the problem exactly: the cheapest implementation of every
layer is a boolean. So the cheap implementation is made unavailable.

`Verdict` is a **declared domain**, not an int8 with a comment:

- it has no boolean context — `if verdict:` is a lint and codegen error;
- it refuses `!=` against a single member, because `verdict != PASS` folds three
  outcomes into one and the error says which three;
- narrowing to anything smaller must go through `narrow()`, whose
  `treat_indeterminate_as=` argument is **required and has no default**.

The five narrowings live together in `modules/verdict/__init__.py`, and their
answers genuinely differ — `REFER` for granting, `CONDITIONAL` for limit
increase, `ABORT_SEARCH` for project 06 (an indeterminate scenario makes the
whole search unsound, which is not the same as an inadmissible one),
`MANUAL_REVIEW` for arrangement, and `False` for business surety, which is the
only consumer for which the conservative answer *is* `False` — and it had to
write that down to get it.

`evidence_sufficiency_code` is a second closed domain with nine members and zero
meaning "not indeterminate". `tests/test_properties.py` parametrises over every
non-zero member, so a code added without a corresponding evidence path fails
collection rather than quietly never firing.

### 3.10 The evidence ladder as a generated artefact

Doc 04 §6 proposes generating the reviewable artefact from the **module data** —
ordered steps, descriptions, declared inputs and outputs, params with bounds.
That is a **rulebook**: what the pipeline does in general. The spec wants a
**receipt**: what happened to one applicant. They are different documents with
different failure modes and a project needs both.

`@rung` is the receipt's mechanism, declared beside the step:

```python
@rung(section="deductions", order=40,
      says="Income tax of {income_tax_cents:money} per month. Gross of "
           "{gross_monthly_income_cents:money} annualised to {annual_gross_cents:money}, "
           "placing the applicant in bracket {paye_bracket_index} of table version "
           "{paye_brackets@version} ... Rebate class {rebate_class:rebate} applied, "
           "being the class for age {applicant_age_years:years} at {decision_date:date}.")
@step(description="Monthly income tax: annual liability less rebates, divided by twelve.")
def income_tax_cents(...): ...
```

Two checks make it a *generated* artefact rather than a maintained description:

- **a rung naming a value no step produces is a build error**, so the narration
  cannot drift from the calculation — and spec §9.1 says a divergence between the
  documented calculation and the deployed one "is itself the finding";
- **every section in `policy/ladder.py` must be reachable by at least one rung**,
  so a stage that stops emitting evidence fails the build rather than quietly
  producing a shorter document.

A design rule falls out of §9.1 and shapes the whole tree: **anything an
adjudicator will ask "why not" about must be emitted as a positive record of its
exclusion.** An excluded month emits `month_exclusion_reason_code`; an excluded
account emits `exclusion_reason_code`; a losing expense basis is retained by
`contest(retain_losers=True)`; a rejected income tier is retained the same way.
Absence cannot answer "why not".

[`sample-evidence-ladder.md`](sample-evidence-ladder.md) is what an adjudicator
receives. The one thing worth pointing at in it is §8, where the residual floor
binds rather than the proportional buffer. A maintained description would say
"the buffer bound", because that is the case everybody thinks of first and it is
right most of the time — the floor only binds for low-income applicants with
dependants, who are exactly the population the floor protects and exactly the
population an ombud hears from. A narration rendered from the value the
computation *selected* cannot make that mistake.

---

## 4. What a non-engineer sees

Four people, four artefacts, and none of them is Python.

**Regulatory Compliance** reads `policy/ladder.py` — ten sections, each with a
`heading` and a `must_answer` string that is spec §9.1's own wording — and a
rendered sample beside it. They sign the ladder, not the code. They also own
`tables/statutory/` and `tests/compliance/*.csv`, and the second of those is the
interesting one: a CSV with `in:` and `out:` column prefixes, run by
`decider spec-test`, tolerance zero. Money is in **rand** in that file because
Compliance reads gazettes in rand, and the framework applies the declared
projection from `vocabulary.py` rather than an engineer applying it in a
translation step that becomes a second source of truth.

**Credit Risk Policy** reads `tables/policy/*/` — CSVs with a comment header
naming the owner, the effective date, the approval reference and what the
behaviour column means — and `config/affordability/policy.json`. That file is
*complete for its owner class*: every field they may set is present and none is
inherited from a code default. It contains no field they may not set. A document
containing one is rejected by name.

**A product team** reads `config/affordability/product/10-flex-loan.json`. Six
fields. The file carries a `_rejected_examples` block listing, with real error
messages, what it cannot contain and who to ask.

**The Credit Committee** reads `config/affordability/overlays/register.json`.
Each overlay carries a description, a rationale in business terms, a target, an
operator, a scope, an approval reference, effective dates and a review date. The
`_expiry_report` block answers spec change scenario 12 — every live overlay with
its age — and names the command that prices unwinding one.

---

## 5. Explaining one assessment to an ombud adjudicator

Four years later, one application, no technical background, one question: *on
what basis did you conclude this person could afford this?*

They are handed `sample-evidence-ladder.md`, generated from execution trace
`AFF-2026-08-14-0009182` by `decider ladder render`. Ten sections in the
regulation's own order. Each income source with its tier, the document that
established it, the haircut and its three components, the months averaged and
the months excluded **with reasons**. All four expense bases, the band and the
dependant cell, the table version. Every account considered — including the four
excluded, with the reason for each, and including A-11, where the Bank's own view
says settled and the bureau does not, so the account stays in and the discrepancy
is recorded. The ladder. The buffer, the floor, which bound. And then §9: the
statutory answer and the Bank's conservatism, side by side, with the overlay that
made the R69.69 difference, its approval reference and its scope.

Spec §9.2 says the adjudicator's sharper question is often not affordability at
all — it is whether the Bank *declined someone the statutory calculation would
have approved, and on what authority*. Section 9 of that document is the answer,
and it is present because `max_affordable_instalment_unadjusted_cents` is an
always-emitted output rather than a diagnostic. Section 9 also renders when it is
empty, with the sentence "No policy overlays applied to this assessment" — an
absent section and a section reporting nothing are different answers to that
question, and only one of them is an answer.

Investigation (spec §9.4) is doc 03 §6's `debug()` with one addition: a changed
run is marked, at the artefact level, so it can never be confused with the
original. `dbg.set("accounts[3].treatment_code", STATED)` produces a document
headed *"Re-derivation with 1 override. Not the assessment that was made."*

---

## 6. What changes when a value moves, versus when structure moves

Doc 08 §2 gives three change classes. This project needs a fourth axis crossed
with them — **ownership** — and a fourth document kind. The grid as it actually
falls out:

| The change | Class | Owner | Costs | Reaches production by |
|---|---|---|---|---|
| Buffer 20% → 23% at grade 6 | value | Credit Risk Policy | nothing | `rt.params.swap()` |
| Credit card `PCT_LIMIT` 5% → 4% | value | Credit Risk Policy | nothing | new dated table file, params pointer |
| Credit card `PCT_LIMIT` → `GREATER_OF` | value *to compile*, interior *to approve* | CRP + Credit Committee | nothing to compile | same, through review |
| New account type 602 appears | nothing | — | nothing | `unknown_key="REFER"` already handles it; it is `indeterminate` until treated |
| A new gazetted norms table | value | Compliance | nothing | a new file in `tables/statutory/expense_norms/`; both live at once |
| Tax tables change mid-year | value | Credit Systems | nothing | same; `2026-09-01.json` beside `2026-03-01.json` |
| A new overlay | **overlay** | Credit Committee | nothing | register validation, then activation |
| Buffer varies by channel | skeleton (one line) + value | engineer + CRP | rebuild | add `channel_code` to the grid key, re-issue the table |
| An eleventh treatment behaviour | skeleton | engineer | rebuild | new registered step, new switch arm |
| A fifth consumer wanting shape (b) | **nothing** | product team | nothing | a sixth line in `consumers.py` and a new product config |
| A new product minimum evidence tier | value | Compliance | nothing | a row in `minimum_evidence_tier` |

The pattern worth naming: **everything the spec lists under "change scenarios"
that will arrive within months of go-live is a value change or a table file.**
The two that are skeleton changes are both genuinely new logic — a new key
dimension and a new behaviour — and both are one-line diffs in code plus a table
re-issue. Scenario 10 (a fifth consumer) is one line and one config file, which
is the load-bearing claim of the whole design: the fifth consumer is the cheapest
thing that happens to this project, so nobody forks it.

What a value change does *not* buy is behavioural invariance, and doc 04 §2 is
right to be blunt about it. Moving the buffer from 20% to 23% changes the outcome
for every applicant near the boundary. `decider2.impact(active, candidate,
sample)` is what makes that reviewable in Credit Risk Policy's own terms — "1.8%
of applications change decision, all in the declining direction" — rather than in
terms of validator bounds. Spec change scenario 11 asks for exactly that, before
the overlay goes live and after.

---

## 7. What this sketch does not solve

Three things are genuinely unresolved and are in FRAMEWORK-DEMANDS rather than
hidden here.

**The annotation's memory shape** (#7). A record-tier kernel emitting a
variable-length output contradicts doc 02 §1's "a kernel consumes named columns
and produces named columns". The sketch's answer is fixed-capacity `(n, 80)`
arrays plus a count, exploded at the boundary — which means every record pays 80
slots' worth of register and cache pressure whether it has 80 accounts or three,
and the modal applicant has six.

**Tax on a joint household.** Tax is per person and the household liability is a
sum of two liabilities, not a liability on a sum. The evidence ladder sample
papers over this with a footnote pointing at an appendix, which is a tell.
Stages 2 and 3 want to run per applicant and stages 4 to 7 want to run per
household, and `over()` does not compose with a *stage boundary*.

**Two major versions of this module live at once** (doc 00 §7.1 requires it). The
sketch has `contract=` files and a vocabulary map and no answer to two
`Assessment` objects in one process with different `Verdict` domains.

---

## 8. Index of inventions

Each appears at least three times in the tree; the count is what makes its
ergonomics visible rather than its existence.

| Invention | Where | Uses | Demand |
|---|---|---|---|
| `profile()` / `.under()` | `policy/modes.py` | 4 | #1 |
| owner-classed params (`statutory`/`policy`/`local`/`overlay`) | everywhere | ~40 | #2 |
| `dated_table(...).asof` | 9 artefacts | 9 | #3 |
| unit projections in `Vocabulary` | `vocabulary.py` | 6 | #4 |
| `behaviour_table` | treatments, haircuts, expense categories | 3 | #5 |
| `over()` | accounts, sources, months, employers, categories | 5 | #6, #7 |
| `contest()` | income tiers, expense bases, capacity constraints | 3 | #11 |
| `cut()` / `hold()` / `resume()` | `pipelines/` | 3 | #12 |
| `overlay_target` / `tightens_when` / `overlay_op` | capacity, expenses | 5 | #8 |
| `Domain` with no boolean context, `narrow()` | `modules/verdict/` | 7 | #9 |
| `@rung` / `ladder()` / `section()` | throughout | 14 | #10 |
| `monotone_in(..., edges_from=)` | `pipelines/affordability.py` | 1 | #13 |
