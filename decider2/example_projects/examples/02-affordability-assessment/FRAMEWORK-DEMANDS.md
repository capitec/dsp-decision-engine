# Framework demands — what this project requires of decider2

Twenty-one demands this project places on the framework, each traced to the
spec section that forced it. Read [`README.md`](README.md) first for the
narrative; this file is the ledger.

Each demand is marked exactly one of:

- **satisfied** — the proposed API already does this; the sketch just uses it.
- **needs extension** — doc 03 doesn't cover this, but nothing here contradicts
  it; the sketch adds a capability doc 03 left open.
- **departure** — doc 03 proposes a specific mechanism for this, and using it
  as proposed would produce something worse than not having it (silent drift,
  lost evidence, a governance hole, or a contradiction with another part of the
  design), so the sketch used a different mechanism instead.

Ten demands are departures, matching the count `README.md` commits to. Nine
need extension, two of which are stated as genuinely unresolved rather than
merely unwritten. Two are already satisfied. `Q13#n` means question *n* of
spec §13; `acceptance n` means criterion *n* of §10; `scenario n` means change
scenario *n* of §11.

| # | Demand | Tag |
|---|---|---|
| 1 | `profile()` instead of `Branch` for the four modes | departure |
| 2 | Owner-classed params (statutory/policy/consumer_local/overlay) | extension |
| 3 | `dated_table(...).asof`, no `.latest`/`.today` | departure |
| 4 | `Vocabulary` carries a unit + dtype projection | extension |
| 5 | `behaviour_table` — a third table shape | departure |
| 6 | `over()` with `group=`/`collapse=` — record-tier ragged collections | departure |
| 7 | The annotation's fixed-capacity memory shape | departure, unresolved cost |
| 8 | `overlay_target`/`tightens_when`/`overlay_op` — a fourth doc kind | departure |
| 9 | `Domain`, no boolean context, `narrow(treat_indeterminate_as=)` | departure |
| 10 | `@rung`/`ladder()`/`section()` — the receipt, not the rulebook | departure |
| 11 | `contest()` — evaluate every candidate, retain losers | departure |
| 12 | `cut()`/`hold()`/`resume()` — a derived third entry point | departure |
| 13 | `monotone_in(edges_from=)`, `declares_edges=` | extension |
| 14 | Money as int64 cents, `round_half_up`, float64 accumulators | satisfied |
| 15 | Null policy tiers 1–3 declared in the signature | satisfied |
| 16 | `no_fill: true` — a hard, column-level forbid | extension |
| 17 | `decider spec-test` over a non-engineer spreadsheet | extension |
| 18 | `sensitivity=`/`masking_profile=` travel with the artefact | extension |
| 19 | A testing API beyond `assert_modes_agree`/`golden.record` | extension |
| 20 | Per-applicant tax inside a per-household pipeline | unresolved |
| 21 | Two live major versions, one process, two `Verdict` domains | unresolved |

---

### 1. `profile()` instead of `Branch` for the four modes — departure

Spec §5.8 ("modes must not become copies... any structure in which a mode can
be updated without the others is a structure that will produce that"), Q13#2,
acceptance 2. `policy/modes.py`, `contracts/affordability.json`
(`declared_properties.single_kernel_across_modes`),
`tests/test_properties.py::test_all_modes_share_one_kernel`.

Doc 03 §8.2's `Branch` is the wrong tool: an arm's whole job is to differ, its
rule that "every arm must produce every declared `modifies` value" pushes
`modifies` toward the union of everything all four modes touch across seven
stages, and nothing stops one arm quietly acquiring a step the others lack —
the exact "tax table fixed in three of them" failure the spec names. `profile()`
instead validates a binding of `params`/`switches`/`emits` against the one
compiled pipeline at build time (binding anything else is an error), and a
`switch` selects an arm already inside the same compiled kernel for every mode,
so "the four modes share the arithmetic" becomes
`len({kernel_fingerprint(Assessment.under(m)) for m in ALL_MODES}) == 1` rather
than a review convention. Doc 03 has neither the concept — a closed enum
selecting an already-compiled arm, checked against a frozen structure — nor
`kernel_fingerprint()` itself.

### 2. Owner-classed params — needs extension

Spec §6.2 ("the statutory class is... unreachable from the consuming
projects, which is a stronger statement than an access convention"), Q13#9,
acceptance 1. `contracts/affordability.json` (`params` block and its `_note`
on the addition to doc 03 §5.1), `config/affordability/product/10-flex-loan.json`
(`_rejected_examples`).

Doc 03 §4.1 namespaces params by module instance (bounding blast radius to one
module); §5.1 freezes an interface. Neither carries an ownership axis — *who*
may set a value, independent of *where* it lives. A param moving from
`consumer_local` to `policy` is a MAJOR contract change even though its name,
type and bounds never move — a product config that used to validate now
doesn't. This needs an `owner_class` per param, `resolve_params(doc,
as_owner=X)` rejecting a field outside the caller's class, `decider export
--params --owner=X`, and semver treating an owner-class change as major. It
adds a second axis to doc 03's namespacing rather than replacing it.

### 3. `dated_table(...).asof`, no `.latest`/`.today` — departure

Spec §5.3 (rebate class "at `decision_date`"), §5.4 (norms), §9.3 (reproduce
to the cent, never reach a live system), scenarios 1, 2, 7, Q13#8, acceptance
5 and 12. `modules/deductions/tax.py`, `modules/expenses/norms.py`
(`declares_edges=`), `pipelines/consumers.py::replay`,
`tests/test_properties.py::test_pinned_and_dated_resolvers_agree`,
`tests/compliance/expense_norms_cases.csv` rows C11–C13.

Doc 03's only table proposal — "Tables (keyed lookups) — provisional" — is
explicitly its lowest-confidence section: a bare `key`, a dense array plus a
present mask, `tables.term.max_loan[term]`, no version axis at all. Nine
artefacts here must resolve by `decision_date` and reproduce identically seven
years later without touching a live system, which a plain keyed lookup has no
opinion about. The sketch replaces it outright: `dated_table(...)` whose only
accessor is `.asof` (resolving against the reserved, frame-only
`decision_date`), which auto-emits its resolved version so acceptance 12 is
not a `taps=` list an author can forget, and whose replay
(`resolver="pinned"`) is a property of the resolver, asserted to agree with
independent re-resolution — so a version file edited in place is detectable
rather than a silent wrong answer.

### 4. `Vocabulary` carries a unit + dtype projection — needs extension

Implicit in the project boundary — the interior must be int64 cents (doc 03
§1.2) while `credit-core` publishes float64 rand and Compliance's own
spec-test spreadsheet is denominated in rand; surfaced by acceptance 6.
`vocabulary.py`, `tests/compliance/expense_norms_cases.csv` header.

Doc 03 §5.2 defines `Vocabulary` as a pure name-to-name map plus prefix
families — no unit axis, no dtype axis. A manual conversion at every
producing or consuming site is the "79 passthrough steps" failure doc 01
§5.1 already recorded once. `vocabulary.py` extends `Vocabulary` with
`projections={name: Projection(target, to=, divide=, rounding=, places=)}`,
generating a real, audited step at the boundary and refusing the inverse: a
float-rand input arriving where cents are expected is a build error naming
the missing declared rounding, never a silent `int(x * 100)`. This extends
doc 03's object; it doesn't replace its name-mapping job.

### 5. `behaviour_table` — a third table shape — departure

Spec §5.5.2 ("the matrix cell selects a *behaviour*, not a value"), §5.2.2,
§5.1, Q13#5, scenarios 4 and 5, acceptance touching quarterly retunes.
`modules/obligations/treatments.py`, `modules/income/haircuts.py`,
`tables/policy/obligation_treatment/2026-07-01.csv`.

Doc 08 §3.4 gives a test for choosing between its two kinds: *"can one
compiled loop evaluate every instance... with the instance supplied as
arrays? If yes, generic kernel. If it needs a switch over node types,
codegen."* This project's matrices fail that test on both sides: every row
applies the same *shape* of rule (not a `ruleset`), but the discriminant
column selects which compiled arm runs, not a value read from a row array
(not a `decision_table`). `behaviour_table` compiles to a switch with one arm
per registered behaviour, so reassigning a cell (`PCT_LIMIT` → `GREATER_OF`)
is a value change (both arms already exist) while adding an eleventh
behaviour is a skeleton change — compile cost and approval class split at
different places, which is why `change_class` is declared **per column**.
Neither of doc 08's two kinds has a column-level `change_class`.

### 6. `over()` with `group=`/`collapse=` — departure

Spec §5.5.1 (two account lists, dedup, "a group with an internal member takes
the internal... figures"), §5.2.3 (nested 0..12-inside-0..6), §5.1, §5.3 (UIF
per employer), Q13#6, acceptance 10. `modules/obligations/assemble.py`
(`AccountList`, `group="dedup_key"`), `modules/income/__init__.py`
(`Sources`, `MonthQualification`), `modules/deductions/statutory.py`
(`EmployerUIF`).

Doc 02 §5 places joins and group-bys in the frame tier because lineage falls
out of the record-tier graph for free and does not survive arbitrary polars —
the right home for a group-by *across* records, the wrong home for one
*inside* a record with no independent row identity. Following doc 02 §5 here
means exploding accounts into rows, grouping, and joining back — the
`filter → group_by → join-back` rewrite doc 03 §8.3 records as having
destroyed early exit in the previous generation's port. `over()` is the
record-tier answer, with a `group=`/`collapse=` phase for the two-source
dedup that doc 03 has no equivalent of.

### 7. The annotation's fixed-capacity memory shape — departure, unresolved cost

Spec §5.5.3 ("the scalar must never be computed by a second code path that
could disagree with the annotation"), Q13#6, acceptance 10.
`modules/obligations/annotate.py` (`materialise_by_default=False`),
`pipelines/affordability.py::assert_materialisation_neutral`, `README.md` §7.

Doc 02 §1's kernel contract — "consumes named columns and produces named
columns" — describes a fixed schema per row, and a 0..80 ragged annotation
contradicts it. Fixed-capacity `(n, 80)` arrays plus a count, exploded only on
request, make disagreement structurally impossible (every scalar reducer
reads the same column the annotation publishes), but do not remove the cost:
every record pays 80 slots of register and cache pressure whether it holds
80 accounts or the modal six. Recorded as unresolved, not solved — doc 02 §1
is missing a genuine variable-length-output primitive, and a padded array is
an approximation of one, not a provision of one.

### 8. `overlay_target`/`tightens_when`/`overlay_op` — a fourth doc kind — departure

Spec §5.6.2 (five numbered requirements: definition-time rejection, scope as
an error not a no-op, declared order, required expiry, a disabled-stack path
through the same implementation), §9.2, Q13#10, acceptance 8 and 9, scenarios
11 and 12. `policy/overlays.py`, `modules/expenses/__init__.py`
(`living_expenses_cents`, `floor="statutory_norm_cents"`),
`config/affordability/overlays/register.json` (`_rejected`, `_disabled_run`).

Doc 08 §2 presents three change classes as exhaustive. An overlay stack fits
none: it changes a value, but composes in a declared order that changes the
answer and whose admissibility is a property of a *pair* (operator, target) —
closer to logic than to a value — and unlike an interior change it never
recompiles. `register.json` names this directly: "a FOURTH document kind."
The sharper demand — conservative-only, provable before any applicant exists,
not bypassable by a negative magnitude — needs concepts doc 03/08 don't have:
an operator declaring a monotone `direction` over a bounded magnitude domain
(so a loosening attempt is a bounds error on the field typed, not a policy
error on the composed effect), a target declaring which direction
`tightens_when()` it, and a target declaring a computed floor that only a
`preserves_floor=True` operator may approach. All three run at
register-validation time, which is what acceptance 8 actually requires.

### 9. `Domain`, no boolean context, `narrow(treat_indeterminate_as=)` — departure

Spec §5.7.1 ("conflating `indeterminate` with `fail` is the most consequential
error available"), §5.1 (never silently drop an applicant), Q13#12,
acceptance 7. `modules/verdict/__init__.py` (`Verdict`, `Sufficiency`, five
`narrow()` calls), `tests/test_properties.py::test_verdict_has_no_boolean_context`.

Doc 03's own worked examples return plain `bool`/`float` throughout and
propose no closed, non-boolean output type — an author following doc 03's
idiom literally writes exactly the int8-with-a-comment the spec calls "the
cheapest implementation of every layer." That idiom has to be made
unavailable: a `Verdict` with no `__bool__`, no `__ne__` against a single
member (`verdict != PASS` folds three outcomes into one, so it raises,
naming them), and a `narrow()` whose `treat_indeterminate_as=` has no default
— forcing each of the five consumers to answer in writing, in one place,
where the five different answers are readable together.

### 10. `@rung`/`ladder()`/`section()` — the receipt, not the rulebook — departure

Spec §9.1 (eight things one artefact must show, in the regulation's order),
§9.2, Q13#11 ("doc 04 §6 ranks this as the top open risk"), acceptance 11.
`policy/ladder.py`, `@rung` on `income_tax_cents`, `sample-evidence-ladder.md`,
`tests/test_properties.py::test_every_rung_references_only_values_that_exist`.

Doc 04 §6 proposes generating the reviewable view from module data — ordered
steps, descriptions, params, the version chain — and flags it as the design's
own least-confident, untested risk. That artefact, even working, is a
**rulebook**: what the pipeline does in general. Spec §9.1 asks for a
**receipt**: what happened to *this* applicant, with the months actually
excluded and why. A rulebook can be complete and correct and still not answer
that, so taking doc 04 §6 as sufficient would leave §9.1 unanswered. The
sketch keeps it and adds `@rung`, tied to one step and rendered from one
execution trace, with two build-time checks doc 04 §6 lacks: a rung naming a
value no step produces is a build error, and every declared section must be
reachable by at least one rung.

### 11. `contest()` — evaluate every candidate, retain losers — departure

Spec §5.2.1 ("weaker contradictory evidence is recorded but does not change
the tier"), §5.4 (all four bases, a declared tie-break), §5.6.2, Q13#7,
acceptance 3. `modules/income/waterfall.py` (`select=STRONGEST,
retain_losers=True`), `modules/expenses/bases.py` (`select=HIGHEST,
tie_break=`), `modules/capacity/__init__.py` (`select=LOWEST`).

Doc 03 §8.2's `Branch` is built around the opposite property a contest needs:
its performance case is that "only the taken arm executes" — a columnar
engine's inability to do that is cited as `Branch`'s 7.8× advantage. A
contest needs every candidate to run and its value retained, because the
adjudicator's first question is "what else did you have, and why didn't you
use it" — a dispatch primitive that only executes the taken arm discards
exactly that evidence. `contest()` also carries a **declared** `tie_break=`
(never argument order — "an accident of how somebody typed the call") and a
`monotone=True` flag feeding `monotone_in()` (#13), neither of which a
conditional-dispatch primitive has reason to carry.

### 12. `cut()`/`hold()`/`resume()` — a derived third entry point — departure

Spec §5.7.2(3) ("without the separation becoming a second entry point that
can drift from the first... the alternative is exactly the fork this project
exists to prevent"), Q13#3, acceptance 4. `pipelines/affordability.py`
(`evidence = cut(...)`), `pipelines/consumers.py::search_consolidation`,
`tests/test_properties.py::test_cut_soundness_is_a_build_time_check`.

Doc 02 §3.5 gives exactly two entry points — `apply()` and `score()` — both
re-running the whole kernel from the top. Building "vary the obligations 400
times without redoing the waterfall" on top of them means either a
hand-cached second composition (the drift Q13#3 warns against) or 400 full
re-runs, failing §5.7.2(3) outright. `cut()` is declared once; `hold()`/
`resume()` are *derived*, entering the same compiled kernels at the cut. The
framework verifies from static lineage that no step upstream of the cut
reads a `resumes=` name — why `Framing` derives `principal_debtor_set` from
identity, not the account list — so a violation is a build error, not a
runtime surprise on the 400th scenario.

### 13. `monotone_in(edges_from=)`, `declares_edges=` — needs extension

Spec §5.7.2(1) ("monotonicity must be tested, not assumed"), §5.6.2 (the
buffer/floor interaction named explicitly), Q13#4, acceptance 3.
`pipelines/affordability.py::monotone_in`, `modules/expenses/norms.py`
(`declares_edges=`), `tests/test_properties.py::test_monotone_in_proposed_instalment_across_declared_edges`.

Doc 03 §1.2 states the one concrete requirement on a `corpus` — it must
contain boundary values, since the recorded int64-overflow defect "was found
by binary search, not by sampling" — and doc 03 §11 *uses* `corpus` without
saying where one comes from. The residual-floor/buffer interaction lives
exactly at a cell boundary — measure-zero under random sampling, routine in
the applicant population. This fills a gap doc 03 itself flags as open:
`declares_edges=` on a keyed artefact plus `corpus_from_declared_edges()`
derive a seeded corpus mechanically rather than by hand.

### 14. Money as int64 cents, `round_half_up`, float64 accumulators — satisfied

Implicit throughout §5; §9.3's "reproduce to the cent."
`modules/income/variable_pay.py` (`component_sum_squares: float64`),
`policy/overlays.py`, `vocabulary.py`.

The one demand doc 03 answers completely, used as proposed. Doc 03 §1.2 names
the exact bugs this project would hit — the `.xx5` rounding disagreement
between njit, CPython and `Decimal`; int64 overflow at realistic loan sizes; a
sum-of-squares accumulator wrapping at 2,667 rows — and prescribes exactly
what this project does. The comment on `component_sum_squares` could be
lifted verbatim from doc 03 §1.2. Nothing here needed inventing.

### 15. Null policy tiers 1–3 declared in the signature — satisfied

Spec §5.2.3 (a payable-and-zero month counts, a no-record month doesn't —
two different nulls), §5.4. `modules/income/variable_pay.py`
(`component_amount_cents: int | None`), `modules/framing/__init__.py`.

Doc 03 §1's three tiers — required, declared fill, explicit `Optional` — map
directly onto this project's "zero counts, absence doesn't" distinction.
`month_qualifies` is written in doc 03's tier-3 style because, as the
module's own docstring says, "here the distinction IS the logic" — doc 03's
guidance applied exactly where it says it should bite.

### 16. `no_fill: true` — a hard, column-level forbid — needs extension

Spec §5.4 ("a category declared as zero and a category not asked... must not
be conflated"), §5.2.4. `schemas/affordability_input.json`
(`declared_category_cents`, `statement_confidence`,
`employer_is_on_verified_register`), `modules/expenses/bases.py`.

Doc 03 §1's tier 2 (`missing_as()`) is a per-call-site declaration — nothing
stops a different step, or a future edit, from filling the same column a
different way where the whole point is it must never be filled. `no_fill:
true` in the input schema is a project-level, column-level, build-enforced
forbid on `missing_as(...)` reaching that column anywhere in the pipeline —
stronger than tier 2's per-signature opt-in, and additive to it.

### 17. `decider spec-test` over a non-engineer spreadsheet — needs extension

Spec §9.1 (Compliance "does not write code and must not have to read any"),
Q13#13, acceptance 6. `tests/compliance/expense_norms_cases.csv`, rows
C11–C13 (replay against a historical `decision_date`).

Doc 03 §11's testing story — `.score(...)` from Python, plus
`assert_modes_agree`/`golden.record` — assumes a programmer. Q13#13 asks for
the opposite. This needs a runner that maps `in:`/`out:` prefixes onto
declared names, applies the project's `Vocabulary` projection (#4) so a
rand-denominated spreadsheet reconciles against cents, accepts a per-row
`decision_date` rather than "now," and fails at zero tolerance. None of it
exists in doc 03; the sketch's CSV is written as though it already does.

### 18. `sensitivity=`/`masking_profile=` travel with the artefact — needs extension

Spec §9.5 (access controlled, non-production masked, a debugging session on
a real assessment is production data access). `policy/ladder.py`
(`sensitivity="applicant_financial"`), `contracts/affordability.json`.

Doc 04 §5.3 states the consequence correctly — trace output inherits the
sensitivity of its input — but treats it as following from where a trace
ends up, not as a property declared on the artefact itself. This project
needs the classification carried on the `ladder()` declaration through
render, storage and a `debug()` session, so masking doesn't depend on an
engineer remembering which bucket a file landed in.

### 19. A testing API beyond `assert_modes_agree`/`golden.record` — needs extension

Acceptance 2, 3, 4, 8, 9, 10, 11 and 12 collectively — nearly all of §10 is a
claim about structure, not about one applicant's numbers.
`tests/test_properties.py`: `kernel_fingerprint()`,
`corpus_from_declared_edges()`, `Assessment.check_cut()`,
`Assessment.check_ladder()`, `assert_materialisation_neutral()`,
`assert_cut_equivalent()`, `derive_admissible()`, `OverlayRegister.validate()`.

Doc 03 §11 offers exactly two framework-provided assertions. This project's
acceptance criteria are almost all claims about structure — four modes share
one kernel, a cut is sound, a scalar can't disagree with its own annotation,
a rung can't reference a value nothing produces — which is exactly the
distinction the README insists on ("a test that asserts one applicant gets
R4 312.18 protects one applicant"). Each name above answers one criterion,
and none exists in doc 03 or doc 04 today.

### 20. Per-applicant tax inside a per-household pipeline — unresolved

Spec §5.1 (a joint application "as one household," incomes summed after
their own individual haircuts), §5.3 (rebate class depends on per-person
age). `modules/deductions/tax.py`, `sample-evidence-ladder.md` §3's footnote,
`README.md` §7.

`over()` (#6) composes for a collection living entirely inside one record. It
does not compose with a boundary that changes what the *record itself*
represents partway through a pipeline: stages 2–3 want to iterate per
applicant (tax is non-linear per person), stages 4–7 want one row per
household. This sketch does not solve it — the ladder papers over the gap
with a footnote pointing at an appendix that doesn't exist in this tree,
which is itself the tell. Missing: a primitive for scoping some stages one
way and the rest another — different from `over()`'s single-record nesting
and from doc 02 §5's cross-record join. Flagged, not hidden.

### 21. Two live major versions, one process, two `Verdict` domains — unresolved

Not forced by this project's spec — forced by the framework's own
multi-version-coexistence requirement, which this is the first sketch to
actually need: a change to the treatment matrix or to `Verdict`'s membership
is a MAJOR contract change, and the framework requires old and new to run
side by side while five consumers migrate on different schedules.
`contracts/affordability.json` (frozen at `3.2.0`), `README.md` §7.

`contract=` fails a build on a breaking change; it says nothing about running
two versions in one process. `Domain` (#9) is `closed=True`; two closed
domains sharing a name with different membership is the same trap doc 08
§4.2 warns about for params `NamedTuple` classes, but nobody has said what
that means for a domain, and `narrow()` has no notion of "which major
version's `Verdict` is this." Flagged, not hidden.

---

## Tally

Ten departures (#1, 3, 5, 6, 7, 8, 9, 10, 11, 12) — matching the count
`README.md` commits to. Nine need extension (#2, 4, 13, 16, 17, 18, 19, 20,
21), two of which (#20, #21) are genuinely unresolved rather than merely
unwritten. Two are already satisfied by doc 03 as proposed (#14, 15).

## Fix made to an existing file while writing this document

`modules/income/__init__.py` referenced `MAX_SEVERITY` in an `aggregate={}`
dict without importing it, and represented one reducer
(`"income_source_code"`) as a bare string —
`"of_max(income_source_tier, by=source_gross_after_haircut_cents)"` — while
every other reducer in the same dict is a typed combinator call
(`SUM("...")`, `MIN("...", where=..., sign=...)`). A string expression beside
typed calls in one dict is the inconsistency doc 08 §3.2 argues against in
config; it has no better standing in code. Fixed by adding `MAX_SEVERITY` and
`OF_MAX` to the import line and rewriting the reducer as
`OF_MAX("income_source_tier", by="source_gross_after_haircut_cents")`,
matching the convention used everywhere else in this project.
