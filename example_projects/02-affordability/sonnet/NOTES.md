# NOTES

## 1. What I built

All seven stages of spec 02, for single and joint applicants, over 00's
five affordability units:

- **Stage 1 (household framing, `assessment/household.py`).** Dependants
  (higher of the two declarations), expense consolidation (higher-of for
  shared categories, sum for personal ones -- a working-depth 4 shared + 3
  personal set, not the spec's full 8-14), income and statutory deductions
  run once per applicant (00's `core.income`/`core.deductions` units,
  `.relabel()`-ed onto each applicant's fields) and combined into the
  household figures, the joint-account dedup (an account reported on both
  applicants' bureau profiles counts once), and the "can't assess on one
  applicant's income alone" rule.
- **Stage 2 (income) and Stage 3 (deductions)** are 00's units, run twice
  and combined -- no new arithmetic.
- **Stage 4 (living expenses)** is 00's `core.expense_norms` unit,
  unmodified, fed the household-consolidated declared/statement figures on
  the household's combined gross income (as the regulation prescribes for
  joint consumers).
- **Stage 5 (obligations, `household.household_obligations`).** Merges the
  two applicants' bureau and internal account lists, dedups by identity,
  shapes the list for the assessment mode (`EXCLUDE_ON_QUOTE` only in
  scenario mode), then calls 00's own per-account arithmetic directly (see
  "Framework friction" for why it can't compose the published
  `obligations.obligations` step here).
- **Stage 6 (capacity, `assessment/capacity.py`).** The buffer grid (12
  grades x 6 products, `DecisionTableConfig`) and residual floor (6
  dependant bands), "the binding one wins", then the tighten-only overlay
  stack (`core.adjustments`, unmodified) over the result.
- **Stage 7 (verdict, `assessment/verdict.py`).** `evidence_sufficiency_code`
  (applicant income gap, stale bureau, `REFER` account, weak tier, no net
  income) and a verdict that is `indeterminate` before it is ever `fail`,
  plus the three answer shapes: shape (a) pass/fail/marginal against a
  supplied `proposed_instalment`; shape (b) capacity only, `pass` or
  `indeterminate`, never `fail`, when none is supplied; shape (c) (the
  maximum loan amount) is out of scope per spec (project 03's job).

**The four modes (§5.8)** share the one pipeline: `assessment_mode_code`
selects `minimum_income_tier` (`assessment/modes.py`) and how the account
list is shaped before the shared obligations unit runs -- never a second
copy of any stage. One acceptance test per mode in
`tests/test_pipeline.py::test_every_consuming_mode_scores_through_the_one_pipeline`.

**Separability (§5.7.2 item 3, acceptance §10 item 4).**
`pipeline.evidence_unit()` (stages 1-4) and `pipeline.capacity_unit()`
(stages 5-7) are built from the same step objects as `pipeline.build()`,
never a second implementation. `tests/test_scenario_budget.py` proves both
agree with the combined pipeline and runs capacity_unit() 400 times over
one evidence_unit() result.

**Monotonicity (§5.7.2(c), acceptance §10 item 3)** is proved, not
asserted, in `tests/test_monotonicity.py`: swept across the buffer bound,
the residual-floor bound, the exact crossover between them, zero/negative
discretionary income, and a broad grid of (discretionary income, buffer,
floor) combinations.

### What I left out

- The full evidence ladder rendering (§9.1) -- SCOPE.md explicitly skips
  it for this slice; the underlying data (every intermediate, every table
  cell, both adjusted and unadjusted figures) is all present in the
  output, just not rendered as a document.
- Variable-pay edge cases beyond one component (SCOPE.md) -- 00's
  `core.income.income_variability_ratio` already carries the one
  component at working depth; 02 does not deepen it.
- 00's `core.income` unit is a **single-source waterfall** per applicant
  (best evidence wins), not spec 02 §5.2.5's "0..6 sources, each
  independently haircut and averaged, then summed". 02 composes two
  applicants' single-source results into a household; it does not deepen
  the per-applicant source model, since that is 00's unit to extend, not
  02's to fork. See "Spec problems".
- The obligation treatment matrix, buffer grid and residual floor are all
  at working depth (00's own ~15-type matrix; a 12x6-product buffer grid,
  not 12x8; a single-version buffer grid, not multi-version
  effective-dated) -- the mechanism (a matrix cell selects a behaviour; two
  constraints race and the binding one wins) is what this slice proves,
  per SCOPE.md's "keep the dominant difficulty... cut breadth" rule; this
  project's own dominant difficulty is reuse across five consumers and
  regulatory versioning, not table size.
- 09 §5.15's full field-level PII classification and the golden-set
  spreadsheet-as-test mechanism (acceptance §10 item 6) are adopted as a
  checklist, not built -- 09-C is a wave-0 requirements document per
  SCOPE.md, not code this slice implements.

---

## 2. Reuse

**From project 00 (`credit_core`), unmodified:**

- `core.income` (`income.gross_monthly_income`, `.income_source_code`,
  `.income_verification_tier`, `.income_haircut_applied`,
  `.income_variability_ratio`) -- run twice via `.relabel()`, once per
  applicant.
- `core.deductions` (`deductions.tax_table_version_step`,
  `.build_tax_table()`, `.statutory_deductions`, `.net_monthly_income`) --
  run twice, same pattern. Tax is individual, never computed on the
  household's combined income.
- `core.expense_norms` (`expense_norms.statutory_version_step`,
  `.internal_version_step`, `.build_statutory_table()`,
  `.build_internal_table()`, `.internal_norm_amount`, `.norm_floor`,
  `.norm_table_version`, `.expense_basis_code`, `.living_expenses`) --
  the whole unit, unmodified, run once on the household's combined figures.
- `core.obligations._process` -- 00's per-account arithmetic (the ten
  treatment behaviours, the treatment matrix), called directly rather than
  through the published `obligations.obligations` step; see "Framework
  friction" for why.
- `core.affordability.discretionary_income` -- the one function reused
  from that unit. `affordability_buffer_applied_step` and
  `affordability_verdict_code_step` are **not** used; see "Spec problems"
  for why 02 builds its own buffer (grid-driven, not a flat param) and
  verdict (mode/shape-aware) instead.
- `core.adjustments` (`AdjustmentRegister`, `Adjustment`,
  `AdjustmentEffect`, `.apply_stack_step`) -- the tighten-only overlay on
  `max_affordable_instalment`, exactly the "same capability, run through a
  `param()` flip for stack-off" mechanism 00 built.
- `core.reason_codes` (`ReasonCodeRegistry`, `ReasonCode`,
  `.resolve_step()`) -- 02's own registry of evidence-gap and
  capacity-exceeded reasons.
- `core.rounding.round_instalment` / `round_instalment_step` -- rounds
  `max_affordable_instalment` to the cent after the overlay's `multiply`.
- `core.evidence.cell_id` -- used inside `capacity.py`'s two new tables,
  the library's own cell-attribution convention.
- `core.dates.EffectiveDatedSet` -- not used directly by 02 (the buffer
  grid and residual floor are single-version at this working depth); every
  effective-dated table this slice reads (tax, statutory/internal norms)
  is 00's own, already resolved by `decision_date`.

**From `decider`'s built-ins:** `DecisionTableConfig` (the buffer grid,
the residual floor); `frame_step` (household expense consolidation, the
obligations merge); `dag`/`flow`/`step`; `param()`/`missing_as()`
throughout; `.relabel()` (per-applicant unit reuse, table output naming)
and `.named()`-equivalent reuse via the applicant-prefixed unit functions.

**Written from scratch:** everything in `assessment/` -- household framing,
mode selection, the bureau-staleness/REFER evidence gates, the
buffer-grid/residual-floor capacity race, and the verdict/shape logic.
DEPS.md's own boundary (00 owns the units, 02 owns the assessment) is
exactly this split.

---

## 3. Gaps in what I consumed

- **`core.bureau`'s published shape doesn't serve this project.** DEPS.md
  and spec 02 §4.1 name `core.bureau` as the source of the bureau account
  list, but 00's actual `core.bureau.normalise_bureau` takes a raw nested
  `bureau_response` (accounts/enquiries/public records) and emits
  *aggregates* (counts, worst status, a fixed 90-day staleness flag) --
  not an account list, and not a product-dependent (7 vs. 14 day)
  freshness window as spec 02 §4.2 requires. `core.obligations` (the unit
  02 actually needs for stage 5) takes a flat per-account list directly,
  matching 00's own `sample_request.json` shape. 02 consumes
  `core.obligations`'s contract (unmodified) and computes its own
  freshness check (`assessment/evidence_gates.py::bureau_is_stale`,
  product-keyed window) directly from a `bureau_as_of_date` field, rather
  than routing through `core.bureau` at all. Recorded here rather than
  worked around silently, per BRIEF's instruction.
- **`core.obligations` has no `EXCLUDE_ON_QUOTE` treatment.** Spec 02
  §5.5.2 requires a settlement quote to zero an obligation *only* in
  scenario mode; 00's `_treat_one` applies `USE_SETTLEMENT_QUOTE`
  (an imputed obligation) unconditionally whenever `settlement_quote` is
  present, with no mode concept. 02 works around this at the caller
  boundary: outside scenario mode, `household.household_obligations`
  strips `settlement_quote` from every account before 00's arithmetic ever
  sees it, so the ordinary stated/imputed treatment applies instead. This
  reproduces the *outcome* spec 02 asks for without editing 00, but it is
  a real behavioural gap between the ten treatments 00 documents and the
  ten spec 02 names.
- **00's `core.income` is a single-source-per-applicant waterfall**, not
  spec 02 §5.2.5's 0..6-sources-summed model (see "What I left out" and
  "Spec problems").

---

## 4. Framework friction

Four items, each confirmed with a minimal reproduction. The first two are
new, deeper instances of gaps 00's NOTES.md already found; the third and
fourth are new.

### 4.1 A genuinely empty top-level ragged field, or a lone `None` scalar, both crash single-record scoring -- and this directly threatens the separability pattern spec 02 *requires*

00's NOTES.md documents that an entirely-absent `missing_as([])` column
crashes (`ShapeError: unable to add a column of length 0 to a DataFrame of
height 1`) and that a present-but-empty list literal with no non-empty
sibling infers `List(Null)` and crashes at the arrow boundary
(`ArrowImportError: Expected array with 0 buffer(s) but found 1
buffer(s)`). This project hits both, one level deeper: a **solo**
application's `applicant2_*` ragged fields (`applicant2_bureau_accounts`,
`applicant2_internal_accounts`, `applicant2_variable_pay_history`) are
genuinely empty by construction, and every combination I tried --
omitting the key, sending `[]`, sending `[]` alongside one non-empty
sibling of a *different* column -- reproduced one of the two errors above.
The only reliable fix was to send one harmless placeholder element
(`account_type_code: 31`, i.e. `EXCLUDE`, all amounts zero, `closed:
true`) for every one of applicant 2's ragged fields on a solo application.
This is now `sample_request.json`'s own convention, documented in
SERVE.md, but it is a real trap for every one of this project's five
consumers the first time they send a solo application.

A **new** finding beyond 00's: a lone `None` value for an *optional
scalar* field is exactly as ambiguous as an empty list, for the identical
reason --

```python
>>> pl.DataFrame([{"x": None, "y": 1.0}]).schema
Schema({'x': Null, 'y': Float64})
```

-- and `decider`'s arrow-import view rejects a `Null`-dtype column against
a declared nullable `Float64` input the same way. This matters more than
it looks: spec 02 §5.7.2 item 3 *requires* a caller to be able to hold
`evidence_unit()`'s output and re-score `capacity_unit()` against it
directly (project 06's 400-scenario pattern) -- and `evidence_unit()`'s
own output routinely contains `None` (e.g. `statement_living_expenses`
when no statement evidence exists). Passing that output dict straight
into a second `.score()` call reproduces the crash. The workaround
(`tests/test_scenario_budget.py`): filter `None`-valued keys out of the
first call's output before merging it into the second call's record. This
is not documented anywhere in `decider`, and a consumer implementing the
exact separation pattern the spec asks for will hit it on the first try.

### 4.2 A `frame_step`'s `list[struct]` output crashes when it flows into a *second* `frame_step`, not only as a terminal output

00's NOTES.md documents that a terminal (unconsumed) `frame_step` output
typed `list[struct]` crashes result materialisation. This project found a
deeper version: the same crash (`ValueError: cannot parse numpy data type
dtype('O') into Polars data type`) fires for an **intermediate**
`list[struct]` column too, the moment it has to flow from one `frame_step`
into a second one. The natural design for stage 5 was: a
`household.merge_household_accounts` frame_step merges and dedups the two
applicants' account lists into `bureau_accounts`/`internal_accounts`
columns, and `obligations.obligations` (00's published step) reads them
next -- exactly how 00's own `sample_request.json` already supplies those
two fields as top-level input. That composition crashes, because the
merged column is now an *intermediate*, and any intermediate
`list[struct]` handed from one frame step to the next round-trips through
the same numpy boundary as a terminal one.

There is no supported way to compose two `frame_step`s through a
`list[struct]` column. The workaround: `household.household_obligations`
merges the two applicants' lists **and** calls
`credit_core.obligations._process` (00's underscore-prefixed, non-public
per-record helper) directly, inside the same frame step, so the merged
list never becomes a materialised column at all. This reuses 00's exact
treatment-matrix arithmetic without forking it, but it means reaching past
the one function 00's own NOTES.md calls "the one entry point" of that
capability into a helper 00 never committed to as stable API -- a real
forking risk if 00's internals change shape later.

### 4.3 `missing_as(None)` and a bare untyped default both raise -- but only one error message says what to do

```
TypeError: missing_as(None) fills a null with a null; annotate the input `T | None` instead
```

is correct but easy to hit live (I did, on `bureau_as_of_date`): the
natural first attempt at "this may be entirely absent" is `missing_as(None)`,
by analogy with `missing_as([])`/`missing_as(0.0)` elsewhere in the same
codebase (00's own `income.py`). The actual answer -- annotate the
parameter `date | None` and give it a bare `None` default, no `missing_as`
at all -- is a different, unrelated-looking mechanism
(`NullPolicy.OPTIONAL` vs. `NullPolicy.MISSING_AS` in
`decider/engine/params/harvest.py`), and nothing in `missing_as`'s own
docstring points at it. Worth a cross-reference in `missing_as`'s
docstring to `T | None` for "the input may be absent" vs. `missing_as` for
"a present null becomes this value".

### 4.4 `.emit()` on an outer `dag` wrapping two already-`.emit()`-ed sub-dags refuses a name neither sub-dag mentions, even though it passes through anyway

`pipeline.build()` is `dag(evidence_unit(), capacity_unit(), name=...)`,
where each of the two members is itself a `dag(...).emit(...)`. Naming
`decision_id` (a top-level input nothing reads, present purely to be
carried through) in the *outer* `.emit(...)` call raised

```
WiringError: affordability_assessment: emit('decision_id'): no step
produces 'decision_id' and it is not a declared input column.
```

-- yet removing it from the explicit list, `decision_id` still appears in
the final output untouched, exactly as flow/dag's own docstring promises
("the output holds the input columns plus the values nothing reads").
Composing two already-scoped sub-dags into an outer one does not
re-expose an untouched input column to the outer `.emit()`'s own name
validation, even though the column itself survives the composition fine.
The fix is simply not to name it -- but the error message, read on its
own, suggests the column has been lost, when it has not.

### What worked well

`.relabel()` running one 00 unit twice (once per applicant) is exactly
the mechanism 00's own NOTES.md recommends, and it composed cleanly for
both `core.income` and `core.deductions` here. `AdjustmentRegister`'s
`apply_stack_step` needed zero changes to layer a second, independent
overlay point (`max_affordable_instalment`, distinct from 00's own demo
overlay on `score`) on top of the same tighten-only mechanism.
`pipeline.parameters().defaults()` again made a correct, complete
`params.json` for a ~25-node pipeline a one-liner. `decider build`
succeeded unchanged once the request-shaping workarounds above were in
place.

---

## 5. Spec problems

- **§5.2.5's per-applicant multi-source income model isn't in 00's unit.**
  00 built `core.income` as a single-source-per-applicant waterfall (best
  evidence wins); spec 02 §5.2.5 describes up to six sources per
  applicant, independently haircut and averaged, then summed. DEPS.md's
  boundary ("00 owns the units' arithmetic... 02 composes them") doesn't
  say which project should have built the multi-source richness. I read
  it as 00's to have built (it is unit arithmetic, not household framing)
  and, since 00 chose working depth here (its own NOTES.md does not flag
  this particular simplification explicitly), left it as a two-applicant
  composition over 00's existing single-source unit rather than forking
  or deepening `core.income` myself -- SCOPE.md's "skip... variable-pay
  edge cases beyond one component" gave licence to trim in this direction,
  but the multi-source-per-applicant gap is bigger than that one phrase
  covers.
- **§5.6's verdict-ownership ambiguity (already flagged in 00's own
  NOTES.md "Spec problems") is real and I had to resolve it by not
  reusing two of 00's five "unit" outputs.** 00 built
  `affordability_buffer_applied_step` (a flat `param()`) and
  `affordability_verdict_code_step` (a two-number comparison) as part of
  `core.affordability`. Spec 02 §5.6.2 requires the buffer to be a
  *grid* (risk grade x product), not a flat param, and §5.7 requires the
  verdict to be mode- and shape-aware with a distinct `indeterminate`.
  Both of 00's corresponding step objects are therefore unused by this
  pipeline; only `discretionary_income` is reused from that unit. A
  reviewer diffing the two projects should not read the unused steps as
  an oversight -- DEPS.md's "Cycles" §2 puts exactly this work on 02's
  side of the line, and 00's own NOTES.md independently reached the same
  reading.
- **`EXCLUDE_ON_QUOTE`'s mode-gating (§5.5.2) isn't representable in 00's
  ten treatment codes as built** (see "Gaps in what I consumed" and
  "Framework friction" 4.2's workaround).
- **`core.bureau`'s contract, as built, doesn't match what DEPS.md says 02
  consumes from it** (see "Gaps in what I consumed").

---

## 6. What I would do next

1. Push 4.1's two boundary bugs (empty ragged top-level field, lone
   `None` scalar) upstream. Every one of this slice's five downstream
   consumers (03, 05, 06, 07, 08) will hit 4.1's separability variant the
   first time they hold this project's `evidence_unit()` output and feed
   it back in, since that is exactly the pattern spec 02 §5.7.2 item 3
   requires of them.
2. A supported way to compose two `frame_step`s through a ragged
   intermediate (4.2) would remove the one place this project reaches
   past a private helper (`core.obligations._process`) to stay within one
   `decider` pipeline.
3. Extend `core.income` to the full 0..6-source-per-applicant model
   (§5.2.5) in project 00, then re-point this project's per-applicant
   income units at it with no change to `assessment/household.py`'s
   combination logic -- the household layer does not care how many
   sources fed each applicant's figure.
4. A genuine `EXCLUDE_ON_QUOTE` treatment code in `core.obligations`,
   mode-aware, so 02 stops stripping `settlement_quote` at the caller
   boundary as a substitute.
5. Effective-date the buffer grid and residual floor (today single-version,
   working depth) the same way 00 effective-dates tax and expense norms,
   to exercise change scenario 3 (buffer varies by channel) for real.
6. The full evidence ladder rendering (§9.1) -- every input it needs is
   already in this project's output; only the rendering is missing.
