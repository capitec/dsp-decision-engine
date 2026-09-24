# NOTES

## What I publish (for 04, 06, 07)

| Capability | Module | Entry point |
|---|---|---|
| The bounded solve | `loan_granting.solve` | `solve_term(evaluator, term_months, risk_grade, applicant_age_years, is_joint, amount_cap, requested_amount, max_affordable_instalment, evaluation_ceiling=24) -> SolveResult` |
| Pricing (rate/fee/credit-life/instalment, one candidate) | `loan_granting.pricing` | `PriceEvaluator(rate_card_index, credit_life_index, statutory_ceiling, credit_life_substitution_declared).evaluate(amount, term_months, risk_grade, applicant_age_years, is_joint) -> PricingResult` |
| The rate card as a fast in-memory index | `loan_granting.pricing` | `RateCardIndex.from_configurable_step(rate_card_flex_loan)` / `RateCardIndex(rows)` |
| The cap waterfall (52 rules, 3 ceilings) | `loan_granting.waterfall` | `cap_waterfall` (a `frame_step`); the register itself is `waterfall.REGISTER`, `waterfall._run_waterfall(app_dict)` for direct reuse outside a pipeline |
| Offer-set construction | `loan_granting.granting` | `build_offer_set(app, term_results, objective, probability_of_default)` |
| Final validation | `loan_granting.granting` | `final_validation(evaluator, app, offer)` |
| Eligibility gates | `loan_granting.eligibility` | `eligibility_gates` (a `frame_step`); `_evaluate_gates(app_dict)` for direct reuse |
| Fraud handoff handling (03 §4.4's stub contract) | `loan_granting.fraud` | `fraud_handling_path_step` and friends |
| 03's own reason registry | `loan_granting.reasons` | `REGISTRY` (a `credit_core.reason_codes.ReasonCodeRegistry`) |

`solve_term` and `PriceEvaluator` are the two 06/07 want (03 §11 items
11-12): both are plain Python, take no `decider`-specific arguments beyond
a `RateCardIndex`/`CreditLifeIndex` pair, and are already what
`loan_granting/granting.py`'s pipeline step calls internally -- there is no
second copy to drift from. `waterfall._run_waterfall` is the equivalent
entry point for the cap-waterfall *mechanism*; 07 wanting it "over limits
rather than amounts" (03 §11 item 12) would mean a new `REGISTER` (a
different rule set) through the same `_run_waterfall`/`_applicable`/
`_candidate` machinery, not a fork of it.

**The `pipeline.py` name collision.** Every project in this set names its
entry-point module `pipeline.py` (BRIEF: "`pipeline.py` with `build(...)`").
03 imports 02's whole assessment (`loan_granting/affordability.py`), and a
plain `import pipeline` from inside 03's own `pipeline.py` self-imports
(Python caches modules by name, and 03's own module is already mid-import
under that name) instead of reaching 02's file. Worked around with
`importlib`, loading 02's file under the distinct name
`affordability_pipeline`, located via `assessment.__file__` (02's package
name is unique on `PYTHONPATH`). **06, which spec 03 §11 item 11 says wants
both 02's assessment and 03's solve, will hit this same collision twice**
the moment it imports both by name -- worth fixing once, upstream (a naming
convention change, or a documented `importlib` helper in the shared
library), rather than reinventing this workaround in every consumer.

---

## 1. What I built

All eleven stages of spec 03 §5, for Flex Loan (product 10) only, real-time
single-application scoring plus a proven real-time/batch identity (`.score()`
vs `.run()` agree -- `tests/test_pipeline.py::test_batch_and_real_time_agree`,
at a demonstration scale; SCOPE.md explicitly skips the 14 M-record timing):

- **5.1 Eligibility** (`eligibility.py`) -- all 14 gates, every one evaluated
  past the first failure, with a genuine third `not_evaluated` state (not a
  collapsed pass/fail) for a gate whose tested field is null.
- **5.2 Consent and fraud** (`fraud.py`, plus `credit_core.consent` reused
  unmodified) -- the four-way fraud-verdict handling table including the
  800 ms timeout's low-risk bypass. The fraud verdict itself is an *input*
  (project 01's contract, stubbed per 03 §4.4 -- see "What I left out").
- **5.3 Bureau data quality** (`bureau_dq.py`) -- DQ-0..DQ-3, the
  never-a-decline rule, per-product staleness tolerance as a `param()`, the
  no-hit/no-accounts/enquiry-failed three-way distinction.
- **5.4 Scorecard, calibration, grading, four overlays** (`scoring.py`) --
  segment precedence (thin-file/new-to-bank/existing-client) computed for
  real; scored through project 00's one reused scorecard (SCOPE.md keeps
  this project's own scorecard count at one, not four); calibration and
  risk-grade tables reused unmodified from 00; the four overlays (score
  shift, scaling change, odds multiplier, boundary shift) composed in the
  declared order through `credit_core.adjustments`, each with its own
  `<target>_unadjusted` value surviving alongside the adjusted one.
- **5.5 The cap waterfall** (`waterfall.py`) -- 52 rules at real volume
  (13 with the spec's own named conditions, including the uplift
  exception; 39 generated fillers at the declared ownership split), full
  chain per ceiling, per-rule attribution (not-applicable vs.
  evaluated-did-not-bind vs. bound vs. raised vs. declined), the
  authority-bounded uplift restrained by a regulatory-class rule when one
  binds.
- **5.6 Affordability** (`affordability.py`) -- consumed from project 02's
  whole assessment, one call per application, mapped from 03's own
  application shape into 02's.
- **5.7-5.8 Pricing and the solve** (`pricing.py`, `solve.py`) -- the
  full-size generated Flex Loan card (96 x 55 x 12 = 63 360 cells, reused
  from `credit_core.rate_card`), the piecewise initiation fee, the credit
  life premium (with the substitution right), the annuity instalment, and
  the bounded, deterministic, banded-bisection search -- see "How I did the
  bounded solve" below.
- **5.9 Offer-set construction** (`granting.py`) -- all five minimum viable
  offer rules, the 2% dedup tolerance, all three recommendation objectives
  (including `best_expected_value`'s PD/LGD/EAD/funding-cost formula),
  ragged 0-9 offers as parallel lists, a suppression record with reasons
  for every term that produced nothing.
- **5.10 Final validation** (`granting.py`) -- a fresh re-derivation of the
  recommended offer from its own amount and term alone (see "Spec
  problems" for the sense in which this is, and is not, "independent").
- **Outcome resolution** (`outcome.py`) -- combines every stage's signal
  into one `outcome_code`/`referral_queue_code`/ranked reason set with a
  declared precedence.

### What I left out

- **Disclosure outputs (§5.11)** -- explicitly out of scope (SCOPE.md).
- **The 14 M-record batch timing** -- proven at demonstration scale only
  (SCOPE.md).
- **Branch-based short-circuiting.** Spec 03 §5.1 wants eligibility gates
  to stop the flow ("no point scoring a deceased applicant"), but I did
  not build conditional early-exit through the rest of the pipeline
  (fraud/DQ/scoring/waterfall/affordability/solve all run unconditionally
  for every application, matching 00/02's own precedent of no
  `branch`-gated skip paths). `outcome.py` applies the correct precedence
  regardless of what the (in that case meaningless) downstream stages
  computed, so the *answer* is correct; only the *wasted computation* for
  an already-doomed application is not eliminated. Given this slice
  doesn't benchmark latency, I judged this an acceptable simplification
  over the added complexity of branch arms threading placeholder values
  through six more stages -- flagged explicitly rather than left silent.
- **Two live scorecard majors simultaneously (§7.1)** and **the challenger
  scorecard (10% traffic split)** -- SCOPE.md keeps this project to one
  scorecard.
- **The full 45-characteristic scorecard, 4-segment scorecard set, 672-cell
  credit life table, 48 calibration/grade boundaries** -- all consumed
  from project 00 as built there (00's own NOTES.md already documents
  these as "working depth, not full size" for every table except the rate
  card).
- **"Coincident" chain status** (§5.5's "two rules reduced to the same
  value, the later is coincident") -- a candidate that doesn't strictly
  improve on the current ceiling is recorded as `evaluated_did_not_bind`
  regardless of whether it happens to equal the current value exactly; the
  distinction is real but adds a second equality-tracking pass I judged
  not worth its complexity for this slice. Documented in `waterfall.py`.
- **The long-term-loading second table (§6.1)** -- not built; see "Gaps in
  what I consumed" for why.

### How I did the bounded solve

Banded top-down bisection, in plain Python (`solve.py`), not
`decider.loop` -- the reasoning is in that module's own docstring, repeated
briefly here because it's the project's central design decision. Within one
amount band the rate is fixed, and every other input to the instalment
(the piecewise-but-non-decreasing fee, the credit life premium on the
amount financed, the annuity itself) is non-decreasing in amount at a fixed
rate and term -- so instalment is monotone non-decreasing *within* a band,
and a bisection there is valid, even though it is provably invalid *across*
bands (spec 03 §5.8's own worked R50 000 example). The search:

1. Tries the domain's top candidate first (trivially the true maximum if
   affordable -- one evaluation, the common case).
2. Otherwise walks amount bands top-down. Each band's cheapest point (its
   bottom) is checked once; if even that is unaffordable the whole band is
   skipped without evaluating anything else in it, and the *next* band to
   look at is chosen by a proportional estimate from the observed
   instalment (not just "one band down"), which converges far faster than
   linear band-stepping on the common case of a wide domain with no nearby
   inversion (spec: only 41 of the card's cells are declared inversions).
   If the band's bottom *is* affordable, a bisection between it and the
   band's (already known infeasible) top finds that band's true maximum
   exactly, in `O(log(band width))` evaluations.
3. Every evaluation counts against the 24-per-term ceiling; reaching it
   without a *proven* maximum returns "no feasible amount, BIND-EXH"
   rather than a plausible guess -- including mid-bisection, where a
   feasible lower bound that hasn't been proven the band's own maximum is
   correctly *not* returned as an answer
   (`tests/test_solve.py::test_evaluation_ceiling_is_never_exceeded_and_refers_when_reached`
   caught this exact bug during development: an earlier version returned
   the unproven lower bound).

Proven, not just asserted: `tests/test_solve.py` names the band-edge
inversion case explicitly (a hand-built card, since 00's own generated
card has none -- see "Gaps in what I consumed"), asserts agreement with
brute-force exhaustive search over that case and five further cases on the
real 63 360-cell card, asserts determinism (five repeated calls, identical
answer), and asserts the evaluation ceiling is a hard bound.

---

## 2. Reuse

**From project 00 (`credit_core`), unmodified:** `rate_card.generate_flex_loan_card`
/ `rate_card_rate` (the full-size card, and the percentage-to-fraction
conversion), `fees.initiation_fee`/`monthly_service_fee`/`*_capped`,
`credit_life.build_credit_life_table`/`credit_life_premium`/`*_cap_applied`,
`instalment.*` (forward and inverse), `rounding.*`, `scorecard.build_scorecard`
/`adverse_action_codes`/`VARIABLES`, `calibration.build_calibration_table`/
`probability_of_default`, `risk_grade.build_risk_grade_table`/`risk_grade_output`,
`consent.consent_verdict_step`, `adjustments.AdjustmentRegister`/`Adjustment`/
`AdjustmentEffect` (all four overlay points), `reason_codes.ReasonCodeRegistry`
(03's own registry, same mechanism), `vocab.FraudVerdictCode`/`SEGMENT_*`.

**From project 02 (`assessment`/`pipeline.py`):** the whole assessment,
called once per application through `loan_granting/affordability.py`'s
adapter (see "What I publish" for the module-name workaround this needed).

**From `decider`'s built-ins:** `DecisionTableConfig` (reused inside 00's
functions, not rebuilt); `ScorecardConfig` (ditto); `frame_step` for every
stage whose per-record work is genuinely row-shaped (eligibility's 14-gate
vector, the cap waterfall's 52-rule register, the solve/offer-set/final-
validation stage, the affordability adapter); `param()`/`missing_as()`
throughout; `.relabel()` for the four overlay points' output-column naming
(avoiding a dag collision on `adjustment_set_id`/`adjustments_applied` --
00's own "column-naming convention" problem, recurring here per-overlay
rather than per-table); `flow(...)` for the one waterfall-shaped pair
(`combined_decline_reason_codes` -> `REGISTRY.resolve_step()`, both writing
`decline_reason_codes`, exactly 02's own `_reasons_unit()` pattern).

**Deliberately not used: `decider.loop`.** Considered for both the solve
and the cap waterfall (both are naturally "run N bounded iterations,
carrying state" problems, which is exactly `loop`'s shape). Both ended up
as plain Python inside one `frame_step` instead -- see `solve.py`'s and
`waterfall.py`'s docstrings for the specific reasoning (multi-phase
per-iteration logic that doesn't reduce to one condition/one body without
an unreasonable number of carried columns, against a correctness bar with
zero tolerance for a wrong answer). I read `loop`'s docstring and its
repayment-schedule example closely before deciding against it; it is a
good fit for a *uniform* bounded iteration, and neither of my two
candidates is uniform enough. Recorded here as a design-fit finding, not a
framework defect.

**Written from scratch:** everything else in `loan_granting/` -- the 14
eligibility gates as one function (see "Gaps in what I consumed" for why
this isn't built on 00's thinner `core.eligibility`), the fraud-handling
table, the bureau DQ verdict, segment assignment, the cap register (13
named rules + 39 generated fillers) and its waterfall engine, the
`PriceEvaluator`/`RateCardIndex`/`CreditLifeIndex` pricing machinery, the
banded-bisection solve, offer-set construction, final validation, and
outcome resolution. DEPS.md's own boundary (00 owns the units and thin
frozen interfaces, 02 owns the affordability assessment, 03 owns
everything specific to granting and pricing) is exactly this split.

---

## 3. Gaps in what I consumed

- **00's generated rate card has zero band-edge inversions.** `generate_flex_loan_card`'s
  rate formula (`base_rate + (grade-1)*1.5 + ai*0.01 + ti*0.02`) is strictly
  increasing in the amount-band index `ai`, so it never produces the
  "Treasury rewards crossing R50 000 with 150 basis points" kind of cell
  spec 03 §5.8 and §6.1 (41 declared inversions) describe and require the
  search to tolerate. I could not edit 00's generator (BRIEF: never edit
  another project's directory), so the non-monotonicity regression test
  (§10 item 6) runs against a small hand-built fixture instead of the real
  card (`tests/test_solve.py::test_band_edge_non_monotonicity_finds_the_true_maximum`).
  The solve's algorithm is exercised against the real card too (agreement
  with exhaustive search, five cases), just not for the inversion property
  specifically, since the real card has none to exercise it with.
- **No long-term-loading second table.** Spec 03 §6.1 describes 55 term
  columns for 6-60 months plus a *separate* 2x12 table for 61-84 (a
  deliberate Treasury-cadence distinction, "must remain visible as such").
  00's generator instead builds all 55 columns spanning 6-84 directly in
  one table. Every term 6-84 is priced from one direct cell read in this
  project; the two-table seam spec 03 asks to keep visible doesn't exist
  to keep visible, because it was never built in 00.
- **`core.eligibility`'s shape doesn't fit 03 §5.1.** 00's interface
  returns a flat fired-reason list with no per-gate identity and no
  `not_evaluated` state (00's own NOTES.md calls it "thin, frozen").
  03 §5.1 explicitly requires "a per-gate verdict vector of 14 entries"
  including the third state. Wrapping and extending 00's version would
  have meant re-deriving most of its logic anyway (9 of the 14 gates
  overlap in substance but not shape), so I wrote all 14 as one function
  instead of forking 00's five and adding nine more beside them.
- **00's `_TIGHTEN_RULES` overlay-kind registry is closed.** Four overlay
  kinds are declared with a tighten direction (`score_shift`,
  `odds_multiplier`, `rate_addon`, `cap_adjustment`, `buffer_adjustment`);
  spec 03 §5.4.1's "boundary shift" and "scaling change" aren't among
  them, and I can't add to 00's dict without editing 00. Scaling change
  (§5.4.1 overlay 2) has a genuinely ambiguous tighten direction anyway
  (flattening a curve isn't uniformly tighter or looser) so `tighten_only=False`
  is arguably correct regardless. Boundary shift (§5.4.1 overlay 4) is
  implemented as a second, distinct `odds_multiplier`-kind overlay on a
  *different* target (`probability_of_default_for_grading`, tested
  against the unmoved risk-grade boundary table rather than moving the
  boundary itself) -- see `scoring.py`'s docstring for why this is a
  faithful mechanical equivalent for a monotone step function, not a
  fudge.

---

## 4. Framework friction

### 4.1 `decider build`'s warm-up still can't handle `list`/`dict`/`date` inputs

Exactly 00's NOTES.md finding, confirmed again here (`TypeError: 'float'
object is not iterable`, from the synthesiser's float-1.0 fallback for
every non-`bool`/`int`/`str`/`bytes` type). Worked around the same way:
`inference.py` replaces `decider.serving.handler._warm` at import time to
warm from `sample_request.json`. Worth restating because this is the third
of three example projects to hit it (00, 02, 03), across three different
implementer sessions -- it is not an edge case.

### 4.2 A `frame_step`'s output columns cannot be renamed with `.relabel(writes=...)`

```
IRError: loan_granting/eligibility_gates: relabel can't rename a frame
step's columns; rename them in the function
```

New finding, not in 00/02's NOTES. `Step.relabel(writes=...)` works for a
`step()`-wrapped scalar function, a `DecisionTableConfig`, a
`ScorecardConfig` -- every relabelling example in 00/02's own code is one
of those. It raises for a `frame_step`. This mattered here because
`eligibility_gates` originally wrote `decline_reason_codes`, the same name
`outcome.py`'s combining step also needs to produce (for the *final*,
ranked set) -- the natural fix elsewhere in this codebase (`.relabel()` at
the call site) isn't available for a `frame_step`, so the column has to be
named correctly *inside the function* instead, which means the producer
and every consumer of a `frame_step`'s writes have to agree on names
up front, with no boundary-level fix-up available later. Worth a line in
`frame_step`'s own docstring, since nothing there flags this asymmetry
with `step()`.

### 4.3 A polars list column mixing `int` and `float` values across rows crashes with a misleading message

```
TypeError: unexpected value while building Series of type Float64; found
value of type Int64: 9

Hint: Try setting `strict=False` to allow passing data with mixed types.
```

New finding. `waterfall.py`'s per-rule "value before"/"value after" audit
column mixes amount/term (float ceilings) and grade (an int ceiling)
across the 52 rows of one application's chain, because most rules touch
one ceiling but the arrears rule touches two and the column reports
whichever a rule happens to have touched last. Polars infers the column's
dtype from an early value (here, float) and then raises on the first `int`
it meets, rather than either promoting to float or reporting which *row*
and which *field* introduced the mismatch -- for a 52-element list buried
inside one column of a one-row batch, the message gives no hint about
where to look. The fix (cast to `float()` at both append sites) was a
two-line change once found, but finding it meant bisecting which of ~20
list-typed outputs across the whole pipeline was the culprit. Worth a
docstring note on `frame_step` (or `DataFrame`-returning steps generally):
a list-typed output column must be internally homogeneous in dtype across
every row of the batch, cast explicitly if the values naturally come from
different-typed sources.

### 4.4 Reordering an unrelated, already-imported name changes whether a column is classified as a pipeline input

The strangest and most concerning finding, and the reason this section is
longer than the others. Building the exact same pipeline object, in the
same process, and immediately checking `Engine().bind(pipeline, ...)._produced`
(the set `_check_shadowing` uses to decide whether an input column is
"really" produced by the pipeline) gave a *different answer* --
`'accounts_in_arrears_count' in exe._produced` was `False` in one script
and `True` in another -- depending on nothing more than whether `from
decider import Engine` appeared *before* or *after* `pipeline.build(...)`
in the calling script, even though `decider`'s own `__init__.py` already
imports `Engine` eagerly (confirmed by reading it), so that statement is a
no-op regardless of where it sits. Reproduced consistently (5/5 each way)
across two minimal scripts differing only in that one line's position. I
could not find the root cause without reading deeper into
`decider.engine.wiring.resolve` than the BRIEF's "internals only if stuck"
licence covers for the time this slice had -- flagging it here with an
exact, minimal repro instead. **This did not reproduce through the actual
`decider build` / `decider serve` CLI path** (confirmed: `decider build`
succeeded cleanly, repeatedly, once `inference.py`'s warm-up fix was in
place), so it did not block this project's mandatory verification -- but
given spec 09 §5.15 item 3's explicit requirement ("no dependence on
iteration order over an unordered collection"), a framework-level bug
where *unrelated import order* changes plan-input classification is worth
someone with internals context looking at directly, before a project that
*does* hit it in its served path discovers it in production.

### Smaller things

- `Engine().score(record)` with no `params` argument uses every step's own
  declared defaults -- convenient for tests, worth stating plainly since
  every worked example I found in 00/02's own code passed `params`
  explicitly and I had to check `Executable.score`'s signature to confirm
  the default path exists.
- `frame_step` has no `param()` injection point (confirmed by 00/02's own
  precedent: every one of their `frame_step`s hardcodes its tunables as
  module constants rather than `param()`s). This project's solve and cap
  waterfall are both `frame_step`s and both need policy-owned tunables
  (the evaluation ceiling, the statutory ceiling, the recommendation
  objective) to be changeable without a redeploy per spec 03 §6.2 --
  worked around by computing those three as ordinary scalar `step()`s
  (which *do* support `param()`) and having the `frame_step` read them as
  ordinary input columns `dag` wires in first. This works, and is
  arguably a reasonable general pattern, but it means "which of this
  project's tunables live in a `param()` vs. a hardcoded constant" tracks
  "was it needed inside a `frame_step`" rather than any policy-relevant
  distinction -- an accident of implementation shape a policy owner
  shouldn't have to know about.

### What worked well

`.relabel()` for the overlay stack's per-target output naming was exactly
the right tool once I knew to reach for it (see 00's own NOTES.md on the
same pattern for tables). `AdjustmentRegister.apply_stack_step` composed
cleanly four times over with no changes needed. `pipeline.parameters().defaults()`
again produced a complete, correct `params.json` for a ~30-node pipeline
with no manual authoring. `decider build`'s cold-start indexing genuinely
does make the 63 360-cell card's repeated in-search lookups fast (the
`RateCardIndex` bisection lookup is sub-millisecond; the whole solve, all
nine terms, ran in under a millisecond in isolated testing).

---

## 5. Spec problems

- **02's shape (b) makes `marginal` effectively unreachable from 03.** 03
  §5.6 says an `affordability_verdict_code` of *marginal* "continues but
  restricts offers to terms of 24 months or less". 03 calls 02 with no
  `proposed_instalment` (shape (b), "capacity only" -- there is no single
  instalment to test before the solve has run), and 02's
  `affordability_verdict_code` function (`credit_core/affordability.py`)
  only returns `marginal` when a `proposed_instalment` *is* supplied; under
  shape (b) it can only be `pass`, `fail` (zero capacity) or
  `indeterminate` (no net income). So the §5.6 clause about marginal is
  written for an interaction the calling convention it also specifies
  (call 02 once, before the solve, for a single ceiling) cannot produce.
  Either 03 is meant to call 02 a second time, with the *recommended*
  offer's instalment, after the solve -- which would make affordability a
  two-pass capability and isn't stated anywhere -- or `marginal`'s clause
  describes 02's shape (a) interaction (not (b)), misplaced in 03's
  section. I implemented the pass-through (03 reads whatever
  `affordability_verdict_code` shape (b) returns and reacts to `marginal`
  if it ever appears), which is correct as written but, per the analysis
  above, dead code given how 03 actually calls 02.
- **Final validation's independence is under-specified, and spec 03 says
  so itself.** §5.10 requires the recommended offer to be "re-derived end
  to end from its own amount and term, carrying nothing forward from the
  solve" and separately §13 Q14 asks "is that genuinely independent of the
  stages it validates, rather than re-running the same code and agreeing
  with itself?" -- posing it as an open question rather than resolving it.
  This project's `final_validation` (`granting.py`) satisfies the literal
  requirement (fresh recomputation from `amount`/`term` alone, no solve
  state reused) but uses the *same* `PriceEvaluator` the solve itself
  calls, because there is exactly one pricing implementation in this
  project (deliberately -- see the solve's own docstring on why "the same
  implementation" is a property, not an oversight, elsewhere in this
  spec). A genuinely second, independently-maintained pricing
  implementation would satisfy §13 Q14's stronger reading but creates
  exactly the drift risk 09 §5.14.3 warns against for the overlay-stack
  case ("it will agree at first and diverge silently"). I judged the
  weaker, spec-explicit reading correct; a reviewer could reasonably
  disagree, which is why spec 03 poses it as a question rather than a
  requirement.
- **CAP-0420's owner doesn't fit the stated four-way split.** §5.5's
  representative rule table lists CAP-0420 under owner "Credit
  Committee", but the paragraph immediately above states the register's
  ownership "split four ways: Credit Risk Policy (31), Unsecured Lending
  Product (12), Credit Systems (6), Financial Crime (3)" -- 52 total, no
  fifth class. I folded CAP-0420 into "Unsecured Lending Product"'s count
  (campaigns are a product concern; "Credit Committee" reads as the
  *approval authority*, recorded separately as `uplift_authority_reference`,
  not the owning team) to keep the stated total consistent, but the spec
  itself doesn't resolve which reading is intended.
- **§5.9's total cost ratio and in duplum rules can contradict each
  other's stated purpose on genuinely cheap, short offers.** Both are
  meant to catch different failure shapes (the worked "6-month
  suppression" example is explicit that TCR alone would miss the flat-fee
  problem), but neither rule references the other, so it's easy to
  construct an offer that trips one but not the other for reasons that
  have nothing to do with either rule's stated intent (a short term with
  a large capitalised fee can fail in-duplum while passing TCR comfortably,
  independent of whether the fee itself is reasonable). Not a defect in
  my implementation, just a note that these two rules' independence is a
  designed property worth a shared regression fixture, which this
  slice's test suite doesn't build (SCOPE.md scope).

---

## 6. What I would do next

1. Push the four-overlay closed-registry gap (`_TIGHTEN_RULES`) upstream
   to project 00, so "boundary shift" and "scaling change" get their own
   declared tighten directions instead of reusing/opting out of an
   existing kind.
2. Ask 00 to add at least a handful of deliberate band-edge inversions to
   `generate_flex_loan_card`, so this project's (and 06's, and 07's)
   non-monotonicity regression tests can run against the real, shared
   card instead of a fixture built just for the test.
3. Chase the import-order-dependent plan-input classification (4.4) to a
   root cause in `decider.engine.wiring.resolve`, given what spec 09
   §5.15 item 3 requires and how easy the repro is once known.
4. A genuine `param()` injection point for `frame_step`, so a project's
   choice of "which tunables need a redeploy to change" tracks policy
   ownership instead of "was this value needed inside row-shaped Python."
5. Widen the cap register's 39 filler rules from narrow-condition
   generated data into a second batch of real, named policy conditions
   (purpose-code exclusions, sector-specific caps), the way project 01
   generated its ~520 rules at real semantic volume rather than
   structural volume alone.
6. A genuine two-call affordability interaction (capacity ceiling before
   the solve, verdict-against-the-actual-offer after it) if the `marginal`
   clause in spec 03 §5.6 is confirmed to mean what it says rather than
   being dead per the "Spec problems" analysis above.
