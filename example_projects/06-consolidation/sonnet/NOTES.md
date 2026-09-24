# NOTES

## What I publish

For project 11 (DEPS.md: "11 consumes real references to ... [06] Obligation
inventory and settleability | [06] §5.2, §5.3 | L5"):

| Capability | Module | Entry point |
|---|---|---|
| Settleability classification | `consolidation.inventory` | `classify_settleability(accounts, decision_date, client_nominated_settle, client_excluded_settle, recently_opened_months=3) -> list[dict]` -- plain Python, one classification per account |
| Settlement amount derivation | `consolidation.settlement` | `derive_settlement_amounts(accounts, settleable_account_refs, ...) -> list[dict]`, and `settlement_buffer(settlement_total) -> float` |
| The concession catalogue | `consolidation.concessions` | `build_concession_catalogue() -> DecisionTableConfig`, `authority_level_for_npv_cost(npv_cost, repeat_concession) -> int` |
| Obligation re-derivation over a reduced inventory | `consolidation.scenario_eval` | `rederive_obligations(accounts, settlement_set, settleability_by_ref) -> dict` (thin wrapper over `credit_core.obligations._process`) |
| Product 11 pricing | `consolidation.pricing11` | `build_product11_evaluator(statutory_ceiling, rows=None) -> PriceEvaluator`, `solve_required_advance(...)` |
| Product 20 pricing | `consolidation.pricing20` | `price_product20(card_index, transferred_amount, risk_grade, promo_duration_months) -> Product20PricingResult` |
| The objective and selection mechanism | `consolidation.objective` | `resolve_weights`, `component_scores`, `rank_and_select`, `shadow_best` |

`inventory.classify_settleability` and `settlement.derive_settlement_amounts`
are plain Python, no `decider`-specific arguments, callable with no pipeline --
the same "solve/price as a plain function the search calls many times" pattern
project 03's NOTES.md documents for `solve_term`/`PriceEvaluator.evaluate`, and
the two entry points 06 §2's own Q4 table says collections (08) and limit
management (07) also want ("Settleability classification, settlement amount
derivation and the before-and-after comparison are this project's own, but
collections and limit management both want the first two").

---

## 1. What I built

The SCOPE.md slice: intake and eligibility (§5.1, all 7 `CON-ELIG` gates),
obligation inventory and settleability (§5.2, all 9 `settleability_code`
values, first-match-wins, per-account rule/cell attribution), settlement
amount derivation (§5.3, the full component breakdown: capital, accrued
interest, fees, early settlement charge, security release cost, rebate),
baseline assessment with the short-circuit to project 03 (§5.4, all six
short-circuit conditions individually recorded), candidate scenario
generation under the declared budget (§5.5, all 8 ordering rules H1-H8 as
prefix generators, the empty set, the client's nominated set, the full
settleable set, a deterministic total order, a recorded termination cause),
per-scenario evaluation (§5.6, re-derived obligations, project 02 called "in
scenario mode", the circular required-advance solve, pricing for two
products), policy interventions (§5.7, 10 of the 14 evaluated as checks --
the other 4 are enforced earlier, at settleability or eligibility, and
recorded `not_applicable` there rather than silently dropped), the objective
and selection (§5.8, all 5 objectives, a blend weight vector, one re-weight
overlay, tie-break, the 2% indifference band, distinct top three, the OBJ-05
shadow result), and the concession catalogue table (§5.9, table only, per
SCOPE.md).

Two product subflows, as scoped: **product 11** (Flex Loan Consolidation --
its own 12x12x12 rate card, the margin-adjustment-by-external-proportion grid,
the circular fixed-point solve for the capitalised advance) and **product 20**
(Everyday Card balance transfer -- the promotional-plus-reversion rate card,
the mandatory-paydown amortisation schedule, the stressed-affordability test
against the reversion rate over 36 months, never the promotional minimum).

Every scenario evaluated -- viable or not -- is recorded with a
`rejection_reason_code` set, actual-versus-threshold values (via
`interventions.InterventionResult`), and the overlay identity when an overlay
changed a threshold (§5.7 requirement 4; §10 acceptance item 4). The evidence
contract items I specifically targeted (09 §5.15): stable decision id
(passthrough from the caller, per addendum convention), no reliance on
"today" (every date-sensitive read goes through `decision_date`, verified by
`tests/test_pipeline.py::test_overlay_does_not_apply_before_its_effective_from_date`),
recorded table/cell versions (`settleability_cell_ids`, rate cell ids, rate
card versions), the overlay stack recorded with unadjusted values alongside
adjusted (`objective_overlay_ids`, each intervention's `overlay_id`),
declared reason codes from a versioned registry with a ranked primary, and
evaluation recorded for every intervention, not only the failures
(`interventions.evaluate_interventions` always returns all 14, `PASSED` /
`FAILED` / `NOT_APPLICABLE`).

### What I left out

- **Products 30 and 40** (SCOPE.md: skip valuation and product 40 entirely).
  `search.route_products` recognises them in the vocabulary (`vocab.
  PRODUCT_DRIVE_REFINANCE`, `PRODUCT_HOME_FURTHER_ADVANCE`) but never routes
  a scenario to them -- no collateral/valuation/bond data is collected this
  slice.
- **The restructure variant (§5.9)** beyond the concession catalogue table.
  No option search over concessions, no NPV-cost/authority-level computation
  on a live scenario, no stressed-affordability test for restructure, no
  debt-review-path output package. `authority_level_for_npv_cost` exists as a
  plain function (the mechanism), unexercised by any pipeline path.
- **Batch identification (§5.11)** -- not built at all, per SCOPE.md.
- **A consolidation-specific scorecard (§5.6.4).** `risk_grade` is taken as a
  request **input**, not re-derived. SCOPE.md's own description of this slice
  ("candidate generation... per-scenario evaluation... configurable
  objective... two product subflows") does not list a scorecard among the
  required parts, and an existing client already carries a grade from an
  earlier assessment -- reusing 00's `core.scorecard`/`core.risk_grade` units
  the way 00 built them would have added a full stage for no proof the
  bounded search needs. Flagged here rather than left silent.
- **New money as a search dimension.** §5.5's candidate space is
  (settlement set x product x term); I do not additionally bracket new-money
  amounts per scenario. `new_money` is derived deterministically per scenario
  (clamped to what CON-INT-08's *base* policy would allow for that
  settlement set -- see "Spec problems" below for why a flat request amount
  broke every scenario at first).
- **Joint applicants.** Single applicant only, throughout (project 02
  supports joint; this project always sends `is_joint_application=False`
  into it). SCOPE.md does not call out joint as required for this slice, and
  02's own household-framing stage is exactly what would be duplicated by
  building it here too.
- **The full before-and-after comparison and execution package (§5.10).**
  The underlying data is all present in the output (baseline measures,
  chosen scenario's full pricing, settlement amounts, runners-up, rejected
  scenarios) but not assembled into the prescribed client-facing document.
- **Products 30/40's own rate cards, the vehicle valuation table, the
  conveyancing tariff.** Not built (SCOPE.md).

---

## 2. Reuse

**From project 00 (`credit_core`), unmodified, called as plain Python inside
the per-scenario loop (not through a pipeline) for speed:**
`obligations._process` (re-deriving obligations over the reduced inventory,
hundreds of times per assessment -- exactly 06 §4.4's "the single most
demanding reuse claim in the slate", now hit for real: `scenario_eval.
rederive_obligations` calls it once per evaluated scenario, so 120-400 times
in one assessment); `credit_core.instalment.instalment_before_fees`;
`credit_core.rounding.round_advance`/`round_instalment`;
`credit_core.adjustments.AdjustmentRegister` (this project's own two overlay
kinds: the anti-harm threshold tightening and the objective re-weight);
`credit_core.reason_codes.ReasonCodeRegistry` (the rejection registry and the
decline registry); `credit_core.evidence.cell_id` (every table this project
builds).

**From project 02 (`assessment`), called two ways:**

1. **Once per assessment, through its real pipeline** (`Engine().score()`,
   via `consolidation.reuse.load_project02_pipeline`): `build_evidence_and_
   baseline` calls 02's `build()` exactly once, with no `proposed_instalment`
   (shape (b), capacity only, over the full unreduced inventory), and that
   one call supplies every invariant figure §5.6.1 requires (`net_monthly_
   income`, `living_expenses`, `evidence_sufficiency_code`) plus the baseline
   `existing_obligations`/`discretionary_income`/`max_affordable_instalment`
   §5.4 needs. This is the "affordability once per assessment" reuse DEPS.md
   describes.
2. **Hundreds of times per assessment, as plain Python, not through the
   pipeline:** `assessment.capacity.max_affordable_instalment_before_overlay`
   and `assessment.verdict.affordability_verdict_code` are already plain
   functions (02's own NOTES.md: "the only stage project 06's 400-scenario
   search needs to re-run per scenario"), called directly once per scenario.
   `assessment.capacity._buffer_pct` and `assessment.capacity._RESIDUAL_
   BANDS`/`_RESIDUAL_AMOUNTS` are **private** module members, called
   directly too -- see "Gaps in what I consumed" below for why, and why
   that's a real gap rather than a style choice.

**From project 03 (`loan_granting`):**

1. **The pricing mechanism, not the solve.** `loan_granting.pricing.
   PriceEvaluator`/`RateCardIndex`/`CreditLifeIndex` are reused completely
   unchanged, built over this project's *own* rate card (`rate_cards.
   generate_product11_card`) -- exactly DEPS.md's "reused four times over
   inputs project 03 never contemplated" (four, because 06's own product 11
   is itself a second product on top of 03's original product 10). I did
   **not** reuse `loan_granting.solve.solve_term` (the max-affordable-amount
   bisection): in consolidation the advance amount is not a free variable to
   maximise, it is the settlement total plus new money plus its own
   capitalised fees, converging via a **fixed-point** iteration (6
   iterations, R1 tolerance, §5.6.2 item 3) rather than a **search for a
   maximum**. `pricing11.solve_required_advance` implements that fixed
   point, structurally the same *kind* of circularity as 03's solve
   (amount<->fee<->rate-band) but a different problem, not a different
   implementation of the same one. This also sidesteps a real reuse blocker
   in `solve_term` itself -- see "Framework friction" #3.
2. **The full pipeline, once, for the short-circuit.** `baseline.plain_
   grant_via_project03` calls `loan_granting.pipeline.build(rate_card_flex_
   loan)` through `Engine().score()`, exactly as `decider serve` would, when
   §5.4's six-condition test says consolidation is unnecessary. This is a
   real call into 03's eleven-stage pipeline, not a stub -- see "What I left
   out" and "Gaps in what I consumed" for the field-mapping cost of that.
3. **The `pipeline.py` name-collision workaround, generalised.** 03's own
   `loan_granting/affordability.py` predicted this exact problem ("06, which
   wants both 02's assessment and 03's solve, will hit this same collision
   twice"). `consolidation/reuse.py` generalises 03's `importlib` workaround
   to load *either* sibling project's `pipeline.py` by its own unique
   top-level package name (`assessment` for 02, `loan_granting` for 03), so
   06 pays the cost once, in one shared module, rather than twice inline.

**From `decider`'s built-ins:** `DecisionTableConfig` (the concession
catalogue, product 11's rate card); `frame_step` (settleability, settlement
derivation, eligibility gates, and the whole search/evaluation/objective
orchestration -- see "Framework friction" #4 for why the search itself could
not be `dag`/`branch`/`loop`); `step` (the final outcome-resolution step);
`param()` (statutory ceiling equivalents, reused via 00/02/03's own
`param()`-guarded functions); `credit_core.adjustments.AdjustmentRegister.
apply_stack` (called directly, not through a pipeline step, for both this
project's overlay kinds).

**Written from scratch:** settleability classification and settlement amount
derivation (this project's own, per DEPS.md's Q4 note that 07/08 want them
too); the eligibility gates; the baseline short-circuit evaluation; the
candidate-generation ordering rules (H1-H8) and the budget/termination
machinery; the per-scenario evaluation orchestration tying obligations,
affordability, pricing and interventions together; the 10 evaluated policy
interventions; product 20's pricing (nothing in 03 has this shape: no term,
no instalment, two rates and a date); the objective/blend/tie-break/
indifference-band/shadow mechanism; the concession catalogue table.

---

## 3. Gaps in what I consumed

- **Project 02 does not publish its two capacity tables (the buffer grid,
  the residual floor) as plain callables, only as `DecisionTableConfig`
  documents.** Every other unit in 02/00 (`income`, `deductions`,
  `expense_norms`, `obligations`, `affordability`, `capacity.max_affordable_
  instalment_before_overlay`, `verdict.affordability_verdict_code`) is a
  plain function a fast loop can call directly. `capacity._buffer_pct` and
  `capacity._RESIDUAL_BANDS`/`_RESIDUAL_AMOUNTS` are not -- they exist only
  as private module members backing `build_buffer_grid()`/`build_residual_
  floor_table()`, which return `DecisionTableConfig` objects meant to be run
  through `decider`'s table-match engine inside a bound pipeline. Running
  either table through the engine 120-400 times per assessment, inside a
  900 ms budget, was not viable (confirmed by the same per-call overhead
  project 03's NOTES.md measured as too slow for its own, far cheaper,
  solve). I reached into the private implementation directly rather than
  either (a) re-deriving the same buffer/floor numbers a second time in this
  project (duplicates the source of truth, the exact defect 02's own spec
  warns against), or (b) eating the Engine-per-scenario cost and blowing the
  budget. **This is a real gap, not a style choice**: a future refactor of
  02's private naming breaks 06 silently, with no test on 02's side to catch
  it. DEPS.md's "06 calls 02 in scenario mode" reads as if the whole capacity
  stage is cheap plain Python; only about half of it actually is.
- **Project 03's `solve_term` bakes `PRODUCT_MIN_AMOUNT`/`PRODUCT_MAX_
  AMOUNT` in as module-level constants** (`loan_granting/solve.py`), not
  parameters, even though 03's own NOTES.md says 06 and 07 want to reuse the
  solve "over inputs project 03 never contemplated". Product 11's bounds
  (R10 000-R500 000) differ from product 10's (R2 000-R500 000); calling
  `solve_term` unchanged for product 11 would silently consider amounts down
  to R2 000, below product 11's real floor. I worked around this by not
  reusing `solve_term` at all (see "Reuse" #1 above -- 06's own problem
  shape does not need a maximum-amount search regardless), so this gap never
  bites *this* project, but it will bite any future consumer of `solve_term`
  itself whose product bounds differ from product 10's, exactly as 03's own
  NOTES.md's own dependency table predicts for 07.
- **Project 03's request shape carries ~50 fields** (scorecard
  characteristics, fraud/DQ inputs, exclusion lists, employer/residency
  detail) that an *existing* consolidation client, by this flow's own
  construction (§5.1: "existing client only"), does not need re-verified.
  `baseline._to_03_request` defaults every field 06 does not itself carry to
  a benign pass value. This is a legitimate scope simplification for a
  short-circuit hand-off (03's own eligibility/DQ/fraud stages were already
  satisfied when this client was originated), not a defect in 03, but it
  does mean the short-circuit path is not exercising 03's own gates the way
  a genuinely new application would -- flagged, not hidden.

---

## 4. Framework friction

### 4.1 A `list[struct]` column where two records have different key sets crashes with a misleading numeric-type error, not a schema error

The most consequential finding this project made. `decider build`/`Engine.
score()` converts a single-record request into a one-row Polars `DataFrame`
and then, for at least the `accounts` top-level `list[struct]` column, calls
`Series.to_numpy()` directly (`decider/engine/run/state.py:58`,
`from_series`) rather than through the typed arrow-import path every other
column in this codebase goes through cleanly. If every element of that list
carries the identical set of struct keys, this works. If even one account
dict in one request is missing a key every other account (anywhere in the
same process, not even necessarily the same request) carries, the *whole*
column falls back to a raw numpy conversion of a struct with heterogeneous
field types, and numpy's `vstack` cannot promote `DateTime64` against
`Float64`:

```
pyo3_runtime.PanicException: called `Result::unwrap()` on an `Err` value:
PyErr { type: <class 'numpy.exceptions.DTypePromotionError'>, value:
DTypePromotionError("The DType <class 'numpy.dtypes.Float64DType'> could
not be promoted by <class 'numpy.dtypes.DateTime64DType'>. ...") }
```

This is a Rust panic surfacing as a raw `pyo3_runtime.PanicException`, not a
`decider` exception -- no `try`/`except` in application code can catch it
usefully, and the message names numpy dtypes, not the missing field. I hit
this twice while building `tests/test_pipeline.py`'s synthetic 18-account
client: once because that client's accounts omitted `quotation_amount`
(present on every account in `sample_request.json`, absent from the test's
own hand-built dicts), and once (in an earlier iteration, before I settled
on a sentinel-not-null convention) because `security_type_code` was `None`
on every one of the 18 accounts with no non-null example anywhere in the
request to establish the field's type from. Both are instances of the same
underlying rule, which is nowhere in `decider`'s docs: **every element of a
`list[struct]` input column must carry an identical, fully-typed key set, in
every request this process will ever score.** Combined with 00/02/03's own
documented findings (a lone `None` scalar, an empty ragged list, both crash
the same way), the practical rule for any consumer sending nested lists of
records is: build every struct-shaped record through one shared constructor,
never a hand-rolled dict literal per call site, and never send `null` for an
optional field -- use a sentinel instead (`SERVE.md`'s own convention:
`"2000-01-01"` for "no such date", `0` for "no code", `""` for "no
reference").

### 4.2 `frame_step` cannot take `param()` overrides, and that blocks 09 §5.14.3's "stack disabled, same implementation" requirement for a search-shaped flow

00's own NOTES.md documents that a `frame_step`'s solve/waterfall internals
can't pick up a `param()`-declared override the way a scalar `step()` can;
03's `pricing.py` docstring names the same limit for its rate-card defaults.
This project hits the same wall at a higher stakes level: 09 §5.14.3
requires every flow to be "runnable with its overlays off, through the same
implementation" -- for every *other* overlay in this codebase (00's score
adjustments, 02's capacity buffer, this project's own anti-harm threshold
and objective re-weight), that's a `param()` flip
(`adjustment_stack_enabled`) a scalar step reads. This project's *entire*
search (`orchestration.py`) is necessarily one `frame_step` (see #4 below
for why), so there is no scalar-step seam to attach that `param()` to, and
no way for a caller to flip the stack off for this project's own search
short of adding a second, explicit function argument threaded all the way
from `pipeline.build()` down through `orchestration._run` to
`interventions.evaluate_interventions`/`objective.resolve_weights` -- which
`evaluate_interventions`/`resolve_weights` already both support
(`stack_enabled=True`), but which nothing in `pipeline.py`'s `build()`
signature currently exposes as a *runtime* toggle, because a `frame_step`'s
closure captures it at *bind* time, not per-request. A caller who wants the
stack-off comparison for this project's two overlay kinds today has to build
a second `Engine` bound with `stack_enabled=False` baked in, not flip one
flag on one running pipeline -- the opposite of "the same implementation".

### 4.3 `RateCardIndex`/`PriceEvaluator` generalise cleanly across products; `solve_term` does not, because of module-level constants

Documented above ("Gaps in what I consumed"), repeated here because it's a
framework-shape finding, not only a 03-specific one: a "solve" built as a
plain Python function with its own domain bounds as **module constants**
(`loan_granting/solve.py`'s `PRODUCT_MIN_AMOUNT`/`PRODUCT_MAX_AMOUNT`) looks
identical, from the outside, to one built with them as **parameters**
(`PriceEvaluator.__init__`'s `statutory_ceiling`, `RateCardIndex`'s rows) --
until a second product tries to reuse it. Nothing in `decider` forces either
choice; the difference only shows up the moment DEPS.md's promised second
consumer actually arrives. Worth a documented convention across this
codebase's "published for reuse" modules: **every bound a second product
might supply differently belongs in the constructor/call signature, never a
module constant**, even when there is currently only one caller.

### 4.4 A `pipeline.py`-named entry point in every project makes cross-project imports collide by construction

Not a new finding (03's NOTES.md documents and predicts it), but this
project is where the prediction landed: importing *both* 02's and 03's
`pipeline.py` from inside this project's own `pipeline.py` needs the
`importlib`-by-unique-package-name workaround twice, which I generalised
into `consolidation/reuse.py` rather than inlining it twice. Every project
in this slate names its entry-point module identically (BRIEF: "`pipeline.py`
with `build(...)`"); a consumer of *three or more* sibling projects (06 is
already the second, after 03) pays this cost once per sibling, growing
linearly with reuse. A cheap fix that would remove the need for this
workaround estate-wide: the BRIEF's own convention could instead ask for a
project-unique entry-point module name (e.g. `<project>_pipeline.py`), or
`decider` could document a supported loader for "the pipeline module of
project X on `PYTHONPATH`" that does not depend on `sys.modules` cache
semantics at all.

### Smaller things

- `pl.DataFrame([{"x": None, "y": 1.0}])` infers `x: Null`, and `decider`'s
  arrow-import view rejects that against a declared nullable type -- 02's
  own finding, reconfirmed here for the `last_consolidation_date` field
  (worked around with a sentinel date, `"2000-01-01"`, per `_make_sample.
  py`'s own comment).
- `decider.cli`'s console-script entry point is `decider.cli:cli`
  (`pyproject.toml`'s `[project.scripts]`), not a runnable `__main__` --
  `python -m decider.cli` fails with `No module named decider.cli.__main__`.
  `tests/test_pipeline.py`'s own CLI-invocation test uses `python -c "from
  decider.cli import cli; cli()" build` instead.

### What worked well

`step()`-wrapped plain functions throughout `credit_core`, `assessment` and
`loan_granting` being genuinely callable outside any pipeline made the
"re-run the same arithmetic hundreds of times per assessment" requirement
(§5.6.2's headline difficulty) tractable at all -- without that convention,
already established by 00/02/03 before this project started, the 900 ms
budget would have been unreachable by any means short of forking the
arithmetic. `AdjustmentRegister.apply_stack` being directly callable, with
no pipeline, made wiring two genuinely different overlay kinds (a scalar
threshold, a five-way weight vector) into this project's own plain-Python
search trivial -- the mechanism (00 §6.22's six properties) held up exactly
as well outside a `dag` as inside one.

---

## 5. Spec problems

- **§5.4's short-circuit gives six conditions but no term to test the plain
  request against.** "Baseline affordability: `core.affordability` for
  `requested_amount` as a plain advance, no consolidation" needs a term to
  turn an amount into an instalment, and §5.4 never names one (§4.1's
  request has no `requested_term_months`). I used a declared reference term
  (36 months, `baseline.REFERENCE_TERM_MONTHS`) and a flat indicative rate as
  a stand-in for "the rate the client would achieve on a plain advance" --
  both parameters, both named, both a reasonable reading, neither
  textually in the spec. A real implementation would need Credit Risk Policy
  to either declare this term explicitly or accept that the short-circuit's
  own affordability check is itself an approximation of what project 03
  would actually offer.
- **§5.8 requirement 2's "blend weight vector over the five measures" and
  requirement 3's "components are ratios to the baseline" are stated for
  OBJ-05's own five sub-components, but the worked example in §6.3
  ("Instalment relief 0.4 -> 0.6") blends **OBJ-01..05 themselves**, which
  have no common unit (rands, a debt-service ratio, a probability-weighted
  margin, a blended score). I applied requirement 3's "ratio to the
  baseline" principle uniformly across all five top-level objectives (see
  `objective.component_scores`) so the blend is well-defined at all -- a
  declared reading, not a literal instruction, and the spec does not say
  what "ratio to the baseline" means for OBJ-04 (Bank expected value) at
  all. I normalised it as `bank_expected_value / settlement_total`, which is
  defensible but not derivable from the text.
- **CON-INT-08's "New money <= Z% of the settlement total" interacts badly
  with a request that names one `requested_amount` for the whole
  assessment, not per settlement set.** The first implementation carried the
  client's full requested amount into every scenario unchanged, which made a
  R6 800 single-account settlement carry a R60 000 advance and fail
  CON-INT-08 by construction on every scenario, regardless of merit (caught
  by inspecting `sample_request.json`'s own rejection-code distribution
  during verification -- initially 100% CON-INT-08 failures). §5.5's "What
  the search must generate" does not list new-money bracketing among the
  candidate dimensions, so the spec does not obviously intend new money to
  vary per scenario either. I read this as: new money is *clamped* per
  scenario to what the base policy would allow for that settlement set,
  which makes CON-INT-08 pass by construction in the common case (still
  evaluated, in case an overlay tightens the cap below the clamp) --
  reasonable, but the spec's own six-hundred-word §5.5 never states it this
  way.
- **§5.6.4's product 11 rate card and §5.6.6's product 20 promotional/
  reversion cards are given dimensions and cell counts in §6.1 (63 072 and
  480+240) but SCOPE.md does not name either as the slice's one
  dominant-difficulty table** (unlike project 00's Flex Loan card, or
  project 03's cap waterfall). I read SCOPE.md's rule 1 ("keep the dominant
  difficulty... at real size... do not shrink it") as applying to *this
  slice's* dominant difficulty (the bounded search), not to every table
  every product mentions, and built both cards at working depth
  (12x12x12 / 8x5x12 / 16x12). A stricter reading would want product 11's
  card at its full 72x73x12 = 63 072 cells; the generator (`rate_cards.
  generate_product11_card`) is parameterised identically to project 00's and
  would need only its band-count constants changed.

---

## 6. What I would do next

1. **Fix the two reuse gaps this project's own existence exposes**, upstream:
   publish `capacity._buffer_pct`/`_RESIDUAL_BANDS` from project 02 as a
   named, public, plain-Python function (not a private module member behind
   a `DecisionTableConfig`), and change `loan_granting.solve`'s
   `PRODUCT_MIN_AMOUNT`/`PRODUCT_MAX_AMOUNT` from module constants to
   constructor/call arguments on `solve_term` (or a `SolveDomain` dataclass
   alongside `PriceEvaluator`). Both are small, targeted changes in projects
   this one cannot itself edit (BRIEF: never edit another project's
   directory), and both are now proven necessary by a real second consumer,
   not speculative.
2. **A genuine new-money search dimension**, if Credit Risk Policy's real
   intent is that a consolidation can deliver *less* new money than
   requested in exchange for a materially better objective score on a
   smaller settlement set -- the clamp this slice uses is a reasonable
   stand-in but forecloses that trade-off entirely.
3. **Products 30 and 40**, in the order DEPS.md's own wave structure implies
   (30's collateral/valuation machinery is the next-largest addition; 40's
   is the one with the sharpest audit stakes, §5.6.7's security warning
   acknowledgement).
4. **The restructure variant's actual search** (concession combinations,
   NPV cost, authority routing, the stressed-affordability double verdict)
   over the machinery this slice already built -- §5.9 itself says "the
   machinery of §5.5 to §5.8 is reused unchanged", and that claim is
   currently unverified: nothing in this slice exercises the search with
   forbearance options in the candidate space instead of settlement sets.
5. **A resolution to friction #4.2** (the stack-off toggle for a
   `frame_step`-shaped search) before project 09's harness tries to run this
   flow's stack-disabled comparison for real -- today that would require a
   second bound `Engine`, not one flag.
