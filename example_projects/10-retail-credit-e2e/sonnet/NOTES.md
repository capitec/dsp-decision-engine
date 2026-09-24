# NOTES

## How I organised a large project

One Python module per phase (`retail_credit/identity.py` = P02,
`retail_credit/cap_waterfall.py` = P09, `retail_credit/affordability_phase.py`
= P10, ...), plus five cross-cutting registries no isolated phase owns:
`phases.py` (the 18-phase identity scheme and ownership map, §5.1/§5.26.1),
`entry_points.py` (the phase-set matrix and resolution, §5.20/§4.6),
`shared_intermediates.py` (§5.21), `ordering.py` (§5.22) and
`decision_record.py` (§5.31). `overlays.py` holds the **one** overlay
register every phase that reads an overlaid value imports (never
re-instantiated per phase -- that is what O-05/O-21 require). `pipeline.py`
is the composition root: it imports every phase module and wires them with
`dag`/`flow`/`loop`, and nothing else in the project calls `decider.dag`
directly except inside a phase module's own `_p0N_unit()` helper.

This mirrors the spec's own claim that "whether a phase becomes one
component or forty is exactly the question this project asks and
deliberately does not answer" (10 §5.1) -- one module per phase is the
smallest structure that keeps that question open without forcing every
phase into either one giant file or forty.

## 1. What I built

The full-width skeleton SCOPE.md asks for: all 18 phases and their
decision-point counts declared (`retail_credit/phases.py`, `TOTAL_
DECISION_POINTS == 1400`), the ownership map (`phases_owned_by`), all 8
entry points' phase-set matrices at (running keep this in ●/◐/○/blank
granularity, `retail_credit/entry_points.py`'s `_MATRIX`, checked against
the spec's own table cell-by-cell), the phase-set identity scheme
(`phase_set_id`/`phases_for_id`, content-derived, round-trip-tested), the
shared-intermediates registry for the twelve values with 5+ consumers
including `existing_obligations`'s two-live-version case (§5.21.1), the
ordering-constraint registry (24 of the ~40, `ordering.py`) checked
against this project's own declared execution order, and the single
decision-record shape (`decision_record.py`, sparse by construction --
`phases_absent_by_entry_point` is populated, not null-padded, when P14
does not fire).

Entry point 1, product 10, fleshed out end to end: P01 (5 structural
checks, not 34), P02 (all four identity-resolution paths with their
confidences), P03 (consent + hard eligibility, every gate evaluated, none
short-circuited), P04 (the one degraded mode this slice implements --
bureau down, code 11, the reduced-envelope gate), P05 (13 fraud rules
across the five families, weighted precedence, the bypass rule -- volume
is project 01's difficulty, not built here at 188), P06 (00's income/
deductions/expense-norms/obligations units, wired exactly as 00's own demo
wires them, plus this project's own segment assignment and evidence-tier
calibration), P07 (this project's own 8-characteristic scorecard, SC-10-A3),
P08 (this project's own calibration curve and grade boundaries, the
overlay register resolved once), P09 (13-entry cap register reproducing
10 §5.10's own worked example nearly verbatim, full attribution -- final
value, binder, chain, evaluated-vs-not-applicable), P10 (00's
`discretionary_income`/`max_affordable_instalment` units reused by
`.relabel()`, this project's own capacity/ratio-ceiling/buffer composition
and verdict policy), P11 (product-10-only eligibility gate), P12 (this
project's own 34 560-cell Flex Loan card at its declared non-uniform band
shape, reproducing the R60 000 band-edge inversion the spec's own worked
example describes, fee/premium/instalment in O-14 order with the fee
capitalised into the financed principal), P13 (a real bounded, bisecting
search -- not division, not a naive scan -- proven against an exhaustive
R250-grid scan and against the spec's own band-edge worked failure), loop
L1 (§5.23.1, a real cross-phase loop via `decider.loop`, with the
actual/hypothetical `existing_obligations` distinction and
`loop_pass_index`/`loop_termination_code` recorded), P16 (offer assembly:
loop result supersedes the initial solve when the loop fired), P17
(independent re-derivation -- never reads `instalment`, recomputes it from
`offer_amount`/`term_months`/`risk_grade` through the same pure-Python
pricing function P13's search uses, per O-18), P18 (this project's own
27-entry reason registry, ranked, `outcome_code`).

### What I left out (declared, not silently dropped)

- **P14 in depth.** One settlement scenario per pass (settle every
  externally held obligation, keep every internal one), not the spec's
  up-to-250-scenario search over subsets. The *loop* is real and end to
  end; the *search inside P14* is not (SCOPE.md explicitly permits this:
  "Skip products 11-40, P14 and P15 in depth").
- **P15.** Never runs on entry point 1 (§5.20); a declared stub only
  (`retail_credit/limit_assignment.py`), proven by nothing more than
  existing in the phase registry.
- **Products 11-40.** This slice is product 10 only throughout; P11's
  fan-out, substitution and preference logic (10 §5.12) collapse to one
  eligibility gate, and the genuine O-09 cycle (P10 runs twice, around
  P11) is real but arithmetically a no-op with one candidate product
  (documented in `routing.py`).
- **Fraud rule volume.** 13 live rules, not 188 -- the mechanism (every
  rule evaluated, weighted precedence, the bypass, not-evaluated-vs-
  did-not-fire) is at full depth; the count is not (that is project 01's
  own dominant difficulty, not this project's).
- **9 scorecards, 3 780 rows.** One scorecard (SC-10-A3, 8
  characteristics). `scoring.select_scorecard` computes which of two
  scorecards *would* apply (including the spec's declared -45 point
  degraded-bureau shift) but the served pipeline always scores on
  SC-10-A3 -- a second `ScorecardConfig` plus a `branch()` on the selector
  is the mechanical upgrade, proven standalone in `tests/`.
- **Batch entry points 3 and 4.** Routing only (`entry_points.py`'s
  matrix + `resolve_phase_set`), not run end to end -- SCOPE.md asks for
  "a routing test", not depth.
- **09-H (the replay/diff/swap-set harness).** Out of scope per DEPS.md's
  wave assignment; this project adopts 09-C (the §5.15 evidence contract)
  as a checklist, same as project 00 did.

## 2. Reuse

**From `credit_core` (project 00), unchanged:** `core.income` (all five
functions, wired exactly as 00's own demo pipeline wires them -- this is
DEPS.md's "hidden coupling", 10 §5.7(b): these units had to live in 00 or
this project stops being standalone, and they do), `core.deductions`,
`core.expense_norms`, `core.obligations` (the `frame_step`, dedup and
ten-treatment-behaviour matrix), `core.affordability`'s
`discretionary_income` and `max_affordable_instalment` (the second reused
by `.relabel()`-ing its `discretionary_income` read to this project's own
`pre_buffer`, not reimplemented), `core.adjustments` (`Adjustment`,
`AdjustmentEffect`, `AdjustmentRegister` -- the whole mechanism, one
register for four overlay kinds across three phases), `core.eligibility`,
`core.consent`, `core.fees`, `core.instalment` (forward direction; the
inverse `solve_advance_for_instalment` is imported but not used --
this project's search bisects rather than inverting, for the band-edge
reason 10 §5.14 gives), `core.rounding`, `core.reason_codes`
(`ReasonCodeRegistry`), `core.dates` (`EffectiveDatedSet`, transitively
through `core.deductions`/`core.expense_norms`).

**From `decider`'s built-ins:** `DecisionTableConfig` for every table
(calibration, risk grade, ratio ceiling, buffer, rate card, credit life);
`ScorecardConfig` for the scorecard; `param()`/`missing_as()` throughout;
`dag`/`flow` for composition, with `flow` used specifically for every
same-name waterfall (`instalment`, `decline_reason_codes`, the L1 seed);
`loop()` for L1 -- the one place this project needed decider's cross-row
iteration primitive, and it did the job: `carries` is exactly the
"actual/hypothetical" state L1 needs to thread through up to four passes,
and the per-row convergence semantics meant a request whose affordability
already passes never enters the loop body at all, at zero extra cost.
`.relabel()` for every table's `cell_id` (namespaced per capability,
exactly as project 00's NOTES.md recommends) and for reusing 00's units
with this project's own field names.

**Written from scratch:** everything domain-specific to project 10 --
segment assignment, the cap register, the rate card generator, the fraud
rule set, the bounded search, the L1 loop body, offer assembly, final
validation, the reason registry, and all five cross-cutting registries
(`phases.py`, `entry_points.py`, `shared_intermediates.py`, `ordering.py`,
`decision_record.py`), because nothing upstream of this project (wave 0,
`credit_core` only) has anything that shapes composition across eighteen
phases -- that is exactly this project's own subject, per 10 §1.1's "how
do you structure one very large project from scratch?"

## 3. Gaps in what I consumed

None that forced a workaround. Every `credit_core` capability this
project needed was cleanly available and composed without a fork -- DEPS.md's
predicted "hidden coupling" (10 §5.7(b)) held exactly as documented: 00
does publish `core.income`/`core.deductions`/`core.expense_norms` as
library-owned units, not folded into project 02's assessment, so this
project stayed standalone as designed.

One near-gap, resolved by not reusing: `core.credit_life`'s age/term rate
table (`_RATES`) is a module-private list, not exported. This project's
search needs the *identical* rate data in two places -- a `DecisionTableConfig`
for the served P12 pricing and a pure-Python lookup for P13's bounded
search -- and there was no public seam to share one table between both
(project 00 never needed this, because its own demo pipeline never
searches). Reaching into `credit_core.credit_life._RATES` would have
worked but is exactly the private-name coupling this project's own
`credit_core` NOTES.md warns against elsewhere. I built this project's own
small credit-life table (`retail_credit/pricing.py`) instead, generated
once and shared between the decider table and the search by construction
(the same pattern already used for the rate card) -- more code, but no
private coupling, and it is what caught the bug in "Framework friction"
below.

## 4. Framework friction

### 4.1 Two projects sharing an environment can silently import each other's `pipeline.py`

Every example project in this set has a top-level `pipeline.py` (`decider
template`'s own convention: `decider.cli` template writer puts one there).
Serving this project needs `credit_core` (project 00's library) on
`PYTHONPATH`, but project 00's *own directory* also contains project 00's
*own* `pipeline.py`. If that directory precedes this project's own on
`sys.path`, `import pipeline` inside `decider build` silently resolves to
**project 00's demo pipeline**, not this one:

```
Error: config version latest failed to build: KeyError: "the config version
has no 'rate_card_flex_loan' document for argument 'rate_card_flex_loan' of
'pipeline:build'"
```

The error names an argument (`rate_card_flex_loan`) this project's own
`build()` does not take -- because the function that actually ran was
project 00's. Nothing about `decider build`'s error points at a
`PYTHONPATH` ordering mistake; I only found the cause by printing
`pipeline.__file__` inside the failing process. This is not a defect in
project 00 or in this project -- it is a structural consequence of every
project in a multi-project example set using the same entry-point module
name, which `decider template`'s own convention guarantees. **Any project
in this set that consumes another project's library will hit this the
first time its own directory is not first on `PYTHONPATH`.** Worth a
`decider template` note, or a documented convention that consumed
projects' library packages (not their whole directory) go on
`PYTHONPATH`, never the directory containing their own `pipeline.py`.

### 4.2 `dag()`'s two-writer collision is easy to trigger with `AdjustmentRegister.apply_stack_step`, repeatedly

Every call to `OVERLAY_REGISTER.apply_stack_step(target, ...)` writes
`adjustment_set_id` and `adjustments_applied` **by default**, regardless
of `target`. This project's one register serves four targets across three
phases (`probability_of_default` in P08, `affordability_buffer` in P10,
`amount_cap` in P09, and `nominal_annual_rate` twice, once to resolve the
addon bps before the solve and once to price the shipped offer for
real) -- five call sites in total. Every one of those calls collides on
`adjustment_set_id`/`adjustments_applied` unless `.relabel(writes={...})`'d,
and the error only appears at `dag()` build time, once per unresolved
collision, one at a time:

```
decider.exceptions.WiringError: dag 'retail_credit_ep1_product10': p08 and
p10 both write 'adjustment_set_id'; use flow(...) to apply them in written
order, the later one winning
```

I hit this five separate times while wiring `pipeline.py` (calibration
vs. risk-grade tables' `cell_id`, the PD overlay's `_unadjusted` output
colliding with its own base step's relabelled name, the buffer overlay,
the rate overlay, and the rate-addon pre-resolution) -- each fixed
correctly, one at a time, by `.relabel()`, but there is no single error
that lists every collision in a pipeline this size at once, so wiring a
34-member dag is an iterate-fix-rebuild loop rather than a single pass.
The fix is exactly the discipline project 00's own NOTES.md describes for
table `cell_id`s; this project's finding is that the **same discipline is
needed for every `apply_stack_step` call**, and `apply_stack_step`'s own
docstring does not flag it (it documents `base_field`/`adjusted_output`/
`unadjusted_output` as the customisation points but not that
`adjustment_set_id`/`adjustments_applied` also need relabelling when a
pipeline calls it more than once). A convenience default -- namespacing
those two outputs by `target` automatically, as `adjusted_output`
implicitly is via the `target` name itself -- would remove the whole
class of collision for the common case (one register, several targets,
one pipeline).

### 4.3 A `list[dict]` terminal output crashes result materialisation on a *plain* `step`, not only `frame_step`

Project 00's NOTES.md documents this for `frame_step`. It reproduces
identically for a plain `step`: `retail_credit.cap_waterfall`'s first
version returned the cap register's full attribution chain
(`list[dict]`) as a decider step output, and any unconsumed
(un-`.emit()`'d) occurrence of it crashed exactly the way project 00's
NOTES.md describes (`ValueError: cannot parse numpy data type dtype('O')
into Polars data type`). The root cause project 00 identified
(`decider/engine/run/state.py`'s generic output path round-trips every
unconsumed output through `np.where(...).tolist()`, which cannot box a
list of Python dicts) is evidently not specific to `FrameStep`'s
`Any`-typed outputs -- it is the *generic* output-materialisation path,
reached by any step, of any kind, whose output type is
`list[struct]`-shaped. **This widens project 00's finding**: any decider
project computing a per-decision-point attribution chain (exactly what
09 §5.15 items 7-8 and 10 §5.10 item 3 both require) cannot return it as
a decider step output at all, regardless of step kind. This project's
workaround is the same shape as 00's: compute the chain via a plain
Python function (`cap_waterfall.run_cap_waterfall`), call it directly
from tests and from evidence-assembly code, and keep the *decider-wired*
step scalar-only (`amount_cap`, the binding rule id as a string, declined,
decline reason). That is a real capability gap for a governance-heavy
domain: the framework can compute the evidence 09 §5.15 requires, but
cannot hand it back as part of a served decision's own output columns.

### 4.4 Cross-step type unification promotes declared `int` columns to `float`, breaking list indexing

`dependants_count` is declared `int` in this project's own
`residual_floor(dependants_count: int)` but arrives as a Python `float`
inside the step's row loop:

```
TypeError: list indices must be integers or slices, not float
  File ".../affordability_phase.py", line 53, in residual_floor
    return _RESIDUAL_FLOOR[idx]
in step retail_credit_ep1_product10/p10/residual_floor, row 0
```

`dependants_count` is read as `int` in three places in this project
(`residual_floor`, `assign_segment`, the scorecard's binning) and
compared as a bound in `core.expense_norms`' `between` table expression
(which resolves `Float64` columns). Somewhere in decider's cross-step type
resolution the column's *declared* type at one reader loses to its
*inferred* type from another, and the promotion is silent -- no warning,
no error, until a caller does something (list indexing) that a `float`
cannot do but an `int` can. Every place this project reads an integer
request field for anything other than arithmetic or comparison
(indexing, `%`, dict keys where the caller cannot rely on Python's
int/float hash equality) now casts explicitly (`int(dependants_count)`).
Worth a docstring note next to `param`/scalar-input typing in `IR.md`
§5.1's table, which documents null-handling per declared type but not
cross-step type unification.

### 4.5 `Engine(...).bind(pipeline, mode=...)`, not `Engine(mode=...)`

A first attempt at `Engine(mode="interpreted")` fails with `TypeError:
Engine.__init__() got an unexpected keyword argument 'mode'`. `mode` is a
parameter of `.bind()`, not the constructor -- correctly shown in
`decider/engine/run/engine.py`'s own docstring examples (`Engine().bind(
pipeline, mode="fused")`) once you read past the first one
(`Engine(params_validation="lazy").bind(pipeline)`, which has no `mode`
and reads, on a skim, like the constructor is where mode-like settings
go). A small thing, but it cost a real iteration during this project's
build, the same class of trap project 00's NOTES.md flags for
`.run()`/`.score()`.

### What worked well

`decider build`'s wiring errors, once past the collision-discovery
iteration in §4.2, are excellent -- `"input 'prior_max_affordable_instalment'
is not produced by any earlier step and is not a declared input column.
Did you mean 'max_affordable_instalment' ... ? Rename it, or
relabel(reads={...})."` found a genuine design mistake (a carry that
should have been the loop's own `max_affordable_instalment_hypothetical`,
not a separate ad hoc input) and named the fix precisely. `loop()`
composed exactly as documented for L1 -- `carries` is the whole
actual/hypothetical state, the condition step is one line
(`not loop_converged`), and a request that never needs the loop pays
nothing for it (the loop body is never called when the seeded condition
is already `False`). `pipeline.parameters().defaults()` again made a
correct, complete `params.json` for a ~30-node dag trivial to generate --
the only manual work was applying this project's own fee/credit-life
calibration overrides afterward, and even that is a five-line patch
keyed on step path.

## 5. Spec problems

- **P13's single-owner claim contradicts its own ownership table.**
  10 §5.26.1's narrative says "Only five of the eighteen phases have a
  single owner: P02, P05, P07, P13 (for its mechanics) and P14", but the
  same section's ownership table lists **both** T1 (`P01, P04, P09, P13,
  P17, P18`) and T7 (`P09, P11, P12, P13, P16`) against P13 -- two owners,
  not one. The table also gives P01 exactly one owner (T1, with no other
  team's row naming it), which the narrative's five-phase list omits.
  `tests/test_entry_points.py::test_at_least_five_phases_have_a_single_owner`
  documents this rather than picking a side.
- **"Applicable" for a cap-register entry is under-specified until you
  read the worked example closely, and I got it wrong once.** 10 §5.10
  item 4 distinguishes "evaluated and did not bind" from "not
  applicable" as "different facts", but never defines what makes an
  entry inapplicable. The worked example (CAP-0131, "applicable: yes (67
  months)") shows that a **threshold not being crossed is "evaluated,
  did not bind", not "not applicable"** -- applicability is about scope
  (does this rule's population include this application at all), not
  about whether its own trigger condition happens to be true. My first
  implementation of `retail_credit.cap_waterfall` encoded the threshold
  *as* the applicability predicate for CAP-0131, CAP-0176 and CAP-0212,
  which is the wrong reading and was caught by my own
  `tests/test_cap_waterfall.py`, not by the spec text. A worked example
  is not a substitute for a definition when the definition is load-bearing
  for a governance requirement (09 §5.15 item 14 depends on exactly this
  distinction for dead-rule measurement).
- **The rate-overlay-participates-in-the-solve requirement and the
  resolve-the-register-once requirement are not reconciled.** 10 §5.13(e)
  says the rate add-on "participates in the solve like any other rate
  movement, including its effect on band-edge behaviour"; O-05/O-21 say
  the overlay register is resolved once, for the whole decision, and two
  phases resolving it independently is a defect. A search that evaluates
  up to 19 candidates per term cannot re-resolve the register per
  candidate without violating O-05/O-21, but also cannot ignore the
  overlay without violating 10 §5.13(e) -- confirmed as a real
  consequence, not a hypothetical one: an early version of this
  project's search ignored the overlay and returned an amount whose
  real, overlaid instalment exceeded `max_affordable_instalment` (10
  §5.14 requirement 7's exact failure mode). The resolution I built
  (resolve the addon's basis-point value once, before the solve, and
  thread the scalar through every candidate evaluation) is not stated
  anywhere in the spec; it is a synthesis of two requirements the spec
  states separately and never connects.
- **Segment precedence (10 §4.3) is unstated.** Twelve segments, several
  with conditions that can overlap (a staff member who is also
  self-employed; a joint application where one applicant is a pensioner),
  and the spec gives a numbered list without saying whether row order is
  precedence order or an implementer's choice. `retail_credit.features
  .assign_segment` picks joint > staff > non-resident > employment-type >
  internal-relationship > bureau-thickness, documented as this project's
  own choice, not derived from the spec.

## 6. What I would do next

1. Extend P14 from one settlement scenario per pass to a real, bounded
   subset search (10 §5.15(c)'s ordered generation rules), keeping the
   loop mechanism unchanged -- `consolidation.l1_pass`'s scenario
   proposal is the one seam that would need to change.
2. Widen P11 to the full six-product fan-out, which makes O-09's
   "P10 runs twice, product-neutral then routed" mechanism arithmetically
   real instead of a documented no-op.
3. Generate `ordering.py`'s `_EXECUTED_ORDER` (what
   `tests/test_ordering.py` checks against) from the built pipeline's own
   dependency graph rather than a hand-maintained list, closing the gap
   between "the order I declared" and "the order dag() actually
   resolves to".
4. A field-by-field audit of `decision_record.build_decision_record`
   against 09 §5.15's 23 items, with a test per item -- this project
   built toward the checklist but never verified it exhaustively.
5. Push §4.1 (PYTHONPATH/`pipeline.py` collision) and §4.3 (list[dict]
   on any step kind, not only `frame_step`) upstream -- both are
   estate-wide findings, not project-10-specific ones, and §4.3 in
   particular blocks any project needing to serve a per-decision-point
   attribution chain as a first-class output rather than an
   evidence-assembly afterthought.
