# NOTES

## 1. What I built

The per-account pipeline (`pipeline.py`, `limit_mgmt/`) for spec 07's slice, plus a
separate population-level allocation stage and a simulation entry point that shares
its implementation, per SCOPE.md:

- **§5.1 population/account state** (`limit_mgmt/population.py`): `utilisation_band`
  (8 bands, closed-lower/open-upper), `mob_band` (6 bands), the 3- and 6-month mean
  utilisation the matrix keys off (not the spot value, per §5.1's oscillation-damping
  requirement), the trailing 6-month 90th-percentile observed spend (§5.1, drives C5),
  and cycle-bucket-count attribution (raggedness recorded, not hidden).
- **§5.2 hard exclusions** (`limit_mgmt/exclusions.py`): 8 of 16 codes at working depth
  (X01, X02, X03, X05, X06, X09, X12, X14, X16), every one evaluated and recorded --
  not short-circuited at the first match. X16 reads `treatment_suspension_active`,
  project 08's stubbed feed (DEPS.md). **Not built**: the "no scoring/affordability
  work performed for an excluded account" efficiency half of §5.2's requirement -- see
  "What I left out" below.
- **§5.3 behavioural scoring** (`limit_mgmt/scoring.py`): an 8-characteristic
  behavioural scorecard (`ScorecardConfig`), a PD calibration table and a 12-grade
  boundary table (`DecisionTableConfig`) keyed by `product_code`, producing
  `behaviour_score`/`probability_of_default`/`behaviour_grade` distinct from any
  `risk_grade`. Overlaid with a score shift and a PD multiplier
  (`limit_mgmt/overlays.py`, `core.adjustments`).
- **§5.4 the limit assignment matrix** (`limit_mgmt/matrix.py`): the full declared
  **1 152 cells** (12 grades x 8 utilisation bands x 6 mob bands x 2 products),
  generated synthetically and externalised as `configs/0.1.0/matrix.json`
  (`generate_configs.py`), loaded exactly as `decider template`'s own
  `build(tree: ConfigurableStep)` shape. The cycle dial (a `multiply` overlay on the
  multiplier's *excess over 1.00*, not the multiplier itself) and a cycle cap (a
  ceiling, which needed its own mechanism -- see "Framework friction").
- **§5.5 caps**: C1 (product max), C2 (income multiple), C3 (total exposure), C5
  (observed spend, the most frequently binding in the spec's own narrative), C6
  (matrix max increase), lowest wins, deterministic tie-break by table order, rounded
  down to the nearest R500 without exception, and the matrix cell's own minimum
  increment floor. C7 (affordability) is folded in by the affordability module, since
  it needs that module's own output.
- **§5.6 degraded-evidence affordability** (`limit_mgmt/affordability.py`): the A-E
  evidence tier waterfall, the imputed notional instalment, CPI indexation for tier C,
  and the automatic/conditional/fail staleness rule -- built entirely on top of
  **project 02's shared assessment**, in `LIMIT_INCREASE` mode, never forked. 07's own
  wider buffer is a `core.adjustments` overlay over 02's buffer grid output, not a
  second buffer calculation.
- **§5.7 the decrease path, one trigger** (`limit_mgmt/decrease.py`): D03 (emerging
  arrears), Immediate class, the universal floor (statement balance + unsettled
  authorisations + accrued unbilled interest), and the client-level simultaneous-case
  precedence (an Immediate-class decrease on one account suppresses every increase on
  the client's other accounts this cycle) -- resolved in `allocation.py`, since it is
  a cross-account, population-level join, not a per-account rule.
- **§5.8 the portfolio budget allocation** (`limit_mgmt/allocation.py`): ranking under
  the risk-adjusted-return default objective (plus expected-value and policy-priority),
  a funding line against three envelopes (applied limit, RWA, expected loss) with the
  over-allocation factor, a tail-skip rule, a 15%-reserve fairness pass over 144
  segments, deterministic tie-break by `account_id`, and a per-cycle summary
  (funded count, envelope consumed, which envelope bound, the funding line).
- **§5.9 (partial)**: rounding, the minimum meaningful increase, `change_type_code`/
  `notice_class_code`, and reason codes are recorded; despatch, channel selection and
  the wording set are out of scope (SCOPE.md: "beyond recording").
- **§5.10 simulation** (`simulation.py`): `run_simulation()` calls the *same*
  `pipeline.build()` + `run_allocation()` the (unbuilt) production scheduler would
  call, over a candidate matrix/params/budget. A self-check and a swap-set function
  are included; see "How I handled the portfolio-level budget" below.

### What I left out

- **§5.2's "no work performed for an excluded account"**. This slice evaluates every
  exclusion (complete attribution) but does not skip scoring/matrix/affordability work
  for excluded accounts -- doing that inside one served `decider` pipeline needs a
  `branch()` arm around every downstream stage (a real structural cost), and this
  slice's acceptance bar (correctness of the exclusion set, not compute avoided) does
  not test for it.
- **§5.7**: 13 of 14 decrease triggers, the notice-class path (only Immediate is
  built), the two-month/45-day agreement test (§5.12), and the real-time event-driven
  path (§5.11) -- all explicitly out of scope per SCOPE.md.
- **§5.8**: hysteresis at the funding line (rule 6) and the separate conditional-list
  sub-envelope charged at a 22% conversion rate. Both are named in the spec's "hard
  parts" list but not in SCOPE.md's own build list for 07 ("ranking, funded line,
  fairness cap, a reason for accounts left unfunded"); conditional-path accounts are
  excluded from the ranked pool entirely at this working depth (see "Spec problems").
- Notice and consent **mechanics** beyond recording the codes (SCOPE.md).
- The full 32-characteristic scorecard (8, working depth, same precedent 00 set for
  its own scorecard) and the full illustrative table sizes in §6.1 other than the
  matrix (buffer/income-multiple/exposure caps are `param()`s, not full tables, at
  this working depth).

---

## 2. Reuse

**From project 00 (`credit_core`), unmodified**: `core.adjustments`
(`AdjustmentRegister`, `Adjustment`, `AdjustmentEffect`, `.apply_stack_step`) for
every overlay in this project (score shift, PD multiplier, the cycle dial, 07's
buffer widening); `core.reason_codes` (`ReasonCodeRegistry`) for this project's own
registry; `core.rounding` is *not* reused for the R500 rounding (see below);
`core.evidence.cell_id` for every table's cell attribution.

**From project 02 (`assessment`), unmodified**: `evidence_unit()` and
`capacity_unit()` (02's own `pipeline.py`) run as-is in `LIMIT_INCREASE` mode, with
`risk_grade` relabelled onto `behaviour_grade` and `proposed_instalment` relabelled
onto the notional instalment this project imputes. The entire seven-stage assessment
(income waterfall, deductions, expense norms, obligations, capacity, verdict) is
02's, not re-derived -- see DEPS.md ("The affordability capability is shared with
project 02 and not forked", 07 §10 item 14).

**From `decider`'s built-ins**: `DecisionTableConfig` for the 1 152-cell matrix (at
full declared size), the behavioural PD-calibration and grade-boundary tables;
`ScorecardConfig` for the behavioural scorecard; `frame_step` for the three
population-level, ragged-history derivations (utilisation means, observed spend,
cycle-bucket count); `param()`/`missing_as()` throughout; `.relabel()` extensively,
both to compose 02's assessment into this project's own field names and to avoid the
column-name collisions documented below; `dag`/`flow` composition, including nesting
an already-`.emit()`-ed external project's units inside this project's own flow (the
same pattern 02 used to nest its own `evidence_unit()`/`capacity_unit()`).

**Written from scratch**: everything under `limit_mgmt/` except the reused calls
above -- the matrix, the caps, the evidence-tier translation, the decrease trigger,
and, centrally, `allocation.py`. `allocation.py` is **not** a `decider` step at all
(see "How I handled the portfolio-level budget"): it is plain `polars`, because the
thing it computes has no meaning for one row in isolation (README §4 Q5).
`caps.round_down_500` is written from scratch rather than reusing
`core.rounding.round_advance` (nearest R100, round-to-nearest) -- §5.5 requires
**down**, without exception, which round-to-nearest cannot guarantee; this is a
project-specific rule, not a fork of the library's.

---

## 3. Gaps in what I consumed

- **02's own mode table (§5.8) lists project 07 under shape (b), "capacity" (no
  proposed instalment, verdict `pass`/`indeterminate` only), but spec 07 §5.6 requires
  a three-way automatic/conditional/fail classification against an imputed
  *notional instalment*.** Those are incompatible: shape (b) never produces a `fail`.
  I called 02's assessment in **shape (a)** instead (supplying `notional_instalment`
  as `proposed_instalment`), which does give `pass`/`marginal`/`fail`/`indeterminate`,
  and layered 07's own staleness gate over `evidence_sufficiency_code` +
  `affordability_verdict_code` to produce automatic/conditional/fail. This uses 02's
  published interface unmodified and unforked, but it is a different shape than 02's
  own table names for this consumer -- recorded here per BRIEF, not worked around
  silently. See "Spec problems".
- **02's buffer grid is keyed on `risk_grade` (12 x 6 products), not
  `behaviour_grade`.** 07 has no independent risk grade; I relabel
  `evidence_unit()`'s read of `risk_grade` onto `behaviour_grade`, which lets 02's
  buffer grid resolve without a fork, but means the buffer grid's cell boundaries
  were tuned against a different (application) scorecard's grade distribution than
  07's own behavioural one. Recorded as a modelling assumption, not corrected.
- **07's own buffer figure (18% vs. origination's 12%, §5.6) doesn't reconcile
  arithmetically with 02's buffer grid**, which is already a 10%-35% curve by grade
  and product, not a flat 12%. I expressed 07's buffer as a **+6 percentage point
  tighten-only overlay** on top of whatever 02's grid produces, which honours the
  *mechanism* both specs ask for (an overlay-adjustable, wider-than-origination
  buffer, visible as such) without claiming a literal "12% becomes 18%" that 02's own
  grid never produced in the first place.
- **C2/C3 (income-multiple, total-exposure caps) name `gross_monthly_income`/
  `net_monthly_income` as their basis (§5.5), but those are §5.6's (affordability's)
  outputs, and caps (§5.5) precede affordability (§5.6) in the spec's own stage
  order.** Wiring C2/C3 to the affordability module's actual output creates a real
  cycle `decider` refuses to build (`WiringError: ['caps', 'affordability_bridge',
  'outcome'] depend on each other in a cycle`). I keyed C2/C3 on
  `declared_gross_income_on_file`/`declared_net_income_on_file` -- a raw account-state
  income figure captured at origination or last refresh, distinct from the
  freshly-assessed figure §5.6 produces -- which breaks the cycle and is a defensible
  reading (a pre-affordability policy cap plausibly uses the income already on file,
  not a number not yet computed), but it is a genuine spec ambiguity, not a
  workaround I'd call obviously correct.

---

## 4. Framework friction

### 4.1 `missing_as(False)`/`missing_as(True)` is not a usable Python default outside `decider`'s own engine -- because `bool` cannot be subclassed

`decider` makes `missing_as(0)`, `missing_as(0.0)`, `missing_as([])` and
`missing_as("")` real subclass instances of `int`/`float`/`list`/`str` (confirmed:
`type(missing_as(0))` is `decider.engine.params.declare.int_MissingAs`, and
`int_MissingAs(0) == 0`), so calling a step function directly in plain Python with the
argument omitted behaves exactly as documented ("the function is still plain
Python", `param`'s own docstring). `bool` is the one builtin Python will not let you
subclass, so `missing_as(False)`/`missing_as(True)` falls back to a generic sentinel:

```pycon
>>> from decider import missing_as
>>> bool(missing_as(False))
True
```

Every boolean `missing_as` default is therefore **truthy when a step is called
directly with the argument omitted**, regardless of the declared default -- the
opposite of what the declaration says. I found this by writing a unit test for
`exclusions.exclusion_codes` that called the function with only the "positive"
booleans set, expecting every other flag to default to `False`; four unrelated
exclusions fired instead. The fix is mechanical (pass every boolean explicitly in
every direct-call test -- `tests/test_exclusions.py`'s `_CLEAN` dict), but it is a
real, silent trap for exactly the testing style the framework's own docs recommend,
and I would not have found it without printing `type()`/`bool()` on the sentinel
directly. Worth a callout in `missing_as`'s own docstring next to its `bool` example
(0.3, above, already shows a bool default in its signature but never shows what
happens when it's *omitted* at a direct call site).

### 4.2 The wiring resolver's typo heuristic produces false positives on legitimately similar, unrelated column names

`decider/engine/wiring/resolve.py::unbound` raises a `WiringError` -- not a warning --
whenever a new, undeclared input name is within edit distance `TYPO_CUTOFF` of any
name already *produced* earlier in the same flow, on the theory that it is probably a
typo. This is right most of the time and wrong here: composing my own
`build_income_evidence_fields` step (which produces `applicant1_declared_income`)
ahead of project 02's `household.consolidate_declared_expenses` (which reads
`applicant1_declared_expenses`, a genuinely different, genuinely necessary field) in
one `flow` raised:

```
WiringError: input 'applicant1_declared_expenses' is not produced by any earlier
step and is not a declared input column. Did you mean 'applicant1_declared_income'
(produced by 'affordability_bridge/evidence_translation/build_income_evidence_fields')?
Rename it, or relabel(reads={'applicant1_declared_expenses': 'applicant1_declared_income'}).
```

Both names are real, both are needed, and they are not a typo of one another -- they
share a 21-character common prefix (`applicant1_declared_`) purely because that is
this project's own (and 02's own) naming convention, not because one was mistyped.
The workaround (`limit_mgmt/affordability.py`'s `_INCOME_EVIDENCE_RELABEL`): produce
under private names (`_translated_declared_income` etc.) and `.relabel()` 02's reads
onto them, so the colliding name is never registered in scope at all. This works, but
it means any two projects composed together whose naming conventions happen to
overlap on a long shared prefix will hit this the first time, with no way to silence
it short of avoiding the produced name entirely. A `# not a typo` escape hatch on
`relabel`, or a looser cutoff scaled to name length, would remove the need for this
workaround.

### 4.3 Composing an externally-owned project's own top-level `pipeline.py` collides on module name with this project's own required `pipeline.py`

BRIEF requires every project's entry point to be `pipeline.py`. Project 02's
`evidence_unit()`/`capacity_unit()` live in *its own* `pipeline.py` (not inside its
`assessment/` package). `SERVE.md`'s own PYTHONPATH deliberately puts 00's and 02's
directories on `sys.path` alongside this project's, so `import pipeline` is
ambiguous the moment both are import-visible: Python resolves one `pipeline` module
for the whole process, silently picking whichever directory sorts first on
`sys.path`, and a naive `from pipeline import evidence_unit` either fails or (worse)
silently imports the *wrong* project's `pipeline.py`. There is no supported way to
import a sibling project's differently-scoped-but-identically-named top-level module.
The workaround (`limit_mgmt/affordability.py::_load_affordability_pipeline`):
`importlib.util.spec_from_file_location` with a private module name, resolving the
path from `assessment.__file__` (a package I *can* import normally) rather than
guessing a relative path. This works and is a five-line function, but every
BRIEF-mandated `pipeline.py` in this repository's example set is a landmine for every
other project's consumer the moment two of them are ever imported in the same
process -- which 07 (consuming 02), 06 (consuming 02, 03), and 11 (consuming
02, 05, 06, 07) all do.

### 4.4 Two `frame_step`s in one `flow` execute in *written* order, not dependency order, and the error message doesn't say so

`decider`'s `flow` docstring says "run steps in written order", but I read that as
describing output precedence (last write wins) rather than a hard execution-order
requirement, since every `dag` in this codebase (including 02's own) resolves by
dependency regardless of argument order. Putting `population.utilisation_band_step`
(reads `revolving_utilisation_6m`) before `population.utilisation_means` (writes it)
in a `flow` raised:

```
WiringError: limit_mgmt/population/utilisation_band reads 'revolving_utilisation_6m'
as an input column, but limit_mgmt/population/utilisation_means, which runs later,
writes 'revolving_utilisation_6m'. Order is execution order: ... saw the input
column, while anything after ... sees ...'s value. Move ... before ..., or rename
one of the two.
```

The error message, once seen, is completely clear and even suggests the fix -- so
this is a smaller item than 4.1-4.3, but the *first* encounter is confusing because a
`dag` (which I used everywhere else in this pipeline) never has this constraint, and
nothing in `flow`'s own docstring states "written order is also execution order,
unlike `dag`" as explicitly as the error message does. A one-line addition to
`flow`'s docstring ("unlike `dag`, members run in the order given, so an intra-flow
producer must be written before its consumer") would have saved the round trip.

### Smaller things

- `AdjustmentRegister.apply_stack_step`'s two extra outputs
  (`adjustment_set_id`/`adjustments_applied`) are hardcoded names with no rename
  parameter (only the *target*'s own adjusted/unadjusted outputs are renameable via
  `adjusted_output`/`unadjusted_output`). This project layers five independent
  overlay targets (score, PD, matrix dial, 07's buffer, plus 02's own instalment
  overlay) in one pipeline, so four of the five needed a `.relabel(writes={...})`
  immediately after construction to avoid colliding on those two names -- the same
  class of friction 00's own NOTES.md already flagged for `DecisionTableConfig`'s
  `cell_id` convention, now confirmed for `core.adjustments` too.
- `core.adjustments`' three effects (`add`/`multiply`/`set`) cannot express a
  ceiling ("no more than X", the cycle cap, §6.6) or an excess-only scaling ("multiply
  the amount over 1.00", the cycle dial, §5.4) without choosing a different *target*
  to apply the existing ops to (the excess, computed as its own column) or abandoning
  the mechanism for a bespoke governed-but-not-`Adjustment` object (`overlays.CycleCap`
  -- same id/owner/scope/dates/review-date shape, different apply function). Both
  workarounds are documented inline in `limit_mgmt/overlays.py`; a fourth `AdjustmentEffect`
  op (`"cap"`/`"floor"`) would remove the second one.
- Inherited from 00/02's own findings, confirmed a third time here at the
  population/account-state layer: an entirely-absent `missing_as([])` column, an
  empty `[]` literal with no non-empty sibling, and a lone `None` for an optional
  scalar all crash single-record scoring differently. `sample_request.json` follows
  02's own documented conventions (see SERVE.md).

### What worked well

`.relabel()` composing two *already-built, already-`.emit()`-ed* dags from a
different project (02's `evidence_unit()`/`capacity_unit()`) into this project's own
`flow`, with a governed overlay (07's buffer) spliced in *between* them by name, is
exactly the composition spec 07 asks for ("the affordability capability is shared...
and not forked") and it worked cleanly once the naming issues above were resolved.
`DecisionTableConfig` generating and validating the full 1 152-cell matrix was
trivial and fast (well under a second). `pipeline.parameters().defaults()` again made
a correct, complete `params.json` for a large composed pipeline (177 output columns)
a one-liner, exactly as 00/02's own NOTES describe.

---

## 5. Spec problems

- **07 §5.6 and 02 §5.8's own mode table disagree about which "shape" 07 calls.** See
  "Gaps in what I consumed" above -- this is the single most consequential ambiguity
  in the slice, because it determines whether the affordability module can ever
  return `fail` for this project at all.
- **C2/C3's basis (`gross_monthly_income`/`net_monthly_income`) creates a real
  build-time cycle against the spec's own stage ordering** (caps before affordability;
  see "Gaps in what I consumed"). This is not a `decider` limitation -- the cycle is
  real in the spec as written, and any implementation composing caps and affordability
  as separate stages, in the stated order, will hit it.
- **§5.8's fairness floor (rule 5) names "three consecutive cycles" as the trigger for
  a starved segment**, but a single monthly cycle (this slice's scope) has no prior-
  cycle state to consult. I applied the floor within the current cycle only
  (a segment below 40% of the population rate *this cycle* is topped up from the
  reserve), which is a strictly *more* generous reading than the spec's own
  three-consecutive-cycle trigger -- worth flagging since a literal implementation of
  the temporal trigger needs cycle history this slice does not model.
- **The observed-spend cap (C5)'s own definition (`max(current_limit, ...)`) makes it
  structurally incapable of binding below `current_limit`**, which means C5 can never
  be the cap that explains a limit staying flat versus one that explains a limit
  falling -- worth stating explicitly in §5.5, since "the most frequently binding cap
  in the programme" (§5.5's own words) is, by construction, never the reason an
  account's limit goes down.

---

## 6. What I would do next

1. **Resolve the shape-(a)-vs-shape-(b) ambiguity with project 02's own maintainers**
   rather than picking shape (a) unilaterally -- see "Spec problems".
2. **Measure batch throughput properly and push on `interpreted` mode's cost.** A
   10 000-account timed batch run measured **~294 rows/second** in `interpreted` mode
   (34 s for 10 000 rows; allocation itself is not the bottleneck at 142 000
   rows/second on the same run). Extrapolated linearly, 4.1 M accounts would take
   **~3.9 hours** -- over §8's 3-hour window. `interpreted` mode is required here for
   the same reason 00/02 document (`norm_table_version` compares two `str` columns,
   which `fused`/`stepped` mode rejects at bind time) -- inherited through project 02,
   not introduced here. Getting under 3 hours needs either a `fused`-mode-compatible
   rewrite of that one comparison upstream (00/02's own "what I would do next"), or
   accepting `interpreted` mode and parallelising the batch run across workers, which
   this slice does not attempt.
3. Build the separate conditional-list sub-envelope (§5.8, "the conditional list is
   ranked separately... at their expected conversion rate") and hysteresis at the
   funding line (§5.8 rule 6) -- both named in the spec's "hard parts" list and both
   skipped at this working depth per SCOPE.md's own narrower build list.
4. Extend the decrease path to the other 13 triggers and the Notice-class timing
   machinery (§6.7), and add the real-time event-driven path (§5.11) and the
   batch/real-time agreement test (§5.12) -- all explicitly out of scope for this
   slice.
5. Widen the behavioural scorecard to a realistic 32 characteristics (mechanical,
   same precedent as 00's own scorecard).
