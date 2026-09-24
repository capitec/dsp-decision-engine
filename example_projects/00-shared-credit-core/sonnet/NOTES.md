# NOTES

## What I publish

Every other example project imports `credit_core` directly. One module per
capability, stable names throughout:

| Capability | Module | Key names |
|---|---|---|
| `core.dates` | `credit_core.dates` | `EffectiveDatedSet`, `EffectiveVersion`, `.resolve()`, `.resolver_step()` |
| `core.reason_codes` | `credit_core.reason_codes` | `ReasonCode`, `ReasonCodeRegistry`, `.rank()`, `.resolve_step()` |
| `core.adjustments` | `credit_core.adjustments` | `Adjustment`, `AdjustmentEffect`, `AdjustmentRegister`, `.apply_stack()`, `.apply_stack_step()` |
| `core.rounding` | `credit_core.rounding` | `round_advance`, `round_instalment`, `round_rate` (+ `*_step`) |
| `core.instalment` | `credit_core.instalment` | `instalment_before_fees`, `instalment`, `solve_advance_for_instalment` (the inverse) |
| `core.fees` | `credit_core.fees` | `initiation_fee`, `monthly_service_fee` (+ `*_step`) |
| `core.rate_card` | `credit_core.rate_card` | `generate_flex_loan_card`, `diff_cards`, `out_of_range` |
| `core.scorecard` | `credit_core.scorecard` | `build_scorecard`, `adverse_action_codes`, `VARIABLES` |
| `core.calibration` | `credit_core.calibration` | `build_calibration_table`, `probability_of_default`, `score_for_probability` (exact inverse) |
| `core.risk_grade` | `credit_core.risk_grade` | `build_risk_grade_table`, `risk_grade_output` |
| `core.income` | `credit_core.income` | `gross_monthly_income`, `income_source_code`, `income_variability_ratio` |
| `core.deductions` | `credit_core.deductions` | `TAX_TABLES`, `build_tax_table`, `statutory_deductions`, `net_monthly_income` |
| `core.expense_norms` | `credit_core.expense_norms` | `STATUTORY_NORMS`, `INTERNAL_NORMS`, `build_statutory_table`, `build_internal_table`, `norm_floor`, `living_expenses` |
| `core.obligations` | `credit_core.obligations` | `obligations` (a `frame_step`; scalar + four parallel per-account lists) |
| `core.affordability` | `credit_core.affordability` | `discretionary_income_step`, `max_affordable_instalment_step`, `affordability_verdict_code_step` (unit only -- 02 builds the assessment) |
| `core.bureau` | `credit_core.bureau` | `normalise_bureau` |
| `core.eligibility` | `credit_core.eligibility` | `decline_reason_codes`, `is_eligible` |
| `core.appetite` | `credit_core.appetite` | `build_appetite_table`, `binding_appetite_limit` |
| `core.exposure` | `credit_core.exposure` | `total_exposure`, `exposure_headroom` |
| `core.consent` | `credit_core.consent` | `consent_verdict`, `ConsentChannel`, `ConsentVerdict` |
| `core.credit_life` | `credit_core.credit_life` | `build_credit_life_table`, `credit_life_premium` |
| `core.adverse_events` | `credit_core.adverse_events` | `event_severity_code` (14 event types, caller-supplied thresholds) |
| shared vocabulary | `credit_core.vocab` | `NullKind`, roles, product codes, `FraudVerdict`, `ConsentVerdict` |
| shared mechanisms | `credit_core.evidence` | `new_decision_id()`, `cell_id()` |

**Import pattern**: `from credit_core import rate_card` then use its
functions directly as decider steps (plain functions auto-convert; `dag`
resolves the wiring). Table-backed capabilities expose a `build_*_table()`
function returning a `DecisionTableConfig` -- call it once, compose it into
your flow. `core.rate_card`'s Flex Loan card is the one capability
externalised as a `configs/<version>/rate_card_flex_loan.json` document
(via `generate_configs.py`), so it can be refreshed without a redeploy
(00 §9); every other table is built directly by its module's `build_*`
function, versioned by the version string constant at the top of that
module (e.g. `RISK_GRADE_VERSION`).

**Column-naming convention consumers must follow**: every table-backed
capability's `cell_id` output is renamed to `<capability>_cell_id`
(`rate_card_cell_id`, `calibration_cell_id`, `risk_grade_cell_id`,
`tax_table_cell_id`, `statutory_norm_cell_id`, `internal_norm_cell_id`) --
`cell_id` alone collides the moment two table capabilities sit in one
flow/dag. See "Framework friction" below.

**Effective dating**: build a resolver with
`EffectiveDatedSet(family, versions).resolver_step()`, put it before the
table lookup, and add an `eq` clause on the resolved version column to the
table's match expression. One JSON document then holds every historical
version's rows side by side (see `credit_core/expense_norms.py`), so an
old decision can never see a new row.

**Adjustments**: build an `AdjustmentRegister` once (see
`pipeline.py`'s `SCORE_ADJUSTMENTS`), then
`register.apply_stack_step(target, adjustment_set_id, base_field=...)`
gives you a decider step producing the adjusted value, the unadjusted
value, the adjustment set id and the applied-ids list, with the stack
disabled by one `param()` flip (`adjustment_stack_enabled`).

---

## 1. What I built

The library-wide mechanisms at full depth: `core.dates` (effective-dated
resolution, used by four different table families in the demo pipeline),
`core.reason_codes` (registry + ranking + primary), `core.adjustments`
(register, scope, declared stack order, tighten-only rejection at
definition, expiry + lapse + review-date surfacing, stack-off run through
the same code path), `core.rounding`, and cell-level attribution
(`credit_core.evidence.cell_id`) threaded through every table.

Wave 1-2 capabilities at working depth: `core.instalment` with its exact
inverse, `core.fees`, `core.rate_card` (Flex Loan at its full declared
96 x 55 x 12 = 63 360 cells, generated, structured lookup result with a
cell id), `core.scorecard` (8 characteristics with per-characteristic
contributions and adverse-action codes -- not the spec's 20-60; see
"What I left out"), `core.calibration` (exactly invertible), `core.risk_grade`,
and the five affordability units `core.income`, `core.deductions`,
`core.expense_norms`, `core.obligations` (dual-list dedup, ten treatment
behaviours, scalar + per-account shapes) and `core.affordability` (unit
only, monotone in the proposed instalment).

Frozen-interface, thin-but-correct: `core.bureau` (one schema),
`core.eligibility`, `core.appetite`, `core.exposure` (no graph),
`core.consent`, `core.credit_life`, `core.adverse_events` (all 14 types,
caller-supplied thresholds).

Proved directly (§10): item 4 (same capability, different settings --
`test_affordability.py::test_same_capability_twice_with_different_settings_in_one_flow`,
and `test_scoring.py`'s calibration-by-segment test), item 6 (re-derive to
the cent -- `test_pipeline.py::test_replay_reproduces_the_decision_to_the_cent`),
item 8 (stack disabled, same implementation --
`test_pipeline.py::test_the_stack_can_be_disabled_through_the_same_pipeline`).

A demo pipeline (`pipeline.py`) wires fourteen of the twenty-two
capabilities into one Flex Loan quote: eligibility -> income -> deductions
-> expense norms -> obligations -> affordability -> scorecard (with one
overlay pair, `segment_code=3` and `channel_code=4` both firing, change
scenario 13) -> calibration -> risk grade -> rate card -> fees ->
instalment -> reason codes -> outcome. `decider build` succeeds and the
sample request scores through the handler (see "Framework friction" for
what that took).

### What I left out

- **`core.scorecard`** carries 8 characteristics, not 20-60. The mechanism
  (bin-to-points, per-characteristic contribution, adverse-action codes)
  is the same regardless of count; extending `VARIABLES` in
  `credit_core/scorecard.py` is mechanical.
- **`core.risk_grade`**, **`core.calibration`**, **`core.appetite`**,
  **`core.credit_life`** carry a handful of (product, segment) cells, not
  the spec's full 8x5, 12x8x6 etc. Only `core.rate_card` is kept at its
  full declared size, per SCOPE.md rule 1 ("generate large tables
  synthetically at their full dimensions... do not shrink them" -- singular
  to the dominant-difficulty table for this slice).
- **Two live majors simultaneously** (§7.1) and **the four bureau
  formats** and **spreadsheet-driven policy tests** -- explicitly out of
  scope for this slice (SCOPE.md).
- The demo pipeline wires 14 of 22 capabilities; `core.bureau`,
  `core.appetite`, `core.exposure`, `core.consent`, `core.credit_life`,
  `core.adverse_events` are exercised directly in `tests/`
  (`test_thin_capabilities.py`) instead, since every capability is
  standalone-testable by design (§7.5) and wiring all twenty-two into one
  demo would only multiply the column-naming bookkeeping (see below) for
  no extra proof of the mechanism.
- `effective_annual_rate` is the compounded nominal rate, not a
  fees-inclusive APR (which needs a numeric root-find over the whole
  payment stream) -- marked with a `ponytail:` comment in `instalment.py`.

---

## 2. Reuse

**From `decider`'s built-ins**: `DecisionTableConfig` for every table
(tax brackets, expense norms x2, credit life, appetite, risk grade,
calibration segments, the rate card); `ScorecardConfig` for the scorecard;
`param()`/`missing_as()` throughout for every tunable and every optional
input; `frame_step` for the three ragged-collection capabilities
(obligations, bureau, exposure) that a scalar `Step` cannot express;
`.relabel()` extensively, both for effective-dating (see above) and for
resolving column-name collisions between table capabilities composed in
one flow; `.named()` (with `.relabel(writes=...)`) to run one capability
twice with different settings.

**From decider's own conventions, not code**: the `configs/<version>/`
pattern for the one table (`rate_card_flex_loan.json`) this project
externalises as a swappable-without-redeploy document, exactly as
`decider template`'s own example does it (`def build(tree: ConfigurableStep)`).

**Written from scratch**: everything under `credit_core/`. There was
nothing to reuse from -- this is project 00, the first wave, with no
upstream dependency (DEPS.md: "hard: none").

---

## 3. Gaps in what I consumed

None -- this project has no hard dependencies (DEPS.md, wave 0). The only
adjacent artefact is 09's evidence contract (spec 09 §5.14-§5.15), adopted
as a checklist rather than code, per SCOPE.md's instruction that 09-C is
"a requirements document adopted by 00... in wave 0."

---

## 4. Framework friction

**This is the most valuable section, so it is long. All four items below
were confirmed with a minimal reproduction before being written up; three
required reading `decider/engine/run/state.py`,
`decider/engine/boundary/_arrow/view.py` and `decider/serving/handler.py`
directly, since the error messages alone did not point at the cause.**

### 4.1 `decider build`'s warm-up cannot handle `list`, `dict` or `date` inputs -- blocks every pipeline with a `decision_date` argument

`decider build`/`decider serve` warm every kernel with a synthetic record
before serving (`decider/serving/handler.py`, `_warm`). The synthesiser is:

```python
_DUMMY = {bool: False, int: 1, str: "", bytes: ""}

def _warm(exe, params):
    record = {v.name: _DUMMY.get(base_annotation(v.annotation), 1.0)
              for v in exe.plan.versions if v.producer is None}
    exe.score(record, params)
    exe.run(pl.DataFrame([record]), params)
```

Every top-level input whose type is not `bool`/`int`/`str`/`bytes` gets
the float `1.0` -- including `list`, `dict`/struct and `datetime.date`.
Since spec 00 §7.3 makes `decision_date: date` mandatory on essentially
every capability in this project ("Every assessment names a
`decision_date`... 'Today' never appears in credit logic"), and several
capabilities take ragged `list` inputs by design (`core.obligations`,
`core.income`'s `variable_pay_history`), `decider build` on this project
failed outright:

```
TypeError: 'float' object is not iterable
  File ".../credit_core/income.py", line 125, in income_variability_ratio
    values = [v for v in history if v is not None]
in step credit_core_demo/income_variability_ratio, row 0
```

There is no `Handler.*_fn` override for this -- `stage()` calls the
module-level `_warm` directly, with no extension point. I could not find
a supported workaround, so `inference.py` monkeypatches
`decider.serving.handler._warm` at import time to warm using this
project's own `sample_request.json` (a real, complete record) instead of
the synthetic one. This is arguably a *better* warm-up input regardless of
the bug, but it is a workaround for a real gap, not a design choice I'd
defend on its own: **any decider project with a `date`, `list` or `dict`
top-level input cannot `decider build` unchanged today.** Given
§7.3/§9.15's date requirement runs through every one of this repo's
twelve specs, I'd expect every other project in this set to hit this too,
unless they also work around it.

### 4.2 A `frame_step` output typed `list[struct]` crashes result materialisation

`core.obligations` needed to return a per-account annotation alongside the
scalar aggregate (00 §6.4). The natural representation is
`list[struct]` -- one dict per account. Any terminal (unconsumed)
`frame_step` output of that shape crashes both `.run()` and `.score()`:

```python
@frame_step(reads=["x"], writes=["z"])
def f(df):
    return df.with_columns(pl.Series("z", [[{"a": 1}] for _ in range(len(df))]))
# ValueError: cannot parse numpy data type dtype('O') into Polars data type
```

Root cause: `FrameStep.to_ir` declares every output `Any`-typed
(`decider/steps/frame.py`); the generic output path
(`decider/engine/run/state.py::_series`) round-trips through
`np.where(...).tolist()` before constructing the final `pl.Series`, and
numpy cannot box a list of Python dicts. A scalar or a *flat* ragged
`list[float]`/`list[int]` output from the same `frame_step` works fine
(confirmed) -- it is specifically nested list-of-struct. Workaround:
`core.obligations` returns four parallel, index-aligned lists
(`obligation_account_type_codes`, `obligation_treatment_codes`,
`obligation_monthly_amounts`, `obligation_is_internal`) instead of one
list-of-dicts column. Functionally equivalent, but it is a real interface
compromise a consumer has to know to zip back together.

### 4.3 `missing_as([])` on an entirely-absent `list` column corrupts a later frame projection

An input declared `variable_pay_history: list[float] = missing_as([])`,
when the column is *entirely absent* from the request (not merely null --
absent), combined with **any** `frame_step` elsewhere in the same
pipeline, breaks frame construction for that later node:

```
polars.exceptions.ShapeError: unable to add a column of length 0 to a DataFrame of height 1
  File ".../decider/engine/run/state.py", line 170, in frame_of
    return base.with_columns(cols) if cols else base
```

This reproduces with a plain scalar `float` output too, as long as its
input was a `missing_as([])`-filled absent list column somewhere upstream
-- the corruption is in that column's own state representation, not in
whatever reads it. Workaround: never omit a declared ragged-list input;
send it explicitly (`"variable_pay_history": []` or with real values).
Separately, a genuinely-empty list *literal* in a single-record JSON
request (`pl.DataFrame([{"x": []}])`) infers as `List(Null)`, which then
fails at the arrow-import boundary (`ArrowImportError: Expected array
with 0 buffer(s) but found 1 buffer(s)`) regardless of `missing_as` --
so an empty ragged list needs at least one same-typed sibling value
anywhere in the record to type-infer correctly on the single-record path.
Both of these are real gotchas for any consumer sending JSON with omitted
or empty array fields, which is an extremely common shape for optional
evidence in this domain (00 §7.4's "not collected" null situation is
*exactly* "the field is absent").

### 4.4 A `list`-typed scalar-step input is a numpy array, not a Python list, in its row loop

`decider`'s interpreted runner passes each row's list-typed value as a
numpy array, not a `list`, to a plain-function step
(`decider/engine/run/runners/interpreted.py::_call`, via
`.tolist()`-then-`zip`, which does not fully convert nested list values).
`variable_pay_history or []` and `decline_reason_codes or []` -- an
idiom that is completely ordinary Python for "use the default if falsy" --
both raised `ValueError: The truth value of an array with more than one
element is ambiguous`. Every place this project reads a `list`-typed
input, it now checks `is None` explicitly, never truthiness. Worth a
docstring note next to `missing_as` in the framework, since nothing in
`missing_as`'s own documentation flags this.

### 4.5 `DecisionTableConfig`'s open-edge rule is table-wide, not per match-group

A `between` expression allows exactly one row with an open lower bound and
one with an open upper bound *in the whole table* --
"only row 0 may have an open lower edge" -- even when other `eq`
conditions (grade, product, cover type, segment) partition the table into
independent groups that each logically need their own open ends. Every
banded table in this project that repeats its band ladder per key
(`core.rate_card` per grade, `core.risk_grade`/`core.expense_norms`/
`core.credit_life`/`core.deductions` per group) hit this at construction:

```
ValueError, Row 11: upper bound unresolvable — only row 47 may have an
open upper edge.
```

Workaround: use `float("-inf")`/`float("inf")` as the outer bounds
instead of `None`, in every group, everywhere. This works and is
documented inline (`rate_card.py::_bands`), but it is not obvious from
`DecisionTableConfig`'s own docstring or `tests/tables/test_tables.py`,
whose examples are all single-group tables. Any table keyed by more than
one dimension where one dimension is banded will hit this the first time
someone writes more than one group.

### 4.6 Compiled modes reject comparing a `str` column to another `str` column

`core.expense_norms.norm_table_version` picks between two version-id
`str` columns (`internal_norm_amount >= amount` decides which). `fused`/
`stepped` mode refuses to bind it:

```
ValueError: credit_core_demo/norm_table_version: reads several `str`
inputs [...]; compiled modes compare a `str` input only with a `str`
param, so split the step or run it in interpreted mode
```

Unlike the others above, this is a clear, actionable, build-time error
(not a runtime crash), so I count it as a documented limitation rather
than friction -- but it does mean this project's demo pipeline can only
be served in `interpreted` mode (SERVE.md explains why), and the
93k-row Flex Loan card only got meaningfully fast (28 s cold compile,
~3 ms/lookup warm) under `fused` mode when I tested it in isolation --
so there is a real tension between "compiles" and "string-selects between
two versions" that a consumer choosing `fused` for throughput will hit.

### Smaller things

- `Engine().bind(step).run(dict)` is a common typo trap: `.run()` wants a
  polars `DataFrame`, `.score()` wants a `dict`. The error
  (`AttributeError: 'dict' object has no attribute 'columns'`) does not
  say which method to use instead.
- The single-record `.score()` path silently falls back to
  `pl.DataFrame([dict(record)])`-based type inference whenever *any*
  `frame_step` is present in the pipeline (see 4.1's docstring: "a
  pipeline without frame steps never builds a polars frame on this path").
  A consumer with no frame steps gets the faster, safer typed `_load`
  path; adding one `frame_step` anywhere silently downgrades every
  request onto the path that hits 4.1 and 4.3. This is not documented
  anywhere I found; I only discovered it by reading `engine/run/engine.py`.
- `DecisionTableConfig`'s cell-level `cell_id`/`*_version` convention
  (this project's own idiom, not the framework's) means composing more
  than one table capability in a flow needs manual `.relabel()`
  discipline (§ "What I publish" above) -- decider gives you `.relabel()`
  to fix it, but doesn't warn you a second table's default `cell_id`
  write silently collides with the first's until the dag build fails.

### What worked well

`.relabel()` and `.named()` are genuinely excellent for exactly the
composition problems this spec is full of (rename at the boundary,
run twice with different settings). `param()`/`missing_as()`'s
documented type table (IR.md §5.1) is accurate and complete for scalar
types. `DecisionTableConfig`'s `rows: {"table": "name"}` params-document
indirection is exactly the "table refresh without a redeploy" mechanism
00 §9 asks for, and `decider build`'s numba warm-up genuinely does
eliminate first-request latency once it succeeds. `pipeline.parameters().defaults()`
made generating a complete, correct `params.json` for a 20-plus-node
pipeline trivial.

---

## 5. Spec problems

- **00-ADDENDUM A5 (fraud verdict) has an unclear owner.** The addendum
  lists it under "capabilities... 00 does not define" and says "publish
  the verdict contract", but DEPS.md's own resolution of the 03-against-01
  soft dependency has *03* stub the contract with its own field names
  (03 §4.4), not 00. I published the vocabulary (`credit_core.vocab.FraudVerdict`)
  since it costs nothing and prevents the exact "invent the same
  three fields twice" problem the addendum is warning about, but the
  spec and its own addendum disagree about whose job this is.
- **"Core.affordability's ownership boundary" is stated but easy to get
  wrong in practice.** DEPS.md's "Cycles" §2 says 00 owns the five
  affordability units' "arithmetic, tables and interfaces" while 02 owns
  "household framing, the four modes, the verdict, the three answer
  shapes". In practice the verdict computation (`affordability_verdict_code`)
  sits right on that boundary -- 00 §6.5 lists it as one of the unit's
  *outputs*, but "the verdict" is also explicitly called out as 02's
  job. I built a single pass/marginal/fail/indeterminate verdict as part
  of the unit (matching §6.5's literal output list) and left the *four
  modes* that would classify differently to 02, but a different
  implementer could reasonably have left the verdict out of 00 entirely.
- **Table sizing instructions conflict for expense norms.** SCOPE.md's
  own rule 1 says "generate large tables synthetically at their full
  dimensions... do not shrink them", but the same document's 00 section
  only names `core.rate_card` for full-size treatment and is silent on
  whether `core.expense_norms`, `core.credit_life`, `core.risk_grade` and
  `core.appetite` (also declared as sized tables in 00 §8) should be full
  size or "working depth". I read "working depth" (the phrase SCOPE.md
  uses for the affordability units) as extending to these tables too,
  since the alternative (all six sized tables at full declared size) is
  not achievable at this slice's stated line budget alongside everything
  else asked for -- but the document doesn't say so explicitly.
- **00-ADDENDUM's own count problems (00 §C1) are real and I hit them
  directly**: I built against "twenty-two" capabilities (00 §6's own
  header) and the addendum's own correction elsewhere says "twenty-one"
  is used just as often. I did not attempt to resolve which count is
  authoritative; I built what 00 §6 enumerates.

---

## 6. What I would do next

1. Widen `core.scorecard` to a realistic 20-45 characteristics and give
   it a second scorecard id, to exercise addendum A6's "more than one
   score per decision, told apart by scorecard and role" for real (today
   only `SCORECARD_ID`/`SCORECARD_VERSION` constants exist as the hook).
2. Externalise every table (not only the rate card) as a
   `configs/<version>/*.json` document, so the "table refresh without a
   redeploy" property (00 §9) is demonstrated for the whole library, not
   one capability.
3. A proper fees-inclusive APR solve for `effective_annual_rate`
   (currently the compounded nominal rate only -- see the `ponytail:`
   comment in `instalment.py`), needed once a consumer wants disclosure-
   grade output.
4. Push on decider's `_warm()` limitation upstream (4.1 above) rather
   than working around it per-project -- every one of the eleven other
   specs in this set names `decision_date` as mandatory, so every one of
   them will hit this the first time they run `decider build`.
5. A second major version of one capability (`core.rate_card` or
   `core.risk_grade`), deliberately, to prove §7.1's "two majors live
   simultaneously" requirement -- explicitly skipped for this slice.
