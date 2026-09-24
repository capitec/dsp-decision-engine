# NOTES

## 1. What I built

The SCOPE.md slice for project 08, all seven pieces it names:

- **Account state** (`collections_treatment/state.py`) -- `arrears_bucket_code`
  computed (not read), honouring §4.3's two policy overrides: held at the
  arrangement-entry bucket while an arrangement performs, floored at bucket 3
  after two cures in 24 months. `balance_band_code` (7 bands) and
  `arrears_to_balance_ratio` alongside it.
- **Suspensions** (`suspensions.py`) -- all 20 codes (§5.2 table), evaluated
  unconditionally every time (no early return anywhere in
  `evaluate_suspensions`), each with its own scope, source, blocking/downgrade
  flag and a computable-or-event expiry. `permitted_treatment_codes` derives
  the empty set only when a hard-block code fired. Nothing here reads an
  `AdjustmentRegister` -- §5.4's "nothing in this stage is overlayable" holds
  structurally, not by convention.
- **Collections score** (`scoring.py`) -- a 10-characteristic `ScorecardConfig`
  (working depth against the spec's 28), `core.calibration`'s sigmoid reused
  unmodified for roll probability, 6-band table, a contact-responsiveness
  band, and recovery-estimate/cost-to-collect curves (bucket x band, working
  depth against bucket x band x family x balance x 24 months). All **four**
  overlay kinds (§5.3 table) run through one `AdjustmentRegister`
  (`SCORE_ADJUSTMENTS`), each on its own `target`: score shift (add),
  scaling change (multiplies the calibration curve's own `scale`), odds
  multiplier (multiplies roll probability), band boundary shift (an additive
  shift subtracted from the probability the band table reads -- mathematically
  equivalent to moving the edge, and it keeps the six-row band table itself
  unmodified across every overlay). The unadjusted score and band are carried
  beside the adjusted ones on every record.
- **Treatment matrix** (`matrix.py`) -- the full 5 376 cells (8x6x7x4x4),
  generated, this slice's dominant-difficulty artefact kept at real size per
  SCOPE.md rule 1. A second, independent `AdjustmentRegister`
  (`MATRIX_ADJUSTMENTS`) for the three matrix-overlay kinds: intensity dial,
  treatment suppression (falls back to no-action; a named per-scope
  substitute is left out, see §6 below), and (consumed downstream, not
  applied here) allocation weighting. Both registers are checked at
  construction (`assert_no_statutory_target`) against a frozen set of names
  (`vocab.STATUTORY_TARGETS`) an overlay may never target -- §5.4's "rejected
  at authoring time, not at run time".
- **Escalation path over time** (`path.py`) -- episode open/close, the five
  reset events as one small declared table (`_RESET_EFFECTS`) rather than an
  if/elif chain, the monotonic intensity floor, the asymmetric roll-mid-
  sequence rule (counters carry, position resets to the new bucket's entry,
  intensity floors at the greater value), and the four-row re-entry-after-
  curing table. Run **twice** through the same function -- once on the
  as-known-on-the-day event flags (drives today's actual treatment), once
  relabelled onto `_as_at_now` variants of the same five flags (§5.5, §10
  item 4) -- and `tests/test_temporal_replay.py` drives a crafted late-
  arriving-payment scenario through both and shows they diverge, and that
  replaying the same event log twice reproduces the same positions.
- **Arrangements** (`arrangements.py`) -- every arrangement tested through
  project 02's `core.affordability`, in ARRANGEMENT mode (`assessment_mode_code
  = 3`, named in the evidence, never implied), reusing 02's own
  `evidence_unit()`/`capacity_unit()` step objects directly rather than
  reimplementing their wiring. On top, this project's own sustainability test
  (§5.6: residual >= R350 and >= 5% of net income; instalment-to-discretionary
  <= 85%, against 65% in granting) -- a second test layered over 02's
  capacity output, not a fork of 02's own buffer/verdict logic.
- **Capacity allocation** (`capacity_alloc.py`) -- one `frame_step` over the
  whole population: pool demand from the treatment codes assigned, a working
  subset of §5.9's 12 capacity pools with daily supply, an 8%-reserved-share
  fairness floor for balance bands 6-7, a switchable ranking basis (value vs.
  priority), and 6 of the spec's 9 non-selection reason codes (200, 205, 210,
  220/230, 240, 250 -- see "What I left out" for 260/270).

### What I left out

- **Settlement and discount offers (§5.7), promise-to-pay acceptance/
  classification (§5.8), the intraday re-run and the real-time path (§5.12),
  and legal handover mechanics** -- all explicitly out of scope per SCOPE.md.
  Suspension 119 (promise in force) is still evaluated from a given
  `promise_date`, since §5.2's 20-code list requires it; what's skipped is
  the *acceptance* of a new promise, not the suspension it creates.
- **Non-selection reasons 260 (sibling account already covered) and 270 (no
  permitted channel available)** -- both need per-client account-grouping
  (260) or per-channel consent-point validation (270) beyond this slice's
  single-account request shape; the other 7 codes are implemented.
- **Intensity-dial scaling of `permitted_retries`/`cooling_off_days`** (§5.4
  says an intensity dial may scale these too, not only shift intensity) --
  only the intensity shift is implemented; the matrix's own
  `matrix_permitted_retries`/`matrix_cooling_off_days` pass through
  unadjusted into `path.interval_and_cap_check_step`.
- **Named substitute treatments on suppression** -- §5.4's "optionally
  falling back to a named alternative"; this slice's suppression always
  falls back to no-action.
- **A business-day calendar with public holidays** -- `suspensions.py`'s
  `_add_business_days` is Mon-Fri only (a `ponytail:`-style note in that
  module documents the upgrade path); spec 08 §6 names a real "10 years x
  public holidays" gazette table as its own parameter artefact.
- **200k-account / 10-day simulation volume** -- `test_temporal_replay.py`
  proves the as-known/as-at-now mechanism and its replay property on a
  handful of crafted synthetic accounts over 10 simulated days, not the
  full population. SCOPE.md's rule 1 ("keep the dominant difficulty at real
  size... generate large tables synthetically at their full dimensions") is
  about *tables* -- the 5 376-cell matrix is at full size, proven in
  `test_matrix.py::test_matrix_is_the_full_5376_cells`. A pytest-driven
  200k x 10-day loop is a load-test volume, not a correctness one; see
  "What I would do next".

---

## 2. Reuse

**From project 00 (`credit_core`), unmodified:** `credit_core.adjustments`
(`Adjustment`, `AdjustmentEffect`, `AdjustmentRegister`, used directly via
`.apply_stack()` rather than the `apply_stack_step()` convenience wrapper,
since this project's overlay scopes -- bucket, product family, agency
placement history -- aren't the wrapper's fixed four keys; its own
docstring explicitly invites this: "a consumer needing a wider scope...
builds its own step from `apply_stack` directly"); `credit_core.calibration
.probability_of_default` (the roll-probability curve, reused as-is --
00's own module built it as a generic score-to-probability sigmoid, not a
credit-granting-specific one); `credit_core.evidence.cell_id` (every table's
cell attribution, the same convention as 00 and 02).

**From project 02, unmodified:** `pipeline.evidence_unit()` and
`pipeline.capacity_unit()` -- 02's own dag-composed units, loaded and called
directly (not reimplemented) from `arrangements.py`. See "Framework
friction" for the module-name collision this took real engineering to work
around safely.

**From `decider`'s built-ins:** `DecisionTableConfig` for every table (the
5 376-cell matrix, the 6-band and unadjusted-band tables, the
contact-responsiveness band, the recovery curve, the balance band);
`ScorecardConfig` for the collections score; `frame_step` for capacity
allocation (the one genuinely population-level stage -- see `path.py`'s
docstring answer to §13 Q5); `dag`/`flow` for composition (`flow` where a
step must overwrite a table's own output column in written order -- the
calibration-table-then-scaling-change pair -- everything else as `dag`,
letting the wiring resolve dependency order); `.relabel()`/`.named()`
throughout, including running `path.resolve_path_position_step` twice with
different settings (the as-known / as-at-now pair) and running the band
table twice with different boundary names.

**Written from scratch:** everything under `collections_treatment/` -- this
is what spec 08 §4.1 says the library does not publish ("The library...
does not publish collections names. This project declares the following").
The escalation-path composition (`path.py`) and the suspension evaluation
(`suspensions.py`) are the two places genuinely bespoke sequence logic
lives, per spec 08 §13 Q1's own framing -- see "Spec problems" for where I
landed on that question.

---

## 3. Gaps in what I consumed

- **Project 02's `pipeline.py` is not importable by its own published name
  from a second project that also needs a top-level `pipeline.py`.** This
  is the single largest gap, covered in full under "Framework friction"
  below since it is squarely a `decider`-convention problem (every project's
  build entry point is named `pipeline.py`, per the BRIEF), not a defect in
  02's own design. Worked around via `importlib.util.spec_from_file_location`
  under a private module name in `arrangements.py`, never a bare `import
  pipeline`.
- **02's `evidence_unit()`/`capacity_unit()` expect a two-applicant
  household shape** (`applicant1_*`/`applicant2_*`), which this project
  reuses unmodified for a single account holder -- applicant 1 carries the
  real data, applicant 2 is the solo-application placeholder convention 02's
  own NOTES.md/SERVE.md already documents (one harmless placeholder element
  per ragged field, never `[]` or an omitted key). No 08-specific
  "single-applicant" mode was requested in 02's own mode table, so this
  reuses the existing solo-application path exactly as 02's five other
  consumers would.
- **Project 00's `pipeline.py` also collides on the same name.** Not
  something this project consumes directly, but it means *any* two projects
  from this set sharing one `PYTHONPATH` -- as 00, 02 and 08 all do here --
  carry the same latent risk if a caller's `PYTHONPATH` ever puts this
  project's own directory anywhere but strictly ahead of `DECIDER_API__CODE_PATH`'s
  guaranteed position. Documented prominently in SERVE.md since it bit the
  build during this project's own development (see "Framework friction").

---

## 4. Framework friction

### 4.1 A consuming project's own directory on `PYTHONPATH` silently defeats `DECIDER_API__CODE_PATH`'s priority, and `decider` then serves the wrong project's pipeline with no error

`decider/serving/handler.py`'s warm-up does:

```python
if code_path not in sys.path:
    sys.path.insert(0, code_path)
```

This guard means `CODE_PATH` is only guaranteed `sys.path[0]` priority the
*first* time it is added. If this project's own directory is *already* on
`sys.path` -- for instance because an operator's `PYTHONPATH` lists every
project directory uniformly, including this one's own, which is an entirely
natural thing to do when consuming two upstream projects the same way (this
project already needs 00's and 02's directories on `PYTHONPATH`, and it is
an easy habit to add the project's own directory too, "for consistency") --
the guard sees it is already present, skips the insert, and this project's
directory keeps whatever position `PYTHONPATH`'s own ordering gave it.
Since this project, project 00 and project 02 *each* publish a top-level
`pipeline.py` (00's and 02's, per the same BRIEF convention this project
follows), `import pipeline` then resolves to **whichever one is earliest in
`PYTHONPATH`**, not this project's.

Reproduced directly while writing this project's own SERVE.md:

```
$ export PYTHONPATH="<00 dir>:<02 dir>:$PWD"   # $PWD = this project's own dir, appended
$ decider build
Error: config version latest failed to build: KeyError: "the config version has no
'rate_card_flex_loan' document for argument 'rate_card_flex_loan' of 'pipeline:build'"
```

`rate_card_flex_loan` is project 00's own `build()` argument -- `decider`
silently built **project 00's** pipeline under this project's config
directory. The error it surfaces (a missing config document) looks like a
config problem and gives no hint that the wrong `pipeline.py` was loaded at
all; a less lucky combination (project 00's or 02's pipeline building
*successfully* against this project's params document, which is entirely
possible since both take mostly optional/defaulted inputs) would mean
`decider serve` silently serves the wrong project's decision logic with no
error whatsoever. Removing this project's own directory from `PYTHONPATH`
(letting `CODE_PATH`'s insert actually run) fixes it -- confirmed, and now
documented as the load-bearing instruction in this project's own SERVE.md.

This is the reason `arrangements.py` never does a bare `import pipeline` to
reach project 02's dag composition, and instead loads it by file path under
a private `sys.modules` key (`importlib.util.spec_from_file_location`,
`sys.modules["_affordability_pipeline_02"] = module`). That workaround
protects *this project's own code* from importing the wrong `pipeline`.
It does **not** protect `decider`'s own `DECIDER_API__PIPELINE` resolution
from the same risk -- that can only be fixed by keeping this project's
directory off `PYTHONPATH` and relying on `CODE_PATH` alone, which is what
this project's SERVE.md now says explicitly and in bold.

**This is worth fixing in `decider` itself**, not only documenting:
`_warm`'s `sys.path.insert(0, code_path)` could be unconditional (`list.
remove` then `insert(0, ...)` if already present, rather than skipping), so
`CODE_PATH` always wins regardless of what else is on `PYTHONPATH` and in
what order.

### 4.2 A `param()`'s key inside the request record is silently ignored, not rejected

`adjustment_stack_enabled: bool = param(True)` is a *param*, set only
through `.score(record, params={node_name: {...}})` -- never through a key
inside `record` itself. Putting it in `record` compiles fine, wires fine,
and runs fine; it is simply ignored, and the step falls back to the param's
own default. This is not surprising once understood (params and columns are
genuinely different channels), but it produced four false-negative-looking
test failures while writing `tests/test_matrix.py` and `tests/test_scoring.py`
-- assertions like `assert out_off["treatment_intensity"] == 2` failed with
`3 == 2` because `"adjustment_stack_enabled": False` inside the record was a
silent no-op, not the intended override. A `param()`-typed key present in
the record dict raising (or at minimum warning) rather than being silently
dropped would have caught this immediately instead of three separate test
debugging sessions.

### 4.3 Chaining `.relabel(writes=...)` on top of an already-relabelled step is a silent no-op, not an error

`build_band_table().relabel(writes={"collections_band_code": "..._unadjusted"})`
does nothing, because `relabel`'s `writes` dict keys on the step's
*original* declared write names (`"band"`, `"cell_id"` for a
`DecisionTableConfig`), not whatever an earlier `.relabel()` call already
renamed them to. Two copies built this way both keep the *first* relabel's
write name, and `dag()` then raises (correctly) `WiringError: ... both
write 'collections_band_cell_id'` -- a real, confirmed collision, but a
confusing one to debug back to its cause, since the error names the
*symptom* (two nodes writing the same name) rather than the *cause*
(chained relabel is not compositional). Fixed here by parameterising the
table-builder function itself on its final output names
(`scoring._build_band_table(read_from=..., code_output=..., cell_output=...)`)
and calling it twice, rather than relabelling once and relabelling again.
Worth a line in `.relabel()`'s own docstring: "keys reference this step's
*original* names, even after an earlier `.relabel()` call."

### 4.4 `missing_as(False)` is not usable as a bare Python default outside `decider`'s own execution path

`decider.engine.params.declare.missing_as(False)` returns a `MissingAs`
marker object that is *not* itself a `bool` (Python's `bool` cannot be
subclassed, so the "carrier trick" the module uses for `float`/`int`/
`str`/`list`/`dict`/`tuple` defaults -- where `missing_as(0.0)` genuinely
*is* `0.0` when called directly -- silently doesn't apply to `bool`):

```python
>>> from decider import missing_as
>>> m = missing_as(False)
>>> bool(m), isinstance(m, bool)
(True, False)
```

Every step signature in this project with an optional boolean input
(`agency_placed_12m: bool = missing_as(False)`, `deceased`, `dispute_raised`,
and a dozen more in `suspensions.py`) works correctly when run *through*
`decider` (`Engine().score()`/`.run()`), because `harvest()` checks
`isinstance(d, MissingAs)` explicitly and reads `.fill`, never relying on
the default's own truthiness. It is calling the plain Python function
*directly* -- the natural first thing to do when unit-testing a step's
logic in a REPL, and exactly what I did while writing `suspensions.py`'s
first draft -- that silently misbehaves: every `missing_as(False)` default
evaluates as truthy, so every optional suspension flag appeared to "fire"
with no arguments passed at all. `decider.engine.params.harvest.
call_with_defaults` exists precisely to call a step function correctly
outside the engine, but nothing points a step author at it from
`missing_as`'s own docstring, and `bool` is the one type among
`missing_as`'s documented carrier types where "the default is also the
value" quietly stops being true.

### 4.5 Chained `dag()` composition over a same-named write needs `flow()`, and the error only appears at `.parameters()`/build time, not at construction time

`dag(scoring.build_calibration_table(), scoring.scaling_change_step, ...)`
constructs without complaint; the `WiringError` ("collections_calibration_segments
and scaling_change both write 'scale'; use flow(...) to apply them in
written order, the later one winning") only surfaces once something calls
`.parameters()`, `.run()` or `.score()` (i.e., forces `to_ir()`). This
matches 02's own NOTES.md precedent for the identical pattern (its
`overlay_unit`), so it is a confirmed, repeatable shape rather than a
one-off -- worth documenting once, centrally, rather than every consuming
project rediscovering it. `dag()`'s own docstring could name this
explicitly ("two steps that both write the same output need `flow()`, not
`dag()`, and this is only checked when the pipeline is built, not when it
is assembled").

### Smaller things

- `AdjustmentRegister` has no public way to enumerate every `Adjustment` it
  holds (only `for_target(target)`, keyed on a target you must already
  know). `matrix.assert_no_statutory_target`/`scoring.assert_no_statutory_target`
  reach into the register's own `_all` attribute (name-mangled-looking but
  not actually private by Python's rules, just undocumented) to validate
  every adjustment's target at construction time. A public `.all()` or
  `.__iter__` would remove the only place this project touches a
  leading-underscore attribute of a library object.
- A genuine JSON `null` for an optional `date` field
  (`debt_review_default_date: date | None = None`) infers as a `Null`-dtype
  polars column on the single-record scoring path and crashes at the arrow
  boundary (`ArrowImportError: Expected array with 0 buffer(s) but found 1
  buffer(s)`) -- the exact "lone `None` scalar" finding 02 NOTES.md 4.1
  made, reproduced here for five different optional date fields rather than
  one. Worked around the same way 02 did: a harmless non-null placeholder
  date, documented in SERVE.md, rather than a genuine `null`.

### What worked well

`frame_step` for capacity allocation is exactly the right escape hatch for
"this decision depends on every other record that ran today" -- the one
place in this pipeline that is not row-shaped, and `decider` has a purpose-
built mechanism for it rather than forcing a workaround. `AdjustmentRegister
.apply_stack()` called directly (not through the convenience wrapper) is
genuinely flexible enough to express four structurally different overlay
kinds on one mechanism, exactly as spec 08 §5.3 asks. `.relabel()` running
one function twice with different bindings (the as-known/as-at-now path
position, the two band tables) is precisely the composition primitive
spec 08 §13 Q1 is fishing for, and it did not need a bespoke "run twice"
mechanism of its own.

---

## 5. Spec problems

- **§13 Q1's own question ("is sequence position a fourth core component
  kind?") doesn't have a clean single answer, and the spec doesn't resolve
  it either.** This project's answer is "no, composed from state plus one
  small declared table of reset-event effects plus a handful of genuinely
  arithmetic rules (the monotonic floor, the roll-carry rule)" -- but that
  composition is one function (`path.resolve_path_position`) with five
  branches, not obviously the shape that scales to "40 hand-written
  filters" the spec worries about avoiding. Whether five reset events stays
  a small table as more products/families are added, or grows into
  something that wants its own `ConfigurableStep` kind, is genuinely open;
  §13 poses the question but the spec gives no acceptance criterion that
  would tell an implementer which answer is right.
- **§4.3's bucket-floor rule and §5.5's re-entry-table floor rule use
  different windows for what reads as the same policy.** §4.3: "an account
  that has cured twice in **six months** is floored at bucket 3." §5.5's
  re-entry table: "Cured twice in **12 months** ... bucket floored at 3."
  This project implemented §4.3's rule using `times_cured_24m >= 2` (a
  24-month field, since no 6-month field is declared in §4.1's vocabulary)
  in `state.py`, and left §5.5's re-entry-position table keyed on
  `times_cured_12m` as literally written. The two may be the same rule
  described inconsistently (six months vs. twelve), or two genuinely
  different rules that happen to share a threshold count -- the spec
  doesn't say, and no declared vocabulary field exists for a genuine
  6-month cure count.
- **Non-selection code 250's own worked example undercuts §5.9's ranking-
  basis switch.** §5.9 says the ranking basis (value vs. policy priority)
  "is a parameter, not a constant... switchable by Collections Strategy
  without a release", but §5.9's own code-250 example ("You were 91 412th
  of 186 000 in the early agent pool") only makes sense as a rank *number*
  if every account in the pool was ranked on the *same* basis that day --
  which is true in this implementation (one basis per allocation run) but
  the spec never states that constraint explicitly, leaving open whether a
  basis change mid-day (e.g., a live re-rank after the intraday re-run,
  itself out of this slice's scope) is meant to be possible.

---

## 6. What I would do next

1. **Fix `decider`'s `_warm` sys.path guard** (friction 4.1) upstream --
   unconditional `sys.path.insert(0, code_path)` rather than skip-if-present,
   so `CODE_PATH` always wins regardless of `PYTHONPATH` order. This is the
   highest-value fix in this write-up: every project in this set that
   consumes another project's directory via `PYTHONPATH` while also
   publishing its own `pipeline.py` (all of waves 1-4, by the BRIEF's own
   convention) carries this same latent risk.
2. Named per-scope substitute treatments on suppression (§5.4's "optionally
   falling back to a named alternative"), rather than always no-action.
3. Intensity-dial scaling of `permitted_retries`/`cooling_off_days`, not
   only `treatment_intensity` (§5.4's full three-field dial).
4. Non-selection codes 260 (sibling account already covered -- needs
   per-client account grouping, which this single-account-per-request
   pipeline doesn't have) and 270 (no permitted channel -- needs
   per-channel consent-point validation against `core.consent`).
5. Scale `test_temporal_replay.py`'s as-known/as-at-now demonstration from
   a handful of synthetic accounts to the full ~200k-account/10-day volume
   SCOPE.md names, under a batch profiler rather than pytest, to also prove
   the 90-minute batch window (§8) at that scale.
6. A real business-day calendar (public holidays), replacing
   `suspensions.py`'s Mon-Fri approximation, for the notice-period and
   promise-window expiry computations that spec 08 §6 names as their own
   quarterly-and-on-regulatory-change parameter table.
