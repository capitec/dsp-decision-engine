# NOTES

## 1. What I built

The 09-H slice SCOPE.md asks for: exact replay (§5.1), single-decision
explanation (§5.2), version diff (§5.4), swap-set attribution across
population/data/logic/overlay (§5.5, acceptance §10 item 20), the overlay
register with ageing and a stack-off run (§5.14), and dead-logic detection
(§5.8) -- run over the evidence of three real flows of different shapes: 01
(real-time rules), 03 (the solve), 05 (nested entities). 09-C (the §5.15
contract) was already adopted by 00/01/03/05 in wave 0, per SCOPE.md; this
project is 09-H alone.

- **`governance/flows.py`** -- a `FlowAdapter` per flow (01, 03, 05): how to
  build its pipeline at a config version, its date fields, its outcome/
  reason field names, which sibling projects it needs on `sys.path`. This is
  the one place a ninth flow would be added.
- **`governance/evidence_store.py`** -- captures a decision's evidence
  (request as received + the full scored record) and persists it, write-once,
  under `evidence_store/<flow_code>/<decision_id>.json`. See "Gaps in what I
  consumed" for why this exists at all: none of 01/03/05 persist evidence
  externally.
- **`governance/replay.py`** -- exact replay (§5.1). Re-scores a captured
  decision from its recorded config version, params and request, with no
  live call anywhere, and diffs every field the flow returned into
  exact/tolerance/diverged, giving the three-way verdict the spec's
  acceptance standard asks for (`reproduced` / `reproduced_within_tolerance`
  / `not_reproduced`), plus the first point of divergence.
- **`governance/explain.py`** -- single-decision explanation (§5.2), three
  audiences (consultant/analyst/adjudicator), built entirely from the
  captured record: the dominant reason and nothing internal for the
  consultant; every reason, the flow-specific mechanism detail (01's fired/
  evaluated rule sets, 03's cap chain, 05's entity/event attribution) and
  the adjusted/unadjusted pairs for the analyst; a narrated, identifier-
  glossed version with the policy version for the adjudicator.
- **`governance/diff.py`** -- semantic (not textual) version diff for the
  three shapes that recur: a `DecisionTableConfig` document (rows keyed by
  match columns, generalising `credit_core.rate_card.diff_cards` to any
  value column), a `TreeConfig` rule-set document (rules keyed by
  `meta.name`, never by list position -- proven against inserting a rule
  mid-priority), and a params document (key-path diff). `summarize_table_diff`
  produces the aggregated "1 840 of 63 360 cells... mean move +0.31... no
  cell moved down" statement §5.4 asks for, not a row list.
- **`governance/swapset.py`** -- runs a population through two versions and
  reports who moved (outcome, reasons appeared/disappeared, unmatched).
  `attributed_swap_set` is §5.5's "a release of n changes supports n+1 runs":
  cumulative increments over the flow's own params document, each one's
  swap set measured against the run immediately before it.
- **`governance/overlay_register.py`** -- the one estate-wide view over the
  `AdjustmentRegister`s 01/03/05 already own (§5.14.1): combined register,
  ageing report (`due_for_review`, reused directly), and the stack-off run
  (§5.14.3) -- through the *same* flow implementation, one params flag
  flipped generically wherever it appears, not a second calculation.
  `unwind_estimate` is the "estimated effect of unwinding" row §5.14.2 calls
  "the one with teeth", over a population of captured decisions.
- **`governance/deadlogic.py`** -- coverage over a batch: rule hit counts
  against each flow's own declared rule universe (01's live-rule catalog,
  03's cap-waterfall register, 05's roll-up rule constants), flagging dead
  (zero firings) and dominant (>40%) rules, plus a measured-not-thresholded
  table-cell-coverage fraction.
- **`pipeline.py`** -- the one capability wired through `decider`'s serving
  path: replay a decision by id. See "Framework friction" for why the other
  five capabilities are a tested Python library instead of `decider` steps.
- **`capture_demo_evidence.py`** -- populates `evidence_store/` from each
  flow's own `sample_request.json` plus a small deterministic population per
  flow, for the coverage/ageing/swap-set capabilities to have more than one
  decision to work over.

### What I left out

- **Drift and outcome monitoring (§5.9, §5.10), the regulator/ombud pack
  (§5.11), fairness testing (§5.12), PII masking beyond classification tags
  (§5.13)** -- explicitly out of scope for this slice, SCOPE.md.
- **The reviewable artefact (§5.7)** -- not named in SCOPE.md's 09-H list
  (§5.1-§5.13 minus the skips above); not built.
- **What-if intervention (§5.3)** -- not named in SCOPE.md's 09-H list
  either. `governance.overlay_register.stack_off_run` and
  `governance.swapset.attributed_swap_set`'s per-increment params transform
  are structurally the same "change one thing, re-run, compare" mechanism
  §5.3 asks for, but neither is built as a general-purpose "change any one
  input or param and compare" tool with §5.3's own non-confusability
  requirements (a distinct identifier, a visible marking, a guarantee it
  cannot be written into the decision store or issued to a client) -- that
  is real, unbuilt scope, not a renamed version of what exists here.
- **Scale.** Every population in this project is 1-9 decisions (the small,
  deterministic sweep `capture_demo_evidence.py` generates), not 09's own
  2 M-record swap-set or 400 000-record certification golden set. The
  mechanisms (`swap_set`, `attributed_swap_set`, `rule_coverage`) are the
  real ones and would run unchanged over a real population; nothing in this
  project has been proven at 09's own stated volumes.
- **Certification (§5.6)** -- not in 09-H's SCOPE.md list (the golden-set
  regression suite, coverage thresholds, the sign-off artefact are a
  separate, unbuilt capability).

---

## 2. Reuse

**From `credit_core` (project 00), unmodified:** `AdjustmentRegister.due_for_review`
and `.apply_stack`'s `stack_enabled` flag (the overlay ageing report and the
stack-off run are direct calls into 00's own mechanism, never a
reimplementation); the "reach past the frozen entry point" pattern for a
private-but-stable attribute (`register._all`), the same trade-off 02's and
05's own NOTES.md document for `core.obligations._process` and
`_ALWAYS_DISQUALIFYING`.

**From 01/03/05, as whole pipelines, not stubs:** every flow this harness
replays, explains, diffs and measures coverage over is the real
`pipeline.build(...)`, loaded and bound through `decider.Engine`, exactly as
that flow's own tests build it. Nothing here re-derives a decision; every
capability either reads a previously captured record or re-scores through
the real pipeline.

**From 01/03/05's own module-level singletons, reached directly (not a
published API, since none of the three publish "list your overlay
registers" or "list your rule universe"):** `fraud_interdiction.overlays.OVERLAY_REGISTER`/
`ADJUSTMENT_SET_ID`, `fraud_interdiction.rules.build_rule_catalog`,
`loan_granting.scoring.SCORE_ADJUSTMENTS`/`ADJUSTMENT_SET_ID`,
`loan_granting.waterfall.REGISTER`, `business_nested.overlays.EVENT_THRESHOLD_OVERLAYS`,
`business_nested.grade._BUSINESS_OVERLAYS`, `business_nested.scoring._ENTITY_OVERLAYS`,
`business_nested.rollup.AE_R_*`, `fraud_interdiction.reasons.REASON_REGISTRY`,
`loan_granting.reasons.REGISTRY`, `business_nested.outcome.REASON_REGISTRY`.

**The `pipeline.py` name-collision workaround** (`governance/paths.py::load_pipeline`,
`importlib.util.spec_from_file_location` under a distinct module name) is
the exact technique 03's `loan_granting/affordability.py` and 05's
`sole_proprietor.py`/`counterfactual.py` already document and use for the
identical problem (every project ships a `pipeline.py`) -- generalised here
to load three flows' worth instead of one project's own dependency, not
reinvented.

**From `decider`'s built-ins:** `Engine().bind(...).score(...)`/`.run(...)`
(every replay, capture and swap-set run); `step()`/`param()` for the one
servable capability; `pipeline.parameters().defaults()` for `params.json`,
same as every other project in this set.

**Written from scratch:** everything in `governance/` -- there is no
existing "governance harness" shape in `decider` or in 00/01/03/05 to build
on; this project's whole job is to be the first thing that reads across
flows instead of building one.

---

## 3. Gaps in what I consumed

- **None of 01, 03 or 05 persist decision evidence externally.** Each
  flow's `.score()` output satisfies 09 §5.15 items 1/6/10 (a stable
  decision id, the inputs as received survive on the same output dict, the
  complete params document is available from `pipeline.parameters().defaults()`),
  but nothing in any of the three writes that output anywhere durable --
  each project's own NOTES.md documents `.score()` returning a complete
  record as *the* evidence, with persistence explicitly out of scope for
  their own slices. `governance/evidence_store.py` is this project's stand-in:
  capturing evidence is one call to the flow's own `.score()` (never a
  second, harness-side derivation), which is the honest way to cover this
  gap without editing any of the three projects' directories (BRIEF: never
  edit another project's directory).
- **03 does not surface a rate-card cell id on its top-level decision
  record.** `loan_granting/pricing.py` computes `rate_cell_id` per offer
  candidate internally, but 03's `pipeline.py` `.emit()` list never
  publishes it (only `risk_grade_cell_id`/`calibration_cell_id` are). §5.8's
  "never-read table cells" measurement over the Flex Loan card (the spec's
  own headline example, 00 §8's 63 360-cell card) is therefore not
  reachable from 03's evidence as captured; `deadlogic.table_cell_coverage`
  is demonstrated against `risk_grade_cell_id` instead (see `tests/test_deadlogic.py`),
  a real but much smaller table. Worth 03 knowing this is a real evidence
  gap for governance, not only a convenience miss -- the rate card is
  exactly the artefact 09 §5.8 names by number.
- **No flow's own tests build a second config version.** Every version-diff
  and swap-set test in this project constructs its "before"/"after" pair by
  hand (a mutated copy of a real config document, or a params transform) --
  there is no second, real `configs/0.2.0/` anywhere in 00/01/03/05 to diff
  against. `tests/test_diff.py` and `tests/test_swapset.py` are honest about
  this: they prove the mechanism works, not that any of the three flows has
  actually shipped a second version.

---

## 4. Framework friction

### 4.1 The one servable capability had to be deliberately narrow, because the other five don't fit `decider`'s typed step model

This is the header finding, and it shaped `pipeline.py` directly. 00/01/03/05's
own NOTES.md all document the same underlying constraint from different
angles: a `frame_step`'s terminal output cannot be `list[struct]` (00) or
`list[list[T]]` (05); `Engine.run()` panics on a `list[str]` output where at
least one row is empty and another is not (01, confirmed with an 8-line
repro). Explanation, diff, swap-set and the overlay register all *produce*
exactly these shapes on purpose -- a cap chain is an ordered list of
(rule_id, value) pairs, a rule-set diff is a list of (rule_id, change,
before, after) records, a swap-set report is a list of (decision_id,
before, after) moves. Forcing any of them through `decider`'s step model
would mean re-encoding every one as parallel flat lists the way 00/01/05's
own multi-valued outputs already do -- workable for a flow's *own* evidence
(where the grain is "per rule of this one flow"), but this project's outputs
have a different, harder-to-flatten grain (a diff is keyed by an id that
may not even exist on one side; a swap-set report's `outcome_moved` list has
no fixed width). I judged that re-encoding purely to satisfy `decider build`
would produce exactly what SCOPE.md and the BRIEF both warn against --
complexity added for the framework's sake, not the domain's -- so
`pipeline.py` wires through `decider`'s serving path the one capability that
is genuinely scalar-shaped (replay: an id in, a verdict and a count out),
and the rest is a plain, tested Python library, called directly by an
analyst's tool or a notebook the way `decider`'s own docs describe testing a
capability "standalone, no pipeline needed" for 00's units. I did not find
a documented way to make `decider` accept a step whose output is "a list of
records of unknown, per-call-varying width" -- if one exists, I missed it,
and it would change this design.

### 4.2 `find_param`'s naive "match any key by name" needed a second look because of a real ambiguity `decider`'s own params-document shape creates

03's `evaluation_ceiling_per_term` step and its one param share the exact
same name (`pipeline.py`: `def evaluation_ceiling_per_term(evaluation_ceiling_per_term:
int = param(24))`), which is unremarkable and probably common (a step whose
whole job is "carry one tunable" is naturally named after that tunable).
But it means a naive "find every dict key named X" walk over a params
document matches *twice* for that one param: once at the step-dict level
(`loan_granting/evaluation_ceiling_per_term`, whose value is itself a dict)
and once at the actual leaf (`.../evaluation_ceiling_per_term/evaluation_ceiling_per_term`).
Setting the first match first corrupts the document for the second (`TypeError:
'int' object does not support item assignment`, confirmed directly -- see
`tests/test_diff.py::test_find_param_does_not_match_a_step_that_shares_its_param_s_name`).
The fix (only match when the value is not itself a dict) is a one-line
guard, but it is worth flagging because the recursive-"find and flip every
`adjustment_stack_enabled`" pattern this project reuses from 03's own
`tests/test_pipeline.py` has the identical latent bug the moment a step and
its param share a name -- 03's own test only avoids it because
`adjustment_stack_enabled` never doubles as a step name there.

### 4.3 `.score()`'s output is not directly JSON-serialisable, because `decision_date` (and any other `date`-typed output) survives as a real `datetime.date`

Persisting a decision (`evidence_store.capture`, the thing 09 §5.15
implicitly asks every flow to be able to do) is the first place in this
project that calls `json.dumps` on a real `.score()` result, and it fails
outright without help: `TypeError: Object of type date is not JSON
serializable`, confirmed directly on all three flows. Not a defect --
`.score()`'s own docstring never claims JSON-serialisability, only "a dict
of the output row" -- but it is a real, immediate wall for the one thing 09
§5.15 needs every flow to be able to do (persist a record durably), and
nothing in `Executable.score`'s docstring or 00/01/03/05's own NOTES.md
flags it, because none of the three actually persists a record itself
(see "Gaps in what I consumed"). `governance/evidence_store.py::json_safe`
recurses through the result once and converts every `date` to its ISO
string, which is enough here; I looked for, and did not find, a numpy
scalar anywhere in any of the three flows' actual `.score()` output despite
00/01/05's own NOTES.md documenting numpy arrays appearing on the *input*
side of a step's row loop -- confirmed empirically (`type(v).__module__ ==
"numpy"` over every field of a real `.score()` call, on all three flows,
found none), so `json_safe`'s numpy-scalar handling is defensive rather
than something this project's own evidence needed.

### 4.4 Same finding as 00/01/03/05's own NOTES.md: `decider build`'s warm-up cannot synthesise this project's own inputs either, for a third, distinct reason

00/01/03's NOTES.md document `_warm`'s dummy-record synthesiser mistyping
`date`/`list`/`dict` inputs. This project's two inputs (`flow_code`,
`decision_id`) are both plain `str`, so it is clear of that specific bug --
but it still needed the same `inference.py` override, because the
synthetic dummy string is `""` for every `str` input, and `""` is neither a
known `flow_code` nor a `decision_id` this harness has evidence for:
`run_replay("", "", ...)` raises `KeyError`/`FileNotFoundError` during
warm-up. Three different projects in this set (00/01/03's `date`/`list`
mistyping, this project's "valid-looking but meaningless" string) have now
independently hit `_warm`'s synthetic record failing to be a usable request
for a real pipeline, for three different reasons -- reinforcing 00's own
"what I would do next" #4: push a real fix upstream (accept the project's
own `sample_request.json` as the documented warm-up source, rather than
every project needing its own `_warm` monkeypatch).

### Smaller things

- `Engine().bind(pipeline, mode=...).plan.versions` (used briefly while
  investigating whether "declared inputs" could be read off the plan
  generically, per 00's own NOTES.md finding) is not part of `Executable`'s
  own documented surface (`Executable.__doc__`/`.score.__doc__`/`.run.__doc__`
  describe `run`/`score`/`report`/`step` only). I ended up not needing it
  (`evidence_store.capture` takes the caller's own request dict directly,
  rather than re-deriving which fields are "inputs" from the bound plan),
  but confirming that took reading past the documented API, the same as
  00's own NOTES.md reports doing for the same attribute.
- `AdjustmentRegister` has no public way to enumerate its overlays (only
  `for_target`, `due_for_review`, `__len__`) -- reasonable for a flow that
  only ever asks "what applies to *this* target", genuinely awkward for a
  harness whose whole job is "list everything, across every flow, that
  exists". `overlay_register.py`'s `._all` reach is the same "reach past
  the frozen entry point" trade-off other projects in this set already
  document elsewhere; a documented `.all()`/`__iter__` would remove it.

---

## 5. Spec problems

- **SCOPE.md's 09-H list omits §5.6 (certification) and §5.7 (the
  reviewable artefact) without saying so explicitly.** The main spec's §5
  numbers fourteen capabilities; SCOPE.md's 09-H sentence names six by
  number/description ("exact replay (§5.1)... explanation (§5.2)... diff
  (§5.4)... swap-set attribution (§5.5)... the overlay register... (§5.14)...
  dead-logic detection (§5.8)") and then says "Skip drift and outcome
  monitoring, the regulator pack, fairness testing and PII masking beyond
  classification tags" -- which names §5.9-§5.13 as skipped but never
  mentions §5.3 (what-if), §5.6 (certification) or §5.7 (the reviewable
  artefact) at all, in either the "build" list or the "skip" list. I read
  their absence from both lists as "not this slice" (SCOPE.md's own
  discipline elsewhere is to name what a slice skips explicitly when it
  matters), consistent with the main spec's own framing that §5.7 in
  particular is "the requirement most likely to be underestimated" and
  "people-blocked, not code-blocked" (§5.7's own text) -- not something a
  single implementer session should attempt without the "real credit risk
  analyst" review the spec itself says the format needs before being fixed.
  A different implementer could reasonably have read the omission as an
  oversight and built one of the three anyway.
- **§5.5's swap-set report table (nine measures) is written for a 2 M-record
  population with grade/product/channel/segment breakdowns; SCOPE.md gives
  09-H no population to run it over.** Unlike 01/03/05, which each consume a
  named upstream artefact (a rule set, a rate card, an assessment), 09-H has
  no data source of its own -- it operates on whatever the three flows
  happen to have evidence for, and SCOPE.md is silent on where a
  "recent representative population" (§5.5's own phrase) comes from for a
  harness with no production traffic behind it. `capture_demo_evidence.py`'s
  small synthetic sweep is this project's own answer, documented as a
  demo-scale stand-in (see "What I left out" -- Scale) rather than the
  spec's own stated case.

---

## 6. What I would do next

1. Widen `capture_demo_evidence.py`'s population from a handful of hand-
   chosen variants per flow to hundreds, and run `swap_set`/`rule_coverage`
   at a scale where "dead" and "dominant" findings are more than an
   artefact of a nine-record sample.
2. Build the what-if capability (§5.3) as its own module, with the
   non-confusability guarantees §5.3 states explicitly (a distinct result
   identifier, a visible marking, a check that a what-if result cannot be
   saved into `evidence_store/`) -- `overlay_register.stack_off_run` and
   `swapset.attributed_swap_set`'s per-increment params transform are close
   in mechanism but were not built to that guarantee.
3. Add the reviewable artefact (§5.7) for one flow (03 is the best
   candidate -- eleven stages, a bounded rule count) as a generated,
   layered rendering, and get it in front of someone who has not read this
   codebase, per §5.7's own "test against a real reviewer before the format
   is fixed" advice.
4. Push 03 to surface `rate_cell_id` (or a `rate_card_cell_id` alias) on its
   top-level decision record, closing the "Gaps in what I consumed" table-
   cell-coverage gap against the spec's own headline example.
5. Chase `find_param`'s step-name/param-name collision fix (4.2) and the
   `.score()` numpy-scalar finding (4.3) upstream; both are small, general
   fixes that remove a real trap for the next project that serialises a
   decision record or writes a recursive params-document walk.
