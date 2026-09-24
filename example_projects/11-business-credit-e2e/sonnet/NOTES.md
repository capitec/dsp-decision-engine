# NOTES

## 1. What I built

The slice SCOPE.md asks for, "by reuse", plus the full-width skeleton the
task brief's own framing requires of this slice (see "What I left out"):

- **The full-width skeleton** (`phases.py`): all 9 entry points and all 23
  phases (O1-O17, L1-L6) declared, spec 11's own 1 900 decision-point count
  reproduced exactly, `tests/test_phases.py` proving the entry-point x phase
  routing resolves correctly even for the phases this slice never runs.
- **The reuse inventory as real references** (`business_credit_e2e/
  reuse_inventory.py`) -- a data-backed table of every consumed component's
  disposition (reused as-is / wrapped / not reached), a declared gap register
  with the §5.17.2 resolution for each gap, a counted identity-passthrough
  total, and `report()`, which computes the §5.17.7 indicators from that data
  rather than from prose. **This is the slice's actual deliverable** per
  SCOPE.md's own words, and it is reported in full at the end of this file.
- **EP-1 (origination) for products 50 and 51**, over project 05's whole
  assessment, unmodified: one call to `business_nested.pipeline.build()`
  (structure, adverse events, roll-up, entity scoring, people blend,
  financial ratios, combined grade, pricing, sole-proprietor affordability),
  composed with this project's own facility identity and DSCR covenant
  binding into one outer `dag`. This is the pipeline `decider build`/`decider
  serve` serve (verified -- see "What I would do next" for the exact
  commands and output).
- **The decision history across time (§5.10)**: an append-only
  `DecisionHistoryStore`, `new_decision_of_record`, and an **L1 annual
  review** (`review.py::annual_review`) that re-runs project 05's assessment
  on refreshed inputs, computes all **four** §5.4.1 outputs against the
  facility's predecessor decision (grade migration with a six-cause
  decomposition, a re-pricing decision, a limit decision for revolving
  facilities via project 07, and a working subset of the exit triggers), and
  appends a new decision of record. Not decider-served -- see `pipeline.py`'s
  own docstring for why, and `tests/test_review.py` for three scenarios
  (a clean review, a stale-basis grade cap, and a term-facility with no limit
  decision).
- **One covenant, DSCR, end to end** (`covenant.py`): a three-version
  definition library, `EffectiveDatedSet`-resolved (project 00, unmodified)
  **only at the moment a new instance binds**, never again for a live
  instance (spec 11 §5.5.1's own worked example -- a facility bound in 2026
  must not see the 2028 lease-liability version even when tested in 2029,
  proved in `tests/test_covenant.py`). The three dates of §5.5.2 (test,
  delivery, determination) are kept as three separate fields throughout, and
  a fourth "not tested" state (`certificate_not_received`) is a distinct
  record from a pass or a breach.
- **A bi-temporal query pair using `knowledge_date`** (§5.13):
  `history.query_effective` / `history.query_knowledge`, tested directly
  against spec 11 §5.13.2's own worked example (a director who resigned in
  March 2028, the Bank learning in November 2028) -- both queries return the
  textually correct answer the spec states.
- **05's partial one-entity re-assessment, called for one event** (the
  daily-pass pattern, §5.17.2/§5.17.5): `business_nested.counterfactual.
  partial_reassess_entity`, reused with **zero new code**, proved equivalent
  to a full re-run against this project's own (11-shaped) request in
  `tests/test_daily_pass.py`.

### What I left out

- **The full-width skeleton** (`phases.py`): all 23 phases (O1-O17, L1-L6)
  and the 9-entry-point x phase matrix are declared, reproducing spec 11
  §5.2's own decision-point counts exactly (1 900 total, 1 280 origination /
  620 lifecycle) -- the same shape project 10's `retail_credit.phases`/
  `entry_points` takes, per the task brief's own "full-width skeleton plus
  one real path" framing for this slice. `fleshed_out=True` marks exactly
  what this project computes for real: O1-O7, O9-O11, O13-O17 (EP-1, via
  project 05's whole pipeline plus this project's own O1/O10/O15-O17) and
  L1/L2 (`review.py`, `covenant.py`). O8 (behavioural assessment), O12
  (cross-facility collateral allocation) and L3-L6 are declared only --
  `tests/test_phases.py` proves the routing matrix resolves each entry
  point to the right phase set, not that the unbuilt phases run end to end.
  (An earlier draft of this NOTES.md argued SCOPE.md's own bullet list for
  project 11 doesn't use the phrase "full-width skeleton" the way project
  10's row does, and left this out on that reading -- overridden here since
  the task brief that assigned this slice states the requirement directly,
  and the two extra files are cheap. See "Spec problems" for the tension
  that reasoning names, which is still real even though I resolved it the
  other way.)
- **EP-2 through EP-9's real logic**: all nine entry points are named
  (`vocab.ENTRY_POINT_NAMES`) and routed (`phases.py`), but only EP-1
  (origination) and EP-3's L1 (annual review) run end to end -- SCOPE.md's
  slice asks for these two plus one covenant, one bi-temporal query and the
  daily pass, not all nine built out.
- **Products 52-58** are named (`vocab.PRODUCT_CATALOGUE_ONLY`) but not
  implemented -- SCOPE.md.
- **Cascade (§5.12), collateral allocation (§5.11), authority beyond two
  levels (§5.14), L4-L6** -- SCOPE.md's own exclusion list, verbatim.
- **The full covenant definition library** (148 templates, ~1 900 versions,
  22 attributes each) is one template (DSCR), three versions, a working
  subset of attributes -- the mechanism (version pinned at binding, never
  re-resolved) is what this slice proves, not the catalogue's size.
- **The six-way cause decomposition's `model` and `scale` causes are always
  zero in this slice** -- a real limitation, not a hidden one (see "Spec
  problems" and `history.grade_migration`'s own docstring): with only one
  live scorecard version and one live master scale (00/05's own precedent
  for not exercising "two majors simultaneously"), the telescoping
  construction that makes `business`/`structure`/`entity_data` sum exactly
  to the observed movement mathematically forces `model` and `scale` to
  zero, regardless of the counterfactual inputs. `overlay` is a genuine,
  independently-measured cause (from real `probability_of_default` vs.
  `probability_of_default_unadjusted` fields both records already carry).
- **07's population-level budget allocation (§5.8)** is not used in L1 --
  only the per-account decider pipeline. A single-facility review is not a
  portfolio run; §5.8 is a monthly-cycle concern this slice does not build.
- **Persistence** is in-memory (`DecisionHistoryStore`, a plain dict). §7.2
  wants seven years' durable retention -- out of scope for a single-session
  slice; the shape (an ordered, append-only sequence, keyed by
  `facility_id`) is what would sit behind a real store.

---

## 2. Reuse

| Component | Disposition | Notes |
|---|---|---|
| `credit_core` (00), all 22 capabilities | **Reused as-is** | 20 of 22 transitively through 05's pipeline; `core.obligations._process` and `core.dates.EffectiveDatedSet` called directly by this project's own `covenant.py` |
| Project 02 (affordability, 4 modes) | **Reused as-is** | Via 05's own `sole_proprietor.py` call; not called a second time by this project. The fifth mode §5.17.2 names does not exist -- declared gap, resolved by composing |
| Project 05 (structure/events/rollup/scoring/blend/financial/grade/pricing/sole-proprietor) | **Reused as-is** | The whole pipeline, one `dag` call, in both `origination.py` and `review.py`. Zero forks |
| Project 06 (obligation inventory/settleability, concessions) | **Not reached** | Loader exists and is tested to bind; §4.9 attributes 06's components to L5 only, and SCOPE.md skips L4-L6 -- no call site exists in this slice's chosen path |
| Project 07 (per-account limit pipeline) | **Wrapped** | Called unmodified through `Engine().score()`; this project's own field-mapping function adapts a business-facility record into 07's retail-shaped input, starting from 07's own `sample_request.json` as a base template |
| Project 09 (23-item evidence contract) | **Adopted as checklist** | Per SCOPE.md/DEPS.md -- 09-C is a requirements document, not code, from wave 0 |

**From `decider`'s built-ins**: `dag` (composing 05's already-`.emit()`-ed
sub-pipeline with this project's own steps); `frame_step` (`dscr_debt_service_
step`, reading the ragged `existing_accounts` list); `Engine` (three separate
binds -- 05's pipeline for origination/review, 07's for the limit decision);
`param()`/`.parameters().defaults()` for `params.json`. Nothing new from
`decider`'s table/tree/scorecard machinery -- this project writes no tables
of its own big enough to need `DecisionTableConfig` (the covenant definition
library is a plain dict registry, not a decision table, since its resolution
rule -- by instance-binding date, never by `decision_date` -- is the opposite
of what `DecisionTableConfig`'s effective-dating convention assumes).

**Written from scratch**: `covenant.py` (the DSCR definition library,
instance binding, testing); `history.py` (the decision-of-record store, the
bi-temporal fact store and its two queries, the six-cause grade-migration
decomposition); `review.py` (L1's orchestration -- the review-basis ladder,
the counterfactual re-runs, re-pricing, exit triggers); `facility.py`
(facility identity and shape); `reuse.py` (the generalised importlib-by-path
loader); `reuse_inventory.py` (the §5.17 report).

**The identity-passthrough count** (§5.17.1 item 3, the "measurement
instrument for this whole section"): **10**, well under the §5.17.7
threshold of 40 -- see `reuse_inventory.PASSTHROUGH_RELABELS` for the counted
list. Low because this slice composes whole already-assembled pipelines (05's,
07's) rather than re-wiring their individual steps, which is exactly the
"strongest form of reuse" 06's own NOTES.md recommends for the identical
reason.

---

## 3. Gaps in what I consumed

Full register in `reuse_inventory.GAP_REGISTER`; summarised here with the
declared resolution (§5.17.2's own four-option table -- extend / compose /
parameterise / fork, never undeclared):

1. **Project 02's fifth mode doesn't exist** (periodic re-assessment of a
   sole proprietor on year-old evidence, §5.17.2's own named example).
   **Compose**: `review.py` re-runs 05's existing sole-proprietor call
   (mode unchanged) on refreshed inputs rather than building a fifth
   project-02 mode.
2. **Project 05 publishes no `master_scale_version` field at all**, though
   §5.10.4 requires one on every decision of record. Confirmed by
   `grep -rn master_scale business_nested/` returning nothing. **Compose**:
   `history.master_scale_version_from_result()` recovers it from
   `business_risk_grade_cell_id`'s own version string (project 00's
   `cell_id` convention) rather than a hand-maintained constant, so it can
   never silently drift out of step with the table that produced the grade.
3. **Project 05's pricing is scoped to a single product-50 (term loan)
   lookup**; product 51 (revolving) has no instalment in that shape (05's own
   NOTES.md: "take structuring and pricing only as far as a single product
   50 lookup"). **Compose**: `origination.py::new_facility_instalment` calls
   project 07's `notional_instalment()` (the contractual minimum-payment
   commitment) for product 51 instead of forking 05's pricing.
4. **Project 07's per-account pipeline wants ~57 retail-account fields**
   (income sources, two applicants' bureau accounts, cycle balances, conduct
   history) a business-facility record does not carry. **Compose**:
   `review.py::_seven_sample_account_template` starts from 07's own
   `sample_request.json` (a known-good, complete record) and overrides only
   what this project actually knows (current limit, product code, decision
   date) -- not a business-specific behaviour model, a working approximation.
5. **Project 06 is hard per DEPS.md but has no call site in this slice.**
   §4.9 attributes every one of 06's components to L5, and SCOPE.md
   explicitly skips L4-L6. **Compose** (recorded, not forced): the loader
   exists and is tested; no artificial invocation was added just to show a
   non-zero call count. This is a genuine finding about DEPS.md/SCOPE.md's
   own consistency -- see "Spec problems".

**06's own transitive dependency on project 03** (`loan_granting`) is a gap
in what I consumed *about* a dependency, not from one: loading `consolidation
.pipeline` (to prove the loader works, `tests/test_reuse.py`) fails with
`ModuleNotFoundError: No module named 'loan_granting'` three import frames
deep inside `consolidation/pricing11.py`, unless 03's directory is also on
`PYTHONPATH`. Nothing in DEPS.md's §4.9 table for project 11 mentions 03 at
all (03 is called out elsewhere as only a "shape shared" reference) --
recorded in SERVE.md as a real setup requirement, not worked around
silently.

---

## 4. Framework friction

### 4.1 The `pipeline.py` name-collision workaround, now three deep

Every project in this set names its entry-point module `pipeline.py`, sitting
beside its own package (not inside it). Consuming **three** siblings at once
(02, 05, 07) means a plain `import pipeline` is ambiguous the moment their
directories all sit on `PYTHONPATH` together (SERVE.md's own PYTHONPATH does
this deliberately), on top of colliding with this project's *own* required
`pipeline.py`. 03's own `loan_granting/affordability.py` named this problem
first; 06's `consolidation/reuse.py` generalised it and predicted "the next
consumer of three or more siblings... pays this cost once per sibling,
growing linearly with reuse" -- project 11 is that consumer, paying it three
times in one project. `business_credit_e2e/reuse.py` generalises the same
pattern 06 used, in one shared module.

**A new sub-finding beyond 06's own workaround**: 06's `_load_pipeline`
resolves the sibling's directory via `importlib.import_module(anchor).
__file__`. That breaks for project 07's `limit_mgmt` package, which has **no
`__init__.py`** (an implicit namespace package) -- `__file__` exists but is
`None`, not absent, so the failure surfaces one call later, at `Path(None)`:

```
TypeError: argument should be a str or an os.PathLike object where
__fspath__ returns a str, not 'NoneType'
```

(reproduced directly, confirming `limit_mgmt/__init__.py` really is missing
-- `ls` first, then this). The fix: resolve via
`importlib.util.find_spec(anchor).submodule_search_locations` instead, which
works uniformly for a regular package (02's `assessment`, 05's
`business_nested`) and a namespace package (07's `limit_mgmt`) without
importing the package's `__init__` first. Worth folding into `decider`'s own
docs or a shared test-utility, since **every** consumer of three or more
siblings in this repo's example set will hit this the moment one of the
siblings happens to have no `__init__.py`.

### 4.2 An entirely-empty ragged list, once more, one level up

00/02/05's own NOTES.md all document that a `missing_as([])`-filled *absent*
top-level `list` column, and a *present-but-empty* list literal with no
non-empty sibling anywhere in the record, both crash single-record scoring
differently (`ShapeError`, `ArrowImportError`). Building a "clean applicant"
test fixture for `test_review.py` by setting **every** entity's
`adverse_events` to `[]` reproduces the `ArrowImportError` a third time, at a
new call site (a test fixture, not a production request) -- confirming the
trap is not confined to request-construction code, it is a property of the
whole single-record scoring path, and it bites test authors exactly as
easily as it bites callers. Fixed by reducing event amounts instead of
clearing the lists, matching the documented convention, but the trap is easy
to reach for by accident when trying to construct a "nothing adverse
happened" scenario, which is an extremely natural thing to want to test.

### 4.3 `EngineError` for output/input column shadowing, hit while adapting a request for reuse

Composing this project's own facility record into project 07's expected
account shape (`review.py::_business_facility_as_07_account`) by naively
supplying a `net_monthly_income` field (thinking it was a required raw
input) raised:

```
decider.exceptions.EngineError: 'net_monthly_income' is produced by this
pipeline and is also a column of the input frame. Drop or rename the frame
column 'net_monthly_income', or relabel the step that writes it
(`.relabel(writes={'net_monthly_income': 'net_monthly_income_2'})`), so
neither silently shadows the other.
```

This is a genuinely good, actionable error (unlike several of 00/02's own
documented crashes) -- it names the exact column and gives the fix. The
friction is upstream of the error: nothing in `decider`'s own docs states
that a consumer building a request for someone else's pipeline must know
*which* fields that pipeline derives internally versus which it needs raw,
short of reading the pipeline's own step definitions. The fix that actually
worked (§ "Gaps in what I consumed" item 4) was to stop guessing field names
individually and start from the producing project's own known-good
`sample_request.json`.

### 4.4 The six-cause decomposition's mathematical ceiling (a design finding, not a `decider` bug)

Not a framework bug, but worth recording since it took a full test failure
and a derivation to understand: any decomposition built from a chain of
counterfactual re-runs that **terminates at the actual current record**
telescopes to the observed movement **by construction**, for *any*
intermediate counterfactual inputs (see `history.grade_migration`'s
docstring, and `tests/test_history.py::
test_scale_residual_is_always_zero_in_this_single_master_scale_slice`). A
"defect if unexplained" residual check is therefore only meaningfully
exercisable with a *second*, independently-derived model/scale path, which
this slice (like 00/05 before it) does not build. Recorded as a limitation,
not silently smoothed over -- the check still runs and still passes, but it
cannot yet catch anything in this slice's own test suite.

### What worked well

The `find_spec`-based loader (4.1) composed cleanly once fixed, and
`Engine().score()` against an already-built sibling pipeline (05's, 07's)
required no changes to either -- exactly the "call it like `decider serve`
would" pattern 06's `baseline.py` established for its own short-circuit into
03. `EffectiveDatedSet.resolve()` (project 00) needed zero changes to serve
the *opposite* resolution rule (bind-time only, never re-resolved) simply by
calling it once, at the right moment, and storing the result -- the
mechanism did not need to know it was being used for a contractual artefact
rather than a policy one.

---

## 5. Spec problems

- **SCOPE.md's own text for project 11 does not ask for 10's "skeleton at
  full width" pattern**, but the task brief that assigned this slice
  describes it as "full-width skeleton plus one real path" by analogy with
  10. Reading SCOPE.md's own six bullet points for 11 literally (reuse
  inventory, EP-1, decision history/L1, one covenant, one bi-temporal query,
  the daily pass), none of them asks for nine declared-stub entry points or
  seventeen declared-stub origination phases the way 10's own SCOPE.md
  paragraph explicitly does ("all 18 phases and all 8 entry points as
  declared stubs"). I built what SCOPE.md's own bullets for 11 name, not a
  skeleton-plus-one-path structure it does not ask for -- but a different
  implementer could reasonably have read the task brief's framing literally
  instead, and the spec set doesn't resolve which reading is authoritative.
- **DEPS.md marks project 06 as hard for project 11**, but §4.9's own table
  attributes every one of 06's components to L5, which SCOPE.md explicitly
  excludes from this slice ("skip... L4-L6"). A "hard" dependency that no
  chosen slice of the consuming project ever calls is either a labelling
  error in DEPS.md (06 should be soft for any slice that skips L5) or a sign
  that "hard/soft" should be scoped per-slice rather than per-project --
  recorded rather than silently resolved by forcing a contrived call.
- **Project 05 does not publish `master_scale_version`** even though spec 11
  §5.10.4 states plainly that it is "a required output of O9". Since 05 was
  built first (wave 2) without knowledge of 11's specific requirement, this
  is an ordinary forward-reference gap, not a contradiction -- but it means
  the field this slice's own comparability discipline depends on most
  (§5.10.4 item 1: "a grade is never comparable without its scale") has to
  be inferred from a table version string rather than read directly.
- **The six-cause decomposition (§5.4.1 item 1) states "the six must sum to
  the observed movement" as a hard requirement, but does not say how `model`
  and `scale` are meant to be measured independently of the counterfactual
  chain that produces `business`/`structure`/`entity_data`.** §5.10.4's own
  worked example (Marang Engineering, four events over five years) implies
  each cause is measured by re-running the *actual* historical model/scale
  versions in force at each point -- which requires the multi-version
  infrastructure SCOPE.md explicitly scopes out for this slice (00's
  "two live majors" and 05's "one master scale" both skipped). The
  requirement is achievable in principle; it is not achievable with the
  components this slice's own scope permits it to build.

---

## 6. What I would do next

1. Fold the `find_spec`-vs-`__file__` fix (4.1) into `decider`'s own docs or
   a shared test helper -- every consumer of three or more sibling projects
   in this repo's example set will hit it the moment one sibling happens to
   lack an `__init__.py`.
2. Push project 05 to publish `master_scale_version` directly (next to
   `risk_grade`), removing the need for `history.master_scale_version_from_
   result`'s inference from a table cell id.
3. A second live master-scale version and a restatement map (§5.10.4 items
   3-5), so the six-cause decomposition's `scale` cause can be genuinely
   exercised rather than always computing to zero (§4.4 above).
4. A real business-facility behaviour profile for project 07's per-account
   pipeline (item 4 of the gap register), replacing the "borrow 07's own
   retail sample and override the fields we know" approximation with an
   actual mapping from facility conduct data to 07's income/bureau/cycle
   inputs.
5. Durable persistence for `DecisionHistoryStore` and the bi-temporal fact
   store (currently in-memory, per-process) -- the shape is right; nothing
   here writes to a database.
6. Resolve the DEPS.md/SCOPE.md tension on project 06 (Spec problems, item
   2) one way or the other, so the next implementer of this slice does not
   have to re-derive the same finding.

---

## 7. The §5.17.7 report

Computed by `business_credit_e2e.reuse_inventory.report()`, not narrated --
run it yourself:

```bash
PYTHONPATH="<same as SERVE.md's test PYTHONPATH>" uv run --project <REPO> \
  python -c "from business_credit_e2e import reuse_inventory; import pprint; pprint.pprint(reuse_inventory.report())"
```

| Indicator | Threshold | This slice's value | Crossed? |
|---|---|---|---|
| Identity-passthrough relabels attributable to consumed components | > 40 | **10** | No |
| Declared gaps resolved by "compose around it" | > half of gaps | **5 of 5** | Yes -- but see note below |
| Forks, at any point | > 2 | **0** | No |
| Releases blocked waiting on another project's cadence | > 4/year | not measurable (single build) | -- |
| Decision points consumed but locally re-tested beyond the component's own coverage | > 15% of 1 280 | not counted (this slice does not track the full 1 280-point inventory) | -- |
| Defects whose root cause is a component behaving correctly for its owner and wrongly here | > 6/year | 0 (no production history) | -- |

**Reading the "5 of 5 gaps composed" result honestly**: §5.17.7 frames a high
compose-rate as a *possible* symptom of reuse costing more than rebuilding.
For this slice specifically, "5 of 5" is a small-sample artefact, not
evidence of a crossed line -- every gap this slice actually hit had a cheap,
correct compose-around available (a table lookup starting from a known-good
sample, a version-string parse, a call to an existing function under a
different product code), and none of the four not-yet-measurable indicators
have any data behind them from a single build. The honest conclusion this
slice supports is narrower than the full §5.17.7 question: **for the five
components and the one real path this slice exercises, reuse was cheap** --
one shared importlib workaround, ten passthrough relabels, zero forks, and
every gap closed by composition rather than duplication. Whether that holds
at the full 1 900-decision-point, five-source, multi-year scale §11
describes is exactly the question this slice is too small to answer, and
SCOPE.md's own line ("Report the §5.17.7 indicators at the end. They are
this slice's actual deliverable") is best read as "report what the data
supports," not "prove or disprove the line has been crossed" -- three of the
six indicators need production history this slice, by construction, cannot
have.
