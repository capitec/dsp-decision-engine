# 11 — Business credit, end to end

This is the reuse half of the deliberate A/B described in the spec's §2.2:
project 10 is built standalone, from nothing; this one is assembled from
components that already exist, and roughly 1,280 of its 1,900 decision points
are consumed rather than written. The 36 files in this tree are a sketch of
what that assembly looks like in a real codebase — no function runs, every
body is `pass  # what it does`, and every construct invented here (`consume`,
`role`, `Compared`, `cross_record`, `Cascade`, `governance_matrix`,
`consumer_contract`, `bound_table`, `append_only_artefact`, `Expand`,
`approval_binds_to`) is a demand this project makes of the framework, not a
feature it can assume.

Read this file with the tree open. Every claim below points at a real path.

---

## 1. The directory, and why it survives 1,900 decision points and 14 teams

```
11-business-credit-end-to-end/
  roles.py                        # THE central invention. Read before any pipeline.
  vocabulary.py                   # the systematic renames — 8 pairs, 2 prefixes
  consumed/                       # everything this project consumes. Spec 4.9, as code.
    __init__.py                   #   the reuse surface: SURFACE = consumed_surface(...)
    manifest.py                   #   per-assessment version pinning (the ugly mechanism)
    core_library.py               #   all 22 project-00 capabilities, bound once
    p02_affordability.py          #   project 02, plus the missing fifth mode
    p05_origination.py            #   project 05's 13 stages as O2–O16, 1,280 dp
    p06_restructure.py            #   project 06's concession search + project 07's limits
    GAPS.toml                     #   the 12 declared gaps: EXTEND / COMPOSE / PARAMETERISE
    FORKS.toml                    #   the fork register — 0 of 0, and the 7 named pressures
  time/                           # the other central invention. H1–H3, in one place.
    dating.py                     #   POLICY vs CONTRACT — two resolution rules, two types
    bitemporal.py                 #   KNOWN vs ACTUAL — bi-temporal views, no default
    comparability.py              #   Compared[T] — "not comparable" as a value
    master_scale.py               #   the one artefact resolved under both rules
  modules/                        # local logic, new to this project
    request/relationship.py       #   O1 — four dates, four state flags
    appetite/facility_types.py    #   O11 — appetite depends on facility STATE
    security/allocation.py        #   O12 — cross-facility collateral allocation (H4)
    covenants/
      definition.py               #   the covenant definition library (H2, the 4th change class)
      schedule.py                 #   a covenant instance as a frame-tier subject generator
      setting.py                  #   O15 — where a policy value becomes a contractual one
      test.py                     #   L2 — contractual=True, no Dated[T] permitted
      waiver.py                   #   waivers as core.adjustments, scoped by instance
    authority/routing.py          #   O17 — the matrix, and the level that moves (H7)
    watchlist/signals.py          #   L3 — 186 signals, six grades, escalation asymmetry
  lifecycle/                      # what happens to a facility after origination
    amendment/reopen.py           #   L4 — the re-open matrix IS the control flow
    annual_review/
      basis.py                    #   L1 part 1 — five review modes, none of them an error branch
      migration.py                #   L1 part 2 — the document's spine (H1, §5.10.4)
    forbearance/classification.py #   L5 — both limbs, including the evidenced negative
    exit/handoff.py               #   L6 — a consumer-driven contract on a flow Recoveries never runs
  pipelines/
    origination.py                #   O1–O17 in one expression — 12 of 17 lines `from consumed`
    entry_points.py               #   the 9 entry points, each `Origination.scoped_by(...)`
    annual_review.py              #   EP-3 — basis chosen first, four comparison outputs
    covenant_test.py              #   EP-4 — 3 lines, 148 decision points, runs almost no origination
    early_warning.py              #   EP-5 — the daily pass, singleton calls, incremental + full
    cascade.py                    #   EP-8 — bounded propagation, dual attribution
    pre_assessment.py             #   EP-9 — structurally incapable of writing a decision of record
  config/manifests/                # (stub) where a frozen batch manifest would be written
  policy/                          # (stub) where policy-owned narrative would sit
  schemas/                         # (stub) input-frame schemas per pipeline (doc 07 §1)
  tables/governance/               # (stub) the matrices: review_scope, amendment_reopen, arod_lifecycle_disposition
  tables/covenants/dscr/           # (stub) one of the 63 tables' worth of actual rows
```

Four organising decisions, and each is defensible only because of what it
buys at scale:

**1. `consumed/` is a directory, not an import scattered through the tree.**
Its own docstring states the payoff as a shell command:

```
grep -rn "^from consumed" ..   ->  the complete list of consumption sites
grep -rn "^from modules"  ..   ->  the complete list of local logic
```

`pipelines/origination.py` makes this literal: twelve of its seventeen phase
imports are `from consumed.p05_origination import (...)`; five are `from
modules...`. The reuse ratio — the project's headline number — is legible
from one import block without reading a phase body. That is only possible
because nothing under `modules/` or `pipelines/` ever imports a project-05 or
project-00 symbol directly; every such symbol is re-bound once, in
`consumed/`, and consumed from there. `consumed/__init__.py` is explicit that
this is the point: *"A consumed component never appears at a use site under
its own import path."*

**2. `roles.py` and `vocabulary.py` live at the project root, not inside
`modules/`.** They are read before any pipeline, by design — the docstring on
`roles.py` says so directly. Burying the vocabulary mapping inside a module
directory would hide the one file a new engineer, a policy analyst or an
auditor most needs on day one: the file that says what `applicant_age_years`
means when it is a surety's age at the final instalment date instead of a
retail applicant's age today.

**3. `time/` is its own top-level package, not a helper module.** H1
(comparability), H2 (two resolution rules) and H3 (bi-temporality) are the
three hardest difficulties named in the spec (§2.5), and all three constructs
that answer them — `Compared`, `dated_table`/`bound_table`, and the
`known`/`actual` views — live in four files that nothing else in the tree can
avoid importing. `modules/covenants/test.py` imports `time.dating.CONTRACT`
precisely because it must; `pipelines/origination.py` imports
`time.bitemporal.KNOWN` for the same reason. Putting these in `modules/time/`
would suggest they are one concern among many; putting them beside
`consumed/` says they are load-bearing for the whole tree, which they are.

**4. `modules/` and `lifecycle/` are split, even though both hold local
Python.** `modules/` is logic a phase inside origination needs (`O1`, `O11`,
`O12`, `O15`, `O17`, the watchlist). `lifecycle/` is what happens to a
facility *after* origination — amendment, annual review, forbearance, exit.
The split matters because it is also the ownership split: everything under
`lifecycle/annual_review/` is Business Credit Risk Policy and Credit Risk
Modelling's; everything under `modules/covenants/` is Covenant Operations,
Legal and Business Credit Risk Policy jointly. A team asked to review its own
surface (spec §9.4 requirement 2, "sliceable by owner") starts by knowing
which top-level directory is theirs, before any tool renders anything.

**5. `pipelines/` stays thin.** `pipelines/origination.py` is seventeen names
piped together with `|`; it contains no credit logic at all, which is
exactly what the spec's §2.2 requirement 1 demands ("no consumed component
may be re-specified here"). `pipelines/entry_points.py` is the one file that
turns one pipeline into nine — every entry point is
`Origination.scoped_by(review_scope, "EP-n")`, so there is no second
expression, anywhere, of which phases an entry point runs.

The empty stub directories (`schemas/`, `tables/governance/`,
`tables/covenants/dscr/`, `policy/`, `config/manifests/`) are deliberate, not
missing: this sketch is the Python authoring surface, and the 63 tables, the
252-cell review scope matrix and the 392-cell amendment re-open matrix are
*data* — CSVs and JSON owned by Product, Treasury, Legal and Credit
Governance — that a policy analyst edits without touching any file in this
listing. Sketching their contents would misrepresent who writes them.

---

## 2. The two central inventions

### 2.1 Roles — vocabulary mapping declared once

`roles.py` opens with the problem stated as the spec states it (§5.17.1): the
library publishes `applicant_age_years`, `gross_monthly_income`,
`existing_obligations` and 28 more, every one named for a retail applicant.
Inside this flow every one of them belongs to somebody with a role, and the
role changes what the name *means* — not just its spelling:

```python
SURETY = role(
    "surety",
    over="known.entities",
    where="is_required_surety",
    maps={
        core.applicant_age_years: "entity.age_years_at_final_instalment",
        core.gross_monthly_income: "entity.personal_income_cents",
        core.existing_obligations: "entity.obligations_excl_this_suretyship_cents",
    },
    nulls={
        core.months_employed: NOT_APPLICABLE.when("entity.is_retired"),
        core.employment_type_code: NOT_APPLICABLE.when("entity.is_retired"),
    },
    params={"criticality_class": "entity.criticality_class"},
    discloses="entity",
)
```

Doc 03 §5.2 offers three layers for adapting a module to different names —
name matching, a project vocabulary map, `.at()` instance relabel — and
`roles.py`'s docstring walks through why each one fails here specifically:
name matching assumes one name means one thing; a vocabulary map is a
function from name to name and this needs one right-hand side *per role*;
and `.at()` is per-instance (40 entities × 3 roles = 120 declarations), lives
in the composition file where a reviewer at the use site can't see which
semantics applied, and can only rename — it has nowhere to put the
parameterisation (`criticality_class`) or the null policy
(`months_employed` is `NOT_APPLICABLE` for a retired surety). Doc 01 §5.1's
79 identity-passthrough steps are what happens when none of the three
layers fit and a project reaches for a rename-shaped workaround anyway.

So `role(...)` is proposed as a **sixth scope**, alongside doc 03 §3's step,
module, branch arm, loop body and pipeline. It carries four things that
travel together — `maps=`, `params=`, `nulls=`, `discloses=` — because they
are useless apart: a role that only renamed a value would still let a
criticality class leak in as an unparameterised assumption, or a
role-inapplicable null get folded into "not established" and silently drag
down the 75% coverage ratio (spec §5.17.3). And a role gives values a
*structural* owner: inside `SURETY`, entity 7's income is addressed
`surety[7].gross_monthly_income`, so two applications of `core.income` to two
entities produce two answers that cannot be confused — spec §5.17.1
property 1, satisfied by the scope rule rather than by a naming convention.

`consumed/p02_affordability.py` shows what the sole-proprietor double-count
(spec §5.17.3(1)) looks like once roles exist: it is one line, at the site
where the affordability check meets the role, not a rule repeated at every
place `business.drawings_cents` might otherwise be read twice:

```python
Affordability = Affordability.under(SOLE_PROPRIETOR).asserting("counted_once")
```

`roles.py` declares `once={"business.drawings_cents": ["O14.debt_service",
"p02.income"]}`, and the framework is asked to refuse to compile a graph
where both paths reach one verdict — a static check, because lineage already
knows both readers.

Six roles are declared (`DIRECTOR`, `SURETY`, `CORPORATE_GUARANTOR`,
`SOLE_PROPRIETOR`, `PERIPHERAL`, `GROUP`), plus three roll-ups
(`ENTITY_VERDICT`, `PEOPLE_BLEND`, `SIGNAL_SCORE`) that name how a value
*leaves* a role scope — three consumers, three roll-ups, one component,
living in one file so "why does this event show as material here and
disqualifying there" (spec §5.17.3) is answered by reading two records in
one place, not two code paths in two.

### 2.2 Time — comparability, two resolution rules, and no default view

Four files, three problems, and a `time/` package because none of the three
constructs generalises to something smaller.

**`time/dating.py` — two resolution rules, expressed as two types with no
accessor between them.** This is called out as "the single most consequential
departure from doc 03 in the whole project." Doc 03 has one effective-dating
story: everything resolves by `decision_date`. Applied to a covenant
definition that produces a plausible number that breaches a covenant the
client did not breach (spec §9.5) — silently, because it does not error. So:

```python
POLICY = resolution("policy", by="decision_date", ...)
CONTRACT = resolution("contract", by="instance_binding",
                       immutable_once_bound=True, ...)

covenant_definitions = bound_table("covenant_definitions", resolution=CONTRACT,
    bind_on="covenant_definition_version", ..., closure=[...], live_versions=1_900)
```

`dated_table(...)` produces a `Dated[T]` that resolves only against
`decision_date` and has no `.bound_to()`. `bound_table(...)` produces a
`Bound[T]` that resolves only against an instance's binding key and has **no
`.in_force_at(date)` at all** — the question that produces the wrong answer
is unaskable, not merely discouraged. `modules/covenants/test.py` doubles
this with a module-level tag: `contractual=True` means the module may not
read any `Dated[T]`, so a future engineer adding "just one more" sector
benchmark to the covenant test fails the build naming the table, not a
reviewer six months later.

**`time/bitemporal.py` — no unqualified name.** A structure has two
plausible readings — `known_from <= knowledge_date` for replay,
`effective_from <= as_at < effective_to` for "what was true" — and neither
errors. `.at()` would produce two modules reading two names with the
difference invisible at the use site. Instead:

```python
EntityFacts = bitemporal_source("entity_facts", ...,
    views={"known": KNOWN, "actual": ACTUAL}, unqualified=None)
```

A step asking for bare `entities` gets a build error naming both candidates
and quoting spec §5.13.2 at the reader. `pipelines/covenant_test.py`'s
`OwnershipChangeTest` reads `actual.entities@test_date` against
`instance.ownership_baseline` in the same test — two temporal reads, two
sources, kept apart by namespace rather than by discipline.

**`time/comparability.py` — "not comparable" as a value, not a missing
one.** The document's namesake requirement (spec §5.10.4): a 2026 grade 6 and
a 2031 grade 7 might be a real deterioration or an artefact of two
recalibrations and a master-scale collapse, and *"a design that always
produces a number will always produce a wrong one."* `Compared[T]` carries a
basis tag beside every grade-bearing value, and `delta()` is the only
subtraction defined over it — refusing, at the type level, to compile a
report that subtracts across bases:

```python
Compared = compared_type("compared", bases={0: "as_graded", 1: "restated",
    2: "not_comparable"}, carries="basis_ref", arithmetic="basis_checked")
```

**`time/master_scale.py` — the one artefact read under both rules.** A
master scale version is never superseded (all four stay readable forever,
because every past decision names one), so it is resolved by `POLICY` when a
*new* grade is being assigned and treated as `CONTRACT`-like — bound by the
decision record — when an *existing* grade is being read. It gets its own
file specifically because it is the one place the project's two dating rules
apply to the same underlying registry, and that needs to be visible rather
than discovered.

---

## 3. Walkthrough — origination, EP-1

`pipelines/entry_points.py`:

```python
EP1 = entry("EP-1", "New-to-bank business application", profile="A",
            volume="900/day", budget="p95 4s, p99 8s",
            pipeline=Origination.scoped_by(review_scope, "EP-1"))
```

`Origination` is `pipelines/origination.py`'s seventeen-phase expression, and
`EP-1` runs all of it — the review-scope matrix row for EP-1 is "FULL" in
every one of its 28 cells. Read down the pipe:

```python
Origination = (
    Relationship                                   # O1  local — 4 dates, 4 state flags
    | StructureResolution.as_known_at("knowledge_date")   # O2  consumed, view=KNOWN
    | eligibility | consent                        # O3  consumed
    | AbsoluteRules                                # O4  consumed, dispositioned per entry point
    | EntityAssessment                              # O5  consumed, under 5 roles
    | PeopleBlend                                   # O6  consumed
    | Spreading                                     # O7  consumed, 1..6 periods
    | scorecard.behavioural                         # O8  consumed
    | CombinedGrade                                 # O9  consumed, + master_scale_version
    | GroupExposure                                 # O10 consumed
    | fuse(Appetite | Constraints)                  # O11 local — the only `fuse()` here
    | Security                                      # O12 local — cross-facility allocation
    | PricingSearch                                 # O13 consumed
    | Affordability                                 # O14 consumed, pinned major, role SOLE_PROPRIETOR
    | CovenantSetting                                # O15 local — the lifecycle begins here
    | Conditions                                     # O16 consumed, extended to 61
    | Authority                                      # O17 local — 7 levels, moving
).with_vocabulary(CREDIT_CORE)
```

**O1 — `modules/request/relationship.py`.** The phase project 05 does not
have. It sets the four `reserved_input` dates (`decision_date`,
`knowledge_date`, `test_date`, `determination_date`) so that no later step
can read "today" by accident — *"there is no `.today()`, no `.now()` and no
`datetime` import anywhere under `modules/` or `pipelines/`."* For a new
client, `existing_state` returns the facility's four flags at their initial
values; `facility_decision_history` returns empty, because this is the
first decision of record.

**O2 through O10 — `consumed/p05_origination.py`.** `EntityAssessment` runs
under all five entity-level roles at once — `[DIRECTOR, SURETY,
CORPORATE_GUARANTOR, SOLE_PROPRIETOR, PERIPHERAL]` — so a single component
call produces five structurally distinct answer sets, one per role, with no
renaming step between them. `StructureResolution` opens `view=KNOWN`
explicitly, because O2 for an origination and O2 for a review resolve
differently (§5.13) and the pipeline author has to say which.

**O11 — `modules/appetite/facility_types.py`, `fuse()`.** This is the one
`fuse()` call in the whole origination pipeline, and the file explains why:
Appetite and Constraints are the two cheapest modules every one of the nine
entry points runs, which is exactly the shape doc 02 §1.1 says benefits from
fusion (small, cheap, always-hot). Everywhere else in `Origination`, modules
stay split — one kernel each — which is the framework's own safe default
(doc 03 §3.2, doc 02 §1.1: "boundary stores are near-free").

**O12 — `modules/security/allocation.py`.** For a first facility with no
shared collateral, `Allocation`'s cross-record machinery is present but
inert — there is nothing yet to reallocate. Its cost shows up at the next
facility, not this one (§4 below).

**O14 — `consumed/p02_affordability.py`.** `Affordability` is
`Branch(p02_major, [p02.at_major(2), p02.at_major(3)], ..., fuse=False)` —
both compiled arms exist in the image, and the manifest resolves which one
this assessment runs against, per row, from `decision_of_record.
component_versions`.

**O17 — `modules/authority/routing.py`.** `base_level` keys off *total group
exposure after the proposal*, and `AuthorityDuringSearch` recomputes on every
change O13's structuring search makes to the proposal — so by the time
`Authority` finishes, `authority_sequence` already records every level the
proposal passed through, and `ApprovalBinding` has fingerprinted the
structure that was actually approved. If the analyst adds a surety after
this to shave the rate, the fingerprint moves, the approval is invalidated
automatically (reason 5894), and the pack — generated, never hand-assembled
— regenerates at the new level.

The decision of record this pipeline produces pins, among the ~180
provenance fields spec §5.10.2 requires: `decision_date`, `knowledge_date`,
every table version read, `master_scale_version`, the p02 and p05 majors
that ran, the allocation id and valuation date behind the security position,
and the authority sequence. There is no predecessor — `comparison_basis_code`
is `as_graded` by construction, because there is nothing yet to compare
against.

---

## 4. Walkthrough — the annual review, five years later

`pipelines/entry_points.py`:

```python
EP3 = entry("EP-3", "Annual review", profile="C",
            volume="180 000/year in monthly cohorts of ~15 000",
            budget="8 hours overnight",
            pipeline=Origination.scoped_by(review_scope, "EP-3") | AnnualReview,
            pins_versions_at="batch_start")
```

Two pipelines concatenated: the origination flow, scoped down by the same
review-scope matrix (EP-3's row is not "FULL" everywhere — O2 runs as an
as-at-date resolution only, per §5.2's phase table), and
`pipelines/annual_review.py`'s `AnnualReview`, which is where the four
outputs origination cannot produce live.

```python
AnnualReview = (
    ReviewBasis                                   # which of 5 modes, and why
    | GradeMigration                              # output 1: 3 grades + 6 causes
    | PricingSearch.as_repricing()                # output 2: contractual vs indicated
    | LimitDecision                               # output 3: p07, revolving only
    | CovenantSetting.as_reset_proposal()         # covenant reset proposal
    | fuse(Exit)                                  # output 4: exit recommendation
    | Authority
).under(REVIEW_BASIS)
```

**The basis is chosen first.** `lifecycle/annual_review/basis.py` declares
five modes — complete, late, stale, turnover-only, not-performed — with
`arithmetic_invariant=True`: a mode selects evidence, parameters and which
outputs are produced, but the DSCR formula is the same in basis 1 and basis
5. Marang Engineering's 2031 review, on late financials, runs basis 2: a 5%
haircut on the spread, and — because its covenant certificate arrived after
the deadline — an information-undertaking breach recorded alongside whatever
the financial covenants say. This ordering is the file's whole argument: put
the complete-evidence path first among five declared arms, not as the happy
path with four error branches hanging off it, because 56% of reviews take a
mode other than 1.

**The grade migration — `lifecycle/annual_review/migration.py`.** This is
the worked case the whole document is named for, reproduced here at the
file that computes it. Marang was graded 6 in March 2026 on master scale v2
by `BUS-COMM-01`. Between then and 2031: `BUS-COMM-01` was recalibrated
(Nov 2027, same scale); the scale moved to v3 (Jun 2028); `BUS-COMM-02`
replaced `BUS-COMM-01` with two characteristics no longer collected
(Feb 2030); the scale collapsed to v4 (Sep 2030). The 2031 review grades it
7.

`three_grades` returns three `Compared` values — origination, last review,
now — each still carrying its own `master_scale_version`. The record **never
says grade 7 for 2026**; it says grade 6 on v2, and, where `map_exists`
returns true for that specific cell, grade 7 by map v2→v4 *beside* it.
Because `BUS-COMM-02` dropped two characteristics, the 2026 assessment
cannot be re-scored under it, so `restated_onto_current` returns basis 2 —
`NOT_COMPARABLE`, reason 5915 — for that leg, and `six_causes`'s
`scheme` axis is exactly this: the movement attributable purely to the
scale, which is not a credit event. The `overlay` axis needs the *difference*
between two `core.adjustments` stacks a year apart — a gap
(`overlay_stack_delta`, EXTEND, owner Credit Systems) the file states
plainly this project cannot resolve alone: *"an acceptance criterion of this
project depends on another team's roadmap item. That is what reuse costs
and it is stated rather than worked around."*

**`comparison_basis_code`** lands on the decision of record — `restated` or
`not_comparable`, chosen per axis and reported as a whole — and
`migration_report_guard` is the mechanism (spec 10 acceptance 3) that makes
a report mixing bases un-buildable rather than merely wrong: `delta()` will
not typecheck across bases, so the report builder has no path to the
offending column in the first place.

**Batch pinning.** `AnnualReview` is wrapped in `batch_pin(AnnualReview,
at="cohort_start", covers=["generation", "component_manifest",
"dated_tables"])`. This is stated as going beyond what doc 08 §4's
generation pointer already gives (one call reads the pointer once, so a
batch cannot straddle a *skeleton* swap): the demand here is that the
*table* and *component-manifest* resolutions, which are separate mechanisms
from the generation pointer, are frozen for the same eight hours, for the
same 15,000-facility cohort, with a version change mid-batch either not
applying or aborting the whole cohort.

**Security, revisited.** If Marang's overdraft (F2) was raised at some point
in the intervening five years, `modules/security/allocation.py`'s
`reallocate` will have already re-tested F1 — a facility a different
relationship manager owns — and recorded whether its `security_type` moved.
The 2031 review reads the allocation as it stands via `Allocation`'s `pins=`
fields (`allocation_id`, `valuation_id`, `valuation_date`, `advance_rate`);
it does not recompute the pool, because the pool's position as at any past
date is a query against those pins, not an archaeology exercise.

The resulting decision of record's `comparison_basis_code`, `six_causes`
and `predecessor` link are what let a 2036 analyst — or a 2031 credit
committee — answer *"did it deteriorate, or did the scale change"* in the
terms spec §9.1 demands, from one artefact, without re-running anything.

---

## 5. How each hard part is expressed

| Difficulty (spec §2.5) | Construct | Where |
|---|---|---|
| H1 — comparability under a changing flow | `Compared[T]`, `rebase()`, `delta()`, six-way `cause_decomposition` | `time/comparability.py`, `lifecycle/annual_review/migration.py` |
| H2 — two version-resolution rules | `dated_table`/`bound_table`, `POLICY`/`CONTRACT`, `contractual=True` module tag | `time/dating.py`, `modules/covenants/test.py` |
| H3 — bi-temporal entity structure | `bitemporal_source`, `unqualified=None`, `KNOWN`/`ACTUAL` views | `time/bitemporal.py` |
| H4 — cross-facility constraints | `cross_record`, declared `invariant=`, `reopens=` | `modules/security/allocation.py` |
| H5 — the group cascade | `Cascade`, four bounds as constructor arguments, `dual_attribution` | `pipelines/cascade.py` |
| H6 — five volume profiles | `fuse()` used once, `parallel()` used once, `incremental()` with a weekly full-pass reconciliation | `pipelines/origination.py`, `pipelines/early_warning.py` |
| H7 — authority that moves mid-assessment | `Loop(..., carries=["authority_level_code", "authority_sequence", ...])`, `approval_binds_to` | `modules/authority/routing.py` |
| H8 — a covenant as a schedule | `Expand` as a frame-tier subject generator, `append_only_artefact` | `modules/covenants/schedule.py`, `modules/covenants/definition.py` |
| H9 — degraded operation as the normal case | `modes(..., arithmetic_invariant=True, exhaustive=True)` | `lifecycle/annual_review/basis.py` |
| H10 — forbearance classified at decision time, including the negative | `@required_output` on both the classification and the negative-case evidence | `lifecycle/forbearance/classification.py` |
| H11 — fourteen teams, one artefact | `governance_matrix` + `assert_matrix_total`, per-node `owner=`/`co_owners=` | `pipelines/entry_points.py`, `lifecycle/amendment/reopen.py` |
| H12 — navigability after five years | `render()`/`lineage()` (framework, unmodified) applied to a tree small enough to grep | throughout — see §1's grep test |
| H13 — the cost of reuse | `consume()`, `gap()`, `Branch(pinned_major(...))`, the fork and gap registers | `consumed/` in full |

Two constructs are named twice in that table because they answer two
difficulties at once, which is itself worth noticing: `approval_binds_to`
(H7) and the governance matrices (H11) are both instances of the same
underlying move — *turn a governance artefact into the thing the framework
already computes (lineage, or the pipeline itself) rather than a second
description that can drift from it.*

---

## 6. What a credit committee sees

A level 6 decision — Credit Committee, quorum-based, twice weekly — renders
from `modules/authority/routing.py`'s 322-cell pack matrix, and three things
about that render are load-bearing rather than cosmetic:

**Consumed logic looks different from local logic, on the page, not just in
the source tree.** `consumed/__init__.py` states three visible differences
and all three land in the pack: a consumed node has no interior document —
`decider export --interiors` skips it, so a policy analyst editing this
project's rules literally cannot open project 05's roll-up from here; it
renders with its **owner and approval reference** in the header, and its
body text comes from the *owner's* published description, not this
project's, so §9.4 requirement 2 ("sliceable by owner") is free because
ownership sits on the node; and its pinned version prints at the use site —
`p05.entity_assessment @ 4.2.1 (pinned by manifest)`. A reviewer reading a
2031 render of this 2026 decision sees `3.9.0`, not `4.2.1`, at that exact
line.

**The pack is generated from the structure that was actually approved, not
from a variable someone set.** `pack_sections` is derived from
`ApprovalBinding`'s content fingerprint. If the proposal changed after
routing — sequence (c) in §5.14.3, where adding a surety drops the level to
the analyst's own — the fingerprint moves, the prior pack is stale by
construction, and a stale pack cannot be presented: *"a level 6 pack
describing a structure that was subsequently reduced to a level 4 decision
is a governance failure that looks like a formatting problem."*

**What the committee reads at re-approval is a diff, not a document.** Spec
§9.4 requirement 1 asks for a policy summary per entry point, a full
reference, and a diff view, "which is the thing actually read at
re-approval." `pipelines/annual_review.py`'s `COHERENCE` check
(`assert_sequence_coherent`, run monthly over the whole book, "expecting"
non-zero exceptions) is the same instinct one level down: a check that never
finds anything is a check nobody trusts, so the artefact is designed to
produce a legible non-zero count rather than a reassuring zero.

For a level 1 decision — the automated mandate, no human — there is no pack
at all: *"the decision record **is** the pack."* The same rendering
machinery that produces 46 sections at level 7 produces nothing at level 1,
because the pack-content matrix says so, not because level 1 is special-cased
in code anywhere.

---

## 7. What changes when a value moves versus when structure moves

Doc 08 §2's three change classes map onto real artefacts in this tree, and
the mapping is the whole point of having named the classes at all:

**A value moves — free, no compile.** `modules/covenants/setting.py`'s
`cover_cushion_pct` (20% convention on cover ratios) is a `param()`. Policy
changes it annually; nothing recompiles; the change shows up as a swapped
bundle. `modules/watchlist/signals.py`'s `signal_catalogue` is a
`decision_table` — 186 rows × 11 attributes — and it is `values`, not
`interiors`, specifically because it passed doc 08 §3.4's test ("can one
compiled loop evaluate every instance of this kind, with the instance
supplied as arrays") — the file states this is "the one place in this
project where the three change classes land cleanly on a real artefact
without argument." Early Warning edits it monthly, and the edit compiles
nothing.

**An interior moves — one background compile, one staged swap, the
interface unchanged.** The watchlist *grade* — six boundaries, seven trigger
overrides, the escalate-automatically/de-escalate-by-human asymmetry — is
the counter-example in the same file: it fails doc 08 §3.4's test, so it is
`skeleton` (Python, reviewed), not `values`. `lifecycle/amendment/reopen.py`'s
`amendment_reopen` governance matrix reads like a values-class table — a
392-cell CSV — but behaves like an interior: changing a row changes *which
phases run for an amendment kind*, which is control flow, not a parameter.
That is why `assert_matrix_total`
exists: doc 08 §2 property 2 promises an interior change "cannot alter the
graph," and a governance matrix that scopes a pipeline is exactly the case
that promise has to be checked against rather than assumed for.

**Structure moves — rebuild and redeploy.** Adding a phase to `Origination`,
or changing what `Security.reallocate` does to the pool, is a Python change
under `modules/` or `pipelines/`, reviewed like any other code change, and it
is the only class of the three that can alter what `lineage()` returns.

**And a fourth class this project needed that doc 08 §2 does not have.**
`modules/covenants/definition.py`'s `covenant_definition_version` is neither
a value (editing it changes the terms of live contracts) nor an interior
(an interior change recompiles and the new body applies to *everything*,
where a covenant definition must apply to *nothing already bound*) nor
skeleton. It is **append-only**: immutable once an instance binds to it,
retired only when the last bound instance closes — up to twenty years for
product 53 — and standardising it across a live book is stated as *"an
amendment programme, not a library edit"*: forty thousand instances, forty
thousand client consents, eighteen months. This is FRAMEWORK-DEMANDS D2, and
it is the sharpest instance in the tree of a change that is real, governed,
and genuinely does not fit any of the three published buckets.

---

## 8. What is fragile, stated rather than hidden

Two things in this tree are explicitly flagged as at-risk by the sketch
itself, and they are worth repeating here rather than leaving buried in a
TOML comment:

**`roles.py`'s `NOT_APPLICABLE` is a project-local construct**, added because
doc 00's three null situations (not collected, collected as zero, could not
be established) have no fourth for "not applicable to this role." `consumed/
FORKS.toml` records this as pressure #7 and marks it `AT RISK`: *"adding a
fourth null situation to core touches six consumers; a local convention
touches none."* This is the closest thing to a fork in the project, and it
is recorded as one rather than left to be discovered.

**`consumed/GAPS.toml`'s `single_entity_rescore` gap is the largest single
fork pressure in the project.** The interim measurement — the daily pass's
whole-entity path holding at 2.1h against a 3h budget, p99 2.7h — is the only
thing keeping a stripped-down local classifier off the table. The gap's own
review date (2026-09-30) and its stated escalation ("2.6h mean and this
becomes a release blocker on project 05") are the mechanism, not a promise:
the question "did we fork?" is designed to be answered by a build, not by
memory.

Both are named in `FRAMEWORK-DEMANDS.md` (D9 and the reuse-cost indicators
of §5.17.7 respectively), because a sketch that hid its own weakest points
would be less useful than one that didn't.

---

## Fixes made to the existing tree

Two self-referential numeric claims in the existing files did not match the
code they described, and both are corrected in place (not rewritten
otherwise):

- **`roles.py`** said "there are eight of them below," referring to the
  `role(...)` declarations. The file declares six (`DIRECTOR`, `SURETY`,
  `CORPORATE_GUARANTOR`, `SOLE_PROPRIETOR`, `PERIPHERAL`, `GROUP`). Changed
  "eight" to "six."
- **`vocabulary.py`** said "Eleven pairs and two prefix families... `grep -c
  '":' vocabulary.py -> 13`." `CREDIT_CORE` declares eight name pairs and two
  prefixes, which is what `grep -c '":'` actually returns against the file
  (ten matching lines). Changed "Eleven pairs" to "Eight pairs" and the grep
  result from 13 to 10.

No other files were altered.
