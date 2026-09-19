# 10 — Retail credit, end to end · design sketch

An ideal-world sketch of what this project would look like if the authoring
surface could be anything. Nothing here runs. The deliverable is a **shape**:
what a codebase implementing the 3 485-line spec would look like, at the
scale the spec insists on — 1 400 decision points, 47 tables, 340 parameters,
12 teams, 8 entry points — if `decider2` could have any authoring surface at
all.

Read alongside [`FRAMEWORK-DEMANDS.md`](FRAMEWORK-DEMANDS.md), which is the
ledger: twenty-six numbered things this project needs from `decider2`, each
traced to a spec section and marked satisfied / needs extension / ugly. This
document is the walkthrough — the layout, why it survives, and what a
reader actually does with it.

The spec is `../../10-retail-credit-end-to-end.md`. Section references below
(§5.10, §5.26, Q17) are to it unless they say "doc" (the framework docs under
`decider2/docs/`).

This project is the **standalone** half of the 10/11 A/B (spec §1.1): built
from nothing, with its own numbers, consuming only `core.*`. The question it
answers is **how do you lay out one very large project from scratch so it
stays navigable** — not how do you reuse what already exists, which is
project 11's question.

---

## 1. The layout, in full

```
retail_credit/
  flow.py                                   # THE flow. 120 lines. One graph, eight entry points.
  ordering.py                               # 24 declared constraints, 2 genuine cycles, checked both ways.
  OWNERS.toml                               # ownership as DATA — twelve teams, none of them a directory.
  .github/
    CODEOWNERS                              # GENERATED from OWNERS.toml + the registry. Never hand-edited.

  phases/                                   # the unit doc 07 doesn't have: a module with a declared envelope
    __init__.py                             # 18 envelopes; 10 written in full, 8 elided "for length"
    p01_admission/
      routing.py                            # Normalise | Validate | FixDecisionDate | Route
    p02_identity/
      resolution.py                         # R1..R4, confidence floor, the related-party set
    p03_consent_eligibility/
      consent.py                            # six consent classes, one pinned read (O-23)
      gates.py                              # 43 hard-eligibility gates, none short-circuited
    p04_acquisition/
      orchestration.py                      # retrieval plan, wave assignment, the arrival contract
    p05_fraud/
      families.py                           # F1..F5 as `ruleset` interfaces — 188 rules are DATA
      precedence.py                         # the verdict, a `decision_table`, plus the bypass
    p06_features/
      bureau_normalise.py
      income.py                             # THE exclusive producer of net_monthly_income
      obligations.py                        # THE exclusive producer of existing_obligations
      bands.py                              # 36 feature bands — the §5.28 blast-radius example lives here
    p07_scoring/
      selection.py                          # 28-rule scorecard selection, taps=characteristic_contributions
    p08_calibration_grading/
      calibration.py                        # resolves adjustment_set_id ONCE (O-05)
    p09_policy_gates/
      register.py                           # RegulatorySeeds | Register | ExposureAndConcentration | PolicyGates
    p10_affordability/
      chain.py                              # one arithmetic
      modes.py                              # four evidence modes, zero `if`s in chain.py
      verdict.py                            # degradation="REFUSED", deliberately
    p11_product_routing/
      routing.py                            # entangled in O-09, the second genuine cycle
    p12_pricing/
      rate.py                               # four cards, three representations, one phase
      fees.py                               # the R14 568 kink search.py's Partition breaks on
      credit_life.py
      annuity.py                            # THE shared_value with all three axes
    p13_solve/
      search.py                             # a Partition over declared breakpoints, not a binary search
    p14_consolidation/
      settleability.py                      # written for P14, read by P15's decrease path (Q1 reuse)
      scenarios.py                          # <= 250 of up to 2M combinations, budget exhaustion RECORDED
      objective.py                          # configuration + one hard-coded anti-harm gate
    p15_limit_assignment/
      assignment.py                         # SAME matrix, funded (EP3) vs unfunded (EP2/EP4) answer
    p16_offer_assembly/
      assembly.py                           # the phase no isolated flow owns — a fourth notion of "best"
    p17_validation/
      assertions.py                         # O-18's isolation: cannot read a shared intermediate, by construction
    p18_disclosure/
      disclosure.py                         # 8 output shapes, 8 record shapes, one phase

  loops/
    l1_consolidation.py                     # THE four-phase feedback loop. Not a copy of P10/P12/P13/P14/P16.

  values/
    bases.py                                # THREE version axes: basis, adjustment, pass
    ceilings.py                             # five ceilings as monotone accumulators, direction checked at runtime
    register.py                             # 41 shared intermediates; 7 in full, 34 elided "for length"

  degradation/
    sources.py                              # 13 declared sources, "refer never decline" as a checked property
    modes.toml                              # (not shown) composite degraded_mode_code table

  budgets/
    phases.toml                             # the 111.5 ms internal budget — DECLARED, not commented
    overrun.py                              # (not shown) abandonable vs non-abandonable, referenced by ep04.toml
    ep01.toml                               # (not shown) EP1's internal + external totals
    ep01_loop.toml                          # (not shown) the 820 ms loop-mode class
    ep04.toml                               # (not shown) the 6-hour window budget
    ep07.toml                               # (not shown) the 50 ms quotation budget

  entrypoints/
    manifest.py                             # the derivation rule + 19 declared divergences
    ep01_new_application.toml
    ep02_limit_change.toml                  # (not shown; same shape)
    ep03_limit_programme.toml               # (not shown)
    ep04_campaign_preapproval.toml
    ep05_consolidation.toml                 # (not shown)
    ep06_reprice.toml                       # (not shown)
    ep07_quotation.toml
    ep08_whatif.toml                        # (not shown)

  navigability/
    index.py                                # where() / resolve() / invert() / trace_loop() / phase_set_bitmap()

  records/
    record_shapes.py                        # ONE schema, EIGHT views. Four absences, never collapsed to null.

  deadlogic/
    candidates.py                           # (not shown) 109 candidates: correct-rare/shadowed/dead/unknown

  certification/
    strata.py                               # (not shown) the 197-stratum golden-set declaration + P02 equivalence proof

  tables/
    cap_register/
      cap_register.toml                     # 11 of 118 entries, representative, incl. the two cycle-relevant ones
    rate_cards/
      flex_loan/
        cells.sample.csv                    # 10 of 34 560 cells, incl. the R60 000 band edge
        manifest.json                       # (not shown)
      drive_finance/                        # (not shown; 71 424 cells — the largest single artefact in the estate)
      home_loan/                            # (not shown; margin-over-reference, not absolute — a different shape)
      revolving/                            # (not shown; promotional + reversion, a third shape again)

  schemas/
    ep01_application.json                   # (not shown) declared input-frame schema; build needs it (doc 07 §5)

  tests/
    phases/
      test_p13_solve.py                     # the R60 000 band-edge inversion, verbatim from spec 5.14
      ...                                   # (not shown) 17 more, one per phase, ~164 000 cases total
    golden/
      manifest.toml                         # (not shown) 197 strata x 200 minimum, synthetic cases flagged
    degraded/                               # (not shown) 14 states x affected entry points x 400
    recheck/                                # (not shown) 180 000 applications, zero tolerance
    replay/                                 # (not shown) 50 000/night, bit-for-bit
```

Fifty files exist on disk; the `(not shown)` annotations are files this
layout implies and the spec names directly (a budget file a `.toml`'s own
`file =` field points at, a Python module `deadlogic/candidates.py` two
existing files already reference by path) but which a sketch of this size
does not need to materialise twice to make its point. Every one of them is
discussed below, in the section its content matters to.

### 1.1 Why this survives where doc 07's proposal, applied literally, does not

Doc 07 §1 is right for the project it was written against — `03 unsecured
loan granting and pricing`, ~30 modules, one product, one pipeline file. Its
four principles (a module directory is the unit of reuse and audit;
pipelines are composition; config mirrors pipelines; tests mirror modules)
are all still true here. None of them is wrong. **What doc 07 does not have
is a unit between "module" and "everything"**, because at 30 modules nobody
needed one. This project needed one before it had written a single step.

The concrete failure, stated once: `phases/p09_policy_gates/` under doc 07's
convention is a directory, and a directory is where doc 07 expects you to
find one owner, one test suite, one config file. Spec §5.10 states P09's 196
decision points split FIVE ways (T4 89, three product teams 41 combined,
Compliance 34, Decision Platform 21, Financial Crime 11), and §5.26.1 states
this is true of thirteen of the eighteen phases, not an exception. A
directory cannot carry that; `OWNERS.toml` does, as data, checked against the
built registry at compile time (`# every decision point names a team that
exists here`, `OWNERS.toml`'s own header). This is `FRAMEWORK-DEMANDS.md`
#11 and #14, and it is the single largest reason this tree has more top-level
directories than doc 07's example (`degradation/`, `budgets/`, `navigability/`,
`records/`, `deadlogic/`, `certification/`, `.github/` all sit beside
`modules/`'s replacement, `phases/`) — each one is a governance concern doc
07 has nowhere to put, made a first-class directory instead of a comment.

The second failure is config. Doc 07 §1's `config/<pipeline>/<env>.json`
mirrors the PIPELINE. At 340 parameters split across five approval routes on
four cadences (§6.2 — statutory on gazette, policy quarterly, product weekly,
pricing monthly-plus-patch, overlay ad hoc), one file per pipeline is one
file five different approval processes edit, which is precisely the
Tuesday/Wednesday/Thursday collision spec §5.26.2 works through in numbers.
This project splits config **by artefact ownership class** instead:
`tables/rate_cards/` (Treasury, monthly-plus-patch, 2-hour SLA) is a
different top-level directory from `tables/cap_register/` (four product/
policy teams, quarterly), even though both are read by the same phase (P09,
P12) and neither is a "pipeline" in doc 07's sense. `FRAMEWORK-DEMANDS.md`
#12 has the full argument.

The third failure is subtler and shows up only once you try to write a test
plan: doc 07 §1 principle 4 ("tests mirror modules") is necessary but
answers only half of §5.30.2's eight-layer testing requirement. Five of
those eight layers — the golden set, the degraded-mode set, the re-check
regression, entry-point agreement, replay — are not module-shaped at all.
They are POPULATION-shaped, keyed by (entry point, product, segment,
degradation state), and doc 07 has no convention for that axis because a
30-module project doesn't have 197 reachable (entry point, product, segment)
triples to worry about. `tests/golden/`, `tests/degraded/`, `tests/recheck/`
and `tests/replay/` sit beside `tests/phases/` as siblings with a completely
different addressing scheme — not a violation of doc 07's principle, just
evidence the principle was only ever half the answer.

---

## 2. Entry point 1, all eighteen phases

`flow.py` is the whole composition, and it is genuinely 120 lines. Reading it
top to bottom, for a new application:

```python
retail_credit = (
    P01
    | P02 | P03
    | P04 | P05
    | P06 | P07 | P08
    | P09.pass_one
    | P10.at_site("product_neutral")
    | P11
    | Map(over=routed_products, as_="product_code",
          body=(P09.pass_one | P10.at_site("routed") | P12 | P13 | P09.pass_two),
          capacity=6, ...)
    | ConsolidationLoop
    | P15
    | P16
    | P17
    | P18
).named("retail_credit")
```

Entry point 1's compiled specialisation runs sixteen of these eighteen (it
skips P15, and P14 — inside `ConsolidationLoop` — is conditional, at 6.8%
share). Follow client V (§5.1's worked example: existing client, segment 4,
requesting R95 000 over 60 months, campaign 7712) through it:

**P01** (`phases/p01_admission/routing.py`) normalises her request, runs the
34 structural/domain/cross-field checks, fixes `decision_date` once, and
assigns `phase_set_id`. Nothing here knows she is applying for a Flex Loan
specifically versus anything else the eight entry points might carry —
`Route`'s `candidate_products` step reads it off the request, and
`phase_set_id` comes from the enumerated table `entrypoints/manifest.py`
derives, not from a literal.

**P02 → P03** (`phases/p02_identity/resolution.py`, `p03_consent_eligibility/`)
resolve her to one `client_id` via R2 (exact identity-number match, 0.97
confidence), build her twelve-identifier related-party set for CAP-0301 five
phases later, and evaluate all 43 hard-eligibility gates — all of them, not
short-circuited, per `gates.py`'s `HardEligibility` module, because a client
who fixes one problem and re-applies is entitled to hear about the second
one the Bank already knew about.

**P04 → P05** (`phases/p04_acquisition/orchestration.py`, `p05_fraud/`) issue
the bureau, device and internal-state calls concurrently (wave 1), and
evaluate all 219 fraud rules (188 live + 31 shadow) with no early exit —
`precedence.py`'s `Verdict` module composes the firing set with the
`decision_table`-based precedence, which Financial Crime edits under
four-eyes approval without touching this file at all.

**P06** (`phases/p06_features/`) is where composition costs the most, per
spec §5.7's own words. `income.py`'s `net_monthly_income` is the SOLE
producer (`exclusive_producer=True`, `phases/__init__.py`) of a value eleven
later phases read; `obligations.py`'s `existing_obligations` likewise, and it
is this exact value that later grows up to 251 simultaneous versions inside
P14's search (§3, below). For client V: R31 400.00 net, R11 280.00 living
expenses (the norm floor binds), R7 640.00 existing obligations.

**P07 → P08** (`phases/p07_scoring/selection.py`,
`p08_calibration_grading/calibration.py`) select SC-A3 (segment 4), score
648, run the 12% challenger (SC-A5, 641, recorded and non-binding), calibrate
to PD 0.0294 unadjusted, apply overlay ADJ-0061 (channel 3 odds multiplier,
approved CC-2027-04) to reach PD 0.0347, and grade — 6, one notch worse than
the unadjusted grade would have been. Both PDs and both grades are kept,
always (`probability_of_default_unadjusted`, `risk_grade_unadjusted` in
`calibration.py`), because Credit Committee asks "what would we have done
without the overlay" every month.

**P09 (pass one) → P10 (product-neutral) → P11** is O-09, the first genuine
cycle client V's path actually exercises: `register.py`'s `Register` runs the
amount/term/grade entries with a product-neutral seed, `chain.py` computes
affordability with the most conservative buffer across her two candidate
products (10, 20), and `p11_product_routing/routing.py` routes her to
product 10 (preferred over the dominated product 21 offer) on the resulting
provisional amount. `ordering.py`'s `O09` names this a declared, approved
compromise (`CC-2027-03`) with a monitored residual
(`routing_provisional_delta`) — not a silent re-run.

**The `Map` fan-out** (`flow.py`'s `routed_products` Collection) re-runs
`P09.pass_one`, `P10.at_site("routed")` (her buffer loosens from the
conservative estimate to product 10's own, 18% raised to 22% by overlay
ADJ-0114), `P12` (`phases/p12_pricing/`: R95 000 at grade 6, 60 months →
15.85% → instalment R2 717.16), and `P13` (`p13_solve/search.py`: her
requested amount is affordable outright, BIND-REQ, 4 evaluations, no need for
the Partition search's harder cases) — once per candidate product, one
implementation, `n` commensurable results for `P16` to compare.

**`ConsolidationLoop`** does not fire for client V (her affordability verdict
is 1, pass) — see §4 below for the four-pass case this loop exists for.

**P15** does not run on entry point 1 at all (declared skip, not derived).

**P16** (`phases/p16_offer_assembly/assembly.py`) assembles two surviving
offers (60 months at R2 717.16, recommended; 72 months at R2 402.88, shown
but not recommended — costs R9 977 more total) plus her Everyday Card limit
increase, deduplicating a 48-month variant against a near-identical 46-month
one along the way.

**P17** (`phases/p17_validation/assertions.py`) re-derives all fourteen
product-10 assertions from the recommended offer's own amount, term and
product — carrying nothing forward, by construction (O-18's isolation, not
discipline) — and all fourteen pass.

**P18** (`phases/p18_disclosure/disclosure.py`) ranks her reason set (empty,
she's approved), assembles the disclosure block, and emits one
`FullAssessment` record (`records/record_shapes.py`) — 58 KB, 71% of it the
bureau payload.

Sixteen phases, eleven of them called exactly once, and nowhere in that walk
does a single line read `if entry_point_code == 1`.

---

## 3. Entry point 7, seven of eighteen, under 50 ms

`entrypoints/ep07_quotation.toml`'s own framing: "the sharpest structural
case in the document." A client asks what a loan would cost; no bureau is
touched, no client is assessed, no decision record is created — and yet
every number must be disclosure-accurate, because a quotation is a
representation the Bank can be held to.

The compiled specialisation for `phase_set_id = 7` contains **P01, P03, P09,
P11, P12, P17, P18 and nothing else** — not P01…P18 with eleven phases
switched off, a seven-phase kernel. This is what `entrypoints/manifest.py`'s
derivation rule buys for free: P02 is inapplicable because there is no
resolved client to identify; P04, P05, P06, P07, P08 are inapplicable
because every one of their declared inputs traces back to a client or a
bureau view that does not exist on this entry point; P10 is inapplicable
because nothing downstream of it (the amount solve) needs a full
affordability verdict for a quotation; P13, P14, P15, P16 likewise. **Nobody
wrote any of those seven exclusions down as a rule.** The two exclusions that
ARE written down (`ep07_quotation.toml`'s `[derived.reduced]` P09 entry,
`approved = "COMP-2027-11"`) are exactly the ones §4.1 calls a policy
decision rather than a data-availability fact: P09 restricts to the twelve
regulatory-class entries because a quotation commits the Bank to a price and
assesses no client, and that restriction needed Compliance's signature,
unlike the other five which needed nobody's.

Follow a quotation for the same product-10 shape client V used, 60 months:

**P01** runs identically to entry point 1 — same `Validate`, same
`FixDecisionDate` — because normalisation genuinely doesn't care which entry
point it serves.

**P03** runs in REDUCED form: `gates.py`'s `HardEligibility` module is not
even in this kernel; only `consent.py`'s product-and-channel-availability
subset applies (9 of 62 decision points), because there is no resolved
`client_id` for the other 53 to test against.

**P09** runs `RegulatorySeeds` only — the statutory rate ceiling, the fee
caps, the credit life cap, the in duplum test, the seven disclosure-bearing
limits — twelve of 196 decision points. `Register`, `ExposureAndConcentration`
and `PolicyGates` are absent from this kernel entirely, not evaluated and
skipped.

**P11 → P12** determine which product's arithmetic applies and price it —
`phases/p12_pricing/rate.py`'s `card_cell` lookup is **the same step object**
entry point 1 uses (`fee_arithmetic_shared_with = [1, 2, 5, 6]`,
`ep07_quotation.toml`), not a re-implementation that happens to agree. This
is the direct answer to spec's warning that a quotation understating the
initiation fee is "a refundable overcharge across every agreement written
from it": there is exactly one `fees.py`, called from two different kernels.

**P17** runs six of its fourteen product-10 assertions (1–6: rate matches
cell, rate within ceiling, fee matches piecewise, service fee capped, premium
matches table, instalment recomputes to the cent). Assertions 7–14 are
inapplicable because they read `max_affordable_instalment`, `amount_cap`, or
`risk_grade` — none of which this kernel's P10-less, P09-reduced traversal
produced. `assertions.py`'s `assertion_set` step derives this the same way
every other applicability question in this project is derived: by asking
whether the input exists, never by an entry-point literal.

**P18** emits a `QuotationRecord` (`records/record_shapes.py`) — 1.1 KB,
`emits_decision_record=False`, `emits_quotation_record=True` — structurally a
different contract from `FullAssessment`, not a filtered view that happens to
be smaller.

What makes this fit under 50 ms with zero external calls is stated in
`ep07_quotation.toml`'s `absent_by_construction` list, and it is a genuine
structural property, not an optimisation: **lazy rate-card resolution, the
full decision record, and per-request overlay-stack resolution are not
linked into this specialisation at all.** The 34 560-cell Flex Loan card is
resident and indexed before the first request (`[tables] resident=`), so
`card_cell` is an array read, not a lookup-then-resolve. Entry point 1 can
afford to resolve the overlay stack once per request because it has a 120 ms
budget; entry point 7 resolves it once per GENERATION (at deployment
activation) because it does not.

---

## 4. Each hard part, in the one place it lives

### 4.1 Eight entry points without eight near-copies

One `flow.py`. `entrypoints/manifest.py`'s `Derived` rule and 19
`Declared` divergences. Worked in §2 and §3 above; the tripwire
(`DECLARED_APPLICABILITY_TRIPWIRE = 35`) is this project's own answer to
when derivation has stopped being enough — see §6.

### 4.2 Eighteen phases, twelve teams, one artefact

`OWNERS.toml` declares the map; `phases/__init__.py`'s envelope carries
`owners=` per phase; `.github/CODEOWNERS` is generated from both plus the
built registry, at file granularity finer than OWNERS.toml's own phase-level
table (its own header explains why: nine of eighteen phase directories list
more reviewers than the phase-level table alone would suggest, because a
single file inside a five-owner phase — `phases/p09_policy_gates/register.py`
— can still belong to one team's slice at the DECISION-POINT level).

What makes T7 (Unsecured Lending) deployable on a Thursday afternoon without
waiting for T4's Wednesday quarterly release (§5.26.2's worked collision):
its changes land in `tables/cap_register/cap_register.toml`'s T7-owned rows
(CAP-0131, CAP-0158, CAP-0248 among the eleven shown) and in
`phases/p11_product_routing/routing.py`'s permitted-term parameter — neither
of which T4's release touches — and `navigability/index.py`'s blast-radius
walk over `narrows=` declarations can, in principle, prove the two changes'
radii are disjoint before either ships. Whether that proof runs in the
60 seconds spec §5.28 demands is `FRAMEWORK-DEMANDS.md` #8's open half.

### 4.3 Forty-one shared intermediates, up to 251 versions, a third axis

`values/register.py` declares the 41 (7 in full: `decision_date`,
`net_monthly_income`, `risk_grade`, `adjustment_set_id`,
`existing_obligations`, `max_affordable_instalment`, `instalment`; 34
elided). `values/bases.py` declares the three axes (`BASIS`, `ADJUSTMENT`,
`PASS`) as data, with a build-time rule
(`BARE_READ_OF_MULTI_BASIS_VALUE = "build_error"`) that makes a bare,
unmarked read of `existing_obligations` inside `basis="explicit"` code fail
to compile, naming the value, its axes, and the three markers available.
`existing_obligations`'s own `basis_contract` in `register.py` is the
concrete artefact: a table mapping each of its seven consuming phases to
which basis IT is permitted to read, checked at build against every actual
read site.

Client V never sees more than one version of anything, because her
affordability passes on the first try. The client whose consolidation
assessment reaches pass 3 of `ConsolidationLoop` (loops/l1_consolidation.py)
does: her `existing_obligations` is R9 340.00 actual, plus up to 250
hypothetical versions from `phases/p14_consolidation/scenarios.py`'s
`candidate_settlement_sets`, each one distinguishable in the record by
`scenario_ref`, and each one's downstream `risk_grade` and
`max_affordable_instalment` carrying BOTH an adjusted and unadjusted twin —
four live versions of `risk_grade` in one decision, not a hypothetical
edge case.

### 4.4 Twenty-four ordering constraints, two genuine cycles

`ordering.py`. O-06 (fraud/bureau inversion, broken per entry point) and O-09
(product/affordability/amount, broken by a declared two-pass with a
monitored residual) are the two; O-10 (the instalment cap's own two-pass
split) is "a third, in miniature," per that file's own naming. All three are
`cycle_break` objects checked in BOTH directions at build — every declared
constraint holds in `flow.py`'s actual graph, and every cycle the graph
contains is named by a `cycle_break` (so a 189th fraud rule that
accidentally closes a new cycle across O-06's break fails the build, naming
both edges). `FRAMEWORK-DEMANDS.md` #4 has the nuance: three of the spec's 24
tabulated constraints (O-15, O-16, O-17) are correctly NOT declared here,
because they're provable directly from `flow.py`'s sequence and
`loops/l1_consolidation.py`'s nesting; one (O-21, "off the critical path") is
a genuine, unaddressed gap.

### 4.5 The four-phase feedback loop

`loops/l1_consolidation.py`. `ConsolidationLoop` is a `Loop` whose body is an
EXPRESSION over the same `P10`, `P12`, `P13`, `P16` objects `flow.py`
composes elsewhere — plus `phases/p14_consolidation/`'s three sub-modules —
never a second copy of any of them. Bounded at 4 passes
(`max_iterations=4`), with a termination guarantee that is a property of the
INPUT (`should_continue`: strictly improve or stop) rather than a clock. The
worked four-pass trace in spec §5.23.1 — pass 1 rejected by anti-harm at
P16, pass 2 rejected by P17's term-cap assertion, pass 3 ships — is exactly
what `carries=["best_offer", "best_objective_so_far", "candidates_remaining",
"discarded_offers"]` retains: the three offers that did NOT ship stay on
file, each with its own `loop_pass_index` and its own rejecting phase,
because "why did you not give me the 72-month option" is a question with an
answer, and the answer is at pass 2.

### 4.6 A latency budget summing to 111.5 ms with 8.5 ms slack

`budgets/phases.toml`. Eighteen rows (sixteen with a value; P14 and P15 have
none, because P14 leaves this budget class entirely when it fires and P15
does not run on entry point 1). `not_abandonable = true` on P13, P17, P18;
`abandonable_component=` on P05 (shadow rules), P07 (challenger scorecard),
P16 (second/third product fan-out) — matching §5.24.3's table exactly,
as data an overrun policy (`budgets/overrun.py`) reads rather than a runbook
an on-call engineer remembers under pressure at 2 a.m.

### 4.7 Thirteen degraded sources, "refer never decline"

`degradation/sources.py`. Each `DegradedSource` carries `entry_points=`,
`continues=`, and a `behaviour=` string rendered into the reviewable
artefact. The subtlest one: `AFFORDABILITY_TABLES` and `RATE_CARD` have
`continues=False` — no degraded mode at all, deliberately, because "a
guessed affordability answer is the basis of a reckless-lending defence."
`degradation="REFUSED"` on `phases/__init__.py`'s P10 envelope is the
matching declaration on the CONSUMING side: REFUSED is a third state,
distinct from `degradation=None` (nothing external to fail) and from a
populated tuple (three sources, three behaviours, as on P03) — one word away
from "no degraded mode" and meaning the opposite thing, which is exactly why
it needed its own token rather than an empty tuple standing in for it.

### 4.8 Five measurable navigability tests

`navigability/index.py`. `where("amount_cap")` walks `values/ceilings.py`'s
`narrows=` declarations plus `tables/cap_register/cap_register.toml`'s own
`narrows` column and returns 132 sites — not because a second index was
hand-maintained, but because the direction-checking mechanism
(`FRAMEWORK-DEMANDS.md` #16) and the navigability mechanism read the SAME
declaration. `resolve()`, `invert()`, `trace_loop()` and
`phase_set_bitmap()` sit in the same file because none of them executes
anything; all of them read a build-time registry or a declaration that
already exists for a different reason.

### 4.9 Blast radius answerable before a change ships

The unresolved half of this project, named honestly rather than solved
badly: `navigability/index.py`'s `phase_set_bitmap()` gives entry-point
conditioning cheaply (a precomputed bit pattern per `phase_set_id`);
product and loop-pass conditioning are named as the remaining work in
`FRAMEWORK-DEMANDS.md` #8, because naive reachability over a graph containing
`ConsolidationLoop` returns "everything can affect everything" the moment
the loop's back-edge (P12→P10, declared in `values/register.py`'s
`instalment` value as `back_edges_permitted_via=("L1","L3")`) is walked
without bounding which pass the query is asking about.

### 4.10 109 dead-logic candidates, four categories

Named in `OWNERS.toml`'s own comment (`# + 2 900 campaign LEAVES: see
deadlogic/candidates.py`) and built from values already declared for other
reasons: `values/ceilings.py`'s `evaluated_did_not_bind` distinction — built
for attribution, spec §5.10's requirement 4 — is exactly the fired-but-
not-bound count spec §5.29.1 needs for the reached/fired/bound three-count
model. The 47 correct-and-rare, 31 shadowed, 22 dead, 9 unknown split
(§5.29) does not need a new measurement mechanism; it needs this project's
existing attribution data read with a different question in mind.

### 4.11 5 650 test states, proportional certification

`tests/golden/manifest.toml` (not shown) declares the 197-stratum
construction, each stratum tagged with the phases it exercises. Certifying
a rate-card patch (radius {P12, P13, P14, P17}, spec §5.30.3's own worked
row) becomes a filter over that tag set, targeting the same 25-minute bound
the spec states, rather than a full 140 000-decision re-run.
`tests/phases/test_p13_solve.py` is the one concrete file this sketch
materialises — the R60 000 band-edge inversion, verbatim from §5.14 — and it
runs on every change to P13 alone, which is what "18 sets, ~164 000 total,
per-phase" (§5.30.2) means in practice.

---

## 5. Q17: is a campaign tree leaf a decision point?

**No — and this project pays for that answer honestly rather than pretending
it was free.** `entrypoints/ep04_campaign_preapproval.toml` states the
position directly: `citable_decision_points = 2900` is tracked SEPARATELY
from the 1 400 baseline, with `path_capture = "first_class_output"` making
"why this offer, eighteen months ago" answerable at the leaf level without
promoting leaves into the number every other statement in this document —
every navigability target, every blast-radius bound, every dead-logic
percentage — is stated against.

The cost of the other answer is worked through in
`FRAMEWORK-DEMANDS.md` #19 and it is not merely "5.86× more nodes." A tree
leaf's "every place it can be set" is a campaign-authoring tool's row, not a
Python declaration or a TOML entry — a third storage shape on top of the two
this project already reconciles (code for structure, TOML/CSV for register
and rate-card data). `navigability/index.py`'s `where()` and `invert()` are
built around declarations; a tree editor does not produce declarations of
that shape. Taking the "yes" position honestly would mean building a SECOND
navigability system for one team's artefact, not extending the existing one
by 5.86×.

---

## 6. What a non-engineer sees

A policy owner at Credit Risk Policy opens `tables/cap_register/
cap_register.toml`, not `phases/p09_policy_gates/register.py`. They see
CAP-0176's `condition`/`effect` strings in plain arithmetic ("3+ month
arrears in 12 -> R35 000 and tighten to grade 9"), its `sequence` number, and
— because this entry is genuinely dangerous — the `note` field explaining
that it re-thresholded eighteen months ago and shadowed twelve later
entries nobody noticed at the time (spec §5.29's own worked history,
reproduced verbatim in that file's comment). They never see `direction=
REDUCE_ONLY` or `pass_derived_from="narrows"` — those are properties of the
`ceiling()` object in `values/ceilings.py`, checked at build, invisible from
the data document a business user edits, exactly as doc 04 §2's params/
structure boundary intends.

A campaign analyst at Campaign Analytics edits a tree definition — a file
this sketch does not materialise, because campaign trees are, by this
project's own Q17 answer, a different artefact class with a different
owner and a different tool. They never open `phases/p16_offer_assembly/
assembly.py` at all; `campaign_trees()`'s `reads=` declaration is the
frozen boundary between what they can move and what they cannot, the same
mechanism `phases/p05_fraud/families.py`'s `ruleset` interfaces use for
188 fraud rules Financial Crime edits under four-eyes approval.

A contact-centre consultant, answering "why was the term capped at 48,"
never opens a file at all. They read the rendered chain
`navigability/index.py`'s `resolve()` and `trace_loop()` produce from the
record — "term_cap 60 (CAP-0011) -> 48 (CAP-0176, grade 9 tightened by
2-month arrears)" — which is N2's whole point (§5.27): the chain connects a
term to a cap entry to a grade to a scorecard characteristic across four
phases and three owners, rendered as one sentence, in under five minutes,
without an engineer.

---

## 7. What changes when a value moves, versus when structure moves

**A value moves** — Treasury patches 3 100 Flex Loan cells after a repo cut
(§5.26.2's Tuesday). `tables/rate_cards/flex_loan/cells.sample.csv` (its real
34 560-cell counterpart) is replaced; `budgets/overrun.py`'s cost model is
untouched; `phases/p12_pricing/rate.py`'s `card_cell` step is untouched;
`ordering.py` is untouched. The statutory ceiling re-check (`rate.py`'s
`statutory_ceiling_check`, O-13) runs against the NEW card automatically
because it reads `statutory_rate_ceiling` and `card_cell` by name, not by a
cached assumption about what the old card contained. Certification radius:
{P12, P13, P14, P17} — a rate-card patch, per `FRAMEWORK-DEMANDS.md` #10's
worked row, targeting 25 minutes, not three hours.

**Structure moves** — Credit Risk Policy reorders the cap register, moving
CAP-0176 above CAP-0104 (§5.26.2's Wednesday). This is NOT a value change
despite touching the same file (`tables/cap_register/cap_register.toml`):
`sequence` is load-bearing (`values/ceilings.py`'s `order_is_load_bearing=
True` note on `worst_acceptable_grade`), so reordering changes which
population CAP-0104's appetite grid sees, for every one of the six products
CAP-0104 applies to, across all four owners of the register. This is a
STRUCTURAL change wearing a data file's clothes, and it is exactly the case
`ordering.py`'s `L7` note calls out by name ("register order is
semantically load-bearing... visible in the reviewable artefact, because it
is precisely the kind of thing a policy owner will not discover by reading
their own rule"). Certification radius: the full register plus the swap set
— {all six products, four owning teams}, 90 minutes, per §5.30.3's own
worked row for "a cap register reorder."

The distinction the layout has to make sharp, and does: doc 04 §2's
params/structure boundary is usually "which file did you edit." Here it is
"did you change a VALUE, or did you change something a value's MEANING
depends on" — and `sequence`, `narrows`, `pass_derived_from` and `direction`
are the four places in this codebase where a data document can quietly
become a structural one, each flagged in its owning file's own comments
rather than left for a reviewer to discover by accident.

---

## Note on the existing files

No file among the eleven this sketch inherited was rewritten. One
discrepancy was found and deliberately NOT fixed by editing code:
`ordering.py`'s docstring frames its `CONSTRAINTS` tuple as "these
twenty-four" (matching `EXPECTED_DECLARED_CONSTRAINTS = 24`), but the tuple
itself has twenty entries — O-15, O-16, O-17 and O-21 from spec §5.22's own
table are absent. Investigating rather than patching: three of the four
(O-15, O-16, O-17) are correctly omitted, because they are provable directly
from `flow.py`'s `\|` sequence and `loops/l1_consolidation.py`'s nesting —
declaring them would be the "second source of truth" `ordering.py`'s own
closing comment warns against for the ~16 genuinely obvious constraints. The
fourth, O-21 ("record emission last, and off the critical path"), is a real
gap: "off the critical path" has no declarative primitive anywhere in this
codebase or in the framework docs. Rather than bolt an ad hoc field onto
`ordering.py` to paper over one instance of a missing mechanism, the gap is
recorded as `FRAMEWORK-DEMANDS.md` #4's closing paragraph, with a proposed
primitive (`off_critical_path()`, checked the same way `isolates()` is). A
future pass on this project should either add that primitive and declare
O-21 with it, or narrow `EXPECTED_DECLARED_CONSTRAINTS` to 23 and record why
O-21 is tracked elsewhere — but that is a decision for whoever owns
`ordering.py`, not one to make silently inside a sketch.
