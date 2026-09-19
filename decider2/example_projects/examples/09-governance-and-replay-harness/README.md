# 09 — Decision governance and replay harness: an ideal-world sketch

An answer to
[`09-governance-and-replay-harness.md`](../../09-governance-and-replay-harness.md),
written as the shape it would take in a codebase if the authoring surface could
be anything. Nothing here runs. Bodies are `pass` with a one-line comment; the
value is in the signatures, the invented constructs, the manifest schema and the
two rendered artefacts in `artefacts/`.

This project is not like the other eight in the set, and the tree is built
around that difference rather than around it being incidental. It has no
applicants, no scorecards, no rate cards, and no decision of its own. Every
other project sketch in this repository is a pipeline: steps, modules, `|`,
`Branch`, `Loop`. This one is a program that reads *other* pipelines' build
artefacts and never imports their code. That single constraint —
`manifest/model.py`'s opening line, "This is the structurally unusual part of
this project" — is why the directory layout looks nothing like `pipelines/` and
`modules/`, and why most of what this sketch asks for is not asked of decider2
at all. It is asked of the eight flows, stated as a numbered contract
(spec §5.15) and checked mechanically rather than assumed. See
[`FRAMEWORK-DEMANDS.md`](FRAMEWORK-DEMANDS.md) for that split; this file is the
shape and the reasoning behind it.

---

## 1. The shape, in one page

```
09-governance-and-replay-harness/
  manifest/
    model.py       The GovernanceManifest — a flow's graph, as data, and nothing else.
    export.py      `decider govern export` — the one command a flow team runs in its own CI.
  contract/
    items.py       The 23 items of spec §5.15, as data: STRUCTURAL / EMPIRICAL / DECLARED.
    check.py       `check(manifest) -> Attestation`. Never imports a flow.
    attest.py      `attest()` and `holds()` — the two constructs a flow wraps itself in.
  replay/
    pin_resolution.py   PinSet — the sealed, total resolution environment for one decision.
    seal.py             `at_pin()` and the four mechanisms against a wrong-but-plausible replay.
    engines.py          Three replay engines (IMAGE / SOURCE / INTERPRETED), one equivalence ladder.
    verdict.py          Comparison over every DECLARED output, first point of divergence.
  explain/
    disclosure.py  Field classification by inheritance over static lineage, not by declaration.
    record.py      DecisionRecord — the one record. Three audiences are projections of it.
    render.py      consultant() / analyst() / adjudicator() — one record, three renderings.
  whatif/
    counterfactual.py   Intervention, Counterfactual — structurally incapable of becoming a decision.
  diff/
    semantic.py     Structural diff over two manifests: rule insertion, re-parenting, reordering, cells.
  release/
    increment.py    Increment / Release — an authored, individually-attributable decomposition.
    swapset.py       n+1 runs over 2M records, ordered, attributed, frame tier.
    certify.py       The golden set, tolerance classes, the no-effect rule, mutation_screen().
  coverage/
    measure.py      Two primitives (branch paths, index reads) covering rules/nodes/gates/cells/codes.
    shadowed.py      Static shadow analysis over the closed interior vocabulary. No data, no execution.
  adjustments/
    register.py     The live overlay register, across all eight flows, answerable as at any date.
    unwind.py        Running a flow with its stack disabled, through the SAME implementation.
    asymmetry.py     Independent verification that a tighten-only overlay cannot be made to loosen.
  pack/
    assemble.py     The regulator/ombud pack. Seven sections, the consistency cohort the hard one.
  harness/
    tolerances.py    The harness's own tunables, versioned and governed like everything it governs.
  artefacts/
    decision-record-FLX-2027-03-0418822.json     One decision, the full evidence record.
    explanations-FLX-2027-03-0418822.md          The same decision, three audiences, rendered.
```

### Why this layout

`manifest/model.py` states the architecture in one sentence: **two planes, and
the separation is the whole thing.**

> STATIC PLANE — manifests only. Contract checks, semantic diff, renderings,
> lineage, shadow analysis, the adjustment register, coverage aggregation,
> drift, the input inventory. No flow code.
> EXECUTION PLANE — replay, what-if, swap-set, certification. Runs the flow's
> OWN pinned image in a network-denied sandbox. Never the harness's
> interpretation of the flow.

The directories fall out of that split almost exactly. `contract/`, `diff/`,
`coverage/shadowed.py`, and the structural half of `adjustments/asymmetry.py`
read a `GovernanceManifest` and nothing else — no corpus, no image, no network.
`replay/`, `whatif/`, `release/certify.py`, `release/swapset.py`, and
`adjustments/unwind.py` are the execution plane: they run the flow's own pinned
build, in a sandbox, over supplied evidence or a declared population.
`coverage/measure.py` sits on production volume rather than a corpus, which is
why it gets its own primitives (branch paths as a free per-record column,
cell reads as a merged counter) instead of reusing either plane's machinery
wholesale.

`explain/` and `pack/` are a third thing this document doesn't name but the
tree needs: a **presentation plane**, built on top of both. `explain/record.py`
is the shape of what the execution plane produces (a `DecisionRecord`);
`explain/disclosure.py` and `explain/render.py` are pure functions over that
shape, using the static plane's lineage to decide what each audience may see.
`pack/assemble.py` is presentation again, one level up: it composes
`explain/render.adjudicator()`, `replay/verdict.py`'s provenance statement, and
a frame-tier query over stored evidence (the consistency cohort), and produces
nothing the other two planes didn't already make possible.

`harness/tolerances.py` is its own small point: **the harness governs itself
under the same regime**. It is not `config/` because it is not a flow's config
— nobody is composing a pipeline from it — and it is not `manifest/` because it
describes the harness, not a flow. It gets one file because it is small (§4
below), and it is versioned like a rate card because "a governance harness
whose own settings are unversioned is not a governance harness" (spec §6).

**Absent, deliberately:** anything named after one of the eight flows.
`manifest/model.py` is generic across flow03's 610-rule waterfall and flow04's
60-tree monthly batch; the only per-flow thing that exists is the manifest each
flow's own CI produces. That is the acceptance test for spec's own framing —
"a ninth flow can be brought under the harness ... with no change to the
harness" — and it is why nothing in this tree imports `pipelines.unsecured` or
anything like it.

---

## 2. The hard parts, and how the tree answers each

### 2.1 A harness operating on flows it does not own

The corollary of the two-plane split: the harness's *only* contact with a flow
is the `GovernanceManifest` its own CI exports (`manifest/export.py`). That
document is not a description that might drift from the flow — it is
`pipeline.to_dict()`, the flow's structure serialised, plus everything doc 03/08
compute statically (interfaces, params schema, capability pins) union'd with
things only this project needs (claims, witness set, rendering ref, identity
ledger ref). `manifest/export.py` calls this out as a deliberate departure from
doc 07 §5's two independently-versioned `--params`/`--interiors` exports:

> `govern export` is a third document, and it is a UNION rather than a third
> independently-versioned thing — it is keyed on the build, and it is immutable.

The loss-of-fidelity risk this creates is checked, not assumed:
`manifest/export.verify_roundtrip()` asserts `from_config(m.graph) == pipeline`
on export, in the flow's own CI, and calls a failure `ManifestDrift` — "an
incident, not a warning" — because a failure there means every static answer
the harness has ever given about that flow (lineage, diff, shadowing, the
rendering the committee approved) was about a different graph from the one that
ran.

### 2.2 The 23-item contract, checked rather than asserted

`contract/items.py` is the load-bearing file. Its single most important line:

> An item is STRUCTURAL, EMPIRICAL or DECLARED ... STRUCTURAL provable from the
> flow's graph with no data and no execution ... 15 of 23. EMPIRICAL needs a
> corpus run ... 4 of 23. DECLARED irreducibly a human sentence ... 4 of 23.

That split *is* the answer to spec §13 Q4 ("can evidence emission be a property
of the framework rather than of each flow?"): fifteen items decider2 makes
provable from `pipeline.schema()`, `pipeline.fingerprint()` and static lineage
alone — no flow author has to remember them, because `contract/check.py` reads
the manifest and finds them or doesn't. Four need a corpus (idempotent
emission, emission-cannot-fail — genuinely runtime properties no static check
reaches). Four are irreducibly a human sentence — declared expected effect,
individual attributability, prohibited-ground justification — and the
framework's whole job there is to make omitting the sentence impossible, which
is exactly what `Increment`'s required `Declaration` field and the input
inventory's `permitted_use_ref` do.

Grading is blunt on purpose: **A** all 23 pass; **B** structural items pass but
an empirical or declared item is stale; **C** any structural item fails.
`contract/check.py`'s honest answer to "what happens when a flow doesn't
comply" (spec §13 Q18) is that the harness has no power to stop a live flow
deciding, and doesn't pretend to:

> grade C -> the flow keeps running; it CANNOT OBTAIN a certification
> artefact ... and — the part with teeth — every explanation, every pack and
> every monthly report generated from that flow's decisions carries the banner
> produced by `degradation_notice()`.

The decision record for `FLX-2027-03-0418822` carries `"governance_grade": "A"`
at the top of `1_identity` precisely so that every rendering, every pack and
every certification artefact can print it without asking the harness again.

### 2.3 Exact replay, and the replay that succeeds with the wrong answer

Doc 08 §6.2 gives one pinning primitive, `resolve_params(doc, complete=True)`,
and stops there — tables are "provisional", `decision_date` is an ordinary
input, capability versions are import-time facts, and overlays are unmentioned.
`replay/pin_resolution.py` names the consequence directly:

> Replaying a decision under that surface means the caller reconstructs seven
> different things from seven different places and hopes. Spec §5.1's worst
> failure mode ... is not a bug in that design, it is that design's ordinary
> behaviour when one of the seven is forgotten.

`PinSet` is the fix: one object holding all seven of spec §5.1's categories
plus the overlay stack (§5.14.5's eighth), and `at_pin(pins)` — described as
"SEALING, not configuring" — returns a pipeline with **no outside**:

```python
evidence = store.fetch("FLX-2027-03-0418822")
pins     = pin.resolve(evidence)              # -> PinSet, or raises Incomplete
sealed   = flow03.at_pin(pins)                # -> a pipeline with no outside
result   = sealed.score(**pins.inputs_as_received)
```

`replay/seal.py` exists entirely for the sentence that gives this project its
sharpest edge: *"A flow that reads the current date anywhere ... will replay
successfully with the wrong answer."* Four mechanisms answer it, in increasing
cost order: (1) no fallback path exists inside a sealed pipeline — a read with
no pin raises `UnpinnedRead`, structural impossibility rather than diligence;
(2) read coverage compared both ways after every replay — an unpinned read is a
hard fail, an unread pin is reported because it means the replay took a
*different path*, which is either the defect or a badly-assembled pin; (3)
**schema pinning**, the one nobody thought of until it bit — a 2027 params
document is complete against the 2027 model and silently incomplete against the
2030 one once a field with a code default has been added, so `seal()` stores
`params_schema_hash` per module and refuses on drift rather than resolving it
(`SchemaDrift`, never silent); (4) clock denial at the sandbox boundary,
honestly incomplete — a frame-tier `pl.date.today()` or a `@breaks_lineage`
region never reaches numba and so never shows in `fallback_set`, which is a
real, admitted gap (framework demand D6).

### 2.4 The reviewable artefact — doc 04 §6's top risk, and an honest verdict

Doc 04 §6 ranks the reviewable artefact as the design's weakest part, for two
reasons: if it can't be met, structure-as-data and version chains lose their
justification, and it is people-blocked rather than code-blocked. This tree
does not solve the readability problem doc 04 §6 names — it does not put a
non-engineer reviewer in front of a 610-rule flow's rendering and test whether
they find a discrepancy against policy. `manifest/model.py`'s
`rendering_ref: str  # the approved plain-language artefact for THIS build`
is a **reference to that artefact**, generated by the flow under doc 03/04's own
mechanism, not something this project produces.

What it does add is the piece doc 04 §6 has no answer for at all: **is the
prose still true?** `contract/attest.py`'s `holds()` is the invention:

```python
@holds("below R5 000 net salary no term above 48 months is ever offered",
       given={"min_net_salary": 499_900}, then=lambda r: r.term_cap <= 48)
def cap_by_income_band(term_cap: float, min_net_salary: float,
                       cap: float = param(48.0, ge=6, le=60)) -> float:
    ...
```

A `holds` is prose that CI runs. The reviewable artefact prints the sentence;
certification prints "verified by 2 assertions, both passing"; and
`release/certify.mutation_screen()` perturbs the rule's params to their
declared bounds and flips its comparison operators, and any `holds` that still
passes is rendered with the words **"claim not verified by test"** instead of
"verified" — because a claim that survives its own rule being mutated does not
describe that rule. That is a genuine, load-bearing answer to spec §13 Q10
("how is the description that makes a rendering readable kept honest"), and it
is honestly scoped: it stops prose from rotting silently; it does not make the
prose readable to Credit Risk Policy in the first place. Spec §13 Q9's demand —
"tested against a real reviewer before the format is fixed" — is still open,
and this tree says so rather than claiming otherwise.

### 2.5 One record, three audiences

`explain/record.py` states the design law plainly: *"The obvious implementation
— three templates over the same dict — fails the moment a new field is added,
because the new field is in all three or none, and the person adding it is an
engineer on a flow team who has never met an ombud."* The fix is
`explain/disclosure.py`'s classification-by-inheritance: every derived value's
`Disclose` level is the join (strictest) of its inputs', computed over static
lineage, never declared on a step. Only a *downgrade* needs a marker, and it is
built on the same idiom as `@breaks_lineage`:

```python
@declassify("bureau_score_band", to=Disclose.CLIENT_FACING,
            because="a band is not a score; a band cannot be reverse-engineered "
                    "to a cut-off. Approved COMP-2026-88, review 2027-10-01")
```

`explain/render.project()` then drops every field above the audience's level
before a template ever runs — "the consultant rendering does not *omit* the
score, it never receives it" — so the explanation-versus-gaming tension in spec
§5.2 resolves as a build-time classification rather than an editorial habit
that a template author has to remember. §3 below walks the three renderings for
`FLX-2027-03-0418822` against the real artefact.

### 2.6 What-if that cannot become a decision

`whatif/counterfactual.py` takes doc 03 §6's `dbg.set(...)` — numerically
faithful because it runs the same compiled step code — and adds exactly the
governance the debugger has none of, because "print it and it IS a decision."
Four mechanisms, and the docstring is explicit that only the last one survives
a spreadsheet: `Counterfactual.outputs` keys are prefixed `wi_`; construction
requires an operator identity and a non-empty `because=`; the identifier lives
in a distinct namespace (`WHATIF-2028-04-00913`, never `FLX-...`) that
`store.put()` type-rejects; and the marking is **per row, not per document** —
`__WHATIF__` as the first field of every serialised row and the counterfactual
id appended to every monetary value (`R118 000 [WHATIF-2028-04-00913]`),
because a banner at the top of a spreadsheet does not survive one row being
copied out of it. The residual hole is named rather than hidden: "a person who
retypes a number into an email defeats all four" (framework demand X2) — this
is not a mechanism the framework can supply.

### 2.7 Semantic diff, and the four shapes value-diffing misses

Doc 08 §6.2's `diff(old, new) -> list[Change]` plus doc 04 §5.2's "a config diff
is an audit record with no extra machinery" is true for params and, per
`diff/semantic.py`, false for most of what actually changes in a release. Four
shapes get a purpose-built comparison, each because a positional or value-level
diff produces a technically-correct answer nobody can review: a rule inserted
mid-priority (`effect_shift()` — which later rules' *effect* changes because
something now runs before them, computed with no population); a tree
re-parented (a node identity map that classifies every node
carried-forward/changed/added/removed, so re-authoring a tree doesn't renumber
the world); a stage reordered (`lineage_delta()` — "risk_grade can now be
affected by campaign_id, which it could not before", pure static lineage); and
a rate card refreshed (`aggregate()`, because 63,360 changed cells is "worse
than no diff, because it trains reviewers to approve without reading" — the
reviewable sentence it produces is quoted in full in `diff/semantic.aggregate`'s
docstring and reused verbatim in the worked example below).

### 2.8 Swap-set attribution when three changes ship together, plus one that touched no versioned artefact

`release/increment.py` calls its own mechanism "the largest deviation in this
project." Doc 08 §5 gives `impact(active, candidate, sample) -> ImpactReport`:
two generations, one number. Spec §5.5 needs *n+1* runs, cumulative, in a
declared order, because "which one of these five cost us the 1,840 approvals"
cannot be answered by comparing the ends — and a release cannot be decomposed
after the fact from `git diff`. So a `Release` **is** an authored list of
`Increment`s, each independently materialisable via doc 08 §4's
`stage`/`activate`:

```python
release = Release(
    base=Pin.from_build("flow03@2027-03-31"),
    increments=[
        Increment.table("rate_card.flex", to="RC-FLEX-2027-04", declares=Moves(...)),
        Increment.param("cap_register.tenure_floor_months", 24, 18, declares=Moves(...)),
        Increment.interior("cap_register", adds=("CAP-0455",), declares=Moves(...)),
        Increment.overlay("ADJ-2026-114", action="expire", declares=Moves(...)),
        Increment.skeleton(build="flow03@2027-04-24", declares=NoOutcomeChange(...)),
    ],
    order_rationale="rate card first because the threshold move is priced through it; "
                    "the overlay expiry last because it is the only one we can unwind "
                    "same-day if the swap set is wrong.",
)
```

The fourth increment is the point named in the task: **an overlay change that
touches no versioned artefact anywhere else in the estate.** `adjustments/register.py`
makes it visible to attribution the only way that actually works — "you cannot
make a git-driven mechanism see a change that is not in git, so the harness
must be driven by the release manifest and the register must be one of its
sources" — by emitting an `Increment.overlay(...)` automatically on every
register change, so an overlay literally cannot be changed outside a release.
`release/swapset.complementary()` runs the reverse order where an interaction
is suspected, and states its own limit rather than hiding it: for *n*
increments there are *n!* orders and two are run, which "detect that an
interaction exists and bound it; they do not decompose it" (framework demand
X3 — a full Shapley decomposition is 120 runs and 60 hours nobody will pay for
monthly).

### 2.9 Release certification: tolerance rules and proportional re-certification

`release/certify.py` inherits its tolerance table directly from spec §5.6 —
exact zero-tolerance on `outcome_code`/reason codes/node path, exact-to-the-cent
on money, 1e-6 absolute on scores, 1e-12 relative on intermediates — and adds
the mechanism spec §5.6 states as a requirement but doesn't design:
`NoOutcomeChange` as a `Declaration` that blocks the release if it moves **any**
golden output, no tolerance band, no override. `mutation_screen()` (§2.4) is
what makes the sign-off artefact's clause-level claims honest rather than
decorative.

**Proportional re-certification** is the second half, and it is static lineage
doing a job nobody asked it to do: `manifest.cone(changed_modules)` gives the
outputs reachable from a change with no execution; intersecting that with the
coverage index (which golden records exercised a branch inside those modules)
gives the re-certification scope. In the worked release, increment 3 touches
`cap_register` alone, whose cone reaches most of the flow — but the coverage
intersection is 8,140 of 50,000 records, so the local run drops from 5 minutes
to 58 seconds. The honest caveat is in the same file: "a change inside a module
that produces `risk_grade` has a cone of the entire flow ... it is not a
general answer; it is an answer for the 60% of changes that are peripheral."

### 2.10 Adjustment governance

Six of eight flows carry overlays that "change answers with no code change, no
structure change and no table change" — spec §5.14's framing, and
`adjustments/register.py`'s reason for existing at all: "an estate with
overlays and without this capability is an estate where the most frequently
changed thing is the least governed." Three pieces answer the three parts of
the task:

**The live register and ageing/expiry.** One versioned, effective-dated
artefact across all eight flows, answerable as-at any past date. Six monthly
reports (`register.py`, lines 99-148) — past review date, renewed without
rejustification, orphaned, never fired, dominant — and the one with teeth,
`unwind_estimate()` (`adjustments/unwind.py`), because "an expiry report that
cannot say what unwinding would cost produces indefinite renewal, which is
indistinguishable from having no expiry at all." It's costed honestly: a
200,000-record stratified sample per overlay per month, ~4 hours across 90 live
overlays, against ~45 hours at full population — "which does not [fit], and the
report would then be produced quarterly, which is how expiry stops having
teeth."

**One implementation for stack-on and stack-off.** Spec §5.14.3 is explicit
that a second implementation "will agree at first and diverge silently." The
answer in `adjustments/unwind.py` is that an overlay is not a branch in the
graph, it's a *value* `core.adjustments` applies — so `disabled(pins)` is a
params swap (doc 08 §4.2, free, no compile), not a second code path:

> `base = flow03.at_pin(pins.with_overlays(DISABLED))` ... The compiled kernel
> is byte-identical. The equivalence is not asserted; there is nothing to
> assert, because there is one kernel.

That only holds if the overlay bundle's *type* is fixed across enabled/disabled
— which is why `disabled` is a magnitude of zero on a fixed-width array rather
than an absent entry (framework demand D11: an absent entry changes a container
field's length, which doc 08 §4.2 lists as a recompile trigger — "run with the
stack off" would otherwise silently become a compile on the request path).

**The conservative-only asymmetry, verified independently.** `asymmetry.py`
does not trust a flow's own enforcement. It proves the property from the
manifest with a two-part check: structurally, by reading the composition at
each overlay consumption point (`min(base, base*(1-m))` composes monotonically
downward for `m` in [0,1]; `base*(1-m)` alone does not, because `m<0` loosens),
done once per shared capability and inherited by every consumer — 21 checks
instead of 8×N; and empirically, as a backstop, by sweeping the magnitude
across its bounds *and their negation* over 10,000 golden records and injecting
the resulting `holds` assertion into the flow's own certification suite, so a
flow "cannot delete it, and a new overlay kind cannot ship without one."

### 2.11 Coverage and dead logic at production volume

`coverage/measure.py`'s answer to spec §13 Q13 ("can coverage be measured
uniformly") is two primitives, not one and not five, and the file states why
that number and not another: **branch paths** — a rule firing, a tree node
taken, a gate evaluating true, a ruleset arm, a routing index — are all the
same measurement (one int64 per branch point, free at any batch size, per doc
04 §4.1), which is itself the evidence that they are one *kind*, answering
Q5 in passing. **Index reads** — a table cell, a scorecard bin — are not
branches, they're lookups, and per-record tracking of them at 22M
decisions/year is "gigabytes of almost never-read evidence"; so cell coverage
is a shared histogram counter incremented in the kernel and merged per
partition, which is exactly why spec §5.6 measures table cell coverage but
never thresholds it — the asymmetry is now a mechanism, not a concession.

`coverage/shadowed.py` answers spec §5.8's "should be answerable without
running anything" almost completely: doc 08 §3's closed 13-operator interior
vocabulary makes feasible-region accumulation over priority order a decidable
fragment, so `shadowed()` runs static, no data, over all eight flows in one
pass. It also says exactly where the fragment ends — comparisons between two
features (need an LP), derived features with an undeclared output range
(framework demand D14), and undeclared string/category universes — and reports
those as `unknown`, never as `not shadowed`, following doc 04 §3's convention
for lineage gaps: "a governance tool that guesses in the safe-looking direction
is worse than one that admits a gap."

### 2.12 PII: traces are production data, and debugging needs them anyway

Spec §5.13's conflict is structural, not a matter of discipline: everything
that makes a decision defensible (§5.2, §5.3, §5.8) wants evidence abundant and
searchable; everything that protects the applicant (§5.13) wants it scarce.
This tree doesn't resolve the conflict — the spec is explicit that nothing
does — it makes the masked path good enough that unmasked access stays rare
without being slow. `explain/disclosure.mask()` preserves exactly what a
`holds` assertion or a rule condition in the manifest actually tests, plus
every table band edge, rather than guessing at what "representative" means; and
it says outright what it does not solve: "a masked record that preserves band
edges, correlations and null patterns over a 45-characteristic scorecard is
re-identifiable by anyone with the bureau file" (framework demand X4). Access
to anything unmasked is named, purposed and logged rather than gated — the
`unmasked_access_log_years` field in `harness/tolerances.py` — because, per the
spec itself, "an organisation that claims to have solved this has usually just
made the gate slow enough that people stopped asking and started copying."

---

## 3. What each audience sees: walking the rendered artefacts

`artefacts/decision-record-FLX-2027-03-0418822.json` and
`artefacts/explanations-FLX-2027-03-0418822.md` are one applicant, real numbers,
and the single most valuable pair of files in this sketch: they are what §2's
mechanisms actually produce, not a description of what they would produce.

The decision: a Flex Loan applicant asked for **R180,000** over 72 months and
was offered **R95,000**. The record's `10_chains.amount_cap` shows a 52-rule
waterfall reduced to five binding events — the product ceiling, an appetite
ceiling by grade, an arrears cap, a group-exposure cap, and a campaign uplift —
each with the exact `holds`-derived clause sentence, its bureau or table
evidence, and its status (`bound` / `evaluated_did_not_bind` / `not_applicable`,
never just "didn't fire"). The record also carries `8_overlay_stack`: three
overlays applied, one of which — `ADJ-2027-008`, an 18-point score shift — moved
the applicant's risk grade from 6 to 7 and is flagged in the record itself as
`"THIS IS THE ONE THAT MATTERS."`

**The contact-centre consultant** (`Disclose.CONSULTANT`, rendered in 1.2s, 41
words) gets:

```
APPROVED FOR LESS THAN ASKED
Asked for R180 000 · Offered R95 000 over 72 months · R3 118.04 a month

MAIN REASON
The amount we can offer is limited by this client's credit record with
other lenders, and by the credit they already hold with us.

WHAT WOULD HAVE MADE THE BIGGEST DIFFERENCE
Reducing what they already owe us ...

ALSO TELL THEM, IF ASKED
Our own lending policy is currently more cautious than usual ...
```

No threshold, no score, no grade, no rule id — not editorial restraint, but
`explain/disclosure.project()` dropping every field above `CONSULTANT` before
the template runs, so the cut-off (a `library-policy` param, classified
`Disclose.ANALYST`) is structurally absent rather than politely omitted. "What
would have made the biggest difference" is *computed* from the chain — the
largest **actionable** delta (CAP-0361, −R44,000), not the largest delta
overall (CAP-0100, −R350,000, which the client cannot act on) — and the last
paragraph is the overlay, present for all three audiences per §5.14.5, because
"a consultant who does not know the Bank tightened will tell the client
something false when asked."

**The credit analyst** (`Disclose.ANALYST`, 7 pages, 4.8s) gets the full
52-rule waterfall table with before/after/status per rule, signed and ranked
score contributions down to the null bin (`vehicle_finance_balance ... NULL ...
−6.0`, "the null bin is a bin, not an error"), every cell read with its table
version, and — the property unique to this rendering — inline intervention
handles: `[i] requested_amount   [p] cap_register.group_headroom_factor   [o]
disable ADJ-2027-008   [s] sweep`. Analysts get all of `2.6`'s intervention
surface *inside* the explanation, because spec §5.3's 5-second budget only
matters if the handle is where the analyst is already looking.

**The ombud adjudicator** gets a narrative, not a table: five policy steps as a
markdown table with the ceiling before and after each, 47 further policies that
did not bind listed in an appendix "because a policy that was considered and
did not apply and a policy that did not apply at all are different facts," and
a dedicated section asking the sharper question first: *what would have
happened without the Bank's own conservatism?* — "Had the tightening not been
in force, the amount offered would still have been R95,000," computed by
`whatif/intervene()` and marked as a simulation with its own `WHATIF-` id in an
appendix, never presented as the decision. Every identifier is glossed by
`explain/render.gloss()`, which "fails loudly if the identifier has no clause
sentence" — a rule with no `holds` cannot be explained to an ombud, and the
gap surfaces to the rule's owner rather than to nobody. The policy language
quoted is not a paraphrase; it is the same `holds` string the Credit Committee
approved and CI verifies, because a paraphrase that drifted from the approved
rendering is, per spec §5.7, a finding on its own.

---

## 4. What changes when a value moves versus when structure moves

Doc 08 §2's three change classes (values / interiors / skeleton) apply to a
flow. This project needs the same question answered for a *release*, because
that's the unit the Credit Committee actually approves, and `release/increment.py`'s
five kinds map onto it unevenly — which is the whole point of authoring them
explicitly rather than inferring them from a diff:

| Increment | Class | Compiles? | Re-certification scope | Who approves |
|---|---|---|---|---|
| `table("rate_card.flex", to=...)` | value | no — `rt.params.swap()` | cone of every module that reads the card | Credit Risk Policy |
| `param("cap_register.tenure_floor_months", ...)` | value | no | cone of `cap_register`'s consumers | Credit Risk Policy |
| `interior("cap_register", adds=("CAP-0455",))` | interior | yes — one background compile | cone of `cap_register` — most of the flow, but coverage-intersected to 8,140/50,000 records | Credit Risk Policy + review |
| `overlay("ADJ-2026-114", action="expire")` | **overlay** — touches no versioned artefact at all | no | cone of the overlay's declared scope | Credit Committee, via the register |
| `skeleton(build="flow03@2027-04-24")` | skeleton | yes — different image | everything; cannot share a compiled kernel with the other four | Engineering + release approver |

Three things this table makes visible that a git diff does not. First, four of
the five increments share one compiled image — `release/swapset.py` notes this
explicitly — because table, param and overlay changes are all params-bundle
swaps and the interior change stages one background compile; only the skeleton
change forces a different image, which is why it must run last or be checked
for interaction with anything it removes (`Release.check_orderable()`). Second,
"the overlay changes an answer without changing an artefact" (spec §13 Q19) is
answered by making the register itself the fifth kind of versioned thing the
release manifest tracks — there is no artefact for a diff to compare, so the
release manifest *is* the artefact. Third, `harness/tolerances.py` applies the
same value/structure axis to the harness's own numbers and does something
value-changes normally don't: several fields have `Field(7, ge=7, le=7)` —
bounds equal to the value — which is a validator expressing a permissions
statement rather than a range. *A statutory retention period is a value change
that nobody may make*, and encoding "nobody" as bounds is cheaper than a
second access-control mechanism, though it can currently only say "nobody",
not *who* (framework demand D17).

---

## 5. What this sketch does not solve

Ten places in this tree are marked `FRAMEWORK-DEMANDS` inline (D3, D6, D8, D11,
D14, D17, X2, X3, X4, X6); see that file for the full trace. Three are worth
naming here because they are structural rather than incidental:

- **The reviewable artefact's readability** (§2.4) is not attempted by this
  project. It inherits doc 04 §6's open risk unchanged and adds only an honesty
  mechanism for the prose that already exists.
- **A wrong-but-plausible replay from a source outside the sandbox's clock
  denial** (§2.3, D6) — a frame-tier date read or a `@breaks_lineage` region —
  is a real, unclosed gap, not a hedge.
- **A masked golden set is not an anonymous one** (§2.12, X4). Preserving what
  a 45-characteristic scorecard needs to still be tested is close to preserving
  what makes a record re-identifiable, and this tree says so instead of
  claiming the masking rules in `harness/tolerances.py` close the question.

---

## 6. A fix made to the tree

`adjustments/unwind.py`'s `absorb()` docstring referred to *"Flow 17's
prohibition (spec §11.17 ...)"* — but the estate has eight flows, numbered 01
through 08; "17" is a *change scenario* number (spec §11, scenario 17: "a flow
merging an overlay into its base table during a refactor"), not a flow number,
and the same file cites `§11.13` and `§11.16-17` correctly elsewhere as scenario
references. This was an outright inconsistency rather than a deliberate
invention, so it is corrected in place to "Scenario 17's prohibition (spec
§11.17 — a flow merging an overlay into a base table during a refactor ...)".
No other file in the tree contained a flow number outside 01–08.
