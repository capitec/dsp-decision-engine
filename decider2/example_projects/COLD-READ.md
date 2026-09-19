# Cold-read study — what the authoring surface communicates

**Method.** Sixteen readers, eight on Sonnet and eight on Haiku, given *identical*
prompts. Six read one mock project each with **no framework docs, no spec, no
`FRAMEWORK-DEMANDS.md` and no other project**; two attempted persona tasks. Each
cold reader answered eight concrete questions — what does this decide, what comes
in from outside, trace an output backwards, where does execution order come from,
where would you change a threshold and is that the only place — and *then* read
the spec and reported where they had been wrong.

**Why this shape.** A reader saying "I found this confusing" is an opinion. A
reader who *confidently concluded something the spec contradicts*, with the line
that misled them quoted, is objective evidence that the artefact failed to
communicate. The divergences are the finding.

**How it differs from [examples/FINDINGS.md](examples/FINDINGS.md).** That study
asked eleven writers to *find gaps* and got gaps — it says so itself. This one
asked readers to *understand* and recorded where they failed. The two share no
assumptions, so **where they agree, treat it as settled**: change classes, the
fourth null situation, tables, and nesting all appear in both.

**Two caveats that must be applied before reading any number below.**

1. **88% of the constructs readers could not name are not part of decider2.** Of
   327 distinct symbols imported from `decider2` across the eleven sketches, 288
   appear in exactly one project. Only ten appear in five or more. Most reader
   confusion measures what an open-ended surface costs, not a decision anyone
   made. The flip side is the valuable part: where six independent agents each
   invented a construct for the same job, the framework has a hole.
2. **91.4% of functions are stubs** (1,080 of 1,182). Every "trace an output
   backwards" therefore ended in a comment. Readers distinguished the two kinds of
   dead end themselves, and the distinction is where the real findings are.

**The Sonnet/Haiku comparison.** Haiku was not obviously worse. It oriented in
20–90 minutes per project, read execution order correctly from the `|` expression
in every case, and reached similar conclusions on where thresholds live. Where the
two diverge is *verification*: Sonnet checked numbers across artefacts by hand and
found six cross-artefact contradictions; Haiku largely did not attempt that. So the
design does not appear to demand unusual expertise to **read** — it demands it to
**check**, and checking is what governance requires.

---

# Synthesis: what six cold reads and two persona tasks say about decider2's authoring surface

## 0. The correction you must apply before reading anything else

I counted every symbol imported from `decider2` across all eleven sketches:

- **327 distinct constructs.**
- **288 of them (88%) appear in exactly one project.**
- Only **ten** appear in five or more: `param`, `module`, `step`, `fuse`, `Branch`, `parallel`, `Join`, `Aggregate`, `Filter`, `ruleset`.

I then grepped docs/01–08 for every construct the readers listed as "unknown." Counts in the doc set: `Gather` 0, `grain` 0, `Search` 0, `verdict(` 0, `Observed` 0, `ceiling(` 0, `shared_value` 0, `isolates(` 0, `overlayable` 0, `carried(` 0, `na()` 0, `Fanout` 0, `precedence(` 0, `stable_blocks` 0, `keyed_materialisation` 0, `Grid(` 0, `allocate(` 0, `Timeline` 0, `Sequence(` 0, `phase(` 0, `EntryPoint` 0 — and so on for essentially the entire list.

**Almost nothing the readers failed to understand is part of decider2.** The real proposed surface is small (`step`, `module()`, `param()`, `|`, `Branch`, `Loop`, `fuse`, `parallel`, `taps=`, `interior=`, `missing_as`, `.at()`, provisional `Table`, `scorecard`/`decision_table`) and readers got most of it approximately right. What they were drowning in is eleven agents' private extensions.

That cuts two ways, and both are actionable:

1. **Discount** every "I couldn't name `keyed_materialisation`" as evidence against decider2. It is evidence about what an open-ended surface costs, not about a design decision you have made.
2. **Promote** the convergence. Where six independent agents each invented a construct for the same job, the framework has a hole. And the ten-symbol convergent core is a cleaner measurement of "what the authoring API must contain" than FINDINGS.md's verdict tallies, because no agent was told to converge.

Second structural fact: **1,080 of 1,182 functions across the eleven sketches (91.4%) have empty bodies** (33,628 Python lines total). Every reader's "trace one output backward" exercise therefore had to terminate in a comment. Keep that in mind when weighting "I lost the thread" — but note that the readers themselves distinguished the two kinds of dead end, and the distinction is where the real findings are (§1.1).

---

## 1. What recurred across independent readers

Six cold reads (projects 01, 03, 04, 05, 06, 10). Counts below are out of six unless stated.

### 1.1 The trace dies at the interior boundary — 4/6, and at a phantom name — 5/6

Two distinct failures, and readers separated them without being asked to (reader 06 explicitly labelled them "Trace A: a real gap" vs "Trace B: the nature of the exercise").

**Dead end at a config-authored evaluator with no visible interpreter — 4/6.** Reader 01: `ruleset()` compiles `evaluate: {all: [...]}` YAML trees into bits with "no interpreter, no compiler, not even a stub." Reader 04: `emit_tree_kernel` is `pass # topological emit...` sitting under a hand-written illustrative `@njit` block that *looks* executable — "the exact point where reading code became reading a comment about code, for the one function that is arguably the entire engine." Reader 06: `Search(plan=, evaluate=, admit=, select=)` and every `@measure` body. Reader 10: `register_pass` is `pass # partition register evaluation by ceiling.pass_derived_from`.

This is not "stubs, obviously." This is the seam between doc 08 §3's bounded interiors (what a business user edits) and doc 04 §3's static lineage (what the framework promises). **Four independent readers hit the same wall at the same architectural joint.** Doc 04 §3 says lineage answers questions "without running anything"; the evidence is that lineage stops at the box where the rules actually live.

**Dead end at a name with no producer anywhere — 5/6.** Verified all of these myself:
- 01: `overlay_scope_restriction_bits` (`decision/applicability.py:45`) — nearest match is an unnamed tuple element in `decision/overlays.py:145`, produced by a module positioned *after* the one that consumes it.
- 03: `rate_card.Lookup` — one occurrence in the whole tree, at the call site, never defined.
- 05: `FallbackGrade`, `Unscoreable` (used as `Branch` arms), `financial_pd`, `behavioural_pd` (consumed by `business_log_odds`, produced by nothing), and `Gather` used without being imported.
- 06: `from output.execution_package import ExecutionPackage` — `output/` exists on disk and is empty.
- 10: `shipped_offer` — named in `ordering.py`'s `isolates()` scope restriction and `basis_of("shipped_offer")`, produced nowhere, absent from the 3,485-line spec.

Every one of these passed the sketches' own `ast.parse` check. That is the design lesson: **the authoring surface's failure mode is a name that resolves to nothing, and syntax checking cannot see it.** Wiring-by-parameter-name (§4.1, the design's best asset) buys greppability at the cost of making an unbound name look exactly like a bound one. Doc 03 §2.2 already says "an unbound input is a typo, and is treated as one" — the evidence says that check is load-bearing beyond its current framing and should extend to module arms, `writes=` targets, `isolates(may_read=…)` names and interior references, not just step parameters.

### 1.2 "Where does this number actually live?" — 5/6 concluded *not one place*

Every reader ran the change-confidence task. Results:

| project | locations for one threshold | what would silently not work |
|---|---|---|
| 01 | 3 (rule YAML `default`, dated entry in `production.params.json`, overlay `ADJ-0031` ×0.70) | editing the YAML default — explicitly never read at runtime for a deployed rule |
| 04 | 3 (tree `slots`, overlay `target`, identity-map `delta` string) | the three copies are copy-pasted; `publish()` that regenerates them is `pass` |
| 05 | 3 (count `param` in file A, window `param` in file B, the fold's `where=` string in file C) | value-search for `12.0` lands on three unrelated params in the same file |
| 06 | **6 numbers in 3 places** (one Python default, four per-product JSON entries, one overlay) | editing the Python default; and `drive_30`'s base is 12.0 where every sibling is 15.0 |
| 10 | 2 (CSV rate-card cell + a hard-coded expectation in `test_p13_solve.py`) | the test's expected numbers do not self-update with the breakpoint |
| 03 | 1 — verified by grep | nothing *enforces* it; the reader established it only by searching |

**This is the strongest recurrence in the exercise**, and it is doc 01 §5.3's own diagnosis (546 inline `pl.lit()` literals against zero uses of the config mechanism) reappearing inside sketches written specifically to avoid it. The mechanism that produced it here is different — not literals, but *layering*: authored default → deployed dated value → overlay. Three layers is the right model. Three layers with no rendered "what is in force now, and why" at the point of edit is the failure.

### 1.3 The Python default is a decoy — explicitly hit by 3, structurally present in 5

Readers 01 and 06 both wrote down "my first instinct was to edit the Python/YAML default" and then discovered it would have no effect. Reader 06's case is the sharpest: `max_cost_uplift_pct: float = overlayable(15.0, ...)` reads as an ordinary Python default, is not one, and is not even the right base value for one of four products.

This lands directly on doc 03 §1.1 + §4.4 — the design's flagship ergonomics win ("one rule should cost one artefact", `param()` in the signature). The win is real. The hazard is that the syntax deliberately borrows Python's default-argument slot to mean something Python's semantics do not. Doc 06's E11 and E6 are still open on `param()`; **add "does a reader believe the default is the operative value?" to E6's brief**, because three of six readers did and two of them were wrong in a way that would ship.

### 1.4 `|` is the best orientation aid in the codebase, and nobody knows what it does — 6/6 and 5/6

Unanimous positive: the pipeline expression was named the fastest way to understand the system by every reader (see §4.3). Also near-unanimous: "I read this as sequential composition by analogy to shell pipes; it is never confirmed in this codebase; my confidence is high on the narrative order, **zero on the mechanism**" (reader 04, and four others in similar words).

Worse, **three readers found the linear reading contradicted somewhere**:
- 01: `Applicability` consumes `overlay_scope_restriction_bits`, which only `OverlayStack` (later in the pipe) could produce. The reader checked the spec and found §5.8 genuinely forward-references §5.9 — so the pipe is lying, and the spec invited the lie.
- 05: `PeopleBlendFacts` composed twice in one pipeline, and `BlendTerms` composed only inside another node's definition. Reader could not decide whether `|` is graph-based (re-listing is a no-op) or these are authoring slips.
- 10: `P10 | P11 | Map(P09.pass_one | P10.at_site("routed") | …)` reads as ordinary sequence; only `ordering.py`'s separate `cycle_break` declaration reveals a declared circularity.

Design consequence: doc 03 §8.1's "ordering is visible here, and only here" is the property readers most relied on and the one the work most often broke. If written order is the contract, **the framework must reject a pipeline whose data dependencies contradict it**, rather than letting the pipe read as documentation.

### 1.5 Sequence-shaped syntax that isn't a sequence — 2/6, but both spent real time on it

Reader 06 named it "the single most counter-intuitive thing in the codebase": `module(a, b, c)` is an unordered DAG and `A | B | C` is a sequence, and "nothing about the syntax itself distinguishes them — you have to already know which construct you're reading." Reader 05's duplicate-composition confusion is the same problem from the other side.

Doc 03 states both rules correctly (§5: interior is a DAG; §8.1: `|` is sequence). Stating it is not enough — two readers with the doc unavailable inferred the wrong thing from shape alone.

### 1.6 The README pre-narrates, and the code cannot stand without it — 3 flagged it, 11/11 verified

Readers 01, 04 and 10 independently reported that the README does the comprehension work before any Python is opened, and that this compromises the cold-read premise. I verified the pattern across all eleven projects: READMEs are **532–737 lines**, every one cites spec sections or `FRAMEWORK-DEMANDS` **33–82 times**, and all eleven have a companion `FRAMEWORK-DEMANDS.md`.

Reader 01's formulation is the one to keep: *"strip the README's rationale and the Python alone is much harder going."* Reader 03 found a file (`modules/caps/register.py`) where stripping the docstring leaves **zero rule content — only a keyword argument pointing at a JSON file in a different top-level directory**.

This is partly an artefact of the brief (agents were told to justify departures). But it also measures something real: **the framework's constructs are currently comprehensible via their rationale, not via their shape.** That is exactly the property doc 04 §6 needs them not to have.

### 1.7 The pull toward FRAMEWORK-DEMANDS.md — 5/6 recorded it as a finding

Reader 04 counted docstring references to nearly the entire numbered list (#2 through #32). Reader 06 listed thirteen items the README pointed at for constructs it left unexplained. Reader 05: "that several core semantic questions are answerable only by that companion document and not by the authoring surface itself is, I think, a finding in its own right." Agreed — and it generalises past this exercise: a construct whose meaning lives in a rationale document is a construct that will be misused by the second team.

### 1.8 Dead imports — measured, not just reported

Reader 05 said "`step` is imported in almost every file and I never once saw it used." I measured it: **229 imported-but-unused occurrences across the eleven sketches**, of which `step` in project 05 alone accounts for 22, plus `step` 5× in 11, 3× in 10, 2× in 01. `param` is imported-unused 9× in 10 and 5× each in 01 and 07.

`@step` is imported by nine of eleven projects and genuinely needed by almost none. Doc 03 §1 says the decorator is "only needed when you want to override defaults." The evidence says optional-and-visible means cargo-culted. Either it earns its place or it comes off the surface.

---

## 2. The divergences, ranked by production cost

A reader concluding something the spec contradicts is an objective failure of the artefact. Ranked by what the wrong belief would do. I verified items 1, 2, 4, 5, 6, 8 and 10 directly.

**1. Project 01's decision record makes a false attribution claim — and contradicts itself in the same sentence.** *(verified)* `fired_on_overlay_ids: ["MS-0208"]` with `live_base_fire_bits` excluding MS-0208. The reader checked all four predicates against all three threshold vintages by hand: R12,400 clears R8,000 / R6,500 / R4,550; band 4 clears both 4 and 3; 1.4h clears both 2h and 3h; 31.2h clears 72h. **MS-0208 fires identically with or without the overlay.** The file's own human-readable note reads: *"At its base threshold of R6 500 it would not have fired on R12 400 — it would have."*

Production cost is the highest in the list because of *what this record is for*: the monthly overlay-review forum and the regulator act on exactly this field. A false "this overlay bought you a catch" causes an overlay to be renewed on fabricated evidence — and the counterfactual block goes on to draw the opposite policy conclusion ("the overlay bought nothing on this event"). The record gives a reader no way to check it; only cross-referencing four files exposed it.

**2. Project 04's absolute-suppression mask disagrees with its own registry.** *(verified)* `registries/suppressions.json` → `0x0000000000002401`; `modules/suppression/__init__.py:86` → `0x0000_0000_0000_2411`. Bit 4 is S05 "legal action in progress," a *measurement*-class suppression wrongly promoted to absolute. The file's docstring calls the regeneration mechanism "the cleanest win in this sketch" — which is precisely why nobody would check it. In production this is a marketing-contact suppression: a one-bit drift decides whether clients under legal action are contacted.

**3. Project 03's README asserts the spec contains an inconsistency it does not.** The README claims §5.4.1's worked example applies two channel-4-scoped overlays to a channel-2 client. The spec says the opposite, explicitly and at length: both overlays are "evaluated and do not apply," and `score`/`probability_of_default`/`risk_grade` equal their unadjusted counterparts. The reader had believed the README in phase 1.

I rank this third despite causing no runtime bug, because of *where the error propagates*: the phantom inconsistency is cited as justification for the three-valued scope outcome (APPLIED / IN_FORCE_BUT_OUT_OF_SCOPE / NOT_IN_FORCE) and was escalated into FRAMEWORK-DEMANDS #4 — i.e. into the input to your framework design. Note the construct is probably right anyway; the spec demonstrates the same three-valued need. But the demand was raised on a misreading, and nothing in the process caught it.

**4. Project 05's README miscounts its own code's rules, twice, in the same direction.** *(verified against spec)* README says "fourteen absolute rules" where the code correctly says twenty (B-AROD-01..20); README says "12 rules × 3 classes" where both the code and its own data table say 14 × 3 = 42. Root cause is instructive: **the spec's prose says 12, the spec's own table lists 14, and the README followed the prose while the code followed the table.** A reviewer counting rules against policy concludes six rules are missing.

**5. Project 04's campaign.yaml says v12 is both a 10% A/B challenger and the champion.** *(verified)* `measurement.variants.challenger_a: {tree_version: 12, share: 0.10}` nine lines above `versions: [{tree_version: 12, variant: champion, ...}]`. Every other artefact about v12 (identity map, validation report, sign-off) assumes champion replacement. Production cost: either a champion gets A/B'd or a challenger ships to 100%.

**6. Project 04's published tree JSON does not match the Pydantic schema that governs it.** *(verified)* `TreeDocument` requires `slot_index: dict[SlotId, int]` and has no `slot_count`; `trees/v11.json` and `v12.json` carry `slot_count`/`leaf_slot_count` and no `slot_index`. The validation is vacuous.

**7. Project 06's `interventions_drive_30` base value is 12.0 where all three siblings are 15.0** — indistinguishable in shape from the *temporary overlay* `ADJ-AH-012` that sets 12.0 for a different product. An analyst unwinding the expired overlay plausibly "restores" drive_30 to 15.0 and silently changes policy.

**8. Project 03 names a reviewable artefact that does not exist.** *(verified)* `review_artefact="review/cap_register.md"` in code, named again in README §4 as the non-engineer's path — no `review/` directory in the tree. Same project: `interior="config/flex_loan/interiors/eligibility_gates.json"` — absent. Same project, from the persona run: `{"table": "appetite_grid"}` and `tables.statutory_rate_ceiling` — both absent. **A reference to a governance artefact is currently indistinguishable from the artefact existing.**

**9. Project 01's latency budget silently comes out at 18 ms against the spec's 20 ms**, because `Velocity` was folded into the concurrent `fan(...)` rather than run serially. The reader's point is the sharp one: every *other* departure in that project is flagged explicitly; this one just produces a different number for a figure acceptance criterion 15 depends on.

**10. Spec content with no code trace at all** — no reader could have known: project 01's §5.16 outcome-feedback loop (yet `metrics()` advertises precision and incremental catch, both label-dependent); project 03's §5.2 fraud-refer behaviour ("complete the assessment, force refer, compute and *hold* the offer set") and its entire §8 latency budget; project 05's `financial_pd`, `behavioural_pd` and the whole BUS-BEH-01 scorecard; project 06's §5.10 output stage; project 04's §5.5 breach-rate monitoring and §9.2 statistical proxy test.

**11. Where the code beat the spec (log these as wins, not divergences).** Project 01's `applicability.py` correctly refuses the spec §6.1 phrase "self-demotes to shadow" — "shadow membership is structural and a runtime demotion cannot cross that line." Project 06's `na()` docstring demonstrates the concrete wrong answer the spec only asserts ("a sentinel term of 0 computes 0 − 31 = −31, compares it to +24, and PASSES"). Project 06's `Fanout` vs `Branch` answers a question the spec explicitly leaves open.

**The pattern under items 1, 2, 4, 5, 6 and 7 is one thing: two artefacts assert the same fact and disagree.** Six of eleven divergences have that shape. These same agents implemented single-file rule sets essentially perfectly — reader 05 verified AE-R-01..12 and PP-06 match the spec one for one including every param default. **The authoring surface's defect is not rule authoring. It is cross-artefact agreement,** and nothing in the design currently checks it.

---

## 3. Constructs that failed to explain themselves

### 3.1 Real decider2 constructs (these are yours to fix)

| construct | readers who got the intent | readers who could state the mechanism | note |
|---|---|---|---|
| `param()` in the signature | 6/6 | **0/6** | 3 concluded editing the default would be wrong; 2 of those learned it only by checking. Doc 03 §4.4's shape is the single most misunderstood thing in the exercise. |
| `\|` (sequence) | 6/6 | 1/6 | best orientation aid *and* 3 found it contradicted (§1.4) |
| `module(*funcs)` | 6/6 | 4/6 | 2 wrongly inferred argument order = execution order |
| `step` / `@step` | 2/6 | — | one reader: "I cannot tell what `step` is actually for." 229 unused imports measured. |
| `taps=` | 3/6 | 0/6 | nobody could say what happens to a non-tapped output |
| `fuse` / `parallel` | 4/6 | 1/6 | 2 readers found `fuse` used but *not imported*; one found it only in a prose comment |
| `interior=` | 6/6 | 3/6 | the *concept* is clear; the **hop is invisible** — buried in a keyword arg pointing to a different directory tree (persona A) |
| `missing_as` vs `param` | 3/6 | 2/6 | syntactically identical (both call-valued defaults); 2 readers could only guess the distinction |
| `Table` (provisional) | 5/6 | 1/6 | see 3.2 — six spellings across seven projects |
| `.at(...)` relabel | 4/6 | 1/6 | "unclear whether it re-executes or is a compiled specialisation" |

### 3.2 Convergent inventions — the framework has no word for these, and the work kept inventing one

These are the high-value entries. Counted across all eleven sketches (not just the six read):

| concept | distinct names invented | projects |
|---|---|---|
| **nesting / ragged collections** | 9 — `grain`, `Gather`, `Each`, `Enumerate`, `Map`, `Collection`, `Fanout`, `Explode`, `Cross` | 6 |
| **keyed lookup table** | 6 — `Table`, `table`, `dated_table`, `versioned_table`, `lookup_table`, `bound_table`, plus `temporal_table`/`IntervalStore` | 7 |
| **overlay / adjustment register** | 7+ — `overlay`, `overlay_point`, `overlayable`, `OverlayPoint`, `OverlaySurface`, `OverlayRegister`, `AdjustmentRegister`, `overlay.position` | 8 |
| **the fourth null situation** | 5 — `Maybe`, `na()`/`NotApplicable`, `AbsenceReason`, `Observed`, role-bound NA | 6 |
| **unadjusted / shadow twin evaluation** | 4 — `shadow()`, `Shadow`, `unadjusted()`/`basis_of`, `overlays_off` | 5 |
| **bounded search** | 2 — `Search`, `Budget`/`Strategy` | 2 |

Nesting is the standout: **O5 is currently marked deprioritised in doc 06 and it is the single most-invented gap in the set.** Tables are second: doc 03 §4's table section is marked "provisional," and provisional cost you six incompatible spellings. Overlays are third and already confirmed by FINDINGS §2.1 through a completely different method.

### 3.3 One-off black boxes (weak evidence individually; listed because a reader lost real time)

`keyed_materialisation` (invalidation and eviction semantics unknown; the code's own comment calls it "the first thing I would load-test"), `stable_blocks`, `data_shaped_kind`, `attribution_spine`, `cycle_break`/`isolates`/`resolves_once`, `Join(how="cross_masked")`, `carried()`, `ceiling(seed="p10.max_affordable_instalment")` (cross-phase string resolution), `build_catalogue.deferred(...)`, `DecisionTable(outcomes=[...])` vs `DecisionTable(outcome=...)` — a reader could not tell whether that last one was two valid shapes or a typo, "and nothing runs to tell me."

Two cross-cutting shapes worth noting because they recurred inside the one-offs:
- **A string that resolves to a program element.** `precedes("p08.adjustments")`, `seed="p10.max_affordable_instalment"`, `shadow("campaign_stage:*")`, `where="is_minor_recent"`, `Budget(name="search_budget.interactive")` matching a JSON key by convention. No reader could verify a single one of these resolves. If the framework adopts any of them, string-to-element resolution needs a build-time check and a render.
- **A flag whose effect is described only in prose.** `identity_bearing=True` — the project's own note calls it "the most consequential flag in this repository" and the reader could find no code path that reads it.

---

## 4. What worked — protect these

### 4.1 Wiring by parameter name is the design's best asset — 4/6 named it unprompted

*"The consistent naming convention (a value's producer function and every downstream consumer's parameter share exactly one name, across file and module boundaries) meant I could grep my way to a value's origin **without any framework documentation at all** — that's a real design win regardless of how the wiring is implemented."* (reader 01)

This is doc 03 §2, and it is the only mechanism in the exercise that let readers make progress without the rationale documents. Corollary, and it is a real constraint: **everything that breaks one-name-one-value cost disproportionately.** Reader 05 could not resolve `best_of(lift=[...])`'s prefixed-vs-unprefixed field naming (found a case where the same source field is read bare in one file and prefixed in another). Reader 03 found `Quotation.writes=["offered_amount", ...]` where the module only *consumes* that name — apparently contradicting the project's own headline lineage guarantee, with no vocabulary distinguishing "originates" from "re-emits." Guard the invariant; `.at()` relabelling is where it will erode.

### 4.2 Rule ID in the function name — 3/6

`ae_r_02_fires`, `b_arod_01_fires`, `fv_01_instalment_recomputes`, `CAP-0118`, `CON-INT-04`, `H1`–`H9`, `MS-0208`. Reader 05: *"I never had to cross-reference a separate mapping to know which spec rule a function implements — I could grep the spec's rule table and the code and get an instant match **or an instant, checkable mismatch**."* Reader 06 called grep-friendly IDs across code/config/README "probably the single biggest comprehension aid in the whole project."

This is the cheapest thing on this list and it is what made spec↔code verification mechanical rather than interpretive. It should be a lint rule, not a convention.

### 4.3 One concrete worked example carried through every artefact — 5/6, the biggest single unlock

Client W (03), client V (10), Ms Dlamini / entity 7 / event 41 (05), client 8,412,907 (04), the R60,000 band edge (10). Every reader named this as the thing that made abstract declarations legible.

Project 04's is the strongest and deserves to be a design requirement, not a documentation habit: the node key `n_a1f45e9c2b70d863` appears **byte-identical across seven independent files in four formats** (tree JSON, node-meta CSV, routes CSV, identity map, overlay register, sign-off YAML, a Python docstring), then is replaced everywhere at once by `n_b7302ce8149fa65d` while every other node key stays identical. The reader could *verify* "identity is stable except where it should break" by string-matching seven files. That is a governance property demonstrated rather than asserted — and it is exactly the class of check that would have caught divergences #1, #2, #5 and #6.

**Design implication: the framework should generate the one-record walkthrough, not leave it to a README.** It is worth more than any rendered diagram, and the three readers who fell for a pre-narrated README were falling for a hand-written version of it.

### 4.4 Tests that exercise framework surface — 2/6, and both were emphatic

Reader 01: `tests/test_shadow_isolation.py` — `order = [m.name for m in Interdiction.modules]; assert order.index("shadow_rules") > order.index("resolve")` — *"more useful than any comment because it exercises real framework surface (`lineage()`, `.modules`, `.writes`) rather than describing intent in prose. I'd ask for more tests written this way, less for correctness than for documentation."* Reader 10: `test_p13_solve.py` was *"the ONE file in the whole project with real numbers in and real numbers expected out, so it was the only place I could check my understanding against ground truth rather than trust a docstring."*

Doc 03 §11 and doc 04 §7 treat tests as the correctness story. The evidence says the introspection API (`lineage`, `.modules`, `.writes`, `rerun_unit`, `writers_of`) is *also* the best documentation medium you have — and it is the one part of the framework readers gained confidence in rather than lost it.

### 4.5 Smaller things, each named once, each cheap

- **`pass  # <one-line pseudocode>`** (reader 06): "even though nothing executes, I almost always knew the intended arithmetic in one glance — far more useful than a bare `pass` or a bare docstring."
- **Docstrings that state the negative first** (reader 06): *"`abstract=True` is NOT a base class"*, *"routing is NOT a `Branch`"* — "reliably headed off a wrong model before I'd formed one."
- **Docstrings that flag their own elisions** (reader 05): `# ... fourteen more. Elided for the sketch.` worked exactly as intended — and their *absence* is what made the two real gaps (`FallbackGrade`, `financial_pd`) costly instead of merely incomplete.
- **Rule YAML / decision-table-as-data** (readers 01, 03): the single most immediately readable file in project 01 — "plain English descriptions, obviously reviewable by a non-engineer, zero framework knowledge required." Direct positive evidence for doc 08 §3. Contrast with §5 below, which is where it stops working.
- **Data shape teaching a concept faster than prose** (reader 06): `contracts/product_offer.json` explained in one read why four wildly different products are comparable — the spec spreads the same content over four prose tables.
- **Two-counter versioning** (reader 01): `version` / `shape_version` on every rule made "was this a retune or a restructure?" visible in the artefact rather than inferable from a diff — which is what spec Q4 asks for and the spec itself leaves open.

---

## 5. The two persona verdicts

### 5.1 Non-programmer credit-risk reviewer (project 03): **fail** — and this is your stated top risk

Reported plainly. The reviewer could not verify policy, and the reason is worse than "the rules are in Python."

The hop chain for "what is the maximum loan term?": `modules/caps/register.py` (no policy content, only constructor syntax) → `interior="config/flex_loan/interiors/cap_register.json"` (a keyword argument buried in a long call, pointing at a different top-level directory) → rule `CAP-0118`, whose description says "Grades 1-4 84, 5-7 72, 8-9 60, 10-11 36" but whose `effect` is `{"reduce_to": {"table": "appetite_grid", "value": "max_term"}}` → **`appetite_grid` is not in the tree.** And the JSON's own header says it shows "eight of the 52 rules," so "is this the only rule touching `term_cap`?" is unanswerable — there is also a seed ceiling in the Python and a `CAP_REDUCTION` overlay point in a directory called `adjustments/` with no naming link to "term."

"Where is the maximum interest rate enforced?" has **three answers in three unrelated-sounding directories** (`pricing/rate_card.py` at card-staging, `adjustments/points.py` as a per-application post-assert, `validation/independent.py` as a final assertion) and **none of them contains the number**; all three name a table absent from the tree. Redundant independent checks are presumably deliberate. The finding is that no naming convention hints at the fan-out, so "where is X enforced" has no single answer and no way to enumerate one.

And the framework's own prescribed answer — the generated artefact a non-engineer reads instead of code — is named in the code (`review_artefact="review/cap_register.md"`) and does not exist.

**I checked whether that is specific to project 03. It is not.** Across all eleven sketches, exactly **two** produced any reviewer-facing artefact at all: project 01's `rule-sheet-MS-0208.md` and project 09's explanation document. Nine of eleven independent designers, each given a spec demanding reviewability and told to design the authoring surface they wished existed, did not produce it.

Two conclusions, and they point opposite ways — take both:

**The format is achievable.** Project 01's rule sheet is the best artefact produced anywhere in this exercise. It renders authored value *and* in-force value side by side with the reason they differ, names the overlay, gives its approvers, its expiry and "47 days remain," distinguishes a threshold change from a shape change with different approval routes, states the missing-input behaviour per feature, gives the circuit breaker and its current state, and closes with performance by version-and-overlay period. It is sized for 635 rules rather than a waterfall of 30 ("nobody reads 635 diagrams, so the unit is a one-page sheet and the index is a table"). **Build E4's prototype from this file.** It is closer to a solved problem than doc 04 §6 assumes.

**And its correctness is not self-evident.** It is the same project whose decision record fails arithmetic against it (divergence #1). So E4's brief needs one more task than it currently has: don't only ask whether a reviewer can read the sheet — **ask the reviewer to check a decision record against it.** That is the test that just failed, and it failed against the best artefact in the set.

One more requirement falls straight out of the persona run: **a generated artefact must render resolved values and may never contain a hand-written paraphrase of a value stored elsewhere.** `CAP-0118`'s description says "36" while its effect points at a table; nothing forces those to agree, and the reviewer correctly refused to certify the change on that basis. Project 01's rule sheet does the right thing here (authored / in-force / why-it-differs as three columns). Make that structural.

### 5.2 Friday-16:30 maintainer (project 01, adapted): **finding it took under a minute; classifying the change was impossible**

Located the site by `grep -rn business_linked` in under a minute — two hits, both in one rule YAML. **That is a direct win for §4.1 and §4.5's rule-as-data**, and it should be recorded as such.

Then it stopped. `ruleset/__init__.py` declares `shape_projection=("evaluate", "applies_to.event_types", "on_absent", "tunables.*.unit")` — the classifier that decides ~2s recompile vs microsecond bundle swap, and therefore which approval route and which SLA. **`applies_to.segments` is not in that tuple**, and `production.params.json`'s dated per-rule entries only ever carry `tunables` and `outcome` — so the mechanism that should hold the change has no worked example anywhere, while the mechanism that visibly works (edit the rule YAML) is documented as costing a compile for *some* of its fields and not this one.

Map that onto doc 08 §2: **the three change classes classify documents; the work needs them to classify fields.** A rule's segment scope decides *who a control applies to* — at least as governance-sensitive as a threshold — and it falls through the gap. This is the field-level form of FINDINGS §2.1's most-corroborated finding, arrived at by a completely different method.

Two secondary findings from the same run, both concrete:
- **No enumerated registry for the categorical dimension.** Event types get a fully-enumerated `VariantSet`; segments get `versioned_table(...)` with no `document:`, no data file and no name-to-bit mapping anywhere. "What is `business_linked`, how many clients is it, what else names it" was unanswerable in the repo.
- **Blast radius was unknowable at the point of edit.** Widening MS-0208's population changes the denominator of its own `max_fire_rate_pct: 0.40` circuit breaker — the rule could self-demote on day one — and extra SCAM-1 holds could cross the 300/hour queue threshold and trigger an automatic reroute. Both consequences of a "one-line scope edit," discoverable only by manually cross-referencing three unrelated files. The project already contains every input needed to compute this (`applicable_counts_by_family`, the backtest, the fire-rate breaker, the queue SLA).

---

## 6. Concrete design changes, ranked by evidence strength

### Tier 1 — multiple independent readers, and I verified the artefacts

1. **One number, one home — and make the home visible at the point of edit.** *(5/6 readers)* The three-layer model (authored default → dated deployed value → overlay) is right. What is missing is a rendered "what is in force now, and why" at the authoring site. Minimum: the framework refuses to let a `param()` default be read as operative, and `decider` can answer "resolve threshold X as at instant T" mechanically. Project 01's rule sheet already demonstrates the rendering; it just isn't available where the edit happens.

2. **Classify change class per field, not per document.** *(maintainer persona + reader 01 + FINDINGS §2.1's five sketches)* Every rule-document field that can vary needs an explicit shape/value classification in one place, and the framework must **refuse a document containing an unclassified field**. Doc 08 §2's table is the right model at the wrong granularity.

3. **Lineage must descend into interiors.** *(4/6 traces died exactly here)* `pipeline.lineage("action_code")` currently stops at the `ruleset(...)` box. Require every interior kind to supply `lineage()` and `render()` over its rows. Doc 04 §3's promise and doc 08 §3's mechanism meet at a wall right now, and that wall is where a business user's edits live.

4. **Extend "an unbound name is a typo" past step parameters.** *(5/6 found a phantom name; all passed `ast.parse`)* Branch arms, `writes=` targets, `isolates(may_read=…)`, interior file references, `Gather` folds' `where=` strings, cross-phase seed strings. Reader 05's `Gather`-used-without-import is the canonical case: the only check actually run was syntax, and syntax cannot see it.

5. **Make written order enforceable or stop claiming it.** *(6/6 relied on `|`; 3 found it contradicted)* If doc 03 §8.1's "ordering is visible here and only here" is the contract, the framework must reject a pipeline whose data dependencies contradict the pipe. And `module(a,b,c)` (DAG) must not look like `A|B|C` (sequence) — two readers inferred the wrong one from shape.

6. **Give nesting one name and undeprioritise O5.** *(9 invented names across 6 projects; FINDINGS independently found `grain` in 2 sketches)* This is now the most-invented gap in the set by a wide margin.

7. **Settle `Table`.** *(6 spellings across 7 projects)* Doc 03 §4's "provisional" marker has a measurable cost.

8. **Give the fourth null situation one name.** *(5 spellings across 6 projects; FINDINGS §2.4 found the same from 3)* Two methods, same answer.

9. **Delete `@step` or justify it.** *(9 projects import it; 229 unused imports measured; 1 reader could not say what it is for)* An optional decorator nobody needs is surface area that gets cargo-culted.

### Tier 2 — one reader, but specific and cheap

10. **A generated artefact renders resolved values; never a hand-written paraphrase of a value stored elsewhere.** *(persona A: CAP-0118's "36" vs `appetite_grid`)*
11. **A pre-change blast-radius report, attached automatically to governance-sensitive field edits.** *(maintainer persona)* Population delta, expected firing delta, distance to the circuit breaker and the queue threshold. Every input already exists in the sketches.
12. **An enumerated, greppable registry for every categorical dimension a rule can be scoped by** — segments, channels, event types. *(maintainer persona; 01 has one for event types and none for segments)*
13. **`writes=` needs two words: *originates* vs *re-emits*.** *(reader 03)* One keyword currently covers both, which makes a module appear to violate the project's own headline lineage guarantee.
14. **A cross-artefact consistency check as a build step.** *(6 of 11 divergences are "two artefacts assert the same fact and disagree")* Schema-vs-data, registry-vs-constant, description-vs-resolved-value, test-expectation-vs-table-cell.
15. **Generate the one-record walkthrough.** *(5/6 named a hand-written one as their biggest unlock; 3 flagged that a hand-written one defeats cold reading)* One record through every stage, with the concrete numbers, produced by the framework.

---

## 7. Where I differ from FINDINGS.md, and which method I trust

FINDINGS.md and this exercise mostly agree, which matters because the methods share no assumptions: FINDINGS asked eleven writers to find gaps; this asked eight readers to understand and recorded where they failed. Change classes, the fourth null situation, tables and `grain` all appear in both. **Convergence across those two methods is the strongest evidence in the whole programme** — treat those four as settled facts about what the work demands.

Three places I would overrule or extend it:

**a) Doc 04 §6. FINDINGS §6.6 says "Do not rewrite doc 04 — it is the best-performing document in the set by a factor of three, despite nominating itself as the weakest." I disagree for §6 specifically, and only §6.**

FINDINGS' evidence is project 09's self-scored 36% satisfaction rate: an agent grading its own generated demands against the document it was told to argue with. That method cannot detect *"nobody attempted this."* My evidence is different in kind and points the other way: **9 of 11 sketches produced no reviewer-facing artefact at all**, one persona test failed outright, one project names a reviewable artefact in its code that does not exist, and the one genuinely good artefact is arithmetically inconsistent with the decision record it accompanies. For this question I trust the comprehension test, because non-production is invisible to a satisfaction count and visible to a reader who goes looking.

FINDINGS is right about the rest of doc 04. Static lineage, `.modules`, `.writes`, version chains and the audit record were the framework surfaces readers gained confidence in — reader 01 called the lineage test "more useful than any docstring," reader 10 called `ordering.py`'s self-honesty "accurate rather than defensive." Keep doc 04. Treat §6 as unstarted rather than weak, and start E4 from project 01's rule sheet.

**b) The verdict distribution should be discounted further than ±10%.** FINDINGS states two biases (the brief asked for gaps; vocabularies differ). There is a third it does not state: **327 distinct constructs with 288 unique to a single project** means the "needs extension / would be ugly" tallies are partly measuring eleven agents each designing their own framework, not eleven readings of yours. The convergent-core measurement — ten symbols at ≥5 projects, nine of which are already in or adjacent to doc 03 — is a cleaner read on the same question and was not computed. **Doc 03's core survived eleven independent attempts to replace it.** That is a more encouraging result than the 14%-satisfied headline, and it is better supported.

**c) FINDINGS §2.3 ("the fourth core kind: 06 says `Search`, 08 says sequence position, 04 says `allocate()`"). The cold reads cannot say which is right — but they add something FINDINGS could not see: three of those four candidate kinds were never traceable by any reader.** `Search`'s strategy was prose in a comment; `allocate`'s `method="rank_cut_repair"` was "pass 1 / pass 2 / pass 3" in a comment; `emit_tree_kernel` was `pass`. Whatever the fourth kind turns out to be, **it must ship with a lineage and render story or it becomes the new wall every trace hits.** Answer change #3 above before answering the fourth-kind question.

**d) On FINDINGS §7's own caution ("the sketches are not designs; nothing runs").** The cold reads price that caution: **91.4% of functions are stubs**, so every trace terminated in a comment, in all six reads. But note *which* findings survived that: the highest-value divergences — the 01 attribution arithmetic, the 04 mask drift, the 06 six-numbers-in-three-places — were all found by **checking numbers**, and readers could only do that where concrete numbers existed. For E2/E4, require one end-to-end path with real values through it, even a fake one. The numbers are where comprehension becomes verification, and verification is the only thing that found the real bugs.