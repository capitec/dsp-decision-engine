# Framework demands — 04 campaign targeting trees

Numbered, traced, marked. **SAT** = satisfied by doc 03 (or 08). **EXT** = doc 03
lacks it but nothing in doc 03 fights it. **UGLY** = doc 03 *has* a way and the
way is bad enough that this sketch did something else — the ones worth arguing
about. **AWK** = genuine awkwardness this sketch could not resolve and is not
hiding.

Every number below is quoted somewhere in the tree — `grep -rn "DEMANDS #" .`
from `04-campaign-targeting-trees/` finds seventeen of them already wired into
docstrings and comments before this file existed. Those seventeen are fixed;
this file had to be written to match them, not the other way round.

---

## The tree kind itself

**1. A data-shaped module kind needs a `publishes=` obligation, not just
`reads`/`writes`.** *Spec §5.4.1(b), §5.9, §9.6. EXT.*
Doc 08 §3's `ruleset(name=, reads=, writes=, interior=, params=)` describes a
module's *interface*. It has no notion of a module whose validation
**produces side artefacts that are not columns** — node metadata, a route
dictionary, an identity map against the previous version, a validation report,
a dependency-free interpreter — every one required by spec before a version
may run. `modules/tree/__init__.py`'s `CampaignTree = data_shaped_kind(...,
publishes=["node_meta", "route_dictionary", "identity_map",
"validation_report", "portable_interpreter"])` makes this a declared,
checkable list rather than five ad-hoc calls a publication script happens to
make in the right order, and failing to emit one is a build error naming which
artefact is missing — the discipline `reads`/`writes` already gets for
columns, extended to a module's documented obligations.

**2. Two fingerprints, not one.** *Spec §5.4.3 req 1, req 6; §11.2. EXT.*
`modules/tree/canonical.py` computes `shape_fingerprint` (topology, operator
kinds, slot-array length — the **compile** key) and `node_key` per node
(condition text plus discriminator — the **identity** key) as genuinely
independent functions of the same document. `v11.json` and `v12.json` share
`shape_fingerprint: "sf_0c7419ae63b8d25f"` byte for byte while node 6's key
changes from `n_a1f45e9c2b70d863` to `n_b7302ce8149fa65d` — one number moved,
one fingerprint held and one broke, which is the entire content of
`validation/v12.report.json`'s `compilation` block ("A threshold move does
not change the shape, so nothing compiled"). Doc 08 §8's `pipeline.fingerprint()`
is one string covering "skeleton *and* interiors"; it was never asked to keep
two identities and cannot express that an edit changed only one of them.

**3. Route capture must be an unconditional structural output, not a tap.**
*Spec §5.4 preamble ("stating it as a property of Stage 3 has, historically,
resulted in its being dropped when the run got tight"), §5.4.1(a). EXT.*
Doc 04 §4.1's taps are declared, cheap and **optional** — a consumer can omit
one. `modules/tree/emit.py` states the opposite: "Not a tap, not a trace
mode... the emitted traversal cannot produce a leaf without producing a
route." The route fold (`_edge(route, edge_id)`, two integer ops per node) is
emitted **inline in the traversal itself**, not as a side-channel a
`taps=[...]` list could leave out. A kind needs a way to say "this write is
the deliverable, not diagnostics" — stronger than anything `taps=` makes.

**4. `Table` needs a declared `identity_bearing` flag.** *Spec §5.3.4 req 1,
§5.4.3 req 6. EXT — doc 03's `Table` sketch is nine lines and provisional.*
`thresholds` (`identity_bearing=True`) and `leaf_values`
(`identity_bearing=False`) are both arrays of numbers indexed by slot, and they
must be treated oppositely: a threshold's value is baked into `node_key` by
`canonicalise()`, so moving it changes identity; a leaf's amount cap is not, so
moving it does not (`canonical.py:leaf_key` deliberately excludes the amount
rule, the channel list and the priority weight — see #13). Doc 08 §4.2's rule
that "table contents, for a generic-kernel kind" are free to change is true for
one of these tables and false for the other, on the *same module*. The flag is
what lets one kind hold both.

---

## Publication: dating, caching and compiling a version

**5. `as_at=` belongs on every document resolution, not only on tables.**
*Spec §5.3.4 req 7 ("resolved by cycle_date, never by today"), §5.9 ("never the
latest"). EXT.*
`modules/overlays/resolve.py` needs the overlay register, the tree version, the
band set and the suppression registry all resolved as of `cycle_date` — a
re-run of March in 2028 must see March's overlays in March's order, including
the ones since expired. Doc 08 §6.2's `resolve_params(doc, *, origin:, complete:)`
has no temporal parameter at all. Effective dating is not a table-only concern
(`00 §6.19` already says so); this sketch needs it on five different artefact
*kinds*, only one of which is a table.

**6. A cross-process, content-addressed compiled-artefact cache, not a
same-process staged generation.** *Spec §8 ("preparation... must not scale with
population size"), §5.9 (publication "does not run in the cycle"). EXT.*
Doc 08 §4's lifecycle — `stage()` compiles in a background worker, `activate()`
swaps a pointer — is scoped to **one process holding one active generation**.
Here compilation happens in `pipelines/publication.py`, a separate job, for
one campaign, possibly weeks before any cycle runs it; the cycle process that
eventually loads it never compiled anything and may not be the same host.
`modules/tree/__init__.py:prepare()` needs an artefact cache keyed by
`shape_fingerprint` that both processes can reach, months apart, with no
"generation" spanning them — closer to doc 08 §6.2's
`pipeline.build(require_cached=True)`, but that call is scoped to one pipeline
built in CI, not sixty independently-published trees.

**7. `prepare()` must batch-resolve many independent artefacts and raise on
any miss — never compile.** *Spec §8 ("preparing 60 trees... must cost under 4
minutes... and must not scale with population size"). EXT.*
`prepare(campaign_ids, as_at)` resolves sixty campaigns' tree versions to
`shape_fingerprint`s, looks each up in the cache, and raises on the first
miss — it is explicitly "a cache **lookup**, never a compile." Doc 08 §6.2 has
the right shape (`require_cached=True` raises rather than silently compiling)
for a single pipeline at build time; nothing in doc 08 covers resolving and
checking sixty independently-versioned artefacts *at run start*, each of which
might have compiled on a different day.

**8. The emitted-code-size guard rail must be declared per module *kind*, not
one framework constant.** *Spec §5.3.1 (20–400 nodes), §8 (six-hour window).
EXT.*
`CampaignTree` declares `max_emitted_lines=6_000` and says explicitly: "NOT
doc 02 §1.2's 500." Doc 02 §1.2's guard exists to catch an accidental `fuse()`
group nobody meant to build; a 400-node tree emitting several thousand lines
for one shape is not a mistake, it is the shape. One global ceiling cannot
serve both a `fuse()` group that should almost never exceed a few hundred lines
and a codegen kind whose natural size is thousands.

**9. The equivalence ladder must also check two *independent computations* of
one artefact, not only execution modes of one declared function.**
*Spec §5.4 (the route as an output), §8 (determinism). EXT.*
`modules/tree/routes.py:route_of()` folds a *list* of edge ids exactly the way
the kernel folds them one at a time during traversal (`fold_route` in
`canonical.py`) — two separately written implementations of the same fold, one
offline and one inside the compiled kernel. Nothing in doc 02 §3.1's
`interpreted ≡ stepped ≡ fused` ladder compares them, because that ladder is
about one declared computation run four ways, not two declared computations
that are supposed to agree. `assert_modes_agree` needs to cover this pair
explicitly, or a divergence between "the digest the kernel wrote" and "the
digest the dictionary says that walk should have" is undetectable until an
analyst notices a route that renders wrong.

---

## Identity, the closed vocabulary, and rendering without the engine

**10. Node identity's collision case needs an explicit, reviewable
discriminator — the framework must not guess.** *Spec §5.4.3 req 1, §13 Q5.
EXT.*
`canonical.py`'s default is honest and stated as a default: the same condition
**is** the same node, because the tree is a DAG and node 6 in the spec's own
worked example is reached from two parents. Where an analyst means two
different tests that happen to canonicalise identically, `{"condition": {...},
"as": "risk_gate_late_tenure"}` is required, hashed into the key, and enforced
by `check_identity_collisions` at publication — "if these are the same test say
nothing, if they are different tests give one an `as` label." Doc 04 §5.1 says
identity "must be deterministic... never auto-generated"; it does not say what
happens when two authored things are, by the letter of the rule, identical.
This sketch's answer — a declared, hashed, validated escape hatch rather than a
synthetic disambiguator — is the part worth lifting into the framework.

**11. A compound, content-derived reference type for a closed vocabulary's
addressable sub-parts.** *Spec §5.9 req 8, §11.18 (scenario 18). EXT.*
`slot_id(key, ordinal) -> "s:<node_key>:<ordinal>"` puts the node key **inside**
the name of the thing an overlay targets. The consequence is not a feature
someone had to write: when node 6's condition moves and its key changes, the
slot id `s:n_a1f45e9c2b70d863:0` simply stops existing, so a volume dial still
pointing at it fails validation by construction (`check_overlay_disposition`)
rather than by a cross-reference check someone remembered to add. Doc 03 has
`param()` and `Table` but no primitive for "a reference whose own text encodes
the identity of the thing it depends on, so that the reference breaks visibly
when that thing changes identity."

**12. The portable interpreter is a fourth rung of the equivalence ladder.**
*Spec §9.1, §5.4.2(4). EXT.*
`emit_portable_interpreter()` ships a ~200-line dependency-free evaluator
inside the artefact itself, because the analyst answering a dispute in
`§9.1` has no decision system available and Doc 08 §1.3 correctly forbids
building a general interpreter for the **hot path**. It does not forbid one
here: the interpreter runs one record at a time, in a spreadsheet-adjacent
context, and its agreement with the kernel is exactly the property doc 02
§3.1's ladder already tests for the other three modes. `portable ==
interpreted == stepped == fused` needs to be a stated fourth rung, not an
unstated fourth thing that happens to also need testing.

**13. A leaf's identity must be declared narrower than its payload.**
*Spec §5.4.2(1) ("response by leaf month on month"). EXT.*
`leaf_key(outcome_code, tier_code, reason_label, discriminator)` deliberately
excludes the amount rule, the channel list and the priority weight — a
campaign owner nudging `priority_weight` from 0.58 to 0.61 must not break the
leaf's response history, while a leaf moving from tier B to tier C is a
genuinely different offer and should break it. This generalises #4's
`identity_bearing` flag on a *Table* to identity-bearing *fields* on a single
authored record: the framework needs a declared boundary between "what makes
this thing the thing it is" and "what this thing currently advertises,"
because conflating them either breaks history on every tuning pass or hides a
genuine change of offer behind an unbroken key.

**14. Closed vocabularies need a paired emit-rule and validate-rule per
member — a registered step per instance is not a usable answer.** *Spec §5.9.2
(reachability), §5.3.1 (node conditions), §5.5 (amount rules). UGLY — doc 08
§3.2 proposes exactly the pattern this sketch rejects.*
Two closed algebras live in this tree: fourteen condition operators
(`canonical.py`'s `Op` literal — `ge`, `in_band`, `linear_ge`, `is_established`,
…) and four amount-rule kinds (`document.py`'s `Literal["none", "fixed",
"capped_preassessment", "scaled_capped"]`). Doc 08 §3.2's answer to "a derived
value needs logic" is "register it as a step, referenced by id" — right for a
value a rule *reads*, wrong for an operator the *reachability prover* must
reduce to an interval before deciding whether a path is satisfiable
(`validate.py:check_reachability`), and wrong for an amount rule the validator
must bound symbolically against the appetite grid (`amount_scale <= 1.0` is a
**reject**, checked symbolically, not sampled). A step is opaque to both.
What is needed is a closed enum where every member ships an emitter *and* an
interval/bound rule together, so "an operator without one cannot be added" is
a build-time property, not a code-review discipline.

---

## Overlays as a governed artefact

**15. `impact()` needs `as_at=`, `overlays="both"`, and a full-population,
two-*version* comparison — not two generations over a sample.**
*Spec §5.9.7–8. EXT.*
`pipelines/publication.py:impact()` is doc 08 §5's
`decider2.impact(active, candidate, sample)` with three differences it needs
to earn: resolution by the publication's own cycle date rather than today; the
with/without overlay pair `§5.9.8` makes mandatory, not optional; and the
comparison is between two **versions of one data-shaped module**, run over a
campaign's full 6.7 M-average candidate population, because a ±25% population
bound is exactly the kind of thing a sample cannot certify. 2 400 publications
a month, each two tree kernels over one campaign's population, is the actual
shape — cheap because both sides are already compiled, not because either side
is small.

**16. The overlay stacking order is its own declared, approved, hashed
artefact — never an emergent property of execution order.**
*Spec §5.3.4 req 3. EXT.*
`overlays/stacking_order.json` states precedence per kind
(`cut_off_shift` at rank 10 through `cap_reduction` at rank 60), names a
per-campaign exception with its own approval reference (campaign 31 swaps
ranks 20 and 40, approved `CRC-2026-04-07`), and is explicit about why: "the
order is part of the overlay definitions, NOT an emergent property of how the
cycle happened to run." Doc 03 has no artefact for "the order two independent
policy documents compose in," because nothing in the params/structure/interior
taxonomy is *about* composition order between otherwise-unrelated documents.

**17. Overlay resolution has (at least) four named failure modes that must
`raise`, never warn.** *Spec §5.3.4 reqs 5–6, acceptance criterion 17. EXT.*
`modules/overlays/resolve.py:resolve()` enumerates `EXPIRED` (a review date
passed with no renewal — scenario 17), `OUT_OF_SCOPE`, `MATCHES_NOTHING` (a
tightening scoped to a retired campaign, caught rather than assumed working),
and `DANGLING_TARGET` (a threshold shift naming a slot the new tree version no
longer has, naming the old node key and the identity-map entry that explains
where it went). Nothing in doc 03 or 08 has a vocabulary for "a document's
reference into another artefact that must fail the whole cycle by name if it
cannot be honoured" — the closest existing idea, an unbound step input
(doc 03 §2.2), is a build-time authoring error, not a runtime failure against
a *second* document that changed underneath the first.

**18. Approval separation must be checked on the overlay's declared *kind*,
not on a free-text field.** *Spec §5.3.4 req 4. EXT.*
"A risk overlay... requires Credit Risk approval. A volume dial requires the
campaign owner and the campaign forum. Neither may approve the other's kind."
`resolve()` checks this against the `Literal["cut_off_shift", "threshold_shift",
...]` kind, so "a volume dial relabelled as a risk overlay to get it past the
forum fails on its target instead" — the mislabelling cannot succeed because
the check reads the typed field the schema enforces, not a description a
submitter writes. This is a governance property doc 04 §1's fourth audience
(compliance, "checking rules against a policy document") needs and doc 04 §2
does not currently give it: a validator bound to a *closed, typed* field is
strictly harder to route around than one bound to a string.

---

## Determinism and attribution

**19. `stable_hash64` must be a frozen, versioned, golden-tested framework
primitive.** *Spec §5.7 req 1. EXT.*
Control, universal holdout and variant assignment are all
`stable_hash64(client_id, campaign_id, design_version)` and nothing else — no
stored table, no run-order dependence. `modules/holdout/__init__.py` is
explicit about the stakes: "a silent change to it moves every client in every
campaign between groups and invalidates every in-flight measurement, and
nothing in the output would look wrong." Doc 03 has no such primitive;
Python's `hash()` is salted per-process, but even a project-supplied hash
needs a pinned algorithm, an explicit version tag, and a checked-in golden
test vector — "whatever the library does this release" is precisely the
failure mode. A small SAT note alongside: there is no `if control: skip`
anywhere in the skeleton, because control enters only as a refusal reason
inside `Arbitration` (#24) — doc 03's composition model needed nothing new to
keep control clients on the same evaluation path as everyone else; the
framework simply offers no shortcut to find.

**20. `ruleset` needs a bitmask `output_kind`, with an auto-published bit
dictionary.** *Spec §5.2 req 1 ("evidence must preserve the distinction"), req
2 (per-suppression attribution). EXT.*
`GlobalSuppressions = ruleset(..., output_kind="flags", flag_dictionary=
"warehouse/suppression_bit.csv")` packs all nineteen suppressions this sketch
enumerates into one `int64` column, bit-indexed for life, exploded against a
published dictionary the warehouse joins with an ordinary `AND`/shift query —
"a client suppressed by four rules shows four" at one column's cost rather than
four. Doc 08 §3's `ruleset` only shows scalar `writes` (`term_cap`,
`decline_code`); a *set* of independently-attributable boolean facts about one
record, at nineteen-to-thirty-four-and-growing cardinality, is different enough
to need its own declared output shape and its own generated dictionary
artefact.

**21. The absolute/measurement split as a values change, not structure — a
place doc 03 already wins.** *Spec §5.2 req 3. SAT.*
Which suppressions skip evaluation entirely and which must still be evaluated
and path-recorded is `absolute_mask`, a single `param()` bound `bit-and`ed
against `supp_mask_global` (`modules/suppression/__init__.py`). Compliance
reclassifying S23 from measurement to absolute is then "no compile, no staged
swap, no engineer, bounded by a validator" — the file calls this out itself as
"doc 03 §4 working exactly as advertised on a requirement it was not designed
for, and it is the cleanest win in this sketch." Worth stating plainly next to
#20's gap: the *value* half of suppressions is handled perfectly by what
already exists; it is only the *shape* of the output (a bitmask, not a scalar)
that needed anything new.

---

## Shared tables and cross-artefact lineage

**22. A shared reference table can be identity-bearing for many consuming
artefacts at once.** *Spec §6.1, §13 Q12. UGLY — doc 08 §4.2's blanket rule is
actively wrong for this table.*
`registries/feature_bands.json` carries its own warning:
`"identity_bearing": true` and a comment calling itself "THE MOST
CONSEQUENTIAL FLAG IN THIS REPOSITORY." Doc 08 §4.2 classifies table contents
as free — no compile, no review, a values change. That is wrong here because
`canonicalise()` *resolves* a band reference to its boundary values before
hashing (`canonical.py` step 5): moving band 5's lower edge from R12 000 to
R12 800 changes the `node_key` of every node in every one of the 38 trees
testing it, simultaneously. The file's own `pending_edit` block shows the
measured consequence — 114 nodes across 38 trees, ~412 000 clients crossing —
and blocks publication "until all 38 campaign owners hold an identity map for
their tree." A rule that treats *all* table content as uniformly free cannot
express a table whose edits are, for named downstream artefacts, structural.

**23. Reverse lineage across independently-versioned artefacts.**
*Spec §11 scenarios 4 and 5, §5.9.6, §13 Q12. EXT.*
`pipelines/publication.py:trees_referencing(feature_id, as_at=)` answers "which
trees use this feature" **before** Data Engineering retires it or Compliance
rules it a proxy — a static, no-execution reverse index maintained at
publication time, exactly the kind of question `check_prohibited` needs to
resolve "which 11" trees a newly-prohibited feature touches (scenario 5).
Doc 04 §3's `lineage(z)` runs forward, from an output to its inputs, *inside
one pipeline*. This runs backward, from one input to every version of every
one of sixty independently-versioned trees that might reference it — a
different query shape the framework does not currently have, needed at the
moment a shared input changes underneath many consumers rather than at the
moment one consumer is reviewed.

**24. `allocate(...)` — a fourth declarative frame-tier kind for constrained
assignment.** *Spec §5.6, §13 Q9. UGLY — doc 02 §5's escape hatch is available
and using it here would be the wrong call.*
Arbitration cannot be decided one client at a time: "whether client 8 412 907
gets an SMS depends on how many other clients want one, which is not knowable
while evaluating that client." Doc 02 §5 ships `join`/`aggregate`/`filter` and
offers `@breaks_lineage` for anything else — and `@breaks_lineage` on the
single most contested output in the project ("how much of my population did
I lose and to whom") would put it behind exactly the lineage gap that
annotation exists to make visible, not acceptable. `allocate(name=,
partition="cycle", demand=, score=, order=tie_break(...),
constraints=[refusal.when(...), allocate.cap(...), allocate.capacity(...),
allocate.fairness(...)], writes=, reports=)` guarantees three properties as
part of the kind rather than as project discipline: **totality**
(`demand == admitted ⊎ refused`, so `refusal=` is required and "not selected,
no reason given" cannot be expressed), **determinism** (the tie-break is
declared, never row order), and **accounting** (per-constraint utilisation and
refused-demand value are an output of the kind, not a report the project
computes afterwards and hopes agrees) — the three properties any future
constrained-assignment problem in this framework will also need.

---

## Where an overlay attaches, and its unadjusted twin

**25. Overlays need a declared application *site*, checked against a declared
*kind* vocabulary.** *Spec §5.3.4 req 1, req 6; §13 Q16. EXT — doc 03 has no
notion of a site at all.*
`modules/overlays/sites.py` names exactly three places an overlay may change an
answer — `ScoreShift` (a record-tier input transform, before the tree),
`CampaignTreeStage`'s thresholds array (substituted elements, never edited
source — this is what makes "an overlay is not an edit" true *by
construction*: `emit.py`'s "no literal is an immediate" means every threshold
reads `thr[17]`), and `AmountCapOverlay` (after pre-assessment, before the
leaf's amount rule) — and all three are modules present in the skeleton
whether or not any overlay is currently in force. Declaring sites in code,
rather than letting an overlay be "some params that some step happens to
read," is what lets `resolve()` say **by name** that an overlay of a kind with
no wired site cannot be applied, rather than silently doing nothing (spec
§5.3.4 req 6's exact failure mode).

**26. `shadow(...)` — a structurally-guaranteed paired evaluation, not a
second pipeline.** *Spec §5.3.4 req 2, §8 (the 5% shadow), §9.5. UGLY — doc 03
offers two options and both are named as bad in the sketch's own comments.*
`pipelines/monthly_cycle.py` runs `shadow("campaign_stage:*", thresholds=
"unadjusted", params={...: 0.0/1.0}, prefix="unadjusted_")` over the whole
per-campaign stage, producing `unadjusted_route_digest` /
`unadjusted_leaf_key` for **every** evaluation, not a sample (the separate 5%
dispatch decision lives in `holdout.policy_arm`). Threading a shadow value
through every step by hand doubles authoring and diverges at the first
careless edit; running the whole pipeline twice is 2× cost and makes
arbitration meaningless, because shared channel capacity would be consumed
twice. `shadow(...)` duplicates the sub-graph with one input neutralised, so
the shadow's graph *is* the primary's graph by construction and the two
cannot drift apart — what §9.5's "what would we have done without the
overlays" needs to be trustworthy rather than merely plausible.

---

## Cycle-scale guarantees and cross-cycle state

**27. The generation-pointer-read-once guarantee needs to be stated at cycle
scope, not just per invocation.** *Spec §8 (determinism, "identical...
⇒ identical output, row for row"). EXT.*
Doc 08 §4 property 1 measures "0 straddled batches... across 11,605 swaps in
3 s" — a guarantee about **one invocation**. `pipelines/monthly_cycle.py`'s
`Cycle(...)` is "tens of thousands of invocations over six hours," and it pins
one generation, one manifest, for the whole run: "a `Cycle` pins ONE generation
and ONE document manifest for its whole life." The existing property is
necessary but not sufficient — a cycle needs the same guarantee scoped to the
*cycle*, not re-derived per call, or a manifest amendment landing mid-run could
straddle a six-hour cycle exactly the way doc 08's own deliberately-wrong
version straddles a batch.

**28. `manifest_inherits=` — a cycle resolving its artefact set from a prior
cycle's already-pinned manifest.** *Spec §5.1 req 3 ("a client processed in a
delta and again in the next monthly run must not be treated under different
definitions"). EXT.*
`pipelines/daily_delta.py`'s `Cycle(..., manifest_inherits="2026-09-M")` is
called out as "THE line that makes §5.1 req 3 true": tree versions, band set,
feature registry, suppression registry and overlay stack all resolve from the
named monthly manifest; only `cycle_date` is free to move. A tree published on
Tuesday is live in Wednesday's delta because publication wrote a manifest
*amendment* — a reviewed act — not because the delta re-resolved "latest."
Doc 08 §6.2 has `resolve_params(doc, origin=)` for one document; nothing
declares that a *second cycle's whole artefact set* is pinned to a first
cycle's, which is what keeps fatigue and capacity consistent across ~20 deltas
between monthly runs (scenario 9).

**29. `enumerate_movers()` — diffing two deterministic-hash design versions
with no cycle run.** *Spec §5.7 req 2, scenario 6. EXT.*
Because control and variant assignment are pure functions of identifiers
(#19), "which clients moved between `holdout_design_version` 7 and 8" is
computable offline for any pair of versions, with neither cycle run —
`modules/holdout/__init__.py:enumerate_movers(old_version, new_version,
client_ids)`. Scenario 6 (control moving from 5% to 3% on twelve campaigns)
needs exactly this to answer "which historical comparisons are now broken,"
and nothing in doc 03's testing story (`assert_modes_agree`, golden traces)
covers diffing two *assignment functions* rather than two pipeline runs.

**30. `carried(...)` — a declared input whose producer is a prior invocation
of the same pipeline.** *Spec §5.2 ("fatigue is evaluated against contacts
already made, including those issued by daily deltas since the last monthly
run"), scenario 9. EXT — a fifth kind of input beyond leaf / wired / params /
shared / `missing_as`.*
`modules/fatigue/__init__.py`'s `CarriedContactState = carried(name=
"contact_state", produced_by="monthly_cycle.dispatch", key=["client_id"],
version_column="as_of_cycle_id")` names an input that is an **output of a
previous run of the same pipeline**, versioned by the cycle that produced it
and pinned in the cycle manifest — `contact_state@2026090` is what a 2029
replay of March resolves, not today's fatigue counts. Doc 03 §1 gives three
null tiers for a value the *current* invocation cannot compute; none of them
covers a value only a *different, earlier* invocation could have computed.
Without this declared explicitly, a replay silently succeeds against today's
state and produces a plausible wrong answer — the same failure class `00 §7.3`
names for dates, arriving here through carried state instead.

---

## Keeping sixty campaigns navigable, and one honest limit

**31. Adding or retiring a campaign must cost exactly one file and one
registry row, and delete nothing, ever.** *Spec §13 Q11, scenarios 3 and 4.
EXT — doc 07's project layout has no notion of "never reuse this id."*
`campaigns/061-drive-settlement/campaign.yaml` states its own cost
explicitly: "a `trees/` directory with one published version, and a row in
`registries/campaigns.csv`. Nothing is edited." `campaigns/041-card-upgrade/
RETIRED.yaml` states the opposite direction with equal precision: a state
change (`state: retired`), nineteen tree versions retained indefinitely, 190
node keys that "remain valid join keys in the warehouse forever," and
`campaign_id 41 is NEVER reused`. Doc 07 §1's `modules/` layout is a
reuse-and-audit unit for *code*; it says nothing about a directory-per-instance
layout for sixty independently-owned, independently-versioned business
artefacts whose identifiers must never be recycled across an organisation's
entire operating life. That is a project-structure guarantee, not a module
concern, and it is worth naming as a demand precisely because nothing in doc
07 would stop someone from reusing campaign id 41 next year.

**32. `partition="cycle"` makes the campaign-recompute dependency explicit —
it does not remove it.** *Spec §13 Q10 ("can one campaign be re-run alone?").
AWK.*
`pipelines/monthly_cycle.py` states the honest answer as two assertions:
`pipeline.rerun_unit("leaf") == "campaign"` and `pipeline.rerun_unit(
"contact_sequence") == "cycle"`. Through stage 5, yes — a single campaign is
exactly re-runnable. From stage 6, no — `Arbitration`'s `partition="cycle"`
means one campaign's contact list is a function of all sixty. Declaring the
partition is what turns "discovered at 03:00" into "assertable in CI," and
that is a genuine improvement. It is not a solution to the underlying
constraint: operations still cannot re-run campaign 23 alone once arbitration
has touched it, and no annotation makes that possible, because the constraint
is in the business problem (shared, finite channel capacity) and not in the
framework's expressiveness. Said plainly rather than implied: `partition=`
values make a real dependency legible; they do not make it optional.
