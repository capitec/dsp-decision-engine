# Framework demands — 09 governance and replay harness

Numbered, traced, marked. **SAT** = satisfied by doc 03/04/08 as written.
**EXT** = the docs have no answer but nothing in them fights this one; it is
additive. **UGLY** = the docs *do* have an answer and the answer is
insufficient enough that the sketch built something else — these are the ones
worth arguing about. **AWK** = a genuine, unresolved awkwardness, named rather
than hidden.

This project is unusual and the split below reflects it. Every other project
in this set asks decider2 for authoring constructs. This one mostly does not
write pipelines — it reads other pipelines' build artefacts and runs their
pinned images. So **Part A** is short and is about what a harness needs that an
authoring surface does not: sealing, static reachability in the *other*
direction, and a claim mechanism honest enough to survive its own mutation.
**Part B** is long, because spec §5.15 states plainly that "most of the
requirements ... are not requirements on this project at all, they are
constraints on the other eight" — and that contract is the single most useful
thing in the source spec. It is restated here against doc 03/08's actual
mechanisms, item by item, because that mapping is exactly what determines
whether the contract is achievable at all or is aspiration with a number
attached.

A third, short category sits between the two: two demands that are not on
decider2 and not on a flow, but on **project 00**, the shared capability
library the eight flows are built from. They are called out separately so they
are not mistaken for either.

---

## Part A — Demands on the framework

**1. `at_pin()` — a sealing transform with structurally no fallback path.**
*Spec §5.1 (all seven pinning categories), §13 Q1. UGLY.*
Doc 08 §6.2 gives exactly one pinning primitive —
`resolve_params(doc, complete=True)` — and stops. Tables are "provisional"
(doc 03 §4.4); `decision_date` is an ordinary input column; the overlay stack
is not mentioned anywhere in the framework docs; capability versions are
import-time facts. Replaying a decision under that surface means the caller
reconstructs seven different things from seven different places and hopes —
which is precisely spec §5.1's worst failure mode, "a replay that succeeds
with the wrong answer," as that design's *ordinary* behaviour rather than a
bug in it. `replay/pin_resolution.PinSet` and `replay/seal.at_pin()` invent a
single, total, self-describing resolution environment and a transform that
seals a pipeline into it: every table read, date resolution, capability lookup
and overlay resolution terminates in the pin, with **no fallback to prefer it
over** — a read with no pin raises `UnpinnedRead`. Doc 08's answer (pin params,
hope for the rest) does not scale to seven independently-forgettable
categories; this is the case where the framework's own mechanism is
insufficient by construction, not merely unfinished.

**2. Read-coverage instrumentation under a seal, both directions.**
*Spec §5.1, §5.15 items 4/5/7/9. EXT.*
`replay/seal.ReadCoverage` needs every table/date/capability/overlay resolver
rewritten to log against the pin and compare both ways after the run
(`unpinned_reads` hard-fails; `unread_pins` is reported because it means the
replay took a different path). Nothing in doc 08 gives resolver-level logging
today — `resolve_params` validates completeness once, at the boundary, and has
no notion of "was this actually read." This is additive to doc 08 §2's
resolution machinery, not in conflict with it.

**3. Schema pinning, not only value pinning (`params_schema_hash`, `SchemaDrift`).**
*Spec §13 Q1. EXT.* [D8 in the tree]
Doc 08 §2.1's `complete=True` proves a document names every field the model
declares — *against the model of the day*. Three years later a module gains a
field with a code default, and a 2027 params document that was complete in
2027 is silently incomplete against the 2030 schema; `complete=True` still
passes, and the replay's answer moves at the margin in a defensible-looking
direction. Nothing in doc 08 detects this because nothing in doc 08 versions
the *schema* independently of the *document*. `replay/seal.py` stores a hash
of each module's params schema in the `PinSet` and refuses to resolve on
mismatch. This is the one gap in this list most likely to be missed by anyone
who has not tried to replay something seven years old.

**4. A structural check that reaches frame-tier and `@breaks_lineage` clock
reads.** *Spec §5.15 item 4. AWK.* [D6 in the tree]
Contract item `_c04_no_today` works by checking that no step reaches
`date.today()`/`now()`, because such a step cannot `njit` and therefore shows
up in `fallback_set`. That check is real but incomplete: a genuine frame-tier
operation (`pl.date.today()`) or a `@breaks_lineage` region never reaches numba
at all, so it never appears in `fallback_set`, and the structural check is
blind to it. Belt-and-braces is provided by the sandbox denying `clock_gettime`
at replay time, but that only catches it *at replay*, not at build, which is
exactly the gap contract item 4 exists to close for everything else. This is
named as unresolved rather than patched over, because a static check that
silently doesn't cover frame-tier code is worse advertised as complete than
admitted as partial.

**5. `holds()` — a claim that is simultaneously a test, a rendering source, and
mutation-verified.** *Spec §5.7, §13 Q9/Q10; doc 04 §6's own top risk. UGLY.*
Doc 04 §6 proposes generating the reviewable artefact from module data plus the
`@step` docstring. `contract/attest.py` argues this doesn't survive contact
with the evidence doc 04 §6 itself cites for why a *hand-maintained* policy
document rots: a docstring nobody runs is exactly that document, just moved
inside the generated artefact where the rot is harder to see. `holds()` is
prose CI executes — attached to a step, printed verbatim by the reviewable
artefact and the certification artefact both, and stress-tested by
`certify.mutation_screen()`, which perturbs the rule's params to their bounds
and flips its comparisons, flagging any claim that still passes as
"non-discriminating." Doc 04 §7 already has ordinary pytest-style assertions
("rule-level assertions expressing intent, which double as something a
reviewer can read") — `holds()` is that idea made structurally load-bearing
rather than a testing convention, and mutation-checked in a way doc 04 §7 does
not attempt. Marked UGLY rather than EXT because doc 04 §6's proposed
mechanism (docstring-as-description) is actively displaced, not extended.

**6. `attest()` — a witness set derived from the graph, with declarative detail
tiers.** *Spec §5.15 items 14-16, §8.3. EXT.* [cost stated as D3]
Combinators that only change codegen already exist (doc 02 §1.2, `fuse()`);
what's new is a combinator whose payload is *derived from the graph rather
than authored* — every value overwritten at more than one boundary, every
`Branch`, every table read, automatically, so "a team forgets one module" is
structurally impossible rather than merely unlikely. Doc 03 §7's `taps=[...]`
is an authored, opt-in list — the right tool for "an engineer wants this one
value in a dashboard," the wrong one for "every flow must record all of item
14-16, forever." `WitnessPolicy.full_detail_when` — a predicate selecting
which decisions get full intermediate detail (declines, referrals, the 0.5%
sample) versus the always-on 3.1 KB tier — is new declarative surface with no
doc 03/08 analogue. The cost is not free: an attested kernel is wider, and the
witness columns are the dominant write-back cost at high output counts (doc 02
§1 already measures 54.7% write-back share at 633 outputs); `attest()` makes
that worse before spec §8.3's storage tiering claws it back, and the docs have
no answer to the write-back problem beyond "optimise it later."

**7. A counting tap, distinct from a record-tier column tap.**
*Spec §5.6, §5.8. EXT.*
Doc 04 §4.1's taps materialise one extra output column per record. Table cell
coverage at 63,360-219,600 cells across 22M decisions/year cannot be a
per-record column — "gigabytes of almost never-read evidence." What
`coverage/measure.py` needs is a shared histogram over `(table_version,
flat_index)`, incremented once inside the kernel and merged per partition, with
**no per-record row at all**. This is a different diagnostic primitive from a
tap — cheaper, coarser, aggregated at the kernel rather than materialised at
the boundary — and doc 04 has nothing between "a tap" and "a frame-tier
group-by after the fact," which is what per-record cell tracking would force.

**8. A declared output range on `@step`.** *Spec §5.8. EXT.* [D14 in the tree]
`coverage/shadowed.py`'s static feasible-region analysis is decidable for
comparisons against declared, typed leaf features (doc 08 §3's closed
13-operator vocabulary) but stops at a registered derived feature whose body is
an opaque step — its range is not declared anywhere. `@step(...) -> float @
range(0, 1)` would extend the decidable fragment to most of them. Small ask,
concrete payoff: shadow analysis currently reports `unknown` for exactly the
class of conditions a policy analyst is most likely to write next.

**9. A declared algebraic property (monotonicity) on a step or a registered
capability.** *Spec §5.14.6, §13 Q22. EXT.*
`adjustments/asymmetry.verify_structural()` proves the tighten-only property by
reading the *shape* of a composition expression at each overlay consumption
point — recognising that `min(base, base*(1-m))` is monotonically downward in
`m` on `[0,1]` while `base*(1-m)` alone is not. That only scales because the
composition is registered once as a shared capability (`core.adjustments`) and
inherited by every consumer; a flow that composes overlays itself forces the
same reading per flow. A declared monotonicity contract on a step or capability
(direction, argument, domain) would make this a build-time lookup instead of a
bespoke reader per registered composition, and would generalise past overlays
to any place spec asks "can this only move one way" (§5.14.6's per-flow
asymmetry register is the concrete instance, but not the only one this estate
will eventually ask).

**10. Forward reachability from a set of changed modules — the transpose of
`lineage()`.** *Spec §5.4, §5.6 ("a change to one phase not requiring
re-certification of everything"). EXT.*
Doc 04 §3's `pipeline.lineage(output)` runs **backward**: given an output,
which inputs and steps can affect it. `manifest.cone(changed)` (used by
`release/certify.py`'s scoped re-certification and by `diff/semantic.lineage_delta()`)
needs the **transpose** — given a set of changed modules, which outputs are
reachable from them. It is the same static graph and almost certainly the same
cost to compute, but it is not the primitive doc 03/04 name, and nothing in
either document states that both directions are available. Worth confirming
explicitly rather than assuming the backward-direction API happens to support
it.

**11. Semantic diff for interiors, skeletons and tables — doc 08 §6.2 plus doc
04 §5.2 oversells this.** *Spec §5.4, §13 Q11. UGLY.*
"A config diff is an audit record ... with no extra machinery" (doc 04 §5.2) is
true for params and actively misleading for everything else, which
`diff/semantic.py` states is *most of what changes in a release*: a rule
inserted mid-priority, a tree re-parented, a stage reordered, a rate card
refreshed. Each of the four needed its own comparison — an insertion's
downstream effect-shift, a node identity map, a lineage delta, a cell
aggregation — none of which doc 08's generic `diff(old, new) -> list[Change]`
provides today. This is the clearest case in the whole tree of the docs stating
a capability as solved when it is solved for one document kind out of four.

**12. `Increment` / `Release` — an authored, individually-attributable
decomposition of a change.** *Spec §5.5, §5.15 items 12-13. EXT, and named in
the tree as the largest deviation in the project.*
Doc 08 §5's `impact(active, candidate, sample)` compares exactly two compiled
generations with one number. Spec §5.5 needs *n+1* cumulative runs in a
*declared order*, because a release cannot be decomposed after the fact from a
`git diff` — that gives a pile of changes, not an ordered sequence of
individually buildable generations. `Increment`/`Release`/`Declaration` are new
authoring surface with no analogue in doc 08; they compose cleanly on top of
`stage`/`activate` and `rt.params.swap()` rather than fighting them, which is
why this is EXT rather than UGLY — doc 08's mechanism is necessary but was
never going to be sufficient on its own, and nothing in it needed to change.

**13. `impact()`/swap-set over two *artefact sets*, not only two compiled
generations.** *Spec §5.5, §5.14.4. EXT.*
The change a policy team makes most often is a table or param swap, which
produces no new compiled generation at all — `release/swapset.py` notes that
four of five increments in the worked release share one compiled image and
differ only in the params bundle. `impact(pipeline, sample, tables_a,
tables_b)` — the same report, no compile — is closer to what's actually run
monthly than doc 08 §5's two-generation form. (Project 07's sketch makes the
identical observation independently — see its `FRAMEWORK-DEMANDS.md` #29 — which
is a second, unrelated data point that this is a real gap rather than an
artefact of this project's shape.)

**14. Validator bounds should carry an owner class, not only a range.**
*Spec §6, §9.2. EXT.* [D17 in the tree]
`harness/tolerances.py` uses `Field(7, ge=7, le=7)` — bounds equal to the value
— to encode "this is a statutory obligation and nobody may change it by
config." The pattern generalises (`ge=6, le=60` states who may set a term cap
and how far), but a `Field`'s bounds are the only permissions statement
available; the reviewable artefact wants to print "who may change this" beside
every value (spec §5.7: "who may set it"), and today it can only print the
range and let a human infer the rest.

**15. `@declassify` — a second conspicuous escape hatch, on the
`@breaks_lineage` pattern.** *Spec §5.13.6. EXT, and cheap.*
`explain/disclosure.py` needed a declared, greppable downgrade for exactly one
reason doc 04 §3 already solved for lineage gaps: `grep -r "@breaks_lineage"`
lists every lineage gap in a codebase, and that is "a governance feature, not
just a warning." `@declassify(..., because=..., approval=...)` is the same
idiom applied to disclosure level instead of lineage. This is the smallest item
in this list because the pattern doc 04 §3 established already answers the
design question; only the second decorator needs writing.

### Two things the framework should not be asked to fix

**X2 — what-if non-confusability stops at the keyboard.** *Spec §5.3.* Every
mechanism in `whatif/counterfactual.py` (namespaced id, `wi_`-prefixed keys,
per-row `__WHATIF__` marking) defends against export and re-issue. None of them
defends against a person reading a number off a screen and typing it into an
email. That is not a framework gap; no authoring surface closes it, and the
tree says so rather than implying a fifth mechanism would.

**X3 — swap-set interaction decomposition is a compute budget problem, not a
framework one.** *Spec §5.5.* `release/swapset.complementary()` runs the
reverse order to bound an interaction; a full Shapley decomposition over five
increments is 120 runs and ~60 hours nobody will pay for monthly. Two orders
detect and bound an interaction; they do not decompose it, and no framework
mechanism changes that arithmetic — it is a cost the business either accepts or
buys down by shipping one change at a time (spec §5.5 says exactly this).

---

## Part B — The eight-flow contract (spec §5.15)

`contract/items.py`'s own framing is the header worth repeating: 15 items are
**STRUCTURAL** (provable from the manifest, no data, no execution), 4 are
**EMPIRICAL** (need a corpus run), 4 are **DECLARED** (irreducibly a human
sentence, and the framework's job is only to make omitting it impossible).
Retrofittable is `False` for sixteen of the twenty-three — items 9, 12, 13, 17,
18, 20 and 22 are the seven marked `True` in the source, meaning the other
sixteen are the ones spec's own warning is about: *"none of this can be
retrofitted."*

| # | Item | Proof | Doc 03/08 mechanism | Verdict |
|---|---|---|---|---|
| 1 | Stable decision identifier | STRUCTURAL | Ordinary step output; a hash of a stable key. No framework primitive needed. | SAT (trivial) |
| 2 | Stable identity for every logic element | STRUCTURAL | Module/step identity is name-based (doc 03 §3); `Branch`/`Loop` identity likewise. Tree-node / matrix-cell / scorecard-characteristic identity for a *data-shaped module* rests on doc 08 §3's interior document schema, which is **still open (O15)** — "the real schema is the flat_rules algebra plus an id ... needs writing before `ruleset` is built." | **needs extension** — this item is only as solid as an open question doc 08 has not yet closed |
| 3 | Deterministic assignment | STRUCTURAL | `Assignment` (derivation + derived value) is ordinary step logic — `sha256(key\|seed) mod N`. No RNG anywhere in a step is a lint, not a framework feature. | SAT |
| 4 | No reliance on "today" | STRUCTURAL | `_c04_no_today` via `fallback_set` (doc 02 §3.4/doc 08 §4.1). Incomplete for frame-tier/`@breaks_lineage` reads — see Part A #4. | **needs extension**, with an admitted gap |
| 5 | Recorded reference-data versions, to the cell | STRUCTURAL | Rests on doc 03 §4.4's `Table`, explicitly flagged there as "provisional, lowest-confidence part of this document." `TableRef` and `cells_read` in `manifest/model.py` assume cell identity and version resolution the framework has not yet designed. | **the docs would be ugly here** — this project builds directly on the least-finished part of doc 03 and cannot wait for it |
| 6 | Inputs captured as received | STRUCTURAL | Doc 03 §1's three null tiers (`required`/`missing_as`/`Optional`), preserved as a type rather than a convention. | SAT |
| 7 | Overlay stack recorded on every decision | STRUCTURAL | No framework concept of an overlay exists at all (doc 03/08 silent). `_c07_overlay_stack` checks that a module consuming `core.adjustments` declares its unadjusted twin in `writes` — a convention `adjustments/register.py` and `unwind.py` invent wholesale. | **needs extension** — entirely new vocabulary, additive |
| 8 | Mutable state snapshotted or addressable | STRUCTURAL | Not a framework question. Flows 07 and 08's book/state stores don't retain history; that is a data-architecture defect in those two flows, and the spec says so directly ("a replay defect in those flows, not in this one"). | **out of scope for decider2 entirely** |
| 9 | No external calls during replay | STRUCTURAL, retrofittable | `@breaks_lineage`/frame-op declarations plus sandbox network denial at replay. No declared "touches network" category exists on a frame op — a minor, cheap addition. | mostly SAT, small gap |
| 10 | Explicit, complete parameter sets | STRUCTURAL | `resolve_params(doc, complete=True)` (doc 08 §2.1) directly. | SAT |
| 11 | Declared reason codes | STRUCTURAL | Ordinary typed output (`tuple[int, ...]`); the taxonomy itself is project 00's vocabulary, not decider2's. | SAT |
| 12 | Declared expected effect per change | DECLARED, retrofittable | `Increment.declares: Declaration` is a required field — see Part A #12. | **needs extension** |
| 13 | Individually attributable changes | DECLARED, retrofittable | Same mechanism; `Release.check_orderable()` makes the one structural ordering constraint checkable rather than reviewed. | **needs extension** |
| 14 | Evaluation recorded, not only firing | STRUCTURAL | `branch_path` as a compile-time immediate (doc 04 §4.1) — free at any batch size. | SAT |
| 15 | Cap chains recorded | STRUCTURAL | Version chains at module boundaries (doc 04 §5.1, `term_cap@*` doc 03 §7). The chain's `cause` string is the one thing the harness adds, from `holds`. | SAT |
| 16 | Score contributions as output | STRUCTURAL | Ordinary declared output; no new mechanism, just a discipline that it must be an output and not a diagnostic. | SAT |
| 17 | Idempotent, at-least-once evidence emission | EMPIRICAL, retrofittable | Store-level concern (dedup by decision id, retry semantics) — outside decider2's compute model entirely, same boundary doc 08 §6 draws for config sourcing ("not the framework's business"). | **out of scope for decider2** |
| 18 | Evidence emission cannot fail the decision | EMPIRICAL, retrofittable | Deployment/infra property (buffering, reconciliation on a degraded store). Not a decider2 concern. | **out of scope for decider2** |
| 19 | Declared input inventory | STRUCTURAL | `pipeline.schema()`'s unbound-input set (doc 03 §2.2) *is* the inventory — "a feature not in the inventory is not possible." | SAT, and elegantly so |
| 20 | Runnable outside production | STRUCTURAL, retrofittable | Pure functions, explicit arguments, no ambient state (doc 03 throughout) — true by construction of the authoring model, not by extra effort. | SAT |
| 21 | Versioned, resolvable logic identity | STRUCTURAL | `structure_fingerprint` + `compiled_artefact_id` (doc 08 §8) directly. | SAT |
| 22 | Declared prohibited-ground usage | DECLARED, retrofittable | Same `pipeline.schema()` leaf set as item 19, plus `permitted_use_ref` on `LeafInput`. | SAT |
| 23 | Field-level PII classification at emission | STRUCTURAL | Classification-by-inheritance over static lineage (doc 04 §3's cone, applied to disclosure rather than to "what can affect this output") — see Part A #15 for the one piece that's new (`@declassify`). | SAT, with one small extension |

Read down that right column and the shape of the whole project falls out: nine
items are flatly satisfied by mechanisms doc 03/04/08 already state (1, 3, 6,
10, 11, 14, 15, 16, 19, 21, 22 — eleven, in fact); two are outside decider2's
remit no matter how the framework evolves (8, 17, 18 — infrastructure and data
architecture, three items); and the remainder — 2, 4, 5, 7, 9, 12, 13, 23 —
are exactly the ones this project's own `manifest/`, `contract/`, `release/`,
and `adjustments/` directories exist to shore up. That is not a coincidence:
the tree is organised around the gaps, not around the spec's numbering.

---

## Demands on project 00, not on decider2 or on a flow

**X6 — the reduction-vs-decline reason taxonomy has no separate field.**
*Spec §5.2, §11.8.* The decision record for `FLX-2027-03-0418822` carries
`decline_reason_codes: [214, 318, 402]` for an application that was
**approved**, at a lower amount than requested — because "00 §4 has no
separate field" for a statutory reduction reason distinct from a decline
reason, and the estate's shared vocabulary conflates the two. This is not a
decider2 problem — the framework does not constrain what a project names its
outputs — and it is not a flow-03 problem either, since every flow inherits the
same shared taxonomy from project 00. It belongs on project 00's backlog, and
is noted here only so it is not lost by being filed under the wrong project.
