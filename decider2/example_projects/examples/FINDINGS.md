# Findings — what eleven independent sketches say about `decider2`

Eleven agents were each given one project spec, the framework docs, and one
instruction: **design the authoring surface you wish existed.** Doc 03 was
declared a proposal rather than a constraint, and every departure from it had to
be recorded and justified.

They worked independently and did not read each other's output. That is the
whole design of the exercise: **what recurs across eleven independent attempts is
evidence, and what appears once is an opinion.**

Output: 406 files, ~53 500 lines, 286 Python sketch files (all parse), a
`README.md` and a `FRAMEWORK-DEMANDS.md` per project.

---

## 1. How to read the numbers

Two biases must be stated before any figure below is used.

**The briefs asked for gaps.** Every agent was told to be harsh and that
departures were the most valuable thing it could produce. A brief that asks for
problems gets problems. These counts measure *what a motivated reader can find*,
not what a neutral one would.

**The counting is approximate.** The eleven agents used four different verdict
vocabularies (`SAT/EXT/UGLY/AWK`, `[SATISFIED]/[EXTENSION]/[UGLY]`, "satisfied by
doc 03 / needs extension / would make this ugly", "satisfied / needs extension /
departure"). Normalising them is regex over prose, so per-project figures move by
a few points depending on the patterns chosen. Treat every number as ±10%.

What survives both caveats is the **shape** of the distribution, and the
recurrences in §2 — which are not sensitive to counting method at all.

---

## 2. The headline: five closed sets that close one element too early

The strongest result, and the one least explainable as briefing bias, is that
independent sketches kept reaching the same conclusion in the same form: *the
docs fix a set at N; the work needs N+1.*

None of these were briefed. Agents were given goals, not set sizes.

### 2.1 Change classes: doc 08 §2 has three, the work needs four

**Found independently by 01, 02, 03, 06, 08 — five projects.**

Doc 08 §2 gives values / interiors / skeleton. Every one of these five argues
adjustments are none of the three: free at runtime like a value, ordered and
collision-checked like an interior, separately approved and separately expiring
like neither.

> *"a fourth doc 08 §2 change class — adjustments: free like a value, ordered and
> collision-checked like an interior, separately approved and expiring like
> neither."* — 06, D3

01 goes further and calls it a fourth *mechanism*, not merely a fourth document:
per-record-varying, and therefore outside doc 02 §4's params/structure binary
altogether. 02 adds a second axis crossed with the three — **ownership**.

This is the single most corroborated finding in the set.

### 2.2 The equivalence ladder: doc 02 §3.1 has three rungs, the work needs four

**Found independently by 01, 02, 03, 04 — four projects, four different fourth rungs.**

Doc 02 §3.1 asserts `interpreted ≡ stepped ≡ fused`. Each project found a fourth
thing it needed proven equal and the ladder did not cover:

| project | the missing rung |
|---|---|
| 01 | the two entry points agreeing with each other |
| 02 | `held-and-resumed` — an assessment cut, held and continued |
| 03 | `score()` and `apply()` agreeing |
| 04 | a portable interpreter |

Different rungs, one structural complaint: the ladder as specified does not cover
what real projects must prove equal. 04 puts it as *"a stated fourth rung, not an
unstated fourth thing that happens to also need testing."*

### 2.3 Core component kinds: three is not enough — this is Q5, answered

**Found independently by 04, 06, 08.**

Each names a different missing kind, and each argues the three existing kinds
compose badly into it:

- **06 — `Search` as a fourth combinator**, not an encoding of `Loop`.
- **08 — sequence position** as a fourth kind. It states this is *"the central
  question the project exists to answer"*, and that composing it from the three
  means a decision table for interval rules, a tree for the escalation decision,
  and something else again for the state.
- **04 — `allocate()`** as a fourth frame-tier kind for constrained assignment.

Scorecards, decision tables and decision trees survive contact with the work.
They are just not sufficient.

### 2.4 Null situations: doc 00 §7.4 has three, the work needs four

**Found independently by 01, 06, 11.**

Not collected / collected as zero / could not be established — plus a fourth. 11's
is the sharpest: bound to **role** rather than to the value, for *not applicable
to this role*. It is also the most honest entry in the whole set, because 11
records it as fork pressure #7, marked `AT RISK`:

> *"adding a fourth null situation to `core` touches six consumers; a local
> convention touches none."*

That is a sketch predicting exactly where reuse will break and why the local
workaround will win. It deserves an answer before implementation, not after.

### 2.5 Scopes: doc 02 §4 has five, the reuse project needs six

**Found by 11** — the only project built by consuming other projects' components.

Doc 02 §4 fixes five scopes: step, module, branch arm, loop body, pipeline. 11
proposes `role()` as a sixth, carrying vocabulary, parameters, null policy and
disclosure together — because when a person-level component is consumed for a
director, a surety and a sole proprietor, all four of those vary by role rather
than by module instance, and there is nowhere in the current model to hang them.

One project, so weaker evidence than §2.1 — but it is the *only* project that
tested reuse at scale, which is precisely the condition under which it appears.

---

## 3. The verdict distribution

Across 384 classified demands (±10%, per §1):

| verdict | count | share |
|---|---|---|
| satisfied by the docs as written | 55 | 14% |
| needs extension | 203 | 53% |
| the docs would make this ugly — did something else | 92 | 24% |
| genuinely unresolved | 34 | 9% |

Per project:

| proj | sat | ext | ugly | unres | % sat |
|---|---|---|---|---|---|
| 01 fraud | 9 | 24 | 7 | 1 | 22% |
| 02 affordability | 1 | 8 | 21 | 11 | 2% |
| 03 granting | 2 | 19 | 14 | 2 | 5% |
| 04 campaigns | 4 | 27 | 5 | 2 | 11% |
| 05 business nested | 2 | 16 | 10 | 1 | 7% |
| 06 consolidation | 6 | 12 | 7 | 2 | 22% |
| 07 limits | 4 | 23 | 5 | 6 | 11% |
| 08 collections | 4 | 24 | 4 | 2 | 12% |
| **09 governance** | **16** | 19 | 6 | 4 | **36%** |
| 10 retail e2e | 4 | 17 | 8 | 1 | 13% |
| 11 business e2e | 3 | 14 | 5 | 2 | 12% |

**Two features of the distribution matter more than the total.**

**09 is the outlier at 36% — roughly triple the median.** It is the governance
project, and the document it leans on is doc 04. The observability and governance
thinking is holding up substantially better against real work than the authoring
API is. That is worth knowing, because doc 04 §6 nominates itself as the design's
weakest part. On this evidence it is not.

**03 is the floor at 5%.** It is the densest concentration of ordinary credit
logic in the set — scoring, policy caps, pricing, a solve. The framework's home
ground is where its authoring proposals fit worst. 02 is comparable and has the
highest departure count of any project (21).

The **203 "needs extension"** entries are the actionable pile: the construct is
right, the specification is thin. Cheap now, expensive after implementation.

---

## 4. Which documents the work argues with

References in `FRAMEWORK-DEMANDS.md` files:

| doc | references | sketches citing |
|---|---|---|
| 03 authoring API | 268 | 11/11 |
| 08 configuration and lifecycle | 101 | 11/11 |
| 04 observability and governance | 60 | 11/11 |
| 02 architecture | 56 | 11/11 |
| 07 project structure | 30 | 6/11 |

Doc 03 absorbs more than the other four combined. That is partly because it was
the declared target — but doc 08 at 101 references from 11/11 sketches was not
briefed as a target at all, and it is where §2.1's four-change-class finding
lands.

Doc 07 is the interesting low number: cited by only six sketches, and the two
that engage with it hardest (10 and 11, the large ones) both say it does not
survive scale. 10's critique is specific — no unit between "module" and
"everything", config mirroring pipelines collapses four change cadences into one
file, and "tests mirror modules" covers about half the testing surface.

---

## 5. What each sketch contributed

| proj | its central invention |
|---|---|
| 01 | a per-event **gain vector** built from a governed adjustment register; a second table kind for membership-as-at-an-instant, stored as intervals not snapshots |
| 02 | `dated_table`, `behaviour_table`, `profile()`, `Domain`/`narrow()`, `contest()`, `cut()/hold()/resume()` — ten declared departures |
| 03 | the bounded non-monotone solve as `partition` + `search`; the cap register as data; overlay application points as code, overlay values as config |
| 04 | the `decision_tree` kind with `publishes=`, dual fingerprints, mandatory route capture, `SlotId` compound identity, `allocate()` |
| 05 | **`grain`** + `Gather` — the answer to doc 06's unresolved O5 — plus `attribution_spine`, `emit_into`, `Enumerate` |
| 06 | **`Search` as a fourth combinator**; `ordering` and `objective` as module kinds; `Fanout`/`Cross`/`Explode` |
| 07 | `Sweep` for population budgets; simulation as three lines sharing production's implementation; `panel(reduce="min")` |
| 08 | `Timeline` (no wall clock), **`Sequence`**, `Grid`, `Allocate`, `Panel`, `OverlaySurface` |
| 09 | `at_pin`/`PinSet` sealing, `holds()`, `attest()`/`WitnessPolicy`, `Increment`/`Release` |
| 10 | entry-point derivation, ownership-as-data, a three-axis version mechanism, the latency budget as a declared artefact, navigability tooling |
| 11 | `Consumed` vs `Module`, **`role()` as a sixth scope**, a `time/` package (`Compared[T]`, bi-temporal namespaces, dual master-scale resolution), `append_only_artefact` |

**`grain` appears in two sketches independently** (05 heavily, 08 lightly) as the
name for the nesting concept. That is the beginning of an answer to O5, which doc
06 currently records as deprioritised.

---

## 6. Recommended next moves

In the order the evidence supports:

1. **Settle the change-class model first.** Five sketches say it is four, not
   three, and one says the missing axis is ownership. Everything else — config
   documents, approval routing, the audit record, the compile key — hangs off
   this. It is also the cheapest thing on this list to decide.
2. **Design `grain` properly, and stop deferring O5.** Two projects converged on
   the concept and a third needs it. It is currently marked deprioritised, and it
   is load-bearing for business credit, consolidation and collections.
3. **Take a position on the fourth core kind.** 06, 08 and 04 each name a
   different one. The question is not which is right but whether the kind system
   is open or closed — and if closed at three, what those three projects do.
4. **Extend the equivalence ladder to name its rungs explicitly**, rather than
   asserting three. Four projects each found a different fourth.
5. **Re-examine doc 07 against projects 10 and 11.** It is the least-cited
   document and the two that stress it both say it breaks. It was written against
   small projects and it shows.
6. **Do not rewrite doc 04.** It is the best-performing document in the set by a
   factor of three, despite nominating itself as the weakest.

---

## 7. One methodological caution for whoever uses this

The sketches are **not designs**. Nothing was built, nothing runs, and no agent
had to live with its own decisions. A construct that reads well in a sketch may
be unimplementable, unfusable, or impossible to compile — none of which this
exercise tested.

Their value is narrower and real: they are eleven independent readings of what
the work demands, produced before the framework was built rather than after. The
recurrences in §2 are worth acting on. The individual inventions in §5 are worth
arguing with.

Two constructs in 05 did not parse as Python at all, which is recorded in that
sketch's README §9. The lesson generalises: **an authoring surface has to be
checked against the host language while it is being designed.** Declaration-shaped
constructs — a collection of heterogeneous rows wanting both positional clarity
and named modifiers — have exactly one legal arrangement in Python, and it is not
always the readable one.
