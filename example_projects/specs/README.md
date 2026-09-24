# Example projects

Twelve project specifications written to be **built on `decider2`**, and chosen so
that building them honestly forces the framework's unanswered questions into the
open.

These are requirement documents, not designs. They describe what a business needs
to happen, in what order, with what evidence — deliberately stopping short of how
any of it should be expressed. A spec that already told you the answer would not
test anything.

---

## 1. Why these exist

`decider` was designed against toy examples and then met real work. The design
docs (01 §5) record what that cost: 79 identity-passthrough steps written to work
around a missing relabel, 546 inline literals where a parameter should have been,
66 modules behind one test file, a decline-reason taxonomy dropped entirely in a
port, and a descending `while` loop replaced by a filter/group-by/join that lost
early exit.

None of those were visible in a three-step example. All of them were visible in
the first real project, by which point the design was fixed.

So `decider2` gets its real projects **first**. Each spec below is sized and
shaped like actual work: hundreds of rules, grids with thousands of cells,
variable-length nested collections, iterative solves, regulated evidence
requirements, and people who are not engineers changing things on a Tuesday.

---

## 2. Ground rules for these documents

**Everything here is fictional.** The institution, its products, the thresholds,
the rule identifiers, the table dimensions and the volumetrics are invented for
this repository. Where a spec cites a real regulatory mechanism — the National
Credit Act's affordability regulations, minimum expense norms, fee caps, credit
life premium limits — it cites the public, published mechanism, and every number
attached to it is illustrative. No proprietary logic, threshold, field name or
process from any real lender appears in these documents, and nothing here should
be treated as a description of one.

The fictional lender is referred to as **the Bank**. Its products are named in
§4 and used consistently across specs so that cross-project reuse is real reuse
and not a coincidence of wording.

**Specs describe requirements, not structure.** No spec names a `decider2`
construct. None says "this is a scorecard module" or "this is a branch". They say
what must be computed, what must be recorded, what must be tunable by whom, and
what must still be answerable in two years. Whether that becomes one component or
nine is the implementer's problem, and watching where implementers land is the
point of the exercise.

**Each spec ends with a set of questions, not answers.** The final section of
every document lists what the implementation will be forced to decide. That
section is the measurement instrument.

---

## 3. The seven questions these projects exist to answer

| # | Question |
|---|---|
| Q1 | What reusable components fall out, and where are the reuse seams? |
| Q2 | How do large lookup tables — rate grids, norm tables, treatment matrices — get integrated, validated, versioned and edited? |
| Q3 | How are parameters structured so they are easy to edit, easy to externalise, loadable from outside systems, and reusable across flows? |
| Q4 | How are custom, project-specific modules developed once and reused across flows without copy-paste? |
| Q5 | What should the common core components be? `decider2` starts with scorecards, decision tables and decision trees. Is that enough? |
| Q5b | Is an **adjustment** — a post-model overlay on a scorecard, calibration, grid or cut-off — a core component kind of its own? See §5.1. |
| Q6 | How is a large codebase organised so it stays maintainable — many products, many policies, many owners? |
| Q7 | How are auditing and debugging handled when a decision is wrong, disputed, or must be reproduced years later? |

Each spec names the subset it stresses hardest. The coverage matrix is §5.

---

## 4. The slate

Read in order. Later projects depend on capabilities published by earlier ones,
which is itself part of the test.

| # | Project | One line | Dominant difficulty |
|---|---|---|---|
| [00](00-shared-credit-core-library.md) | **Shared credit core library** | The capabilities every other project consumes, and the discipline that keeps them stable | Reuse, contracts, versioning, vocabulary |
| [01](01-transaction-fraud-interdiction.md) | **Transaction fraud and scam interdiction** | Hundreds of flat rules over real-time payment and channel events | Rule volume, priority resolution, which rules fired, minutes-to-live config |
| [02](02-affordability-assessment.md) | **Affordability and obligations assessment** | The regulated affordability calculation, shared by four other flows | Regulatory versioning, one component serving different consumers |
| [03](03-unsecured-loan-granting-and-pricing.md) | **Unsecured loan granting and pricing** | Score, price, and solve for the largest affordable offer | Circular amount↔rate↔instalment↔affordability solve, very large rate grids |
| [04](04-campaign-targeting-trees.md) | **Retail campaign targeting trees** | Sixty deep decision trees over the full client base, with the path taken recorded | Path capture, stable node identity, batch scale, analyst-authored structure |
| [05](05-business-credit-nested-entities.md) | **Business credit granting with nested entities** | A business made of entities, each made of adverse events, scored and rolled up | Ragged two-level nesting, roll-up semantics, per-level attribution |
| [06](06-consolidation-and-restructure.md) | **Consolidation and restructure** | Search over which existing debts to settle, across four products with separate rate tables | Bounded search, product subflows, competing objectives |
| [07](07-credit-limit-management.md) | **Credit limit management** | Monthly limit increases and decreases across a revolving book, inside a portfolio budget | Portfolio-level constraint over record-level decisions, simulation before deploy |
| [08](08-collections-treatment.md) | **Collections treatment assignment** | Daily treatment for a delinquent book, with arrangements and legal thresholds | Temporal state, treatment history, operational capacity, regulatory suspensions |
| [09](09-governance-and-replay-harness.md) | **Decision governance and replay harness** | Replay, diff, swap-set, certification and the regulator pack — operating on all of the above | Exact reproduction across versions, attribution of difference |
| [10](10-retail-credit-end-to-end.md) | **Retail credit, end to end** — *built standalone* | Every retail credit decision the Bank makes, as one deployed flow with eight entry points | Scale: ~1 400 decision points, 12 owning teams, one decision record |
| [11](11-business-credit-end-to-end.md) | **Business credit, end to end** — *built by reuse* | A business facility from origination through five years of reviews, covenants and restructure | Scale plus **time**, plus what reuse costs at scale |

### 4.1 Two kinds of project, deliberately

Projects 00–09 are **isolated**: each takes one part of the credit estate and
goes deep on it. They are how a specific difficulty gets examined without the
noise of everything around it — the nested roll-up in 05, the solve in 03, the
path capture in 04.

Projects 10 and 11 are **end to end**: one flow, one deployment, one decision
record, covering everything a retail or a business client needs. They exist
because that is what real credit systems look like. The model being replaced does
the whole job in one place, and a framework that composes beautifully at six
modules and collapses at six hundred has not been tested by anything in 00–09.

The two are built **by deliberately opposite strategies**, and the contrast is
the experiment:

| | **10 — retail** | **11 — business** |
|---|---|---|
| Construction | **Standalone.** Rebuilt from nothing, as its own project in the codebase | **By reuse.** Assembled from components that already exist, referenced rather than rebuilt |
| Numbers | Its own throughout — different thresholds, table shapes, rule counts and segment definitions from every other spec | The consumed components' own numbers, unchanged; new numbers only for what it genuinely introduces |
| Consumes | The `core.*` library only | The `core.*` library plus the components of projects 02, 05 and 06 |
| Asks | How do you structure one very large project from scratch? | How easy is it to reuse components that already exist, at scale? |

So the overlap between project 10 and projects 01–07 is **intentional
duplication**, not an oversight: a standalone build is the control. And project
11's most valuable section is the one no isolated spec can produce — what reuse
actually costs at scale: vocabulary mismatch where a person-level component is
consumed in a business context, components that give a consumer 90% of what it
needs, semantics that differ per consumer, version coupling across three release
cadences, and the specific points where forking would be the cheapest thing the
team could do.

Beyond that, the new material in both is the connective tissue — the part no
isolated spec can show:

- values computed once and consumed by eleven later stages, and what happens when
  one consumer needs a different version of the same concept;
- ordering constraints between phases, including the non-obvious ones;
- feedback loops that span whole projects — affordability fails, consolidation
  searches, affordability re-runs, pricing re-runs;
- one latency budget divided across eighteen phases and a dozen external calls;
- eight entry points sharing most of a flow but not all of it, with different
  outputs and latency budgets three orders of magnitude apart;
- degraded operation when one of a dozen data sources is down;
- twelve teams owning one artefact, and three of them changing it in the same
  week;
- whether a new engineer can find where a number is decided, and whether an
  analyst can answer "why 48" without one.

Project 11 adds an axis project 10 does not have: **time**. A business facility
is re-decided every year for five years, its covenants tested quarterly, its
group re-assessed whenever any related entity changes. The flow that decides it
in 2031 is not the flow that decided it in 2026, and the two must still be
comparable.

If the framework has a scale wall, these two are where it is — and if reuse has
a breaking point, the difference between them is where it shows.

---

## 5. Coverage

`●` primary stress, `○` secondary.

| | Q1 reuse | Q2 tables | Q3 params | Q4 custom modules | Q5 core kinds | Q6 codebase | Q7 audit |
|---|---|---|---|---|---|---|---|
| 00 library | ● | ○ | ● | ● | ● | ● | ○ |
| 01 fraud | ○ | ● | ● | ○ | ● | ○ | ● |
| 02 affordability | ● | ● | ● | ● | ○ | ○ | ● |
| 03 granting | ● | ● | ● | ● | ● | ○ | ○ |
| 04 campaigns | ○ | ○ | ● | ○ | ● | ○ | ● |
| 05 business | ● | ○ | ○ | ● | ● | ● | ● |
| 06 consolidation | ● | ● | ● | ● | ● | ● | ○ |
| 07 limits | ● | ● | ● | ○ | ○ | ○ | ○ |
| 08 collections | ○ | ● | ● | ○ | ● | ○ | ○ |
| 09 governance | ○ | ○ | ● | ○ | ○ | ● | ● |
| 10 retail e2e | ● | ○ | ● | ● | ○ | ● | ● |
| 11 business e2e | ● | ○ | ● | ● | ○ | ● | ● |

### Structural axes, by project

| axis | where it bites |
|---|---|
| Flat rule sets at volume | 01, 05, 08 |
| Deep trees with path capture | 04, 08 |
| Scorecards and calibration | 03, 05, 07 |
| Large keyed grids | 02, 03, 06, 07, 08 |
| Ragged per-record collections | 05, 06, 08 |
| Bounded iteration and solving | 03, 05, 06 |
| Branching into product subflows | 06, 03 |
| Set-shaped work around record-shaped work | 04, 07, 08 |
| Real-time single record | 01, 03 |
| Large batch | 04, 07, 08, 09 |
| Structure authored outside engineering | 01, 04 |
| Regulated evidence | 02, 03, 07, 09 |
| Post-model adjustments and overlays | 03, 05, 07, 08, 09 — and cross-cutting, see §5.1 |
| Composition at scale | 10, 11 — the only two that test it |
| Standalone construction of a large project | 10 |
| Reuse of existing components at scale | 11 |
| Re-deciding the same subject over years | 11 |

### 5.1 Adjustments — the cross-cutting one

One mechanism runs through almost every project and does not fit the axes above:
**post-model adjustments**, defined in
[00 §6.22](00-shared-credit-core-library.md#622-coreadjustments--post-model-overlays).

A scorecard is validated, signed off and then left alone for twelve to eighteen
months. Reality moves faster. When observed default rates drift from predicted,
when one channel needs tightening for a quarter, or when a macroeconomic view
says the book is riskier than the model believes, the response is not a model
rebuild — it is a named, approved, effective-dated **overlay** layered on top of
the artefact: a score shift, a change to the points-to-double-the-odds scaling,
a probability multiplier, a grade-boundary move, a cut-off shift, a rate add-on,
a cap reduction, a matrix intensity dial.

It is in this set because it sits awkwardly across the design's categories, and
that awkwardness is informative:

- it changes values without changing logic, which makes it look like a
  parameter;
- it composes in a declared order that changes the answer, which makes it look
  like logic;
- it is separately owned, separately approved, separately versioned and
  separately expiring, which makes it look like neither;
- and it must be **removable** — every flow has to be runnable with its overlay
  stack off, using the same implementation, because that is how the base model's
  own performance is monitored and how a validator separates model output from
  policy decision.

The failure mode it guards against is specific and common: a temporary
tightening applied during one bad quarter, still silently in force four years
later, with nobody able to say what unwinding it would do.

---

## 6. The house template

Every spec follows the same shape, so they can be compared and so gaps are
visible:

1. **What this is** — the business problem, in a paragraph.
2. **Why it is in this set** — which of the seven questions it exists to stress.
3. **Actors** — who runs it, who changes it, who has to answer for it.
4. **Inputs** — sources, shapes, cardinalities, nullability, freshness, volumes.
5. **The flow** — numbered stages. Each stage states its preconditions, what it
   determines, what it emits, and what must be recorded. This is the bulk.
6. **Parameters and tables** — every tunable and every table, with its size,
   its owner, its source, and how often it changes.
7. **Outputs** — what the caller gets, and what is persisted.
8. **Non-functional requirements** — latency, throughput, determinism,
   availability.
9. **Audit, evidence and explainability** — what must be answerable, by whom,
   how long after the fact.
10. **Acceptance criteria** — what "done" means, testably.
11. **Change scenarios** — real changes that will be requested after go-live.
    A good implementation makes these cheap; a bad one makes them a rewrite.
    These are the maintainability test.
12. **Out of scope** — so the implementation does not sprawl.
13. **Questions the implementation must answer** — the measurement instrument.

§11 deserves a note. Every spec carries a list of changes that will arrive within
months of go-live, because that is where framework design actually gets tested. A
structure that makes "add one rule" cost a release is a structure that will be
worked around, and doc 01 §5 is a catalogue of what those workarounds look like.

---

## 7. Shared vocabulary

Cross-project consistency is deliberate: the same concept keeps the same name in
every spec, so that a capability published by project 00 and consumed by projects
02, 03, 06, 07 and 08 is visibly the same capability. The canonical names, and
the published interfaces of the shared capabilities, are in
[00-shared-credit-core-library.md](00-shared-credit-core-library.md) §4 and §6.

Where a spec needs a name the library does not publish, it says so explicitly and
declares it locally. Those cases are worth counting: a high count means the
library's vocabulary is wrong.

---

## 8. How to use these

- **As implementation targets.** Build one. Then build a second that shares
  capabilities with the first, and see what the seam costs.
- **As design pressure.** Read a spec against a proposed interface and ask which
  stage of the flow the interface cannot express cleanly.
- **As a checklist for the core component set (Q5).** Keep a tally, per spec, of
  logic that fits none of scorecard / decision table / decision tree. The
  recurring shapes in that tally are the missing core components.
- **As the source for experiment workloads.** The experiments in doc 06 —
  particularly E1, E2 and E4 — need realistic shapes. These specs are where they
  come from.
