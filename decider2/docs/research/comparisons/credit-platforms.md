# Bank-specialist credit-decisioning platforms vs `decider2`

**Date:** 2026-09-21
**Scope:** section §3 of `decider2/docs/research/decision-engine-landscape.md`, re-verified
against vendor documentation where reachable. The landscape note's fraud engines (its §4), open-source
engines (its §5) and enterprise BRMS (its §6) are referenced only where one of them
demonstrates a pattern the §3 set does not.
**Status:** research note. Nothing here is a decision. Every vendor claim carries
the URL that owns it; every `decider2` claim carries `file:line`. Vendor
performance figures are vendor claims, not measurements.

---
## 0. The decider2 baseline this report measures against

Distinguish three things throughout, because the landscape note's §3 compares
*products* against a *document set* and that flatters decider2:

**(a) Built and working** (`decider2/src/decider2/`: 10 240 lines across 43 modules, 383 tests):
- Python steps harvested into modules, wired by name; `flow(...)` sequencing
  (`graph/pipeline.py:251`), `.bind()`/`.rename()`/`.relabel()`
  (`graph/module.py:201-239`), interface inferred then materialised
  (`graph/interface.py:46`, `:149`).
- **Decision tables** as a validated document, decider-1 vocabulary:
  `between`/`in`/`is_true`/`eq` leaves under `and`/`or`
  (`tables/schema.py:246-546`), N rows × outputs + `default`
  (`tables/schema.py:551-590`). **First match wins** and nothing else
  (`tables/schema.py:561`). Row edits recompile nothing — rows ride in the
  `shared` bundle (`tables/__init__.py:19-21`).
- **Decision trees** as a v3 document: positioned nodes + multi-source edges,
  unary/cases(ranges|string_match|isin)/composite nodes, leaves selecting a row
  of a leaf-value table by `result_idx` (`trees/schema.py:653-877`, `:1006-1043`,
  `:1055-1073`). 500-line emitted-source cap (`trees/codegen.py:86`).
- **Params**: `param()` harvested from the signature, pydantic-validated,
  namespaced per module instance, `extra="forbid"` (`params.py:250`, `:455`,
  `runtime/invoke.py:227`). Four null tiers, declared in the signature
  (`boundary/nulls.py:8-17`, `params.py:278-288`).
- **Three execution modes over one graph** — `interpreted`/`stepped`/`fused`
  (`runtime/modes.py:1-30`) — and the **equivalence ladder as a shipped
  assertion** with rung attribution (`testing/equivalence.py:218`,
  `:50-57`), plus a boundary-value corpus generator (`testing/corpus.py`) and
  `assert_no_recompile` (`testing/recompile.py`).
- **Two entry points, one kernel**: `apply(frame)` batch and `score(dict)`
  single-record with no polars (`runtime/invoke.py:432`, `:649`;
  `docs/02-architecture.md:485`).
- **Serving**: ASGI app, 8 routes — `/invocations`, `/params`,
  `/params/schema`, `/params/preview`, `/rollback`, `/health`, `/ping`
  (`serving/dispatch.py:137-145`); `ServeHandle` with
  `stage`/`activate`/`rollback`/`preview` and a `structure_fingerprint`
  (`runtime/serve.py:90`, `:253-352`).
- Measured: kernel ~1.38 µs at 400-in/633-out; `activate()` 0.177 µs, rollback
  3.36 µs with zero compiles (`docs/EXPERIMENTS.md:434-435`); a rule-shape
  change lands in 2.5–7.3 s with 97.9% serving throughput retained
  (`docs/EXPERIMENTS.md:40-42`); single-record framework overhead 971 µs p50 as
  naively implemented, 4.86% of a 20 ms budget, reducible to 0.73%
  (`docs/EXPERIMENTS.md:684-688`, `docs/README.md:29-36`).

**(b) Specified in docs, not built.** `Branch` and `Loop`
(`docs/03-authoring-api.md:1202`, `:1256` — both carry an explicit "NOT BUILT"
banner), `ruleset()` / `decision_table()` / `scorecard.py` interiors
(`docs/08-configuration-and-lifecycle.md:344`, `docs/02-architecture.md:704`,
`runtime/serve.py:12-20`), `pipeline.lineage()` / `.render()` / `.debug()` /
trace / `@breaks_lineage` / `golden.record()` / `decider2.impact()`
(`docs/04-observability-and-governance.md:126-205`,
`docs/08-configuration-and-lifecycle.md:613`) — **grep confirms none of these
symbols exists in `src/`**. `observe/` does not exist as a package.

**(c) Required by decider2's own example projects, with no mechanism anywhere.**
Champion/challenger parallel runs on a stable hash of a client id, traffic share
as a parameter (`example_projects/03-unsecured-loan-granting-and-pricing.md:365-371`);
per-characteristic scorecard point contributions as *required output* for
adverse action (`:330-341`, `example_projects/00-shared-credit-core-library.md:279-288`);
"evaluation recorded, not only firing" and cap chains
(`example_projects/09-governance-and-replay-harness.md:1024-1033`);
cell-level attribution and diffability for 17 reference tables up to 219 600
cells (`example_projects/00-shared-credit-core-library.md:537-565`);
ragged per-record collections — O5, the most-invented gap
(`docs/06-open-questions-and-experiments.md:215`).

Every "vs decider2" paragraph below is scored against (a), and names whether the
gap is (b) or (c).

---

## 1. Per-platform comparison

Every vendor claim below is quoted from a page the landscape note fetched on
2026-09-20 or re-fetched for this report on 2026-09-21; §6 lists what could not be
reached. **No vendor publishes an SLA, and no figure here was independently
measured.**

### 1.1 SAS Intelligent Decisioning (on SAS Viya) — the deepest comparison

**Sources.** Unlike the landscape note's pass, the *User's Guide* was reached this
time, through the PDF API endpoint rather than the HTML doc pages:
`https://documentation.sas.com/api/collections/edmcdc/v_040/docsets/edmug/content/edmug.pdf?locale=en`
— *SAS Intelligent Decisioning: User's Guide* 2023.11, 14 November 2023, internal
build tag `v_033-P1:edmug`. Everything below with a page number is from that
document; marketing pages are cited separately and flagged where they disagree with
it. The scorecard, adverse-characteristics and Model Manager consumption facts below
are from a separate sub-research pass over eight further SAS documentation books —
Intelligent Decisioning AG/Macro Reference/Data Grids/Custom Node Types/CLI/What's
New, SAS Model Manager User's Guide (2023.07 and 2026.07-09), and the VDMML Machine
Learning Node Reference/User's Guide v_035 and EM 15.4 Reference Help — recorded in
full in the scratchpad's `sub-sas-scorecards.md`, whose section letters (§A/§B/§C)
are cited below. That pass also probed the SAS Micro Analytic Service Admin Guide
(`masag` docset, all versions v_016–v_020 reachable) for published throughput or
latency numbers: none exist — the only numeric guidance in the book is a Kubernetes
troubleshooting note ("By default, Kubernetes runs a liveness probe... every 10
seconds") on pod resource exhaustion, not a performance figure
([About Resources for SAS Micro Analytic Service Pods](https://documentation.sas.com/doc/en/masag/v_020/n1wm5ztu39l1ran1vomimt7q7q2k.htm)).
A SAS Container Runtime (SCR) administration docset could not be located under any
guessed collection or docset id (`scr`, `scrag`, `contrag`, `containerruntime`,
`sasscr` all 404); SCR performance remains unverified (§6).

**The object model.** A *decision* is a diagram composed of other first-class,
separately versioned objects. The guide gives each its own chapter: **rule sets**
(ch. 2), **treatments and treatment groups** (ch. 3), **lookup tables and
functions** (ch. 4), **global variables** (ch. 5), **custom code files** (ch. 6 —
DS2, Python, data query, custom context), **custom functions** (ch. 7), **value
lists** (ch. 8), **segmentation trees** (ch. 9), **decisions** (ch. 10). That is
nine object kinds where decider2 has two.

**Rule sets.** Three types (p. 15): *filtering* (`IF condition` only, no action —
selects records for downstream processing, and doubles as a treatment's eligibility
rule set), *assignment* (`ASSIGN var value`, `IF … THEN`, `IF … THEN … ELSE`), and
*common* (shared rules included into assignment rule sets; "Any change to a common
rule set affects all assignment rule sets that include the common rule set").
**Default semantics are collect, not first-match:** "By default, the condition
expressions for all rules in a rule set are evaluated sequentially regardless of the
results of previous rules", controllable with `ELSE`, `OR` or a `RETURN` action
(p. 34).

**Expression language.** DS2-based, "up to 1024 characters", containing "numeric
constants, character strings, variables, operators, SAS DS2 functions, data grid
functions, and the SAS Intelligent Decisioning LOOKUP and LOOKUPVALUE functions"
(p. 35). Operators include `IN (value-list)`, `NOT IN`, `LIKE 'pattern'` with `_`
and `%` wildcards (case sensitive; the guide tells you to `UPCASE` first), `AND`,
`OR` (pp. 37–39). **Nulls:** a `MISSING()` function "returns 0 (false) or 1 (true)",
and the guide cross-references "How DS2 Processes Nulls and SAS Missing Values"
(p. 41) — i.e. the DS2 null / SAS missing-value distinction is exposed to the rule
author, not hidden.

**Variable types, and data grids.** Variable types are "Boolean, character, **data
grid**, date, datetime, decimal, and integer" (p. 25). A data grid is a nested table
*inside one record's variables*. Rules can iterate it — "rule sets that iterate over
a data grid (in other words, the rule sets score the rows in the data grid)" (p. 71)
— and a separate manual, *SAS Intelligent Decisioning: Using Data Grids*, documents
the data-grid functions. Character and data-grid variables carry explicit lengths,
with different rules for input-only, output-only and input-output (pp. 25, 46).

**Treatments and offer allocation — built on the data grid.** "A treatment is a set
of attributes, eligibility rules, and effective dates" (p. 88). Attributes are
name-value pairs, *fixed* (set at definition, uncustomisable) or *dynamic* (set per
treatment group, or "set by the decision at run time"), and are explicitly intended
to carry "profitability, risk, cost, priority, or order" so they can be arbitrated.
Adding a treatment group to a decision "creates an output decision variable **of
type data grid** that contains a column for" each attribute (p. 88). **Arbitration**
(p. 89) is then four listed methods: filtering rule sets; models that "score
treatments"; "**data grid functions to sort or subset the treatments according to the
values of their attributes**"; or DS2 code. Channels come from a predefined
`Treatment Channels` lookup table. Treatment groups are separately versioned,
checked out/committed, compared, and **activated** (pp. 93–112).

**Lookup tables.** "Tables of key-value pairs", imported from CSV "such as those
created by spreadsheet applications", re-importable "as needed to refresh the lookup
tables"; "Lookup keys must be unique within each lookup table"; read with the
`LOOKUP` and `LOOKUPVALUE` functions (p. 116). Keys and values carry **labels** —
e.g. `Country Abbreviation` / `Country Name` — which surface in the editor (p. 117).
Predefined tables ship with the product. Lookup tables have versions, checkout/commit,
version tags, **activation**, and PDF documentation generation (pp. 118–140). Note
the shape limit: **one key to one value**, not an N-dimensional grid.

**Segmentation trees.** "A hierarchical structure that consists of a root node,
branches, and final leaf nodes called **outcomes**" (p. 238). An outcome is "a named
set of actions" — assignment statements, DS2 code files, or both — and "You can reuse
the same outcome in multiple places in a tree". Three **branch element types** (p. 239):
a *Boolean expression* (two paths, True/False); a **matrix** — "a table in which each
cell identifies a different combination of the values of two different variables",
both variables bound to value lists, with an automatic `other` row and column, where
each cell is given a *result string* and one branch is created per unique result
string; and a *variable*, which "creates a branch for each value in the value list
that is associated with the variable", plus an `Other` branch. SAS's own rationale:
"Segmentation trees can be easier to design than using multiple rule sets… Matrices
are particularly useful when each variable has a large number of possible values."

**Scorecards: not authored by Intelligent Decisioning — confirmed at zero across the
whole product, not just the User's Guide.** "Scorecard" occurs zero times in the
User's Guide and zero times across the Administration Guide, Macro Reference, Data
Grids, Custom Node Types, CLI and What's New books (`sub-sas-scorecards.md` §A). The
primitive lives one product tier up, in SAS Model Studio, as the **Scorecard node**,
and it is licence-gated: "Risk Modeling Add-On for SAS Viya is not included with the
base version of SAS Viya. If your site has not licensed Risk Modeling Add-On for SAS
Viya, the Scorecard node is not available"
([Scorecard overview](https://documentation.sas.com/doc/en/vdmmlcdc/v_035/vdmmlref/p01pk3qxkpcuvkn1hjup0j4n5ngz.htm)),
and it must be preceded by an Interactive Grouping node that performs the WOE
binning: "An Interactive Grouping node must precede a Scorecard node." The scaling
is the textbook points-to-double-odds transform, stated exactly — "A linear
transformation is applied to the predicted ln(odds) to compute a score for each
attribute of each characteristic… score = ln(odds) * factor + offset" — with the two
properties an author actually sets being **Odds** ("the nonevent to event odds that
correspond to the score value… e.g. Odds=40, Scorecard points=220 means a score of
220 represents odds of 40 to 1. The default value is 50") and **Points to double
odds** ("the increase in score points that results in the score that corresponds to
twice the odds… e.g. 30… doubles the odds… The default value is 20")
([Scorecard Properties](https://documentation.sas.com/doc/en/vdmmlcdc/v_035/vdmmlref/p15lrmgnt7b0qrn1paqu1ddfw7mo.htm)),
plus Precision (0–4 decimal places, default 0 — integer points), a choice of
bucketing method, and an interactive "Manage Scorecard Points" override for manual
point adjustment. **This factor/offset law is a concrete, published scaling decider2
has no equivalent of and should adopt as-is** rather than inventing a points scale
for recommendation 4.4's `Scorecard` kind.

**Adverse characteristics: SAS's actual answer to reason codes, and it is a
training-time aggregate, not a per-record field — the single most load-bearing fact
here for decider2's own reason-code gap.** The feature is named "Adverse
Characteristic Options" on the Scorecard node's train properties, producing an
"Adverse Characteristics" results table (`sub-sas-scorecards.md` §B). Mechanism,
verbatim: "It is required by law to explain why an application is rejected. The
Adverse Characteristics properties enable you to prepare an explanation for rejected
applications. This is done by comparing the actual value of a variable to the
weighted average score or **neutral score**… in order to identify the
characteristics that are deemed adverse"
([Scorecard Properties](https://documentation.sas.com/doc/en/vdmmlcdc/v_035/vdmmlref/p15lrmgnt7b0qrn1paqu1ddfw7mo.htm)).
The neutral score is defined precisely as "the score points for an attribute when
the attribute's WOE is equal to 0" — i.e. **the per-characteristic adverse
contribution SAS computes is `actual_points(characteristic) − neutral_points
(characteristic)`**, ranked worst-first (SAS's own worked example orders "OWN_RENT,
INCOME, AGE, and then EDUCATION"), capped at a configurable count (1 to 5, default
3). **But the output is an aggregate report, not a per-applicant column**: "Adverse
Characteristics — Displays the number of times each input variable is cited for each
adverse reason. This table is available only when you select Generate report…" — a
validation-set diagnostic, not something the node emits per scored record. None of
the Scorecard node's seven documented output data sets (Train/Validate/Test/Rank/
Score Distribution/Fit Statistic Report/Estimate) is a per-applicant
adverse-characteristic table, and its two scoring artefacts are only "Node Score
Code" and "Path Score Code" (DATA step). **No macro and no PROC anywhere in the
checked SAS documentation emits reason codes**; a site must replicate the
neutral-score differencing itself, in score code or an Intelligent Decisioning rule
set. This is directly reusable for decider2: the *formula* — actual minus neutral
points, per characteristic, ranked — is exactly what a `neutral_points` field on
decider2's planned `Characteristic` (recommendation 4.4) needs to compute **per
record**, which is more than SAS's own product does out of the box.

**Consumption by Intelligent Decisioning: as an opaque registered model, never as a
scorecard object.** ID's own answer to "what can I include in a decision?" is
"models that are registered into the SAS Model Manager common model repository"
(`sub-sas-scorecards.md` §C) — "scorecard" does not appear on that page. Whether the
registered model actually runs in a decision depends on its **score code type**: a
plain Scorecard node produces DATA step score code (Import/Score/Publish/Monitor/
Compare/Retrain all "Yes"); a Viya analytic-store model is rewritten to "DS2
multi-type" at registration and is equally well supported; but "Scoring and
publishing of decisions that contain a model with a score code type of SAS program
is not supported by SAS Intelligent Decisioning"
([Model Manager User's Guide, score code types](https://documentation.sas.com/doc/en/mdlmgrcdc/v_071/mdlmgrug/p0492ot181mptzn15zsert8gl3vj.htm)).
There is also a genuine **PMML route**: `PROC PSCORE` supports the PMML
**Scorecard** model type by name, and any valid PMML 4.2 model — including a
Scorecard — has its score-code type set to DATA step on import, after which "PMML
models with a score code type of DATA step can be scored and published" like a
native model. This PMML Scorecard interchange is the second concrete artefact worth
decider2's attention: it is a small, already-published XML shape for exactly the
characteristics/bins/points object recommendation 4.4 needs to build, and SAS itself
uses it as an import path rather than a novel one.

**Champion/challenger: still nothing more than a Model Manager project label.**
Re-confirmed on this pass across all eight checked Intelligent Decisioning books
(UG, AG, Macro Reference, Data Grids, Custom Node Types, CLI, What's New):
"challenger" occurs zero times, and the only "champion" hit anywhere is "set the
model as a project champion in SAS Model Manager" — one flagged model version per
Model Manager project, not a traffic-split or parallel-arm mechanism inside a
decision. Marketing's "champion/challenger" claim
([features list](https://www.sas.com/en_us/software/intelligent-decisioning/features-list.html))
remains uncorroborated.

**Rule-fired data — the concrete mechanism.** "If a rule's condition expressions
evaluate to true, the rule is said to have fired… By default, every time a rule
fires, it generates a rule-fired record", suppressible per rule with a "Record
rule-fired data" check box (p. 34). Scope limits are stated: recorded for rule sets
including nested decisions and directly-included filtering rule sets, **not** for
other object types, and **not** for eligibility rule sets inside treatments; a
data-grid-iterating rule set fires "once for the entire data grid instead of… once
for each row" (p. 71). The data lands in a `ruleFiredFlags` column in the test-results
output table, or in the *subject contact history*, read with the
`%DCM_GET_SUBJECTCONTACT_HISTORY` and `%DCM_RULEFIRE_DETAIL` macros; a
**Rule-Fired Analysis** view gives a Rules Fired Count per output record, drill-down
to which rules fired, and a bar chart of fire counts per rule (pp. 71–72). It is
gated by a configuration option,
`sas.decisions.nodetraces.includeRuleFiredPathTrackInfoInVariableAssignmentLogging`.
**Note what this is not:** it records rules that *fired*, not conditions that were
evaluated and failed.

**Versioning, approvals, testing, documentation.** Every object kind has major/minor
versions, **check-out/commit** gated per folder by configuration, version tags,
content *and generated-code* comparison between versions, and "Determine Which
Objects Use a …" reverse-dependency queries. Publishing "locks and publishes the
selected version and creates a new minor version" (p. 61). A predefined **SID Asset
Approval** workflow can be enabled, and "a new instance of the workflow is started
each time a new decision or a new version of an existing decision is created", bound
to that version (p. 284). Scenario tests compare actual against expected values,
highlight mismatches and export CSV. And — directly relevant to decider2's top risk
— the product **generates PDF documentation** for a rule set, a treatment group, a
lookup table, a value list and a decision, plus "Generate a Report for a Published
Decision" (pp. 46, 101, 125, 226, 327–328).

**Publishing and execution.** Four destination classes, each doing something
different (p. 60): CAS / Teradata / Apache Hadoop ("Adds a row to the model table for
the destination"), SAS Micro Analytic Service ("Writes a Micro Analytic Service
module in the service"), **Git** ("Creates a directory in the remote Git repository…
and writes the generated code to a file named `scoreResource.txt`"), and container
destinations ("Creates a SAS Container Runtime container", to AWS, Azure, GCP or a
private Docker repository). "The rows in the model tables, the Micro Analytic Service
modules, and the SAS Container Runtime containers all become callable REST API
endpoints, independent of SAS Intelligent Decisioning." Published figures, from the
marketing page only: "More than 7,000 real-time transactions per second" and
"Response times of 5 to 10 milliseconds per transaction"
([features list](https://www.sas.com/en_us/software/intelligent-decisioning/features-list.html)),
with no hardware or decision size stated.

**No equivalence guarantee across destinations — and the guide says why not.**
Nothing in the User's Guide asserts that the same published decision produces
identical results in CAS, MAS, SCR and Git-deployed form. The opposite is stated:
"decisions that use certain model score code types cannot be published or can be
published only to SAS Micro Analytic Service destinations" (p. 4), and filtering rule
sets "can be published to SAS Micro Analytic Service destinations and to container
destinations, but you cannot publish the filtering rule sets themselves to SAS Micro
Analytic Service" (p. 18). Destination reach is per-content-type, so "one decision,
five runtimes" is a deployment claim, not an equality claim.

**South African evidence.** Absa runs SAS Viya on AWS for credit-risk *reporting and
model monitoring*, not decisioning
([press release](https://www.sas.com/fi_fi/news/press-releases/2026/august/absa-bank-modern-credit-risk.html));
Nedbank appears for customer analytics
([Nedbank](https://www.sas.com/en_za/customers/nedbank.html)). No SAS page names a
South African bank running Intelligent Decisioning.

**vs decider2.** SAS wins decisively on *authoring breadth* and on *reviewer-facing
artefact generation*; decider2 wins on *proof*, on *latency class*, and on *free
reference-data edits*.

| axis | SAS Intelligent Decisioning | decider2 |
|---|---|---|
| object kinds | 9 (rule set, treatment, treatment group, lookup table, value list, global variable, code file, custom function, segmentation tree) | 2 data-shaped (`DecisionTable` `src/decider2/tables/schema.py:551`; `Tree` `src/decider2/trees/schema.py:1055`) plus Python steps. `ruleset()`/`decision_table()`/`scorecard.py` are named in `docs/08-configuration-and-lifecycle.md:344` and confirmed absent: "Layer 4 … is BLOCKED — nothing like `ruleset()`/`decision_table()` exists yet" (`src/decider2/runtime/serve.py:12-20`) |
| rule-set semantics | collect by default, `ELSE`/`OR`/`RETURN` to narrow (p. 34) | first-match only, unnamed (`src/decider2/tables/schema.py:561`) → §3 item 7 |
| nested per-record collections | **data grid is a variable type** (p. 25), iterable by a rule set (p. 71), with its own manual | **none.** O5, "the single most-invented gap": nine names across six mock projects (`docs/06-open-questions-and-experiments.md:215-228`) → §3 item 8 |
| treatments / offer allocation | attributes (fixed or run-time dynamic), eligibility rule sets, effective dates, channels; output is a **data grid**, arbitrated by sorting/subsetting it (pp. 88-90) | **none.** decider2's example set expresses this as a 5 376-cell table output (`example_projects/08-collections-treatment.md:54`), which covers the *matrix* but not the offer catalogue, its effective dates, or ranking |
| reference data | key→value lookup tables, CSV import/refresh, labels, versions, activation (pp. 116-140) | rows ride in `shared`, so **a row edit recompiles nothing at all** (`src/decider2/tables/__init__.py:19-21`) — stronger than activation — but the keyed-lookup authoring surface is still O4, spelled six ways by cold readers (`docs/06-...:82-99`), and there is no CSV import and no effective-dating |
| segmentation tree | Boolean / matrix / value-list-variable branch elements; reusable named outcomes carrying assignment statements or DS2 (pp. 238-240) | v3 tree with unary/cases(ranges, string_match, isin)/composite nodes and a leaf-value table (`src/decider2/trees/schema.py:653-877`, `:1006`). No `other` branch generated automatically; no reusable named outcome object; **the matrix-from-two-value-lists construction has no analogue** |
| scorecards | **not in the product** — lives in Model Studio behind the Risk Modeling Add-On licence, with points-to-double-odds scaling and neutral-score "Adverse Characteristics" as a *training-time* report, not a per-record field (`sub-sas-scorecards.md` §A–B) | also absent, but decider2's own spec needs one computed **per record** (`example_projects/00-shared-credit-core-library.md:279-288`) → §3 item 1 and recommendation 4.4. *Neither ships it as a decision-engine primitive; decider2's requirement is strictly harder than SAS's own report* |
| champion/challenger | claimed on marketing, **absent from the User's Guide** and from seven further ID books (`sub-sas-scorecards.md` access note); the only "champion" hit anywhere is a Model Manager "project champion" label | absent from `src/`, and invented three different ways by mock projects → §3 item 2. *This is closer to a tie than the marketing implies* |
| rule-fired trace | `ruleFiredFlags`, Rule-Fired Analysis, fire-count drill-down and chart, history macros (pp. 34, 71-72) — fired only | taken path only: `<Name>_path` (`src/decider2/trees/build.py:71-74`), matched row or −1 (`src/decider2/tables/build.py:55-59`). *Both record firing, not evaluation* — and decider2's own spec demands evaluation (`example_projects/09-governance-and-replay-harness.md:1024-1028`) → §3 item 5 |
| versioning / approval | major-minor, check-out/commit, version tags, content and code diff, reverse-dependency query, SID Asset Approval workflow per version (p. 284) | `structure_fingerprint` + `stage`/`activate`/`rollback` at 0.177 µs / 3.36 µs with zero recompiles (`src/decider2/runtime/serve.py:90`, `:253-317`; `docs/EXPERIMENTS.md:434-435`) — faster, values-only, no approver field (→ §3 item 12), no reverse-dependency query |
| reviewer artefact | **generates PDF documentation** for five object kinds and a published-decision report (pp. 46, 101, 125, 226, 327) | **nothing.** `render()`, `debug()`, `trace` do not exist in `src/`; two reviewer tests both failed (`docs/04-observability-and-governance.md:277-340`). SAS is ahead by default |
| batch ≡ real-time | not claimed, and contradicted per content type (pp. 4, 18) | **asserted and tested four ways** — `interpreted ≡ stepped ≡ fused ≡ score()` with rung localisation (`src/decider2/testing/equivalence.py:200-216`, `:218`) |
| latency | 5–10 ms per transaction (marketing) | 1.38 µs kernel; 971 µs p50 framework overhead as naively implemented, 4.86% of a 20 ms budget, reducible to 0.73% (`docs/EXPERIMENTS.md:684-712`, `docs/README.md:29-36`). Different class: in-process vs service |

### 1.2 FICO Platform (Blaze Advisor, Decision Modeler, Originations Solution)

**Authoring.** The platform page lists "Decision Model & Notation (DMN)",
"Decision flows", "Decision tables", "Decision trees", "Rulesets",
"Scorecards", "Functions"
([platform/decisioning](https://www.fico.com/en/platform/decisioning)) — the only
§1 vendor naming both **DMN** and **scorecards** on a fetched page. The
Originations launch blog describes "decisioning strategies (rules, decision trees,
decision tables, etc.) in low / no-code environment"
([blog](https://www.fico.com/blogs/fico-launches-groundbreaking-fico-originations-solution-powered-fico-platform)).
**Governance.** "Robust lifecycle management", "extensive testing functionality",
"champion/challenger execution options or even end-to-end business outcome
simulations" (same page); Originations: "Test and simulate decisions before they
are put into production" ([originations](https://www.fico.com/en/solutions/originations)).
**Execution/deployment.** "Flexible options for cloud and on-premises delivery";
no latency or throughput figure on any page that rendered. Blaze Advisor's SDK
surface, rule language and embedding model remain **unverified** (§6).

**vs decider2.** FICO is the closest match to decider2's *intended* primitive set —
tables, trees, rulesets, scorecards — three of which decider2 lacks. FICO's DMN
claim is the practical argument for recommendation 4.15 (a CL2 table importer): it
is the one platform whose exports decider2 might have to read. On the other side,
nothing FICO publishes addresses batch/real-time equality, and decider2's zero-cost
reference-data edit (`src/decider2/tables/__init__.py:19-21`) has no FICO analogue on
any readable page.

### 1.3 Experian PowerCurve (Originations, Strategy Management, Customer Management)

**Authoring.** "Online PowerCurve Strategy Design Studio", "no-code/low-code
decisioning from simple to complex"
([DaaS](https://www.experian.co.uk/business-products/decisioning-as-a-service/));
"simple-to-use drag-and-drop building blocks", "assisted strategy design tool"
([strategy management](https://www.experian.co.uk/business/customer-insights/strategy-management)).
**Decision trees, decision tables, scorecards and node-level statistics are not
named on any page that rendered** — the strategy-tree-with-statistics capability
that makes PowerCurve interesting for comparison is an industry expectation, not a
verified Experian statement (§6).
**Execution.** "PowerCurve Decisioning as a Service (DaaS) on the Experian cloud",
"API and microservices architecture", "returns real-time or batch results", and
"1+ billion real-time integrated decisioning transactions we process" with no
period given ([originations](https://www.experian.com/business/products/powercurve-originations)).
**Governance.** "Impact assessment and strategy simulation prior to strategy
changes", "auditable decision processes", "Deploy machine learning models" (DaaS
page). Champion/challenger and versioning not found.

**vs decider2.** The one capability worth borrowing is the *simulation-before-change*
posture, which decider2 specifies in more detail than Experian states — exact
boundary intervals rather than a sampled count
(`docs/08-configuration-and-lifecycle.md:626-641`) — and has not built
(`src/decider2/testing/` holds `equivalence`, `corpus`, `recompile` only). PowerCurve
is also the clearest *alternative* case: as a bundled data-plus-strategy service it
removes work decider2 does not attempt (bureau integration) rather than work it does.

### 1.4 Equifax InterConnect (Decision Hub) and Ignite

**Authoring.** "Deploy configurable decision rules, predictive analytics, and
scoring models"; "Empower business teams to manage and deploy rules quickly"
([risk decisioning suite](https://equifax.com/business/product/risk-decisioning-suite));
"pre-built business workflows … decisioning templates and patterns" and —
the phrase that matters — "**regulatory-required reason code generation**"
([product sheet PDF](https://assets.equifax.com/marketing/US/assets/interconnect-decision-hub-ps.pdf)).
**Governance.** "Leverage challenger/champion testing", "auditable, transparent
decision codes", "decision history reports", "Collaborative, role-based access"
(same PDF). **Execution.** "Secure, scalable SaaS platform", "delivery via API, UI,
or batch"; no latency numbers. **Ignite** is model development, not a rule engine
([Ignite](https://www.equifax.com/business/product/ignite/)).

**vs decider2.** Equifax is the only §1 vendor that names **reason-code generation as
a product feature**. decider2's doc claims "reason codes need no new machinery"
(`docs/04-observability-and-governance.md:175`) and its own review refutes that at
realistic scale (`docs/REVIEW.md:673-677`): a decline taxonomy is an ordered
variable-length list, usually top-k, and a step returns one scalar. decider2's own
example set needs 380 registered codes with a primary
(`example_projects/00-shared-credit-core-library.md:554`, `:128-129`). This is
recommendation 4.5 and it is a must-have.

### 1.5 Provenir Decision Intelligence Platform

**Authoring.** "Visual, drag-and-drop interface" for "decisioning flows" with an
"in-built AI Assistant to help create new logic"
([platform](https://www.provenir.com/platform/)). Tables, trees and scorecards are
**not named**. **Execution.** "RESTful APIs for real-time decisioning", "batch
processing for high-volume scenarios", "webhook support"; "4+ billion decisions
annually", "Millisecond decisions" (same page); "Sub-second decisions, 795M actions
a year" ([decisioning](https://www.provenir.com/platform/decisioning/)).
**Governance.** "Every workflow is auditable and version-controlled", "built-in
testing, simulation and code management", "plain-English explanations of any rules-
or model-based decision". African adopters on the vendor's own site: Carbon
(Nigeria) and DeltaPay (Kenya); no South African bank.

**vs decider2.** The only borrowable idea is the plain-English explanation, which
decider2 should treat as *one renderer over the decision record*, consistent with its
own settled position that "trace is data; every rendering is replaceable"
(`docs/04-observability-and-governance.md:415-445`). decider2 has neither the record
(`observe/` does not exist) nor a renderer.

### 1.6 ACTICO Platform / ACTICO Rules — the closest architectural analogue

**Authoring.** "Rule types such as flow rules and decision tables — graphically
yourself or with an AI agent"
([modeling](https://www.actico.com/platform/rule-decision-modeling/)); "Graphical
modeling of rules and decision tables without coding"
([ACTICO Rules](https://actico.com/en/products/actico-rules)). No scorecard object
named. **Deployment.** "REST, JSON/XML, SOAP and OpenAPI interfaces"; "native Java
API allows direct embedding of decisions into Java applications"; "Integration as a
service, via Java API and via batch processing"; "scaling across multiple engines
with elastic load balancing (e.g. in Kubernetes)"
([execution](https://www.actico.com/platform/decision-execution/)).
**Execution.** Real-time plus "Command-line interface for batch execution …
Supports parallel execution"; "Low latency web stack for thousands of decisions per
second" (same page). **Governance — the richest in the set.** Model Hub "versions
every change, records approvals in an audit-proof manner"; "Use familiar Git tools
to clone, check out, commit and push"; "**Simultaneous hosting of multiple model
versions (dynamic routing, e.g. for champion/challenger)**"; "One-click deployments
of models and roll-backs without downtime"; "**Visualisation of the full execution
path**"; trace data "published to external systems (e.g. Kafka)"; "Model simulation
scenarios such as Monte Carlo or Champion/Challenger"; an AI agent finds "redundant
rules, conflicts and unreachable branches"
([model management](https://www.actico.com/platform/model-management/)).

**vs decider2.** ACTICO is the platform decider2 most resembles in shape — embeddable
library plus batch CLI plus REST — and the one whose governance surface decider2
should be measured against item by item:
| ACTICO | decider2 |
|---|---|
| multi-version hosting with dynamic routing | `ServeHandle` holds generations; activation 0.177 µs, rollback 3.36 µs, zero recompiles (`docs/EXPERIMENTS.md:434-435`) — **but no routing between them** |
| one-click rollback without downtime | `ServeHandle.rollback()` (`src/decider2/runtime/serve.py:305`) — **present, values only** |
| approvals recorded audit-proof | **absent**; "Approval is the caller's policy" (`docs/04-observability-and-governance.md:94`) |
| full execution path visualised | taken path only, as an integer column (`src/decider2/trees/build.py:71`) |
| trace to Kafka | **absent** — and note decider2's own finding that per-record trace is PII by construction (`docs/04-...:262-276`) |
| redundant/conflicting/unreachable rule detection | per-table gap/contiguity (`src/decider2/tables/schema.py:376-383`) and per-node range order (`src/decider2/trees/schema.py:882-910`) only; no reachability |
| Git as the model-versioning interface | decider2's skeleton *is* Python in Git (`docs/08-configuration-and-lifecycle.md:110`) — ahead |
| Java embed | Python in-process `score()` — equivalent |

### 1.7 GDS Link (Decision Studio) and Zoot Enterprises

**Sources.** A full crawl of both vendors' live sites plus archive.org snapshots,
recorded in the scratchpad's `sub-sas...` sibling file for this pair (48 live English
GDS Link pages, 42 live Zoot pages, both crawled exhaustively for the terms this
report tracks — "decision table", "hit polic[y]", "matrix", "reason code", "adverse
action", "Kafka", "REST", "simulat/backtest").

**GDS Link.** The brand has moved on: "Modellica" occurs **zero times** across the
48 live English product pages — it has been replaced by "GDS Link Decisioning
Platform" and "Decision Studio", and old Modellica-era URLs now 301 to the renamed
pages. **Authoring.** "Decision Studio is the central application for configuring
decision strategies, rules, scorecards, workflows, and data integrations"
([design](https://gdslink.com/platform/design/)); "Low-Code Decision Logic – Design
rulesets, scorecards, segmentations, and decision trees visually in Decision Studio"
([decision engine](https://gdslink.com/decision-engine-software/)); "Easily
integrate Python, PMML, and R models while leveraging a low-code interface"
([product overview](https://gdslink.com/product-overview/decision-engine/)). No
page names a hit-policy vocabulary, an in-engine expression language, or a
treatment/offer arbitration mechanism — the closest is an unelaborated "Upsell &
Counter Offers" bullet. The one substantive PMML data point is a customer quote, not
a spec: "We had a 50% reduction in time to implement a Logistical Regression Model
by using GDS's PMML tool... Previously it took approximately 20 hours and with this
tool we can deploy a model in approximately 10"
([archived page](https://web.archive.org/web/20230206193054/https://www.gdslink.com/solutions/loan-originations-and-decisioning/)).
**Node-level statistics:** not found; only strategy-level "Policy Monitoring" is
described ("simulate outcomes using live data and historical scenarios... without
jeopardizing current operations",
[policy monitoring](https://gdslink.com/advanced-analytics/policy-monitoring/)).
**Champion/challenger** is named repeatedly ("Champion/Challenger Testing – Compare
the performance of strategies and models side-by-side before rolling out changes in
production") but **no traffic-split percentage, shadow-run mechanism, or report
content is described on any page** — the same gap as SAS's marketing claim.
**Execution.** "processes real-time and batch credit decisions"
([decision engine](https://gdslink.com/decision-engine-software/)); "1 billion+
decisions... are made every year"
([product overview](https://gdslink.com/product-overview/)); "delivers credit
decisions in milliseconds" appears on several industry pages, and the strongest form
— "Maintains sub-second decisioning at the point of sale", "Full audit trails
automatically logged at the rule, data, and model level" — is from
`gdslink.com/competitor-web-page-template/`, an **unfinished comparison template
page still carrying "[Competitor Name]" placeholders**; it is vendor-owned but this
report treats it as draft sales collateral, not a product claim, and that page is
also the *only* place "reason code" or "adverse action" appears anywhere on
gdslink.com. No REST, Kafka, or p50/p99 figure was found anywhere in the crawl.
**Governance.** "Compare strategy versions side-by-side... to understand how changes
affect decision outcomes before production release"
([design](https://gdslink.com/platform/design/)); "audit trails and role-based
controls"; no approval workflow or maker-checker mechanism found. **Deployment.**
"Built to deploy in the cloud or on-premises"
([release](https://gdslink.com/latest-release-of-our-integrated-decisioning-analytics-platform/));
"cloud-native, containerized architecture"; no embeddable-library or code-generation
claim found; the gated Modellica datasheet landing pages
(`info.gdslink.com/modellica...`) are dead — the subdomain no longer resolves and
has no archive.org snapshot. **The only verified South African bank running a
vendor decisioning platform:** TymeBank, "onboarding over 100,000 new customers per
month" ([case study](https://gdslink.com/case-study/tymebank/)).

**Zoot Enterprises.** The vendor's own product split is explicit and maps onto a
dev/prod separation decider2 should note: "WebRules Builder® is focused on the
development and testing of decision rules, while WebRules Live® is used for the
execution and management of these rules in a live production environment"
([WebRules Builder](https://zootsolutions.com/webrules-builder/)). **Authoring.**
"powerful decision flows, scorecards, strategy tables, and more"
([advanced decision management](https://zootsolutions.com/advanced-decision-management/));
across all 42 crawled pages, **"decision table", "decision tree", "lookup table",
"matrix" and "hit policy" are never named** — "strategy tables" is the closest term
and is never structurally described, and no expression language or scripting surface
is documented (Python/PMML/R integration, which GDS Link states, is absent from
Zoot's pages). **Node-level statistics:** not found — only generic "Robust
Reporting... detailed analytics and insights into rule performance." **Champion/
challenger** is a named package feature, built on "three orchestration types –
Simple, Waterfall, and Parallel" for comparing data providers and flows
([application fraud](https://zootsolutions.com/application-fraud/)); as with GDS
Link, **no split percentage or report content is documented**. **Execution — the
single hardest number in the entire vendor set:** "This fully transparent, easily
configurable, scalable environment processes decisions in as little as 4
milliseconds" ([WebRules Live](https://zootsolutions.com/webrules-live/)) — "as
little as" marks it a best case, not a guarantee, and Zoot explicitly declines a
throughput figure: "Performance metrics vary based on client platform and
transaction types... We work closely with you to gather specific throughput and
capacity needs as part of the project"
([data connection API](https://zootsolutions.com/data-connection-api/)).
**Governance.** "Version Control — Tracks changes and manages different versions of
rules"; an "Audit Trail" and named "Audit Viewer" package inclusion; no approval
workflow found; and — checked explicitly across all 42 pages for "simulat",
"backtest", "back-test", "replay" — **Zoot documents no simulation or backtesting
capability at all**, the only vendor in the full §1/§1.10 set with a confirmed
absence rather than an unreached page. Reason codes: "Adverse Action Enabled" and
"Standard & Custom Override Reasons" appear as origination package feature bullets,
but the term "reason code" itself is never used. A useful change-lead-time
comparator: "Model changes can take as few as a couple days to less than two weeks
to implement within Zoot's platform" — against decider2's measured 2.5–7.3 s
(`docs/EXPERIMENTS.md:40-42`). **Deployment.** Hosted **private cloud only** — "All
of Zoot's solutions run within our private cloud environment" across "five global
data centers"; the Builder authoring tool is still a **desktop/thick client**, with
a browser version "currently in development"; no on-premise, container, embeddable
library or code-generation claim was found (the developer portal itself is
login-walled — §6).

**vs decider2.** Nothing to borrow on authoring primitives from either vendor — both
explicitly avoid the tables/trees/hit-policy vocabulary the rest of §1 uses, which is
itself informative: two vendors selling into exactly decider2's market (credit
origination, account management, collections) do not find that vocabulary necessary
to sell the product. The relevance is §5.3: GDS Link's TymeBank case study is the
existence proof that a South African digital bank runs origination decisioning on a
vendor platform at scale, and Zoot's explicit "no simulation, no backtesting" gap is
the sharpest evidence in the whole survey that even a mature, decades-old rules
vendor can ship without the capability decider2 ranks as a **must-have**
(recommendation 7/8) — buying does not make that problem go away.

### 1.8 Taktile, Oscilar, Alloy — cloud-native "decision OS"

- **Taktile.** "Pre-built 'nodes'", "low-code components, Python, and reusable
  logic", "Leverage AI to generate Python"
  ([decision engine](https://taktile.com/decision-engine)); governance: "Model how
  changes to your logic will perform using real or historical data before going
  live", "Experiment with multiple versions of a flow in production", "Review and
  sign-off" with "audit-ready traceability"
  ([credit platform](https://taktile.com/credit-decision-platform)). The node
  catalogue, whether decision tables or scorecards exist, and latency are
  **unverified** — docs.taktile.com is an authenticated GitBook (§6).
- **Oscilar.** "Describe your workflow and rules logic in natural language" plus
  drag-and-drop; "120K+ requests per second, <100ms latency"
  ([home](https://oscilar.com/)); "one-click backtesting", A/B of workflow versions,
  RBAC per change, "human-in-the-loop approvals"
  ([platform](https://oscilar.com/platform)). docs.oscilar.com is
  password-protected (§6).
- **Alloy** — the only one with public developer docs, and now the most thoroughly
  verified vendor in §1 after SAS. Its model is **Journeys (orchestration graph) →
  Workflows (decisioning unit) → nodes**: "A workflow is the core unit of
  decisioning logic in Alloy … a configurable container for the rules, data
  sources, and policy logic that get run against an entity or event to produce an
  evaluation outcome"
  ([workflow types](https://developer.alloy.com/public/docs/workflow-types.md)).
  Rules are called **thresholds**, outputs are **tags**, and a workflow's rule rows
  are ordered and groupable — "Add a New Row of Logic... Reorder Placement of Rows...
  Adding and Removing a Group" — which is functionally a decision table but is never
  once called one
  ([threshold editor](https://help.alloy.com/en/articles/16820278-workflow-editor-thresholds-and-dependents)).
  A grep of all 115 guide pages and 129 API-reference pages found **zero occurrences**
  of "decision table", "scorecard", "decision tree", "hit polic[y]", "matrix" (as a
  primitive), "arbitrat[ion]" or "treatment" — Alloy has no scorecard primitive at
  all; scoring is done by external/custom models or by **JQ** expressions (Alloy's
  expression language is the open-source JQ query language, not a proprietary DSL:
  `if .riskScore > 700 then "high risk" elif .riskScore > 500 then "medium risk"
  else "low risk" end`,
  [attribute tool](https://developer.alloy.com/public/docs/attribute-tool.md)).
  Reference data has its own versioned object, **Custom Lists**, with major/minor
  version semantics distinct from SAS's lookup tables: "To create a new major
  version of a list: define or update columns... To create a new minor version:
  add/delete/modify entries... The actual list version used in a realtime evaluation
  will be the latest minor version for the *activated* major version"
  ([custom lists](https://developer.alloy.com/public/reference/post_custom-lists-customlisttoken-versions.md)).
  **Champion/Challenger is the single most concretely specified split mechanism in
  the whole survey:** "deploys two or more (up to 5) versions... evaluations are
  probabilistically run through a selected workflow version... percentage
  allocations (at least 5%) applied to each version"
  ([champion challenger](https://help.alloy.com/en/articles/16820342-champion-challenger)),
  with start/stop themselves audit-logged actions — but note it is **per-request
  probabilistic** routing, not a deterministic hash of an entity id, which is the
  opposite of what decider2's own spec requires (recommendation 6: "never `random`").
  A separate, distinct **Shadow Testing** mechanism also exists, flagged on the
  application record itself (`is_shadow_app`, `is_part_of_shadow_test`). Backtesting
  is three named tools, not one — historical replay against another version
  (recommended 500–5,000 sample size, "reprocessed through your new version" using
  *cached* third-party responses, not live calls), a portfolio-snapshot
  re-evaluation ("PE Backtesting"), and a "What If Analysis" with documented limits
  (cannot add new data sources; sandbox evaluations unsupported)
  ([backtesting guide](https://help.alloy.com/en/articles/16820270-real-time-backtesting-guide)).
  Governance is the deepest of the three: parent-linked version lineage, new
  versions inactive by default, rollback, an explicit **approval-to-pin** audit gate
  ("Approval requested to pin new active Application Version... Approval confirmed",
  [audit actions](https://help.alloy.com/en/articles/16820362-roles-settings-audit-actions)),
  ~100 named audit action types, and a per-decision "Rule Explainability" view
  ("click on a tag to open Rule Explainability and see exactly which rules fired and
  why"). Reason codes are a first-class, typed, audited object — `outcome_reasons`
  carries an explicit `"type": ["adverse_action", null]` enum on the evaluation
  response
  ([API schema](https://developer.alloy.com/public/reference/post_journeys-journey-token-applications.md))
  — but Alloy is explicit that this stops at the code, not the notice: "Alloy does
  not offer out-of-the-box solutions or customization for adverse action notices
  (AAN's) sent to denied applicants"
  ([FAQ](https://help.alloy.com/en/articles/16818551-does-alloy-provide-support-for-adverse-action-notices)).
  Deployment is SaaS-only, and — correcting the loose assumption a "sandbox vs
  production" heading invites — the two are **not separate environments**: "They are
  per-request processing modes only — Alloy does not enforce a data boundary. Most
  records... live in a single shared data store for your account, regardless of the
  mode used"
  ([sandbox vs production](https://developer.alloy.com/public/docs/sandbox-vs-production.md)).
  No latency or throughput SLA is published, but Alloy uniquely exposes a per-decision
  timing-diagnostics endpoint (`GET /diagnostics/timings/journey-applications/...`)
  the caller can query for their own numbers.

**vs decider2.** Four things are worth taking, one thing is worth avoiding. (i)
Alloy's **manual-review / step-up node** is the one node type in the whole §1 set
that answers a gap decider2's own review names: "The `required` null policy is
all-or-nothing — one bad row raises for the whole batch. No row-level rejection,
quarantine or route-to-manual-review, which every production batch needs"
(`docs/REVIEW.md:682-684`), against `route_required_nulls`
(`src/decider2/boundary/nulls.py:311`), which routes but has no downstream
destination concept. (ii) Alloy's **percentage-allocation champion/challenger shape**
— named arms, a minimum share per arm, up to N concurrent versions — is a better
template for recommendation 4.6's `arms={name: pct}` than ACTICO's vaguer "dynamic
routing" language, **provided the assignment is switched from Alloy's per-request
probabilistic draw to decider2's own required deterministic hash of a stable id**
(`example_projects/03-unsecured-loan-granting-and-pricing.md:365-371`) — copy the
shape, not the randomness. (iii) Alloy's **typed `adverse_action` reason-code field**,
paired with its honest admission that it does not generate the notice itself, is the
cleanest scoping precedent for decider2's own reason-code recommendation (4.5):
compute and expose the typed code, and treat notice generation as a downstream
renderer, consistent with decider2's own "every rendering is replaceable" position
(`docs/04-observability-and-governance.md:415-445`). (iv) Alloy's versioned
**Custom Lists** (major/minor, activate-a-major-version) is a lighter-weight
alternative shape for decider2's still-unspelled keyed-lookup problem (O4, §3 item
10) than SAS's full lookup-table object. **What is explicitly not a match**: Alloy's
sandbox/production split turns out to be a request-header flag over one shared data
store, not a real environment boundary — weaker than `ServeHandle.preview`
(`src/decider2/runtime/serve.py:332-351`), which genuinely scores one record against
two distinct generations side by side; the earlier framing of "preview is a
one-record version of Alloy's sandbox" undersold decider2's own mechanism. Oscilar's
and Taktile's backtesting remains recommendation 4.7. Natural-language authoring
(Oscilar, Taktile) is explicitly *not* worth taking — see the closing note of §3.

### 1.9 Pega Customer Decision Hub

**Sources and access.** `docs.pega.com` is a JavaScript-only shell to every fetcher,
but its backend content API is open: `docs-be.pega.com/api/bundle/{bundle}/page/
{navPath}` returns the full topic HTML, and `docs-be.pega.com/api/search?q=...`
full-text-searches every Pega doc. Everything below without another citation is
from that route (bundles `platform`, `customer-decision-hub`,
`credit-risk-decisioning`), cross-checked against Pega's static legacy help
(`community.pega.com/sites/.../help_v73` and `help_v84`) where the current docs
omit structural detail the legacy pages still carry. This closes the landscape
note's Pega gap entirely — nothing below is marketing-only.

**Authoring — a strategy is assembled from named component categories, not two
primitives.** The Strategy canvas groups shapes into: **Substrategy** (external/
embedded sub-strategy, prediction); **Import** (data import, proposition data,
interaction history); **Business rules** (decision table, decision tree, map value,
split); **Decision analytics** (adaptive model, predictive model, **scorecard**);
**Enrichment** (data join, decision data, set property); **Arbitration** (filter,
prioritize); and, CDH-only, **Selection** (contact policy, geofence filter, segment
filter, **champion challenger**, exclusion, switch) and **Aggregation** (group by,
iteration, financial calculation)
([strategy components](https://docs.pega.com/bundle/platform/page/platform/decision-management/decision-strategy-components.html)).
**Decision table hit policy — Pega does not use the term "hit policy" at all** (0
doc-search hits); the equivalent is a single checkbox: "To continue processing
after one row in the decision table evaluates to true, select the **Evaluate all
rows** checkbox... If you leave this checkbox clear, the application performs the
result from the first row that evaluates to true"
([decision table options](https://docs.pega.com/bundle/platform/page/platform/app-dev/decision-table-additional-options.html))
— i.e. exactly decider2's missing `first`/`collect` distinction (recommendation 1),
independently converged on by a second vendor. Table, tree and map value are given
an explicit side-by-side definition: "Decision table - Returns a single value for
the row... Decision tree - Returns a single value for the branch... Map value -
Converts one or two input values into a single-value result"
([table vs tree vs map value](https://docs.pega.com/bundle/platform/page/platform/case-management/types-decision-logic.html)).
**Scorecard rule authoring — directly credit-relevant.** "You can use scorecards to
derive decision results from a number of factors, for example, for credit risk
assessments... The output of a scorecard is a score and a segment"; predictors
carry a weight (default 1) and a **combiner function** — "Sum / Min / Max /
Average" across predictors — and cutoffs map score ranges to decision results, with
an "Audit Notes" checkbox to "capture scorecard details in the case history"
([scorecard main](https://community.pega.com/sites/pdn.pega.com/files/help_v84/rule-/rule-decision-/rule-decision-scorecard/main.htm),
[cutoffs](https://community.pega.com/sites/pdn.pega.com/files/help_v84/rule-/rule-decision-/rule-decision-scorecard/results.htm)).
A dedicated **Credit Risk Decisioning** accelerator exists on top of this: "a
three-level Decision strategy for automating loan application decisions"
including "hard-stop eligibility checks that rule out any customers who do meet
certain criteria, for example, their total liabilities exceed the limit"
([overview](https://docs.pega.com/bundle/credit-risk-decisioning/page/credit-risk-decisioning/hub/credit-risk-decisioning-overview.html)).
DMN is **not used**: Pega's own BPMN reference is the only process-standard
citation found; "DMN" returns nothing.

**Node-level statistics — real, but scoped to Strategies, not to tables or trees.**
After a batch case test run, "a label displaying the test result appears at the top
of each shape", including "Number of records with Decisions... Time spent...
Throughput (decisions)"
([batch case runs](https://docs.pega.com/bundle/platform/page/platform/decision-management/configure-batch-case-runs.html),
[labels](https://docs.pega.com/bundle/platform/page/platform/decision-management/test-run-labels.html)),
and the sample can be a **migrated production data snapshot**, not a synthetic one
([sampling production data](https://docs.pega.com/bundle/platform/page/platform/decision-management/sampling-production-data.html)).
But this is explicitly *not* available for the sub-components a credit decision
actually runs: "Decision simulation tests do not support all components. This
includes adaptive and Predictive Models, **scorecards, Decision trees, Decision
tables**, and others" (same batch-run page) — no vendor page describes per-node
volumes on a decision-table row or a decision-tree branch. This corrects the
landscape note's inference: Pega's node-level statistics are a *strategy-shape*
feature, weaker than what decider2's own spec asks for at tree-node granularity
(`example_projects/04-campaign-targeting-trees.md:531-560`), not a stronger one.

**Champion/Challenger — the mechanics are exact, and it is a random per-call draw.**
"randomly allocate customers between two or more alternative components... you can
specify that 70% of customers receive offers for product X and 30% offers for
product Y... During each run the Champion challenger component randomly selects one
of the alternate paths based on the defined percentages... After 1000 runs...
product X for 69% of customers and product Y for 31%... This deviation becomes
smaller as the number of runs increases"
([champion challenger](https://docs.pega.com/bundle/platform/page/platform/decision-management/champion-challenger-component.html)).
Reporting comes from the shape-level test-run labels above and from **Decision
funnel simulation**, which has champion-challenger as a named breakdown dimension.
**Note the contrast with decider2's own spec**: Pega's split is a live random draw,
re-randomised per call, whereas decider2's mock projects require a **stable hash of
`client_id`**, "never an RNG" (`example_projects/03-...:365-371`) — Pega's own
described statistical deviation ("69/31 instead of 70/30") is exactly the
non-reproducibility a deterministic hash exists to avoid, and is a concrete argument
for decider2's stricter design.

**Execution modes and a genuine architectural parallel.** REST (`POST/GET
/v4/container` for real-time inbound decisions), Data Flow batch and streaming
(Kafka-backed) runs, and the credit accelerator confirms both explicitly: "responds
to real-time requests... The application can also run batch decisions to calculate
pre-approved offers"
([overview](https://docs.pega.com/bundle/credit-risk-decisioning/page/credit-risk-decisioning/hub/credit-risk-decisioning-overview.html)).
Kafka is a **hard external dependency** for containerised/streaming deployments
("Kafka Required to support the streaming functionality") — an operational cost
decider2's in-process model has no analogue of, in either direction. Pega's
performance path, **Globally Optimized Strategies (GOS)**, is architecturally the
closest thing to decider2's fused mode found in the whole survey: "it brings
several types of decisioning artifacts into a single program and runs them all
together. Because Rules run together rather than one by one, overall Strategy run
performance greatly increases"
([GOS](https://docs.pega.com/bundle/platform/page/platform/decision-management/globally-optimized-strategies.html)),
compiled via "Single Static Assignments" — conceptually decider2's `fused` mode
(`runtime/modes.py:1-30`) compiling several steps into one kernel, though GOS is
undocumented as to whether it is bytecode, generated source, or something else, and
Pega publishes **no equivalence guarantee** between GOS and non-GOS execution of the
same strategy (the corresponding decider2 claim, `testing/equivalence.py:218`,
remains decider2's alone).

**Latency: real docs numbers exist, and they disagree with the marketing.**
Product documentation gives **150 ms** as a load-test bucket boundary — "% of
transactions processed under 150 milliseconds and % of transactions processed over
150 milliseconds"
([load tests](https://docs.pega.com/bundle/customer-decision-hub/page/customer-decision-hub/implement/cdh-implementation-load-tests.html))
— and "thousands of requests per second" as the only throughput scale claim
([deployment requirements](https://docs.pega.com/bundle/customer-decision-hub/page/customer-decision-hub/install/deployment-requirements.html)).
Marketing pages give **three different numbers**: "less than 200 milliseconds", "all
within 200 milliseconds", and "220 milliseconds or less" on three different pages.
Cite the docs figure (150 ms bucket boundary, not a guarantee), not the marketing
one. The phrase "billions of decisions" is **not a Pega phrase**: the verified forms
are "billions of **interactions**" and "5 Billion next best actions each month" —
mark "billions of decisions/day" unverified if it recurs elsewhere in this report's
source material.

**Governance — the most detailed public approval-workflow specification in the
whole survey.** **Revision Management** lets "business users (such as marketers)...
make controlled changes in the BOE and test them before deploying... within the
boundaries defined by IT", via a **Change Request** case with a named stage
sequence: "Submit for testing → Pending-Testing (simulate and test)... → Submit
Changes → Approval Required? → Approve Changes → [Reassign CR] / Withdraw / Reject"
([stages](https://docs.pega.com/bundle/customer-decision-hub/page/customer-decision-hub/cdh-portal/change-request-stages.html)).
Compare this to SAS's one-line "SID Asset Approval" workflow (§1.1) — Pega's is the
richer public spec and the better model for decider2's recommendation 12 (an
approver field on `StagePlan`). Elsewhere: **Rule history** ("view the saved
history... compare the current version with a previous version or restore a
previous version"); **Branches** — "a naming convention that includes a Ruleset ID,
the word Branch, and a Branch name... create a Branch review to ensure Rules... are
guardrail-compliant"; a **Deployment Manager** CI/CD pipeline. On the **audit trail
for a single decision**, Pega ships three distinct mechanisms at three different
costs: **Execution Audit** (a GOS-only production recorder, near-real-time);
**Explainability Extract** — "records Decision outcomes... and stores them in
Parquet files in Pega Cloud File storage", "5 GB of data for a million... inbound
Decisions", **5-day default retention**, and explicitly "a Preview Release feature"
gated to GOS
([explainability extract](https://docs.pega.com/bundle/customer-decision-hub/page/customer-decision-hub/cdh-portal/explainability-extract.html))
— a concrete number worth citing alongside decider2's own finding that per-record
diagnostics are PII by construction (`docs/04-observability-and-governance.md:262-276`);
and **scorecard explanations** stored per-call in `pxExplanations` when explicitly
enabled — plus a documented cost warning on the adjacent mechanism: "Explaining
results [for a proposition filter] does not filter propositions and should be used
for testing only. This setting also has a negative impact on strategy performance"
([arbitration](https://docs.pega.com/bundle/platform/page/platform/decision-management/arbitration.html)).
**Simulation/backtesting** is named and scoped precisely: **Distribution test**,
**Decision funnel** ("can take significantly longer... up to six times the runtime
of a basic distribution test", recommended sample sizes 100,000 → 500,000 for a
final check), and **Scenario Planner**, which is explicitly disabled on production:
"the Scenario Planner menu item is inactive on any environment that is designated a
Level 5 Production environment"
([scenario planner](https://docs.pega.com/bundle/customer-decision-hub/page/customer-decision-hub/cdh-portal/scenario-planner.html)).
Credit-specific unit tests validate "changes to Decision Tables, Decision Trees,
Strategies, and When Rules."

**Reason codes / adverse action — absent as a named primitive, despite the domain
fit.** Pega's own full-text doc search returns **zero results** for "adverse
action" across every bundle; the only "reason code" hits are Visa/Mastercard
chargeback codes in Smart Dispute, unrelated to ECOA/Reg-B. The credit-decisioning
accelerator's actual output is coarser: "The final Decision (approve, decline, or
investigate)... Any eligibility flags raised for the customer. The score segment of
the customer"
([loan origination decision](https://docs.pega.com/bundle/credit-risk-decisioning/page/credit-risk-decisioning/loan-origination-decision-consumer.html)).
A site would assemble its own reason codes from `pxExplanations` (scorecard),
proposition-filter explanation text, and eligibility flags — Pega does not ship a
named adverse-action primitive any more than SAS does, which reinforces that
Equifax (§1.4) is the only §1 vendor naming this as a product feature at all.

**Model import.** PMML 3.0–4.4, with an explicit algorithm allow-list that includes
"**Scorecard**" by name (the same PMML Scorecard type SAS's `PROC PSCORE` consumes,
§1.1) alongside decision tree, regression, ruleset, SVM and ensemble methods; H2O-3
and Driverless AI `.mojo` models; and ONNX-exported scikit-learn models (logistic
regression, random forest, gradient boosting, XGBoost-style ensembles)
([supported models](https://docs.pega.com/bundle/platform/page/platform/decision-management/supported-models-import.html)).

**Deployment.** Pega Cloud is "the recommended and preferred deployment option";
both on-premises and client-managed cloud are explicitly discouraged by Pega's own
docs — "not recommended due to limitations in features and security advantages
compared to Pega Cloud" — stated for *both* alternatives
([deployment options](https://docs.pega.com/bundle/customer-decision-hub/page/customer-decision-hub/implement/deployment-options.html)),
worth flagging as a vendor steering its own customers toward its highest-margin
option rather than a neutral architecture recommendation. Kubernetes deployments
require Kafka, Elasticsearch/OpenSearch, and (for CDH) Cassandra as external
services. No page documents embedding the decision engine as a library inside a
host application, and no page documents exporting strategy logic as external
source code — only GOS's internal SSA compilation, which stays inside Pega.

**vs decider2.** Pega is not a credit engine, but four things transfer directly.
(i) It still owns the best single sentence in the survey for decider2's purposes —
"Every decision records what it chose, what it rejected, and why" — exactly the
evaluated-but-not-fired requirement
(`example_projects/09-governance-and-replay-harness.md:1024-1028`) that decider2's
`_path`/row-index capture does not satisfy (recommendations 2 and 13). (ii) Its
Change Request stage table is the best public model for recommendation 12's
approver field — richer than SAS's one-line approval workflow. (iii) Its "Evaluate
all rows" checkbox independently confirms decider2's `hit_policy` naming gap
(recommendation 1) from a second, unrelated vendor. (iv) Its Explainability
Extract's stated cost (5 GB/million decisions, 5-day retention, GOS-only, still
"Preview Release" after years in market) is concrete evidence that Pega itself
treats granular per-decision recording as expensive and provisional — which
supports rather than undercuts decider2's own caution that per-record diagnostics
are PII by construction. Two things are worth explicitly *not* copying: Pega's
Champion/Challenger is a random per-call draw with documented statistical drift,
the opposite of decider2's required deterministic-hash design (recommendation 6);
and its node-level statistics do not reach table/tree granularity, so it is not
in fact a stronger precedent for recommendation 9 than assumed on the landscape
note's first pass.

### 1.10 The rest, one line each

- **TransUnion DecisionEdge** — every product page 403s. From TransUnion's own
  10-K: "a software-as-a-service decisioning offering which allows businesses to …
  apply customer-specific criteria to facilitate real-time, automated decisions at
  the point of consumer interaction"; "more than 600 active solutions in over 10
  countries"
  ([10-K](https://investors.transunion.com/~/media/Files/T/Transunion-IR/documents/investor-services/tru-2016-form-10-k.pdf)).
  Authoring model, standards and latency **unverified** — nothing to compare.
- **Scienaptic AI** — ML-first underwriting: "Application from your LOS … JSON · 1
  call", "Decision string – Returned in < 1s", plus "Reasons" and "Audit trail" per
  decision ([site](https://www.scienaptic.ai/)). No rule-authoring model described;
  relevant only as evidence that per-decision reasons are table stakes.
- **Zest AI** — ML-only underwriting, "over 600 active models", integrated with
  Temenos LOS
  ([announcement](https://www.zest.ai/company/announcements/zest-ais-credit-decisioning-and-fraud-detection-now-seamlessly-integrated-with-temenos-loan-origination-solution/));
  monitoring includes "Reason code stability" and "Fair lending analysis"
  ([MRM blog](https://www.zest.ai/learn/blog/heres-how-ml-underwriting-fits-within-federal-model-risk-management-guidelines/)).
  No rule engine; **"reason code stability" is a monitoring metric decider2 has no
  concept of** and would need once recommendation 4.4 exists.
- **nCino / FullCircl** — LOS suite; FullCircl "offers a powerful business
  rules-engine that can simplify the complexity of onboarding"
  ([acquisition](https://www.ncino.com/news/ncino-to-acquire-fullcircl)); identity
  products describe an "easily configurable low-code / no-code rules engine"
  ([DueDil](https://www.identitysolutions.ncino.com/duedil)). LOS-adjacent: the
  rules layer serves onboarding, not credit strategy.
- **Temenos Loan Origination** — "Build risk, price, and decision models using rules
  and **matrices** … using the simple, intuitive Business Language Editor"
  ([2025 factsheet](https://www.temenos.com/specialized/credit-unions-and-community/temenos-origination/nam-temenos-loan-origination_factsheet_2025/)).
  "Matrices" is the one term here that maps directly onto decider2's decision table;
  SA adopter is Barko Financial Services (microfinance, not a bank), "more than
  125,000 loans per month"
  ([release](https://www.temenos.com/press_release/leading-south-africa-microfinance-institution-barko-goes-live-on-temenos-saas-in-under-6-months-to-deliver-finance-faster/)).
- **Finastra Originate** — 403s to fetchers; one page retrieved by curl says
  "real-time decisioning" and "powerful integration to Finastra and select
  third-party LOS solutions"
  ([page](https://www.finastra.com/solutions/fusion-loan-origination)). Rule
  authoring **unverified**.

## 2. Feature inventory

One row per feature found anywhere in the §3 set. "Does decider2 have it" is
scored against **built code**, not docs; a doc-only design counts as **no** with
the doc named. `(b)` = specified in decider2 docs but unbuilt; `(c)` = required by
decider2's own example projects with no mechanism designed.

| # | Feature | Platform(s) | What it does | decider2? | Adopt? |
|---|---|---|---|---|---|
| 1 | **Rule set** as a first-class object | SAS, FICO, ACTICO, Pega, Zoot, IBM ODM | a named, versioned collection of if/then rules evaluated as one unit, editable by a business user | **no** (b) — `ruleset()` is doc-only; Layer 4 `interiors/` is "BLOCKED" (`docs/00-BUILD.md:190`), `runtime/serve.py:12-20` states nothing like it exists | **yes** — the only primitive all six competitors have that decider2 has no form of |
| 2 | Decision table with a **named hit policy** | FICO, ACTICO; SAS rule sets are **collect by default** — "all rules … evaluated sequentially regardless of the results of previous rules" ([UG p. 34](https://documentation.sas.com/api/collections/edmcdc/v_040/docsets/edmug/content/edmug.pdf?locale=en)); **Pega** uses no "hit policy" term at all (0 doc-search hits) but ships the identical binary choice as a checkbox — "Evaluate all rows" vs. default first-match, "If you leave this checkbox clear, the application performs the result from the first row that evaluates to true" ([decision table options](https://docs.pega.com/bundle/platform/page/platform/app-dev/decision-table-additional-options.html)); vocabulary standardised by DMN, DecisionRules (`FIRST_MATCH`/`EVALUATE_ALL`), AWS (`FIRST_MATCHED`/`ALL_MATCHED`) | first / unique / priority / collect semantics, stated on the artefact | **partial** — `DecisionTable` exists (`tables/schema.py:551`) but is first-match-only and does not name it (`tables/schema.py:561`) | **yes** — naming costs nothing; `collect` is the one genuinely missing semantic, and two independent vendors (SAS, Pega) converge on exactly this binary |
| 3 | Decision tree / graphical tree | FICO, Experian, ACTICO, Decisions, Sparkling Logic | multi-way tree, leaves carry outcome rows | **yes** — v3 tree document (`trees/schema.py:1055`), leaf value table (`trees/schema.py:1006`) | n/a |
| 4 | **Node-level statistics in the authoring tool** | Experian PowerCurve Strategy Management, FICO (strategy design); **Pega** — confirmed, but scoped to Strategy shapes, not to the tables/trees/scorecards inside them: after a batch case run "a label displaying the test result appears at the top of each shape", showing volume, throughput and time-spent ([test-run labels](https://docs.pega.com/bundle/platform/page/platform/decision-management/test-run-labels.html)), but "Decision simulation tests do not support all components. This includes... **scorecards, Decision trees, Decision tables**" ([batch case runs](https://docs.pega.com/bundle/platform/page/platform/decision-management/configure-batch-case-runs.html)) | while editing, each node shows historical volume / response / rate, so an author sees the population effect before publishing | **no** — `TreeModule.explain()` reports emitted lines, leaves, depth, features (`trees/build.py:110-133`); nothing population-derived | **yes** — `example_projects/04-campaign-targeting-trees.md:531-560` specifies this at **tree-node** granularity, which is finer than any vendor in this survey documents (Pega's is the closest, and it explicitly stops one level up) |
| 5 | **Scorecard object** | FICO, **Pega** (predictors with weights, a Sum/Min/Max/Average combiner function, cutoffs mapped to results, an "Audit Notes" checkbox to capture scorecard detail in case history), **GDS Link** and **Zoot** (both name "scorecards" as an authoring primitive on their live product pages), Sparkling Logic, FlexRule. **Not** SAS Intelligent Decisioning itself — "scorecard" occurs zero times across eight checked ID documentation books; it lives one tier up in Model Studio's **Scorecard node**, gated by the Risk Modeling Add-On licence, with a published points-to-double-odds scaling law (`score = ln(odds) * factor + offset`, set via **Odds** and **Points to double odds** properties) and a "neutral score" per-characteristic adverse-contribution formula computed as a *training-time report* (`sub-sas-scorecards.md` §A–B). **Not** Alloy either — confirmed zero occurrences across 115 guide pages and 129 API-reference pages; Alloy scores via external/custom models or JQ expressions instead | characteristics → bins → points → score, with per-characteristic contributions | **no** — `scorecard.py` is a planned generic-kernel kind only (`docs/08-configuration-and-lifecycle.md:344`, `docs/02-architecture.md:704`, `docs/00-BUILD.md:214`); no module in `src/` | **yes, must-have** — `example_projects/00-shared-credit-core-library.md:279-288` makes contributions a *required output*, and 9 scorecards × 45 characteristics × 8 bins is a table, not code. SAS's factor/offset scaling law and its `actual_points − neutral_points` per-characteristic contribution formula are ready-made designs decider2 should copy rather than invent (recommendation 4) |
| 6 | Reason-code / adverse-action generation | Equifax ("regulatory-required reason code generation"), Scienaptic ("Reasons"), Zest ("Reason code stability"), **Alloy** — a typed, first-class `outcome_reasons[].type: "adverse_action"` field on every evaluation response, plus audited "reason code group" objects ([API schema](https://developer.alloy.com/public/reference/post_journeys-journey-token-applications.md)), but explicit that it stops at the code: "Alloy does not offer out-of-the-box solutions or customization for adverse action notices (AAN's)" ([FAQ](https://help.alloy.com/en/articles/16818551-does-alloy-provide-support-for-adverse-action-notices)). **Not** Pega — "adverse action" returns **zero** hits across Pega's own full-text documentation search. **Not** SAS's Intelligent Decisioning — SAS's only mechanism is the Model Studio Scorecard node's training-time neutral-score report (§1.1/row 5), and **no macro or PROC anywhere in the checked SAS documentation emits a reason code**. GDS Link and Zoot each mention "reason code"/"adverse action" exactly once — an unfinished draft comparison page (GDS Link) or an unelaborated package-feature bullet (Zoot) — with no mechanism documented | derive client-facing decline reasons from scorecard contributions or fired rules | **partial** — doc says "a reason code is a step output" (`docs/04-observability-and-governance.md:175`); `docs/REVIEW.md:673` refutes that at scale ("an ordered variable-length list, usually top-k … A step gives one scalar") | **yes, must-have** — no vendor in this survey ships a complete, documented, per-record reason-code mechanism derived from a rule or scorecard object; Alloy's typed field is the closest, and even it explicitly excludes notice generation |
| 7 | **Treatments / treatment groups** | SAS: "a set of attributes, eligibility rules, and effective dates"; attributes fixed or run-time dynamic, intended to carry "profitability, risk, cost, priority, or order"; the treatment group's output is a **data grid** arbitrated by "data grid functions to sort or subset the treatments" ([UG pp. 88-90](https://documentation.sas.com/api/collections/edmcdc/v_040/docsets/edmug/content/edmug.pdf?locale=en)); Pega actions with arbitration | a versioned, activated offer catalogue that the decision selects from and ranks | **no** | **maybe** — decider2's example projects express this as table outputs (`example_projects/08-collections-treatment.md:54`, a 5 376-cell treatment matrix), which a table already covers; the missing part is versioning/activation of the action catalogue |
| 8 | **Champion / challenger** | FICO, Equifax, ACTICO (dynamic routing), GDS Link, Zoot ("three orchestration types – Simple, Waterfall, Parallel" for comparing flows), Taktile, Oscilar, FlexRule, **Pega** (a named component: a live **random** per-call draw on a percentage split, with documented statistical drift — "70%... 30%... After 1000 runs... 69%... 31%. This deviation becomes smaller as the number of runs increases", [champion challenger](https://docs.pega.com/bundle/platform/page/platform/decision-management/champion-challenger-component.html)), **Alloy** (the most complete public spec in the survey: up to 5 concurrent versions, percentage allocations with a **minimum 5% per arm**, start/stop themselves audit-logged — also a per-call **probabilistic** draw, [champion challenger](https://help.alloy.com/en/articles/16820342-champion-challenger)). SAS claims it on marketing but "challenger" appears **nowhere** across eight checked Intelligent Decisioning books — it is a Model Manager "project champion" concept, re-confirmed on this pass | run two versions in parallel or on a traffic split, record both, compare | **no** — no occurrence in `src/` or `docs/` outside the landscape note | **yes, must-have** — required in detail by `example_projects/03-unsecured-loan-granting-and-pricing.md:365-371` (deterministic 10% by **stable hash** of `client_id`, share is a parameter, both answers recorded) and `example_projects/09-governance-and-replay-harness.md:973-977`. **Every vendor with a documented mechanism (Pega, Alloy) uses a live random/probabilistic draw, not a deterministic hash** — copy their percentage-allocation *shape* (named arms, a minimum share, a count cap) but not the randomness; decider2's requirement is stricter than any vendor's shipped design |
| 9 | **Lookup / reference tables** with activation | SAS: "tables of key-value pairs", CSV import and re-import "as needed to refresh", unique keys, `LOOKUP`/`LOOKUPVALUE` functions, key/value **labels**, versions, check-out/commit, activation ([UG pp. 116-140](https://documentation.sas.com/api/collections/edmcdc/v_040/docsets/edmug/content/edmug.pdf?locale=en)) — but **one key to one value**, not an N-dimensional grid; Sparkling Logic "lookup models from spreadsheets"; **Alloy Custom Lists** — a lighter, explicitly two-tier versioning scheme: a **major** version changes columns, a **minor** version only adds/deletes/modifies entries, and "the actual list version used in a realtime evaluation will be the latest minor version for the [activated] major version" ([custom lists](https://developer.alloy.com/public/reference/post_custom-lists-customlisttoken-versions.md)) | keyed reference data held outside the logic, versioned and lifecycle-managed | **partial** — a decision table's rows ride in `shared` and edit for free (`tables/__init__.py:19-21`), but pure keyed lookup is O4, still open, and was spelled six different ways by cold readers (`docs/06-open-questions-and-experiments.md:82-99`) | **yes** — 17 tables up to 219 600 cells with cell-level attribution required (`example_projects/00-shared-credit-core-library.md:537-565`); Alloy's major-changes-shape/minor-changes-values split is a simpler starting model than SAS's flat version-plus-activation scheme when O4 is finally spelled |
| 10 | **Data grids** (nested per-record collections) | SAS: `data grid` is one of seven **variable types** ("Boolean, character, data grid, date, datetime, decimal, and integer", [UG p. 25](https://documentation.sas.com/api/collections/edmcdc/v_040/docsets/edmug/content/edmug.pdf?locale=en)), iterable by a rule set (p. 71), with its own manual and function library — and it is what treatments are built on | a table inside one record, so a decision can iterate an applicant's existing loans | **no** — O5, "the single most-invented gap", nine invented names across six projects (`docs/06-open-questions-and-experiments.md:215-228`) | **yes, must-have** |
| 11 | Rule-fired / decision-path capture per record | SAS: a `ruleFiredFlags` column, a **Rule-Fired Analysis** view with per-record fired counts, drill-down and a per-rule bar chart, plus `%DCM_RULEFIRE_DETAIL` over subject contact history ([UG pp. 34, 71-72](https://documentation.sas.com/api/collections/edmcdc/v_040/docsets/edmug/content/edmug.pdf?locale=en)) — **fired only, not evaluated**; ACTICO ("visualisation of the full execution path", trace to Kafka); **Pega** — three mechanisms at three different costs: a rule-level **Tracer** for tables/trees/when-rules, a GOS-only **Execution Audit** for near-real-time production recording, and an **Explainability Extract** writing Parquet files at "5 GB of data for a million... inbound Decisions" with a 5-day default retention, still labelled "a Preview Release feature" ([explainability extract](https://docs.pega.com/bundle/customer-decision-hub/page/customer-decision-hub/cdh-portal/explainability-extract.html)); **Alloy** — a per-evaluation "Rule Explainability" view plus a typed event timeline (`visited_node`, `executed_branch_change`, `reached_auto_decision_node`, …); Corticon Rule Trace | which rules fired and which path was taken, per record | **partial** — a tree emits `<Name>_path` (`trees/build.py:71-74`), a table emits the matched row index or −1 (`tables/build.py:55-59`); nothing records *evaluated-but-not-fired*, which `example_projects/09-governance-and-replay-harness.md:1024-1028` makes a hard requirement | **yes** — Pega's own published cost figures for its richest option (5 GB/million decisions, 5-day retention, still Preview-Release) support decider2's cautious, verbosity-gated design (`docs/04-observability-and-governance.md:436-442`) rather than arguing against it |
| 12 | Simulation / back-test on historical data | FICO, Experian ("impact assessment and strategy simulation prior to strategy changes"), Equifax, Provenir, GDS Link, Taktile, Oscilar ("one-click backtesting"), **Pega** (Distribution test; Decision funnel — "up to six times the runtime of a basic distribution test", recommended sample sizes 100,000→500,000; Scenario Planner, explicitly disabled on production: "inactive on any environment that is designated a Level 5 Production environment"), **Alloy** (three named tools — historical replay against another version using *cached* third-party responses rather than live calls, a portfolio-snapshot "PE Backtesting", and a "What If Analysis" capped at 5,000 evaluations with documented limits such as "cannot add new data sources"). **Zoot names none of this** — a crawl of all 42 public pages for "simulat", "backtest", "back-test" and "replay" returned **zero hits**, a confirmed absence rather than an unreached page | run candidate logic over history and report what moves | **no** — `decider2.impact(active, candidate, sample)` is specified (`docs/08-configuration-and-lifecycle.md:613-655`) and absent from `src/`; `testing/` holds only `equivalence`, `corpus`, `recompile` | **yes, must-have** — and Zoot's confirmed gap is the sharpest evidence that even a decades-old, still-selling rules vendor ships without it; buying a platform does not guarantee this capability |
| 13 | Approval workflow / role-based sign-off | SAS: a predefined **SID Asset Approval** workflow, enabled by configuration, instantiated per decision *version* ([UG p. 284](https://documentation.sas.com/api/collections/edmcdc/v_040/docsets/edmug/content/edmug.pdf?locale=en)); ACTICO ("records approvals in an audit-proof manner"), Equifax, Oscilar, Taktile | a change cannot activate without a recorded approver | **no** — explicitly out of scope: "Approval is the caller's policy" (`docs/04-observability-and-governance.md:94`) | **maybe** — O17; the framework should at least carry an approver field on a staged plan (`runtime/serve.py:60-79`) |
| 14 | Artefact versioning + one-click rollback | SAS, ACTICO ("roll-backs without downtime"), Zoot, Alloy | version chains, promote, revert | **partial** — `structure_fingerprint` + `stage`/`activate`/`rollback` measured at 3.36 µs with zero recompiles (`runtime/serve.py:90`, `:253-317`; `docs/EXPERIMENTS.md:434-435`) — but values only | **yes** — extend to interiors when they land |
| 15 | Effective-dating of rules and table rows | SAS: **effective dates are part of a treatment's definition** ([UG p. 88](https://documentation.sas.com/api/collections/edmcdc/v_040/docsets/edmug/content/edmug.pdf?locale=en)); lookup-table activation; Alloy workflow versions | a row or rule valid between two dates, resolved against a business date | **no** — and `example_projects/09-...:980-983` forbids reading "today" | **yes** |
| 16 | One artefact published to many execution targets | SAS: four destination classes — a row in a CAS/Teradata/Hadoop model table, a Micro Analytic Service module, a **Git directory holding the generated code as `scoreResource.txt`**, or a SAS Container Runtime container pushed to AWS/Azure/GCP/private Docker; all become "callable REST API endpoints, independent of SAS Intelligent Decisioning" ([UG p. 60](https://documentation.sas.com/api/collections/edmcdc/v_040/docsets/edmug/content/edmug.pdf?locale=en)) | one decision, several runtimes | **partial** — `apply()` batch + `score()` single-record + ASGI app (`runtime/invoke.py:432`, `:649`; `serving/dispatch.py:137-145`); no streaming, no in-database | no — decider2's answer is stronger (row 17), and in-database/ESP are not in its problem statement |
| 17 | **Guaranteed equivalence between execution modes** | none. SAS states the opposite per content type: some model score-code types "can be published only to SAS Micro Analytic Service destinations" and a filtering rule set cannot be published to MAS on its own ([UG pp. 4, 18](https://documentation.sas.com/api/collections/edmcdc/v_040/docsets/edmug/content/edmug.pdf?locale=en)) | proof that batch and real-time give identical answers | **yes, and it is the differentiator** — `interpreted ≡ stepped ≡ fused` as a shipped assertion with rung attribution (`testing/equivalence.py:218`, `:50-57`; `runtime/modes.py:1-30`) | n/a |
| 18 | Embeddable in-process library | ACTICO ("native Java API allows direct embedding"), InRule (.NET SDK) | no server in the call path | **yes** — `score(dict)` with no polars and no HTTP (`docs/02-architecture.md:485-510`) | n/a |
| 19 | Container / OCI packaging | SAS SCR, ACTICO (K8s), IBM, Decisions | ship a runtime image | **partial** — `docker/` exists at repo root; no decider2 image spec, and compilation is specified to happen at image build (`docs/02-architecture.md:428`) | **maybe** |
| 20 | Code generation to a second language | InRule (rules → JavaScript for offline execution), jDMN (DMN → Java/Python) | run the same logic where Python cannot | **no** | no |
| 21 | DMN import / export | FICO (states DMN); Trisotech, Drools, Camunda in the wider field | interchange with third-party modelling tools | **no** | **maybe** — landscape §9.4 argues CL2 decision tables only |
| 22 | PMML / ONNX model import | GDS Link, Pega (PMML + H2O) | execute a trained model inside a decision | **no** | **maybe** — a Python step already covers it, at the cost of a kernel split |
| 23 | Arbitrary code node inside a flow | SAS (Python/SAS/DS2 code files), Taktile, Alloy, GDS Link | escape hatch for anything the primitives cannot express | **yes, and better** — steps *are* Python, and an un-compilable step splits the kernel rather than failing (`docs/05-boundary-and-compilation.md:522-561`) | n/a |
| 24 | Natural-language / AI rule authoring | Oscilar, Provenir, ACTICO, Taktile | describe logic in prose, tool emits it | **no** | no — not a gap worth closing before the reviewable artefact (`docs/04-...:277`) |
| 25 | Conflict / redundancy / unreachable-branch analysis | ACTICO ("redundant rules, conflicts and unreachable branches"), Corticon (completeness checker) | static detection of dead or overlapping logic | **partial** — per-table contiguity/gap validation (`tables/schema.py:376-383`) and per-node sorted/contiguous range validation (`trees/schema.py:882-910`); no cross-rule or reachability analysis | **yes** |
| 26 | Visual rule-trace viewer | Corticon (Rule Trace Viewer), ACTICO | a rendered execution path for one record | **no** — `.debug()`, `.trace`, `.render()` are doc-only; grep finds none in `src/` | **yes** |
| 27 | Trace published to an external stream | ACTICO (Kafka) | per-decision trace leaves the engine as events | **no** | **maybe** — and note `docs/04-observability-and-governance.md:262-276`: per-record diagnostics are PII by construction |
| 28 | Per-decision audit record as structured data | Pega ("records what it chose, what it rejected, and why"), DecisionRules (audit logs), Scienaptic | a stored, queryable record of the decision | **no** — `observe/record.py` is the planned "ONLY guaranteed output" (`docs/02-architecture.md:726-731`); the contents are specified (`docs/04-...:239-261`) | **yes, must-have** |
| 29 | Plain-English explanation of a decision | Provenir ("plain-English explanations of any rules- or model-based decision") | prose rendering for a non-technical reader | **no** | **maybe** — decider2 correctly treats renderings as replaceable (`docs/04-...:415-445`); this is one renderer |
| 30 | Published latency / throughput figures | SAS (5–10 ms, >7 000 tps), Oscilar (120K rps, <100 ms), Pega (<200 ms), Provenir ("millisecond"), ACTICO ("thousands per second"), Scienaptic (<1 s) | a number a buyer can hold you to | **yes, and measured rather than claimed** — kernel 1.38 µs, single-record framework overhead 971 µs p50 (4.86% of a 20 ms budget) reducible to 0.73% (`docs/EXPERIMENTS.md:684-712`, `docs/README.md:29-36`) | n/a |
| 31 | Sandbox / non-production environment separate from production | Alloy (sandbox vs production), SAS (test scenarios) | try a change against real traffic shape without affecting decisions | **partial** — `preview(record, proposed)` scores one record against active and proposed generations side by side (`runtime/serve.py:332-351`) | **yes** — extend `preview` from one record to a sample (this is row 12) |
| 32 | Scenario tests with expected values | SAS: per-object tests comparing actual against expected, highlighting mismatches, CSV export ([UG pp. 63-77, 354-376](https://documentation.sas.com/api/collections/edmcdc/v_040/docsets/edmug/content/edmug.pdf?locale=en)); Sapiens "auto-generated test cases" | the tool writes and runs the regression suite for the artefact | **partial** — `testing/corpus.py` generates a boundary-value corpus per declared input | **yes** — extend to table rows and tree leaves (coverage, not just boundaries) |
| 33 | **Generated PDF documentation per artefact** | SAS generates PDF documentation for a rule set, treatment group, lookup table, value list and decision, plus a report for a published decision ([UG pp. 46, 101, 125, 226, 327-328](https://documentation.sas.com/api/collections/edmcdc/v_040/docsets/edmug/content/edmug.pdf?locale=en)) | a printable artefact a reviewer can be handed | **no** — `render()` does not exist in `src/`; two reviewer tests failed (`docs/04-observability-and-governance.md:277-340`) | **yes** — this is decider2's top-ranked risk and SAS ships an answer of *some* quality; `observe/render/` is the planned home (`docs/02-architecture.md:727-731`) |
| 34 | **Reverse dependency query** ("which objects use this?") | SAS has "Determine Which Objects Use a …" for every object kind (rule set, treatment, lookup table, code file, global variable, decision) | before changing shared logic, see what breaks | **partial** — static lineage is specified (`docs/04-...:121-150`) and `graph/lineage.py` does not exist; `pipeline.versions()` (`src/decider2/graph/pipeline.py:74`) is the nearest built thing | **yes** — cheap, and it is what makes a shared module safe to edit |
| 35 | **Content diff *and* generated-code diff between versions** | SAS offers "Compare … Content" and "Compare … Code" for rule sets, treatment groups, lookup tables, value lists, segmentation trees, code files and decisions | see both the authored change and its compiled consequence | **partial** — "A config diff is an audit record" is asserted (`docs/04-...:255-258`) with no implementation; generated source is real files (`docs/05-boundary-and-compilation.md:340`) so a code diff is available for free | **yes** — S effort, high value for the change-class story |
| 36 | **Shared/common rule fragment included by reference** | SAS *common rule sets*: "Any change to a common rule set affects all assignment rule sets that include the common rule set" ([UG p. 15](https://documentation.sas.com/api/collections/edmcdc/v_040/docsets/edmug/content/edmug.pdf?locale=en)) | reuse rules without copying them | **yes, and better** — a module is the unit of reuse, with `.bind()`/`.rename()`/`.relabel()` (`src/decider2/graph/module.py:201-239`) and blast radius bounded by namespaced params (`docs/04-...:57-63`) | n/a |
| 37 | **Value lists bound to a variable** | SAS value lists are a versioned object; a variable associated with one drives automatic branch generation in a segmentation tree, and an `Other` branch is added for unlisted values ([UG pp. 218-240](https://documentation.sas.com/api/collections/edmcdc/v_040/docsets/edmug/content/edmug.pdf?locale=en)) | the domain of a categorical variable is data, and the tree is generated from it | **no** — string literals are hoisted to int32 codes per distinct literal, and a *new* literal recompiles (`src/decider2/tables/__init__.py:26-30`) | **maybe** — the automatic `other` branch is the part worth copying, because an unhandled category is a live defect class |
| 38 | **Matrix as a tree branch element** | SAS: a matrix is "a table in which each cell identifies a different combination of the values of two different variables", both bound to value lists, `other` row and column added, one branch per distinct result string ([UG pp. 239-240](https://documentation.sas.com/api/collections/edmcdc/v_040/docsets/edmug/content/edmug.pdf?locale=en)); Temenos calls the same thing "matrices" | collapse a 2-D segmentation into a grid an analyst can read, then branch on named cells | **partial** — a `DecisionTable` *is* the grid, but it cannot be a tree node; `Branch` is unbuilt (`docs/03-authoring-api.md:1202`) | **yes** — this is how a 5 376-cell treatment matrix stays reviewable (`example_projects/08-collections-treatment.md:54`) |
| 39 | **Filtering rule set / row-level routing out of a flow** | SAS filtering rule sets: "Only the records for which the conditions evaluate to True are processed by the remaining objects in the decision" ([UG p. 15](https://documentation.sas.com/api/collections/edmcdc/v_040/docsets/edmug/content/edmug.pdf?locale=en)); Alloy step-up and manual-review nodes | drop or divert a record mid-flow without failing the batch | **no** — `route_required_nulls` routes null-violating rows (`src/decider2/boundary/nulls.py:311`) but there is no general row-level divert, and `docs/REVIEW.md:682-684` names this: "one bad row raises for the whole batch … No row-level rejection, quarantine or route-to-manual-review, which every production batch needs" | **yes, must-have** |
| 40 | Off-the-shelf embedded expression language instead of a bespoke DSL | Alloy uses **JQ**, the open-source JSON query language, verbatim as its rule/attribute expression syntax rather than inventing one: "JQ is an open-source language for querying and transforming JSON... The JQ filter is a complete expression run against the integration's raw API response" ([attribute tool](https://developer.alloy.com/public/docs/attribute-tool.md)), e.g. `if .riskScore > 700 then "high risk" elif .riskScore > 500 then "medium risk" else "low risk" end` | reuse a well-known, independently-documented language instead of a proprietary one, trading some reviewability for zero design cost | **no, and by design** — `tables/schema.py:246-546` deliberately restricts conditions to `between`/`in`/`is_true`/`eq` under `and`/`or`, a closed set chosen for reviewability, not an open expression grammar | **no** — a general query language is the opposite direction from decider2's stated top risk (reviewer verification, `docs/04-observability-and-governance.md:277-340`); noted here as the one vendor design this report actively recommends *against* following |
| 41 | Runtime fusion of several rule/model artefacts into one execution unit for speed | **Pega Globally Optimized Strategies (GOS)** — "brings several types of decisioning artifacts into a single program and runs them all together. Because Rules run together rather than one by one, overall Strategy run performance greatly increases", compiled via an internal "Single Static Assignment" step ([GOS](https://docs.pega.com/bundle/platform/page/platform/decision-management/globally-optimized-strategies.html)) — but Pega publishes **no equivalence guarantee** between GOS and non-GOS execution of the same strategy | combine multiple authored artefacts into one compiled/interpreted unit rather than calling each separately, for latency | **yes, and proven equivalent** — `fused` mode compiles multiple steps into one kernel (`runtime/modes.py:1-30`), with `interpreted ≡ stepped ≡ fused ≡ score()` asserted and rung-localised (`testing/equivalence.py:200-218`) | n/a — decider2's version is strictly stronger than the one documented vendor analogue, because it is proven equal, not merely faster |

---

## 3. Improvements for decider2 — ranked, de-duplicated

Ranked by **cost of late discovery**, decider2's own ordering principle
(`docs/REVIEW.md` §1), not by how many vendors ship the feature. Effort is S (days),
M (1–3 weeks), L (a month or more) for one engineer. Section 4 turns each of these
into a concrete change; this section is the *why*, with the platform that
demonstrates the better pattern.

| # | Gap in decider2 | Demonstrated better by | Maps onto | Effort | Bank |
|---|---|---|---|---|---|
| 1 | **No scorecard kind.** `docs/08-configuration-and-lifecycle.md:344` classifies one ("bins → points, uniform" → generic kernel, free interior change); `src/decider2/` has none | FICO lists "Scorecards" as a platform primitive ([decisioning](https://www.fico.com/en/platform/decisioning)); Pega — predictors, weights, Sum/Min/Max/Average combiner functions, cutoffs mapped to results ([scorecard main](https://community.pega.com/sites/pdn.pega.com/files/help_v84/rule-/rule-decision-/rule-decision-scorecard/main.htm)); GDS Link and Zoot both name scorecards on their live product pages; Sparkling Logic; FlexRule. **Not** SAS Intelligent Decisioning itself — the primitive lives one tier up, in Model Studio, with a published points-to-double-odds scaling law and a per-characteristic "neutral score" contribution formula computed at training time (`sub-sas-scorecards.md` §A–B) | `interiors/scorecard.py` (`docs/02-architecture.md:704`), templated on `tables/` | M | **must** — contributions are a *required output*, not a diagnostic (`example_projects/00-shared-credit-core-library.md:285-288`), and computing them **per record** is a genuine improvement on SAS's own training-time-only report |
| 2 | **No champion/challenger.** Zero occurrences in `src/`; three mock projects invented three mechanisms — `Module.when(pred)` (`example_projects/examples/03-.../pipelines/flex_loan_granting.py:111`, and `.when` does not exist), a `variant_index` step (`examples/04-.../modules/holdout/__init__.py:53-60`), and a full `ExperimentRegistry` (`examples/08-.../cohorts/assignment.py:45-70`). Three spellings is decider2's own evidence standard for a missing primitive (`docs/06-open-questions-and-experiments.md:95-99`) | ACTICO "simultaneous hosting of multiple model versions (dynamic routing, e.g. for champion/challenger)" ([model mgmt](https://www.actico.com/platform/model-management/)); SAS; FICO; Equifax; GDS Link; Zoot; Taktile; Oscilar; **Pega** — a random per-call draw on a percentage split, with documented statistical drift ("70%/30%... after 1000 runs... 69%/31%"); **Alloy** — the most complete public spec (up to 5 versions, minimum 5% per arm), also a per-call probabilistic draw | a combinator beside `Branch` in the planned `graph/combinators.py`; the arm index is already the `_path` value shape (`docs/03-authoring-api.md:1132-1142`) | M | **must** — specified in detail at `example_projects/03-unsecured-loan-granting-and-pricing.md:365-371` (stable hash of `client_id`, never an RNG) — **stricter than every vendor's shipped mechanism**, all of which randomise per call rather than hash deterministically |
| 3 | **No simulation on a sample.** `decider2.impact()` is fully specified (`docs/08-configuration-and-lifecycle.md:609-655`), including exact boundary intervals rather than sampled counts, and absent from `src/`; `ServeHandle.preview()` does it for one record (`runtime/serve.py:332-351`) | Experian "impact assessment and strategy simulation prior to strategy changes" ([DaaS](https://www.experian.co.uk/business-products/decisioning-as-a-service/)); FICO "end-to-end business outcome simulations"; Oscilar "one-click backtesting"; Taktile | `testing/impact.py` (`docs/02-architecture.md:741`) | M | **must** — without it doc 04 §2 cannot claim a param change is reviewable |
| 4 | **No decision record.** `observe/` does not exist; `observe/record.py` is described as "the ONLY guaranteed output" (`docs/02-architecture.md:726-731`) with contents already enumerated (`docs/04-observability-and-governance.md:239-261`) | Pega "Every decision records what it chose, what it rejected, and why" ([decision hub](https://www.pega.com/products/decision-hub)); SAS rule-fired data with detailed auditing; DecisionRules audit logs; Alloy audit endpoints | `observe/record.py` + `observe/render/` | L | **must** |
| 5 | **Only the taken path is captured.** `<Name>_path` for a tree (`trees/build.py:71-74`), matched row or −1 for a table (`tables/build.py:55-59`); nothing records evaluated-and-not-fired | SAS rule-fired data; ACTICO "visualisation of the full execution path"; Progress Corticon Rule Trace Viewer | a verbosity level on `tables/codegen.py` and `trees/codegen.py`, per `docs/04-...:436-442` | S–M | **must** — `example_projects/09-governance-and-replay-harness.md:1024-1028`: without it *"'why did rule 212 not fire' is unanswerable"* |
| 6 | **No node-level statistics.** `TreeModule.explain()` reports emitted lines, leaves, depth, features (`trees/build.py:110-133`) — structure only | Experian PowerCurve Strategy Management's "assisted strategy design tool" and Design Studio ([strategy mgmt](https://www.experian.co.uk/business/customer-insights/strategy-management)) — marketing-level only, see §6 | a polars group-by over `Tree.children()` (`trees/schema.py:1095`); no kernel change | S–M | nice-to-have, but `example_projects/04-campaign-targeting-trees.md:531-560` specifies it precisely (1 340 of 9 200 nodes dead; volume movement attributable to drift vs version vs overlay) |
| 7 | **No named hit policy; no `collect`.** `DecisionTable` is first-match-only and does not say so in the industry's words (`tables/schema.py:551-563`) | the whole field agrees on the vocabulary: DMN Unique/Any/Priority/First/Collect, DecisionRules `FIRST_MATCH`/`EVALUATE_ALL`, AWS `FIRST_MATCHED`/`ALL_MATCHED`, JDM first/collect | one field on `DecisionTable`; a second kernel body in `tables/codegen.py` | S | nice-to-have — except that `collect` is what a decline taxonomy needs |
| 8 | **No ragged output (O5).** `docs/04-...:175` claims "reason codes need no new machinery"; `docs/REVIEW.md:673-677` refutes it — *"an ordered variable-length list, usually reported top-k. A step gives one scalar of fixed type."* Nine invented names across six mock projects (`docs/06-...:215-228`) | SAS **data grids** are the nearest named vendor analogue; Equifax states "regulatory-required reason code generation" as a feature; Alloy ships a typed `outcome_reasons[].type: "adverse_action"` field but no vendor in this survey — including SAS's own scorecard and Pega, which returns zero doc-search hits for "adverse action" — documents a complete per-record mechanism from a rule or scorecard object | undecided; blocks items 1 and 6 | L | **must** |
| 9 | **No effective-dating.** Nothing in `tables/schema.py` or `trees/schema.py` carries a validity interval | SAS lookup-table "activation and locking" is the nearest vendor mechanism; the rest of the field treats it as the customer's problem | reserved `valid_from`/`valid_to` on `ParametersConfig`, resolved before the kernel so it stays date-free | M | **must** — `example_projects/00-...:541-546` and `09-...:980-983` ("no reliance on today") |
| 10 | **Keyed lookup still has no spelling (O4).** Seven of eleven mock projects needed one and spelled it six ways (`docs/06-...:82-99`) | SAS lookup tables as an object distinct from rule sets; Sparkling Logic "lookup models from spreadsheets" — the spreadsheet path matters, since decider2's rate cards *are* spreadsheets (`example_projects/00-...:546`) | `params/tables.py`, "ONE spelling" (`docs/02-architecture.md:709`) | M | **must** |
| 11 | **No cross-rule conflict or reachability analysis.** Per-table gap/contiguity (`tables/schema.py:376-383`) and per-node range order (`trees/schema.py:882-910`) only | ACTICO's agent finds "redundant rules, conflicts and unreachable branches"; Corticon ships a completeness checker | `graph/lineage.py` plus a per-kind analyser; unreachable-leaf detection is a graph walk | M | nice-to-have |
| 12 | **No approver on a staged change.** `StagePlan` carries `klass`/`recompiles`/`fingerprint`/`eta`/`doc` (`runtime/serve.py:60-79`) and no identity | SAS "role-based approvals and governance workflows"; ACTICO "records approvals in an audit-proof manner"; Taktile "Review and sign-off"; Oscilar "human-in-the-loop approvals" | one optional field, carried not verified — the posture `origin` already has (`docs/04-...:90`) | S | nice-to-have; O17 |
| 13 | **No container story.** Compile-at-image-build is specified (`docs/02-architecture.md:428`) and `--verify` asserts zero runtime compiles (`docs/05-...:671`), but there is no decider2 Dockerfile | SAS Container Runtime ("OCI-compliant containers"); ACTICO on Kubernetes; IBM's published ODM container repo | `decider2/docker/` + a CI gate | S | nice-to-have |
| 14 | **No interchange format.** | FICO is the only §1 vendor naming DMN ([decisioning](https://www.fico.com/en/platform/decisioning)) | a CL2 table importer mapping S-FEEL unary tests onto the existing expression classes (`tables/schema.py:300-546`) | M | nice-to-have — buys migration-in, not capability |
| 15 | **No row-level divert.** `docs/REVIEW.md:682-684`: the `required` null policy is all-or-nothing — "one bad row raises for the whole batch. No row-level rejection, quarantine or route-to-manual-review, which every production batch needs". `route_required_nulls` (`boundary/nulls.py:311`) routes but has no destination concept | SAS **filtering rule sets** — "Only the records for which the conditions evaluate to True are processed by the remaining objects in the decision" ([UG p. 15](https://documentation.sas.com/api/collections/edmcdc/v_040/docsets/edmug/content/edmug.pdf?locale=en)); Alloy's Step-Up and manual-review nodes ([journeys](https://developer.alloy.com/public/docs/journeys-overview.md)) | a declared divert output beside the main frame, in `graph/pipeline.py` + `boundary/nulls.py` | M | **must** |
| 16 | **No reverse-dependency query and no artefact diff.** "A config diff is an audit record" is asserted (`docs/04-...:255-258`) with nothing implementing it; `graph/lineage.py` does not exist | SAS has "Determine Which Objects Use a …" for every object kind, plus **Compare … Content** *and* **Compare … Code** between versions | `graph/lineage.py`; generated source is already real files, so a code diff is nearly free (`docs/05-boundary-and-compilation.md:340`) | S | nice-to-have, but it is what makes editing a shared module safe |

**Two things decider2 already does better than every platform in §1, and should not
trade away.** (i) The **four-way equivalence assertion** — `interpreted ≡ stepped ≡
fused ≡ score()` as a shipped, reusable function that localises a divergence to the
rung that introduced it (`src/decider2/testing/equivalence.py:200-216`, `:218`). No
vendor page found states any equality guarantee between its batch and real-time
engines, and SAS publishes to five destinations without one. (ii) **Free
reference-data edits** — a decision table's rows live in the `shared` bundle, so a row
change recompiles nothing, not even a staged compile
(`src/decider2/tables/__init__.py:19-21`), which is stronger than SAS's "activation
and locking" and matters at 63 360 and 219 600 cells refreshed monthly
(`example_projects/00-shared-credit-core-library.md:546-549`).

**Explicitly not recommended.** Natural-language rule authoring (Oscilar, Provenir,
ACTICO, Taktile): decider2's top risk is that a reviewer cannot verify a rule *a human
wrote* (`docs/04-observability-and-governance.md:277-340`), and generating rules from
prose makes that worse. In-database and streaming publication (SAS CAS/ESP): outside
the stated problem. Code generation to JavaScript (InRule): no stated need.


## 4. Actionable recommendations for decider2

Each item is a change someone could start on Monday. Nothing here says "improve
governance". Where an item depends on another, that is stated.

1. **Add `hit_policy: Literal["first", "collect"] = "first"` to `DecisionTable`**
   (`src/decider2/tables/schema.py:566-570`) so the table's semantics are stated in
   the field name the rest of the industry uses, and update the class docstring's
   "First match wins" (`tables/schema.py:561`) to name it. Emit the `collect` shape
   as a second kernel body in `tables/codegen.py` that writes a per-row hit bitmask
   instead of a row index. **Effort S** (the `first` half is naming only).
   *Motivated by:* DMN Unique/Any/Priority/First/Collect, DecisionRules
   `FIRST_MATCH`/`EVALUATE_ALL`, AWS `FIRST_MATCHED`/`ALL_MATCHED`.

2. **Emit an evaluated-conditions mask from both existing kinds.** In
   `tables/codegen.py`, alongside the matched-row index, write one `uint64` per row
   whose bits mark which DNF groups were evaluated; in `trees/codegen.py`, write the
   node ids visited (path already exists — add the *failed* sibling tests). Expose
   them as `TableModule.evaluated_column` / `TreeModule.evaluated_column` beside the
   existing `row_column` (`tables/build.py:55-59`) and `path_column`
   (`trees/build.py:71-74`), under a verbosity flag per
   `docs/04-observability-and-governance.md:436-442`. **Effort M.**
   *Motivated by:* SAS rule-fired data; ACTICO full-execution-path visualisation.
   *Required by:* `example_projects/09-governance-and-replay-harness.md:1024-1028`.

3. **Give a table row a stable id, and stop accepting id-less inner nodes.**
   Tree *graph* identity is already correct — `PositionedNode.id` is required and
   duplicate-checked (`src/decider2/trees/schema.py:976`, `:1079-1082`). Two gaps
   remain: (a) `ParametersConfig.data` rows have no identifier at all
   (`src/decider2/tables/schema.py:159`), so a table's matched-row index is
   positional and renumbers when a row is inserted; (b) the inner node/condition
   `id` is still optional (`LeafNode:658`, `UnaryNode:697`, `CasesRanges:744`,
   `CasesStringMatch:779`, `CasesIsIn:814`, `CompositeNode:842`,
   `CompositeCondition:585`). Add a required `id` per table row and derive the
   emitted row identity from it rather than from position, because
   `example_projects/09-...:964-972` makes stable, non-positional element identity
   the single most common way an audit trail is destroyed, and
   `docs/08-configuration-and-lifecycle.md:248-253` already records decider 1's
   `uuid.uuid4()` default as a defect to fix in the port. **Effort S.**
   *Blocks:* items 2, 6 and 9 — none of them work with unstable ids.

4. **Write `src/decider2/scorecards/schema.py` + `codegen.py` + `build.py`**, mirroring
   `tables/` exactly: a `Scorecard` document of `characteristics: list[Characteristic]`
   where a `Characteristic` is `{name, feature, bins: list[Bin], neutral_points}` and a
   `Bin` is `{min, max | values | is_null, points}`; a generic row-scan kernel whose
   bins live in `shared` so a points edit recompiles nothing (the property
   `tables/__init__.py:19-21` already delivers); outputs `score` **plus one
   contribution column per characteristic**. Register it as the third generic-kernel
   kind promised by `docs/08-configuration-and-lifecycle.md:344`. **Effort M** (the
   table machinery is a direct template; the contributions output is the hard part
   and may need item 5 first).
   *Motivated by:* FICO, Pega (predictors/weights/combiner functions), GDS Link,
   Zoot, Sparkling Logic, FlexRule all list scorecards as a primitive. *Required by:*
   `example_projects/00-shared-credit-core-library.md:279-288`.
   Note `Bin.is_null` is not optional: *"Null is its own bin, always"*
   (`example_projects/03-unsecured-loan-granting-and-pricing.md:329`).
   **Two SAS mechanics to copy outright, not reinvent** (`sub-sas-scorecards.md`
   §A–B, §1.1): (a) the industry's own points-to-double-odds scaling law —
   `score = ln(odds) * factor + offset`, with `factor`/`offset` derived from two
   author-facing properties, **Odds** and **Points to double odds** — as an optional
   named scaling on `Scorecard` rather than a raw, unscaled `points` sum; (b) the
   `neutral_points` field this recommendation already carries is exactly SAS's
   "neutral score" — "the score points for an attribute when the attribute's WOE is
   equal to 0" — so the per-characteristic contribution to compute is
   `actual_points(characteristic) − neutral_points(characteristic)`, ranked
   worst-first; SAS only ever computes this as a training-time aggregate, so
   emitting it **per record** (already this recommendation's design) is a genuine
   improvement on the one shipping precedent, not merely a copy of it.
   *See also recommendation 20 (PMML Scorecard import) for a second interchange path
   into the same object.*

5. **Decide and document the ragged-output convention (O5) in
   `docs/06-open-questions-and-experiments.md:215` before writing item 4's
   contributions or any reason-code mechanism.** Concretely: pick between (a) a
   fixed-width contribution matrix with a declared maximum characteristic count —
   which works in numba today and matches the dtype-grouped 2D output convention of
   `docs/05-boundary-and-compilation.md:241-275` — and (b) a genuine offsets+values
   pair. Write the choice into doc 05 §3.1 and delete the claim at
   `docs/04-observability-and-governance.md:175` that "reason codes need no new
   machinery", which `docs/REVIEW.md:673-677` has already refuted. **Effort S** for
   the decision, **L** for (b).

6. **Add `Experiment` / `assignment` to the graph layer** as
   `src/decider2/experiments/schema.py` + a `variant()` step factory, taking the
   schema a mock project already designed:
   `Experiment(id, name, salt, arms={name: pct}, eligible={field: values}, opens, closes, measures, approval_reference)`
   with an `ExperimentRegistry(max_concurrent, max_per_account, assignment_order, interaction_recorded)`
   (`example_projects/examples/08-collections-treatment/cohorts/assignment.py:45-70`).
   The assignment step must be `hash(stable_id + salt) % 100` — never `random` — and
   must emit the arm index as an ordinary value so it lands in the output frame like
   any `_path` (`docs/03-authoring-api.md:1132-1142`). Add a lint in
   `src/decider2/lint.py` that fails a build if any step imports `random` or calls
   `np.random`. **Effort M** (the lint is S and worth doing on its own).
   *Motivated by:* SAS, FICO, ACTICO dynamic routing, Equifax, GDS Link, Zoot,
   Taktile, Oscilar. *Required by:* `example_projects/03-...:365-371`.
   *Shape to copy, mechanism to reject:* Alloy's `arms={name: pct}` design is close
   to this schema already — "up to 5 versions... percentage allocations (at least
   5%) applied to each version"
   ([champion challenger](https://help.alloy.com/en/articles/16820342-champion-challenger))
   — so borrow the named-arms-plus-minimum-share shape (e.g. validate `min(pct.values())
   >= 5` on `Experiment.arms`) but not the assignment mechanism: both Alloy and Pega
   assign with a **live random/probabilistic draw per call**, and Pega's own docs
   show the resulting drift ("70%/30%... after 1000 runs... 69%/31%"). decider2's
   `hash(stable_id + salt)` requirement is strictly stricter than either shipping
   vendor design, because it is reproducible on replay — which a probabilistic draw
   is not.

7. **Implement `decider2.impact(active, candidate, sample) -> ImpactReport` in
   `src/decider2/testing/impact.py`** by generalising `ServeHandle.preview`
   (`src/decider2/runtime/serve.py:332-351`) from one `record` to a
   `pl.DataFrame`, returning: the fraction of rows whose declared terminals changed,
   the per-output distribution of the change, and which table rows / tree leaves
   newly matched or stopped matching (available from the `row_column` and
   `path_column` diffs). Leave the exact boundary-interval solve
   (`docs/08-configuration-and-lifecycle.md:626-641`, open as O24) for a second pass
   and have the report *say* it sampled. **Effort M.**
   *Motivated by:* Experian strategy simulation, FICO outcome simulation, Oscilar
   one-click backtesting, Taktile pre-live modelling.

8. **Add `POST /impact` to the serving route table** (`src/decider2/serving/dispatch.py:137-145`)
   next to the existing `/params/preview`, taking a proposed params document plus a
   sample reference and returning item 7's report. **Effort S** once item 7 exists.
   *Motivated by:* every vendor in the set makes simulation a UI affordance, not a
   library call.

9. **Add a `tree_node_stats(frame, tree, path_column)` function to a new
   `src/decider2/trees/stats.py`** that, given an output frame carrying the emitted
   `_path` column and the `Tree` document, returns per-node entry counts, per-edge
   exit counts, per-leaf counts, and the set of nodes with zero traffic. This is a
   `polars` group-by over `Tree.children()` (`src/decider2/trees/schema.py:1095`) — no
   kernel change. **Effort S.**
   *Motivated by:* Experian PowerCurve Strategy Management's node-level design aids.
   *Required by:* `example_projects/04-campaign-targeting-trees.md:535-541`
   (dead-branch identification: 1 340 of 9 200 nodes).

10. **Add `valid_from` / `valid_to` as reserved columns on `ParametersConfig`**
    (`src/decider2/tables/schema.py:152-197`) and resolve them in
    `src/decider2/tables/build.py` *before* the kernel is called, against an explicit
    `as_of` argument — so the compiled code never sees a date and `decision_date`
    stays an input, not an ambient read. Document the rule in
    `docs/08-configuration-and-lifecycle.md` §3.4. **Effort M.**
    *Motivated by:* SAS lookup-table activation. *Required by:*
    `example_projects/00-shared-credit-core-library.md:541-546` and
    `example_projects/09-...:980-983`.

11. **Add an unreachable-node check to `Tree`'s validator**
    (`src/decider2/trees/schema.py:1077-1093`): walk from `root_id()` over
    `children()` and raise if any declared node is not reachable, and if any
    `sourceIndex` in `0..arity-1` has no outgoing edge. Today the validator checks
    duplicate ids and dangling edge endpoints only. **Effort S.**
    *Motivated by:* ACTICO's "redundant rules, conflicts and unreachable branches";
    Corticon's completeness checker.

12. **Carry an approver on a staged change.** Add
    `approval: str | None` and `approved_by: str | None` to `StagePlan`
    (`src/decider2/runtime/serve.py:60-79`) and thread them through
    `Dispatcher.post_params` (`src/decider2/serving/dispatch.py:81`, routed at `:142`). Record, do not
    verify — the same posture `origin` already has
    (`docs/04-observability-and-governance.md:90`). **Effort S.**
    *Motivated by:* SAS role-based approvals; ACTICO audit-proof approvals; Taktile
    review-and-sign-off.

13. **Write `src/decider2/observe/record.py`** emitting the audit record whose
    contents `docs/04-observability-and-governance.md:239-261` already enumerates —
    pipeline identity (`structure_fingerprint`, `src/decider2/runtime/serve.py:90`),
    resolved params with `origin`, framework/module versions, inputs as received,
    outputs, emitted intermediates — plus the two items only decider2's own specs
    demand: the evaluated-conditions mask from item 2 and the cap chain
    (`example_projects/09-...:1030-1033`). Ship exactly one renderer in
    `observe/render/rule_sheet.py`, modelled on
    `example_projects/examples/01-transaction-fraud-interdiction/fraud_interdiction/artefacts/rule-sheet-MS-0208.md`,
    and state in the module docstring that it is replaceable
    (`docs/04-...:415-445`). **Effort L.**
    *Motivated by:* Pega's "what it chose, what it rejected, and why"; SAS auditing;
    DecisionRules audit logs.

14. **Add a `decider2/docker/Dockerfile` that runs `decider2 build --verify` as a
    build step** so the "zero runtime compilations" guarantee
    (`docs/05-boundary-and-compilation.md:671`) is asserted in the image rather than
    in a developer's shell, and record the resulting artefact id in the image
    labels. **Effort S.**
    *Motivated by:* SAS Container Runtime; ACTICO on Kubernetes; IBM's published
    ODM container repo.

15. **Add a DMN CL2 decision-table importer as `src/decider2/tables/dmn.py`** mapping
    S-FEEL unary tests onto the existing expression classes — a range onto
    `BetweenExpression` (`src/decider2/tables/schema.py:300`), a list onto
    `InExpression` (`:403`), a literal onto `EqExpression` (`:489`), `-` onto
    "no condition" — and rejecting anything outside CL2 with a message naming what
    it found. Do **not** add FEEL. **Effort M.**
    *Motivated by:* FICO is the only §3 vendor naming DMN
    ([platform/decisioning](https://www.fico.com/en/platform/decisioning)); the
    landscape note §9 item 4 argument that CL2 covers the exporters that matter.

16. **Correct three doc claims that the code has overtaken**, so the doc set stops
    over-promising: (a) `docs/04-observability-and-governance.md:175` on reason codes
    (see item 5); (b) `docs/06-open-questions-and-experiments.md:447-456` (E3) and
    `example_projects/examples/FINDINGS.md:64-81` both treat "`score()` agrees with
    `apply()`" as a missing fourth rung — it is now implemented and asserted
    (`src/decider2/testing/equivalence.py:200-216`), and that is decider2's single
    strongest differentiator against every platform in §1, so it should be stated as
    done; (c) `docs/03-authoring-api.md:720-748`'s `Table` sketch, now superseded by
    `decider2.tables` — replace the sketch rather than annotate it. **Effort S.**

17. **Add a declared divert output to `Pipeline`.** Give `flow(...)` a second return
    channel — `apply()` returns `(kept, diverted)` when any step or null policy marks a
    row — implemented in `src/decider2/graph/pipeline.py:188` and
    `src/decider2/boundary/nulls.py:311` (which already computes the routing mask and
    has nowhere to send it). Every diverted row carries the reason. **Effort M.**
    *Motivated by:* SAS filtering rule sets ("Only the records for which the conditions
    evaluate to True are processed by the remaining objects in the decision", UG p. 15);
    Alloy's Step-Up / manual-review nodes. *Named as a defect by:* `docs/REVIEW.md:682-684`.

18. **Add `pipeline.used_by(name)` and `diff(v1, v2)` to `src/decider2/graph/`.** The
    first answers "which steps and modules read this value" by walking the interface
    graph `graph/interface.py:46` already builds — the reverse of the lineage query
    `docs/04-observability-and-governance.md:126-130` specifies. The second diffs two
    params documents *and* the two generated driver source files, which are real files
    on disk (`docs/05-boundary-and-compilation.md:340`) rather than `exec`'d strings, so
    the code diff costs a `difflib` call — `runtime/serve.py` already imports `difflib`
    (`:31`). **Effort S.**
    *Motivated by:* SAS's "Determine Which Objects Use a …" plus paired
    "Compare … Content" / "Compare … Code" on every object kind.

19. **Generate a per-artefact reviewer sheet for the two kinds that already exist.**
    Add `TableModule.sheet()` and `TreeModule.sheet()` beside the existing
    `explain()` (`src/decider2/tables/build.py:98`, `src/decider2/trees/build.py:110`),
    rendering the document plus its in-force `shared` arrays as Markdown in the shape of
    `example_projects/examples/01-transaction-fraud-interdiction/fraud_interdiction/artefacts/rule-sheet-MS-0208.md`
    — authored value beside in-force value, the reason they differ, per-input missing
    behaviour, and the approval row. This is the smallest honest step on decider2's
    top-ranked risk, it needs no new subsystem, and it gives E4
    (`docs/06-open-questions-and-experiments.md:457`) something concrete to put in front
    of a reviewer. **Effort M.**
    *Motivated by:* SAS generates PDF documentation for five object kinds and a
    published-decision report (UG pp. 46, 101, 125, 226, 327-328) — the only vendor in
    §1 shipping a printable artefact per rule artefact.

20. **Add a PMML Scorecard importer as `src/decider2/scorecards/pmml.py`**, once
    recommendation 4's `Scorecard`/`Characteristic`/`Bin` schema exists: parse the
    PMML 4.x `Scorecard` element (`Characteristics`/`Characteristic`/`Attribute`,
    each carrying a `partialScore`) into that schema, mapping PMML's `SimplePredicate`
    ranges onto `Bin.min`/`Bin.max` and its categorical predicates onto `Bin.values`.
    This is not a hypothetical interchange format: `PROC PSCORE` "supports PMML model
    types including 'Scorecard'" and Pega's own model importer names "Scorecard" in
    its explicit PMML algorithm allow-list alongside decision tree, regression and
    ensemble methods
    ([supported models](https://docs.pega.com/bundle/platform/page/platform/decision-management/supported-models-import.html)) —
    **two unrelated vendors both use PMML Scorecard as a live import path**, which is
    stronger evidence for the format than DMN CL2 has for recommendation 15. Do not
    attempt general PMML (all model types); scope to the `Scorecard` element only,
    the same discipline recommendation 15 applies to DMN. **Effort M.**
    *Motivated by:* SAS `PROC PSCORE` (`sub-sas-scorecards.md` §C); Pega's PMML model
    import allow-list; GDS Link's and Zoot's Python/PMML/R model-integration claims
    (§1.7), which name PMML without a version or profile detail decider2 would still
    need to nail down against a real exported file before trusting this importer.

---

## 5. Genuine alternatives — where buying beats building

Stated with decider2's own numbers and requirements, not as a recommendation to
buy. In each case the question is whether the *part* of decider2's scope is worth
building given what it costs.

**1. The champion/challenger and experiment-registry layer — buy or borrow the
pattern, do not invent it.** This is not really a build-vs-buy call, because no
vendor sells the layer separately; but two vendors publish a concrete-enough shape
to copy. ACTICO states "simultaneous hosting of multiple model versions (dynamic
routing, e.g. for champion/challenger)"
([model management](https://www.actico.com/platform/model-management/)) without
mechanics; Alloy is far more specific — named arms, "up to 5 versions", a "minimum
5%" per arm, start/stop as audited actions
([champion challenger](https://help.alloy.com/en/articles/16820342-champion-challenger))
— and is the better template for the `arms={name: pct}` shape, **provided the
assignment itself is not copied**: both Alloy and Pega assign with a live
random/probabilistic draw per call, and Pega's own docs show the resulting sampling
drift (a configured 70/30 split landing at 69/31 after 1000 runs). decider2 already
has the two hard halves: `ServeHandle` holds several generations with 0.177 µs
activation and 3.36 µs rollback and zero recompiles (`docs/EXPERIMENTS.md:434-435`),
and the arm index has the same shape as a `_path` value. What it lacks is the
registry and the deterministic assignment, both of which a mock project already
specified
(`example_projects/examples/08-collections-treatment/cohorts/assignment.py:45-70`).
Building it is item 4.6; **the alternative here is to copy a naming/allocation
shape, not a product, and explicitly not the randomness.**

**2. Scorecard *development* — already not decider2's scope, and should stay that
way.** Equifax Ignite ("build models in weeks, not months",
[Ignite](https://www.equifax.com/business/product/ignite/)), Zest AI
("over 600 active models") and SAS Model Manager all own model *development*.
decider2's example projects treat a scorecard as a table of 9 × 45 × 8 cells
authored by the Model team and *executed* by the engine
(`example_projects/00-shared-credit-core-library.md:552`). Recommendation 4.4 builds the
executor, not the developer. There is no case for building the development side.

**3. Bureau-integrated hosted origination decisioning, for a greenfield product
line.** GDS Link's TymeBank case study is the one verified South African bank
running a vendor decisioning platform, at "over 100,000 new customers per month"
([case study](https://gdslink.com/case-study/tymebank/)). Experian PowerCurve
DaaS bundles the data and the strategy layer
([DaaS](https://www.experian.co.uk/business-products/decisioning-as-a-service/)).
If the requirement were "decision a new unsecured product with bureau data and
nothing bespoke", either is faster than building. **What they do not do** is the
thing decider2's example set is built around: an in-process engine that runs the
*same artefact* over a 22-million-decision-a-year batch estate
(`example_projects/09-governance-and-replay-harness.md:11-13`) and a real-time
request, with the equality asserted. No vendor in §1 publishes any such guarantee.

**4. The reviewable-artefact problem — nobody sells a solution, so this cannot be
bought.** decider2's top-ranked risk is that a credit-risk reviewer cannot verify
a rule from a generated view, and it has been **tested twice and failed twice**
(`docs/04-observability-and-governance.md:277-340`, `:334-375`). Every vendor
claims readability; none of them publishes evidence that a reviewer succeeded. The
honest position is that buying a platform relocates this risk rather than removing
it — the reviewer would then be reading *the vendor's* rendering, with no ability
to change it, whereas decider2's settled position ("trace is data; every rendering
is replaceable", `docs/04-...:415-445`) at least makes iteration possible. This is
an argument for building, and it is the strongest one in the set.

**5. A visual editor — borrow GoRules' MIT editor rather than build one.** Not a §3
vendor, but the relevant comparison: `jdm-editor` is MIT-licensed React
(landscape §8, [github.com/gorules/jdm-editor](https://github.com/gorules/jdm-editor)),
and decider2's tree document is already a positioned node/edge graph with `Position`
and `MultiSourceEdge` (`src/decider2/trees/schema.py:970-1004`) — i.e. it was
designed for a canvas editor. Writing an exporter to JDM is cheaper than writing an
editor. **Effort M, and it removes the largest chunk of "UI work" from the roadmap.**

**6. Where buying is clearly *worse*.** Three of decider2's stated requirements have
no vendor answer in §1: (i) the four-way equivalence assertion
(`src/decider2/testing/equivalence.py:200-216`); (ii) an in-process µs-scale kernel
— the field's published in-process numbers are Higson 0.23 ms and ZEN µs/op, while
hosted decisioning clusters at 5–100 ms, against decider2's measured 1.38 µs kernel
(`docs/EXPERIMENTS.md:706-712`); (iii) free reference-data edits — a decision
table's rows change with no compile at all
(`src/decider2/tables/__init__.py:19-21`), which matters at 63 360 and 219 600 cells
refreshed monthly by Treasury (`example_projects/00-...:546-549`).

---

## 6. Unverified / unreachable

Recorded rather than dropped or asserted. Every item below is a claim this report
**does not** make, or makes only on the stated secondary basis.

**Resolved on this pass — the landscape note's largest gaps are closed.** The SAS
Intelligent Decisioning *User's Guide* was reached by ignoring the HTML doc pages and
requesting the PDF through the collections API with a browser user-agent:
`curl -L -A 'Mozilla/5.0 …' 'https://documentation.sas.com/api/collections/edmcdc/v_040/docsets/edmug/content/edmug.pdf?locale=en'`
→ 200, 3.8 MB, *2023.11*, build tag `v_033-P1:edmug`. §1.1 is therefore
documentation-grade, not marketing-grade, and it **contradicts the marketing pages in
two places** (champion/challenger; scorecards). The same route works for `v_035`,
`v_030` and `v_024`. A follow-up pass reached SAS **Model Manager** (`mdlmgrcdc`),
the Viya **VDMML** node reference and user's guide (`vdmmlcdc`), and SAS 9.4
Enterprise Miner reference (`emref`) the same way, closing the scorecard and
adverse-characteristics gap entirely (§1.1, `sub-sas-scorecards.md`). The **MAS**
admin guide (`masag` docset, versions v_016–v_020 all reachable via the TOC
endpoint) was also reached, but contains no throughput or latency numbers at all —
only Kubernetes pod-resource troubleshooting guidance — so "MAS performance" moves
from unattempted to **checked and confirmed absent**, not merely unverified. **SAS
Container Runtime (SCR)** administration content could not be located: `scr`,
`scrag`, `contrag`, `containerruntime` and `sasscr` all 404 as both collection and
docset ids, and no public collection index exists to enumerate the real one — SCR
performance stays genuinely unverified. Separately, `docs.pega.com`'s backend
content API (`docs-be.pega.com/api/bundle/{bundle}/page/{navPath}`, open, no auth)
and `docs-be.pega.com/api/search` close the Pega gap completely — §1.9 is now
developer-documentation-grade, not Academy-page-grade. GDS Link, Zoot and Alloy were
each crawled exhaustively on their public sites (48, 42 and 244 pages respectively);
Alloy remains developer-docs-grade, GDS Link and Zoot remain marketing-grade but are
now **exhaustively marketing-grade** — absences reported for them (e.g. no decision
table, no hit-policy vocabulary, no simulation for Zoot) are confirmed zero-hit
crawls, not gaps from an unreached page.

**Blocked by the vendor (fetch failed on 2026-09-20 and again on 2026-09-21).**

| Vendor / page | Failure mode | What stays unverified |
|---|---|---|
| fico.com Blaze Advisor, FICO Platform, Origination Manager product and newsroom pages; investors.fico.com; community.fico.com | bot wall, 403, timeouts, error shell | Blaze Advisor's rule language (SRL), Java/.NET/COBOL SDK surface, whether rules compile or interpret, Rule Maintenance Application for business users, any latency figure, scorecard authoring detail, reason-code generation |
| transunion.com / .co.za / .co.uk / .ca product pages | 403 to all fetchers | DecisionEdge's entire authoring model, standards, execution modes, latency. The only usable source is a 2016 10-K |
| docs.taktile.com | authenticated GitBook | node catalogue; whether decision tables or scorecards exist; how "experiment with multiple versions of a flow in production" is implemented (traffic split vs shadow); latency |
| docs.oscilar.com | password-protected | rule-authoring form, backtesting mechanism, versioning/approval detail. The "120K+ rps, <100 ms" figure is from the marketing home page only |
| docs.actico.com | DNS failure | product documentation; the "12,000 REST calls per second" figure exists only in a search snippet and is **not cited in this report** |
| actico.com/platform/dmn-decision-model-notation | generic content returned | whether ACTICO imports or exports DMN |
| finastra.com lending pages and PDFs | 403 | Finastra Originate's rule-authoring surface |
| Experian PowerCurve product-sheet PDFs; experian.co.za case studies | 404 | whether PowerCurve names decision trees, decision tables, scorecards or **node-level statistics**; on-prem availability; champion/challenger; versioning |
| equifax.com/business/product/interconnect/ | 404 | named editors (Rules Editor, Attribute Navigator) appear only in search snippets and are **not cited** |
| `info.gdslink.com/modellica-solution-overview`, `.../modellicapro-solutionoverview-landpage` | DNS failure (subdomain does not resolve; no archive.org snapshot exists) | the two gated Modellica datasheet landing pages; no PDF datasheet exists in archive.org's 36-file `.pdf` listing for gdslink.com either — this is now a confirmed absence, not just an unreached page |
| `zootsolutions.com/developer-portal/{documentation,api-reference,extract-guide,changelog}/` | login wall (302 to `/login/`) | Zoot's actual API reference and rule-authoring schema; everything in §1.7's Zoot block is from public marketing pages only |
| `documentation.sas.com` collections `scr`/`scrag`/`contrag`/`containerruntime`/`sasscr` | 404 on both collection and docset id | SAS Container Runtime performance and administration detail — no public collection index exists to find the real id |
| documentation.sas.com/doc/en/edmcdc/**default**/edmug/… (HTML user-guide pages) | 404 / JS shell, on both passes | nothing — see below: the same content was retrieved as a PDF |
| documentation.sas.com … /docsets/edmug/content/edmug.pdf (`default` version segment) | 404 | nothing — `v_040`, `v_035`, `v_030` and `v_024` all returned 200 |
| *SAS Intelligent Decisioning: Using Data Grids* and *: Macro Guide* (separate docsets) | not attempted | the data-grid function catalogue, and the exact output of `%DCM_RULEFIRE_DETAIL` |
| SAS Event Stream Processing as a publishing destination | "Event Stream" appears **nowhere** in the 2023.11 User's Guide | the landscape note's ESP/in-stream claim comes from the marketing features list only and is **not** corroborated by the product documentation |
| SAS Global Forum proceedings / communities.sas.com | WebSearch budget exhausted | whether any user-written macro or third-party package emits reason codes from a SAS scorecard — the documentation set has none (`sub-sas-scorecards.md` §B), but a community package cannot be ruled out |

**Still marked as secondary-source in this report.** §1.1 (SAS) and §1.9 (Pega) are
now primary for everything carrying a page number or a `docs.pega.com`/
`docs-be.pega.com` citation; §1.1 is marketing-sourced only for the latency/
throughput figures and the ESP destination, and §1.9 only for the three conflicting
"~200ms" marketing figures (the docs' own 150 ms load-test boundary is primary).
§1.8's Alloy entries are developer-docs-grade throughout. Every claim in §1.2
(FICO), §1.3 (Experian), §1.4 (Equifax), §1.5 (Provenir), §1.6 (ACTICO), §1.7 (GDS
Link, Zoot), §1.8 (Taktile, Oscilar) and §1.10 (TransUnion, Finastra) remains vendor
**marketing** copy, not documentation — but for GDS Link and Zoot specifically, that
marketing corpus was crawled exhaustively (48 and 42 live pages respectively), so an
absence reported for them (no hit-policy vocabulary, no simulation feature for Zoot,
no champion/challenger split mechanics for either) is a confirmed zero-hit result,
not a page this report failed to reach.

**Claims deliberately not made.**
- That any vendor guarantees batch and real-time equality. No page found states it;
  the absence is recorded as an absence, not as a denial.
- That Experian or FICO implements strategy trees with node-level statistics. The
  capability is inferred from product category and is **unverified** (§1.3). The
  improvement at §3 item 6 is justified by decider2's own specification
  (`example_projects/04-campaign-targeting-trees.md:531-560`), not by the vendor claim.
- Any statement about SAS null semantics, SAS lookup-table size limits, or which
  node types SAS refuses to publish to a given destination.
- Any latency comparison between decider2 and a hosted platform as like-for-like.
  decider2's numbers are in-process kernel and framework overhead on one machine
  (`docs/EXPERIMENTS.md:684-712`); vendor numbers are end-to-end service response
  times on undisclosed hardware. The classes are different and the report says so
  each time.
- That Pega's marketing "sub-200ms" / "billions of decisions" phrasing is a Pega
  quote. It is not: the verified forms are "less than 200 milliseconds", "within 200
  milliseconds", "220 milliseconds or less" (three different numbers on three
  pages), and "billions of **interactions**" / "5 Billion next best actions each
  month" — never literally "billions of decisions". The docs' own 150 ms load-test
  bucket boundary is what this report cites as Pega's defensible figure (§1.9).
- That any vendor's champion/challenger mechanism is deterministic. Pega and Alloy
  are the only two vendors in §1 with a documented mechanism, and both are a live
  random/probabilistic draw per call, not a hash — the absence of a deterministic
  vendor precedent is recorded as an absence, and decider2's own required design
  (recommendation 6) is stricter than either.

**decider2-side uncertainties that affect this comparison.**
- `docs/README.md:130-140` warns that doc 01's `decider/...:line` citations resolve
  only against an unpushed local branch. Nothing in this report depends on a doc 01
  citation.
- **Re-checked on this pass, 2026-09-21, against the live working tree:** `git status`
  now shows only `src/decider2/trees/schema.py` and `trees/codegen.py` modified
  (plus their tests) — `tables/schema.py`, `tables/codegen.py` and everything under
  `boundary/` are clean against `HEAD` and their citations were spot-checked exact
  (`DecisionTable` at `tables/schema.py:551`, "First match wins" at `:561`,
  `route_required_nulls` at `boundary/nulls.py:311` all confirmed verbatim). **The
  drift in `trees/schema.py` is not "a few lines"**: `git diff --stat` shows +168/-58
  lines, and node-class anchors have moved by roughly 85–90 lines each
  (`LeafNode` cited at `:658`, now at `:747`; `UnaryNode` `:697`→`:787`;
  `CasesRanges` `:744`→`:832`; `CasesStringMatch` `:779`→`:868`; `CasesIsIn`
  `:814`→`:903`; `CompositeNode` `:842`→`:931`; the `Tree` class itself moved from
  roughly `:1055` to `:1149`, `children()` from `:1095` to `:1189`, `root_id()` to
  `:1198`). Every `trees/schema.py` and `trees/codegen.py` line citation in this
  report (§0, the SAS/ACTICO comparison tables, §2 rows 3/25/37/38, §3, §4 items 3/9/
  11) should be treated as **directionally correct but numerically stale by
  roughly +90 lines** until decider2's tree work is committed and the anchors are
  re-run; the class and function names themselves were not renamed, so a
  name-based grep will still find each one.

---

## 7. Sources

Every vendor claim in §1–§5 carries its own inline URL at the point of use; this
section is an index by vendor/topic, not a duplicate of every citation. Access
methods worth reusing are noted where they mattered.

**decider2 (primary source for every "vs decider2" claim).** `decider2/docs/README.md`,
`02-architecture.md`, `03-authoring-api.md`, `04-observability-and-governance.md`,
`05-boundary-and-compilation.md`, `06-open-questions-and-experiments.md`,
`08-configuration-and-lifecycle.md`, `EXPERIMENTS.md`, `REVIEW.md`; source under
`decider2/src/decider2/{graph,tables,trees,params.py,runtime,boundary,testing,serving}/`;
`decider2/docs/example_projects/{00,03,04,07,08,09}-*.md`. Read against the working
tree on 2026-09-21 (see the uncommitted-file caveat above).

**SAS.** Intelligent Decisioning *User's Guide* 2023.11 (`edmcdc/v_040/edmug`, and
`v_035`/`v_030`/`v_024`), Administration Guide, Macro Reference, Data Grids, Custom
Node Types, CLI Reference and What's New (2026.07 line), all via
`documentation.sas.com/api/collections/<collection>/<ver>/docsets/<docset>/content/
<docset>.pdf?locale=en`; SAS Model Manager User's Guide 2023.07 and 2026.07-09
(`mdlmgrcdc`); VDMML Machine Learning Node Reference / User's Guide / Advanced
Topics v_035 (`vdmmlcdc`); SAS Enterprise Miner 15.4 Reference Help (`emref`); SAS
Micro Analytic Service Admin Guide (`masag`, v_016–v_020, via the TOC and per-page
content endpoints — no performance numbers found); SAS features list and press
releases (`sas.com`) for marketing-only claims. Full quote inventory in the
scratchpad's `sub-sas-scorecards.md`.

**Pega.** `docs.pega.com` bundles `platform`, `customer-decision-hub`,
`credit-risk-decisioning`, read via the open backend API
`docs-be.pega.com/api/bundle/{bundle}/page/{navPath}` and
`docs-be.pega.com/api/search`; legacy static help at
`community.pega.com/sites/.../help_v73` and `help_v84`; `academy.pega.com`;
`www.pega.com` marketing pages, enumerated via `sitemap.xml`. Full quote inventory
in `/home/sholto/.claude/projects/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/1835b6ef-3ce0-40a6-a48b-1a24de526ae1/tool-results/toolu_01Urcwiyt66zzydVVxy7H7tN.txt`.

**GDS Link, Zoot, Alloy.** `gdslink.com` (48 live English pages) plus
`web.archive.org` snapshots for retired Modellica-era copy; `zootsolutions.com` (42
live pages; developer portal login-walled); `developer.alloy.com` (115 guide pages +
129 API-reference pages) and `help.alloy.com` (Intercom help centre). Full quote
inventory in `/home/sholto/.claude/projects/-home-sholto-Documents-Workspace-capitec-dsp-decision-engine/1835b6ef-3ce0-40a6-a48b-1a24de526ae1/tool-results/toolu_01E4nAcmf11nWo8pGwHoMy3w.txt`.

**FICO, Experian, Equifax, Provenir, ACTICO, TransUnion, Scienaptic, Zest, nCino/
FullCircl, Temenos, Finastra, Taktile, Oscilar.** Vendor marketing pages only, cited
inline in §1.2–§1.6 and §1.10; reachability and failure modes recorded in §6.

**Landscape note (starting point).** `decider2/docs/research/decision-engine-
landscape.md` §3 (bank-specialist platforms) and §10 (source list), re-verified
rather than re-cited wholesale.
