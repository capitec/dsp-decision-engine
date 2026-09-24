# Project dependencies and build waves

**Hard** means the project uses something another project publishes (a capability, an
assessment, a component, a batch output), so the other project has to exist first
or be stubbed against its published contract. **Soft** means the project only
mentions the other one: out-of-scope notes, reuse it might want later, "same shape as",
a runtime data feed whose shape is fully described in the consuming spec, or the
reverse direction.

Quotes are verbatim, and each is followed by `spec §section`.

---

## Per project

### 01 Transaction fraud: hard 00 only
- **00**: "Consumes `core.reason_codes`, `core.dates`, `core.consent`, `core.rounding` and `core.adjustments`, and needs almost nothing else" (01 §2).
- Soft: 02, 03 and 07 are named only as out of scope ("Credit decisioning of any kind. See projects 02, 03 and 07", 01 §12). 08 appears in a §13 question. 09 governs 01, not the other way round.
- **01 does not publish what 03 and 10 expect from it.** 03 §4.4 consumes "project 01's application-fraud engine", but 01 covers only *transaction* fraud and excludes credit decisioning (§12). That makes 03→01 **soft**: stub it with the three-field verdict contract in 03 §4.4 (`fraud_verdict_code` 1–4, `fraud_reason_codes`, `fraud_response_ms`).

### 02 Affordability: hard 00 only
- **00**: "Bureau account list | `core.bureau`"; "`risk_grade` | `core.risk_grade`"; "Adjustment set in force | `core.adjustments`" (02 §4.1).
- 02 is also where the full flow of five library capabilities lives: "Full flow in [project 02]" (00 §6.5), and "Supplied by **project 02** through the library capabilities `core.income`, `core.deductions`, `core.expense_norms`, `core.obligations` and `core.affordability`" (06 §4.4). See the ownership note under Cycles.
- Soft: 06 supplies settlement quotes as runtime input ("Settlement quotes | Project 06 / settlement desk | 0..80 | null outside consolidation", §4.1). That is a data feed and can be stubbed. 03, 05, 06, 07 and 08 are consumers, and 09 stores the output.

### 03 Unsecured granting: hard 00, 02
- **00**: "This project consumes ten capabilities from project 00 — `core.eligibility`, `core.bureau`, … `core.dates`" (03 §2). The sentence says ten but lists thirteen.
- **02**: "This project does not compute affordability. It consumes project 02's published assessment" (03 §5.6).
- Soft: 01 (see above). 04, 06 and 07 are downstream consumers or want 03's parts later ("Project 06 wants the solve", "Project 07 wants the cap waterfall", 03 §11 items 11–12). 08 and 09 are out of scope.

### 04 Campaign trees: hard 00, 03 (03 can be stubbed)
- **00**: "`core.consent` … `core.eligibility` … `core.scorecard` … `core.appetite` … `core.adjustments`" (04 §4.3).
- **03**: "Project 03, batch mode | Bulk pre-assessment producing the amount the Bank would actually grant" (04 §4.3). Also "Every amount a tree puts in front of a client must be backed by a pre-assessment produced by project 03 running in batch" (04 §5.5 req 1).
- 04's dominant difficulty (trees and path capture) does not need a real 03. A stub that emits `pre_assessed_amount`, `pre_assessed_term`, the binding constraint, `risk_grade` and a validity window (04 §5.5 "Emits") lets 04 move up one wave.
- Soft: 07 ("Card campaigns must take their amounts from project 07's limit management run", 04 §11 item 13, which is a future change) and 08.

### 05 Business nested entities: hard 00, 02
- **00**: "`core.bureau`, `core.adverse_events`, `core.scorecard`, `core.calibration`, `core.risk_grade` and, for sole proprietors, `core.affordability`" (05 §2).
- **02**: "a statutory affordability assessment is required on the natural person behind the facility, through [project 02]" (05 §5.2). Also "the statutory affordability assessment on the natural person, through [project 02] … A statutory fail is a hard fail" (05 §5.12 item 7).
- Soft: 03 and 06 appear only as "the same shape" (§13 Q10). 08 and 09 are out of scope.
- **Dangling claim.** 05 §2 says its components are reused "in [project 07]'s business limit review", but 07 covers only retail revolving products (20, 21) and has no business review. That reuse actually happens in 11 L1.

### 06 Consolidation: hard 00, 02, 03
- **00**: "`core.obligations`, `core.affordability`, `core.rate_card`, `core.fees`, `core.credit_life`, `core.instalment` … `core.adjustments` and `core.rounding`" (06 §5.6.2).
- **02**: "Supplied by **project 02** through the library capabilities …" (06 §4.4). Also "Affordability is project 02's, invoked hundreds of times per assessment" (06 §2).
- **03**: "Granting and pricing per product is project 03's logic, reused four times over inputs project 03 never contemplated" (06 §2). Also "This is project 03's circular solve, appearing here once per scenario" (06 §5.6.2 item 3).
- Soft: 07 and 08 want settleability later (§13 Q15). 09 does replay.

### 07 Credit limit management: hard 00, 02
- **00**: "`core.affordability`, `core.income`, `core.obligations` and `core.expense_norms` are consumed here in a **degraded-evidence mode**" (07 §2).
- **02**: "The affordability capability is shared with project 02 and not forked" (07 §10 item 14). Also 02 §5.8 defines the "Limit increase (07)" mode.
- Soft: 08. "Project 08 treatment state | … | 0.31 M accounts | Daily" (07 §4.1) is a data feed, used only by exclusion X16 (§5.2), so a stub is enough. 03 is only a comparison point for the buffer (§5.6). 09 does replay.

### 08 Collections: hard 00, 02
- **00**: "`core.adjustments`" (08 §2 Q3), "`core.scorecard` publishes these" (§5.3), "`core.consent`" (§4.2).
- **02**: "Every arrangement is tested with `core.affordability` … The distressed mode must be a parameterisation of one capability, not a fork" (08 §5.6). Also "Consumes it in arrangement mode (project 08)" (02 §3) and the "Arrangement (08)" column of 02 §5.8.
- Soft: 07 ("Project 07 has the same…", §2), 01, 04 and 07 in §13, and 09 for evidence.

### 09 Governance harness: hard 00, 01–08 (see the cycle below)
- **01–08**: "This project's inputs are other projects' outputs" (09 §4). The §4.1 table lists flows 01–08.
- **00**: "Every versioned artefact named in 00 §8" (§4.2). Also "each of the twenty-one shared capabilities: a build identifier" (§4.3).
- 10 and 11 are not in 09's table, but both adopt 09's contract (below).

### 10 Retail end to end: hard 00 only (standalone)
- **00**: "The only thing it takes from outside is the shared library's published capabilities (project 00 §6) and the canonical vocabulary (project 00 §4)" (10 §1.1).
- Contract, not code: "The estate-wide harness is specified in project 09; this flow's obligation is to emit what it needs" (10 §3.2).
- Soft: the overlap with 02, 03, 06 and 07 is "intentional, not an oversight" (10 §1.1). 11 is the A/B partner.
- **Hidden coupling:** 10 §5.7(b) "Consumes `core.income`, `core.deductions` and `core.expense_norms`". Those units must therefore be implemented in 00 and not only inside 02, or 10 is no longer standalone.

### 11 Business end to end: hard 00, 02, 05, 06, **07, 09**
- **00**: "All **22** published capabilities are consumed" (11 §4.9).
- **02**: "Regulated affordability, four assessment modes | [02] §5, §5.8 | O14" (§4.9).
- **05**: twelve components, from "Entity structure resolution … [05] §5.1" to "Final validation, conditions, committee pack … [05] §5.13" (§4.9).
- **06**: "Obligation inventory and settleability | [06] §5.2, §5.3 | L5"; "Concession catalogue and NPV authority model | [06] §5.9" (§4.9).
- **07**: "Portfolio-budget limit decisioning | [07] | L1's limit decision on revolving facilities" (§4.9).
- **09**: "The 23-item evidence contract | [09] §5.15 | Every phase"; "Replay, diff, swap-set, certification … | [09] §5.1–5.7 | Governance, not runtime" (§4.9).
- **README §4.1 is out of date.** It says 11 consumes "the `core.*` library plus the components of projects 02, 05 and 06". 11 §4.9 adds **07** (runtime, phase L1) and **09** (contract and governance). 11 §5.17.4 says "components from **four** other projects", which is 00, 02, 05 and 06, so the spec disagrees with its own table. 03 is only a "shape shared" reference.
- Also, 11 is about **1 900** decision points, not 1 400 (11 §1 and §4.9: "1 280 of the 1 900 … consumed").

---

## Dependency table

| Project | Hard (build first) | Soft (mentions, feeds, future reuse) |
|---|---|---|
| 01 | 00 | 02, 03, 07, 08, 09 |
| 02 | 00 | 03, 05, 06 (quote feed), 07, 08, 09 |
| 03 | 00, 02 | 01 (fraud verdict, unpublished), 04, 06, 07, 08, 09 |
| 04 | 00, 03 (stubbable) | 07, 08, 09 |
| 05 | 00, 02 | 03, 06, 07 (dangling), 08, 09 |
| 06 | 00, 02, 03 | 07, 08, 09 |
| 07 | 00, 02 | 03, 08 (treatment-state feed), 09 |
| 08 | 00, 02 | 01, 04, 07, 09 |
| 09 | 00, 01–08 | 10, 11 (they adopt its contract) |
| 10 | 00 (+ 09 §5.15 as a contract) | 02, 03, 06, 07, 11 |
| 11 | 00, 02, 05, 06, 07 (+ 09 §5.15 contract, 09 harness for governance) | 03, 08, 10 |

---

## Cycles and how to break them

1. **09 against 01–08.** 09 operates on every flow, but its §5.15 contract applies to all of them from the first release: "none of this can be retrofitted" (09 §1). **Break it by splitting 09**:
   - **09-C, the contract** (§5.15's 23 items plus §5.14 adjustment governance requirements): a requirements document adopted by 00 and by every flow in their own acceptance tests. It goes in wave 0.
   - **09-H, the harness** (§5.1–5.13: replay, diff, swap-set, certification, rendering, monitoring): built last, over whichever flows exist.
2. **00 against 02 (ownership, not a true cycle).** 00 §6.1–6.5 publish five capabilities, and 00 §6.5 says "Full flow in project 02". If 02 owns their implementation, then 03, 05, 06, 07, 08, 10 and 11 all depend on 02, and 10 stops being standalone. **Break it so that** 00 implements the five units (their arithmetic, tables and interfaces), while 02 implements the *assessment*: household framing, the four modes, the verdict, the three answer shapes and the evidence ladder, composed from those units. 10 then needs only 00. See the 00-ADDENDUM.
3. **03 ↔ 06, 03 ↔ 07 (future reuse only).** 06 and 07 want 03's solve and cap waterfall (03 §11 items 11–12). No cycle exists today, because 03 publishes and 06 consumes. **To keep it that way**, 03 must publish the solve and the waterfall as reusable parts, and 06 must not modify them.
4. **07 ↔ 08 (data only).** 07 reads 08's treatment state, and 08 points to 07 only as "the same shape". Stub the feed. No build ordering is needed.
5. **11 against 09 (governance).** 11 consumes 09's harness only for governance, not at runtime. Build 11 against 09-C, and put 09-H in the same wave or a later one.

---

## Waves

Each wave depends only on earlier waves. The waves are kept as wide as possible.

| Wave | Projects | Depends on |
|---|---|---|
| **0** | **00** (with 09-C adopted as its evidence requirement) | none |
| **1** | **01, 02, 10** | 00 |
| **2** | **03, 05, 07, 08** | 00, 02 |
| **3** | **04, 06** | 03 (and 02) |
| **4** | **09-H, 11** | 01–08 (09-H); 02, 05, 06, 07 (11) |

Variants:
- **Stub 03's batch output for 04.** 04 moves to wave 2, and wave 3 contains only 06.
- **Stub 02 with its §7 output contract.** 03, 05, 07 and 08 move into wave 1 alongside 02. That is riskier, because 02 AC1 ("Five consuming projects use one implementation") is exactly what a stub hides.
- **10 has no consumers.** It can run in any wave from 1 onward. Put it wherever there is spare capacity.
