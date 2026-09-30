
### 3. A decision trace and reason codes

**What it is.** A contract in which every component declares the points at which it will record what it did; the engine captures those cheaply as it runs; and a separate step later expands them into a readable document — which rule matched, which table row fired, which scorecard band applied, every rounding step and its mode, and the reasons attached.

**Why it matters.** This is the part a bank cannot do without, and `decider` currently has none of it. Reason codes appear only as a comment in its source. Its one piece of audit machinery, in `plan.py`, records which columns a step added, and its own specification describes that as dead code. Tracing exists as a proposal to emit observability spans, marked as not yet approved. Observability and evidence are different things: a five-year regulatory record cannot live in a monitoring system's retention budget, and a span tree cannot say which row of which table matched.

**What was measured.** On a synthetic 2,000-node flow: capturing a full trace added about 26 µs to a 14 µs decision and produced 10.7 KB per decision. Expanding it to a readable document *inside* the request cost roughly eight times as much time and twenty-nine times the bytes — so the expansion belongs downstream, with the compact form archived alongside a small map that decodes it. About two thirds of the recorded events were personal data, and the design separates those so that a deletion request can be honoured without decoding the record.

**One finding worth passing on regardless.** An optimisation in the study's own engine silently deleted 1,102 of 1,953 trace points while every output stayed identical. The lesson generalises to any compiled pipeline: trace-point conservation has to be checked, or an optimiser will quietly remove the evidence.

**Where it fits in `decider`.** Trace points declared per step (its steps already declare what they read and write, which is the hard part), raw capture in the kernels, and expansion outside the request path. Its closed four-node representation is an advantage here: there are only four places to instrument.

---


### 4. Whole-path checks over a flow

**What it is.** Automated checks that a property holds on *every route* through a flow — for example, that no path can reach an approval without the affordability assessment having been performed, and that every adverse outcome carries a reason that may be disclosed to the customer. When a check fails, the message names a concrete path in business terms rather than printing a graph.

**Why it matters.** These are the properties a regulator asks about, and they cannot be tested by example: a flow with twenty branches has more paths than anyone will write tests for. The study's prototype caught all 22 deliberately planted defects of this kind — a new branch that skipped the check, a re-pointed sub-flow exit, a whole leg written without it — **with no false alarms** on two correct flows.

**A correction worth inheriting.** The companion rule, as the study's own architecture first wrote it, was unusable: implemented literally it flagged every adverse ending of two correct flows and nothing that was actually wrong. Three refinements fixed it. Anyone building this independently would hit the same wall, so the refinements are as valuable as the rule.

**Where it fits in `decider`.** As passes over `engine/ir/`: `BranchNode` and `LoopNode` give the control-flow graph these analyses need. The rules themselves are best supplied as governed modules attached by a flow's classification, rather than written by the flow's author — otherwise the author can weaken the rule they are being checked against.


---

### 5. Proving that a change did not change behaviour

**What it is.** Two tiers. The first compares two versions of a flow structurally and, for a class of edits — renaming, extracting a sub-flow, reordering independent steps, merging calculations — *proves* they cannot behave differently. The second generates test cases concentrated at every threshold in **both** versions and compares the outcomes.

**Why it matters, with the evidence.** The first tier proved eight of ten behaviour-preserving refactors in about nine milliseconds each, and never once declared a genuinely changed flow equivalent — including across 283 randomly generated edits, two kinds of which were designed to trick it. The second caught all ten deliberately seeded behaviour changes, but only after two fixes that are findings in themselves: the generated cases must include a record in every *regime* of the flow, not merely at every threshold; and the thresholds must be taken from both versions, because a moved threshold is a number the old version does not contain.

**The finding that should worry anyone running a decision platform:** four of those ten behaviour changes passed **all 42 of the flow's own tests**. Separately, retuning one parameter — a dependant allowance from R450 to R475 — also left every test green. A flow's test suite is not a safeguard against an unintended behaviour change, and a lighter approval path for "just a parameter" needs something stronger behind it.

**Where it fits in `decider`.** `decider/testing/equivalence.py` already asserts that its execution modes agree on the same pipeline. The extension is to compare *versions*: same harness, cases generated from both versions' constants, reported per level (same decision / same decision and reasons / same path).

>>> Not sure how possible this is always but it might be good to do a spike to see what is possible <<<



---

### 6. Common Defect detection


- **Nineteen credit policy rules read the wall clock**, A common defect its using the systems clock for determining things like age which may be incorrect when running in a docker environment
- Over-underflow detection or detecting places that are near boundaries to these
- Places where floating point errors start making big differences the ability to suggest a different datatype for these steps

>>> My read is maybe decider needs also a compliance api to run tests and diagnostics and to be able to compare different revisions and maybe we should expose these through to the vscode extension as well <<<



---

### 8. Change control for the parts of a flow that are data

**What it is.** Treating a flow as data with stable identifiers, so that: every element has an identifier that survives edits; two fingerprints exist per release (one for the logic, one for the whole package); a change can be described in business terms rather than as a text difference; and two people editing the same flow are merged by *meaning* rather than by text.

**What was measured.** Across 15 concurrent-editing scenarios, 8 merged with no author action, 7 produced a business-level question ("you set this cell to 2.10, the release set it to 1.75, which stands?"), none was unsupported, and **no text conflict marker ever reached the author**. Plain `git merge` on the same 15 conflicted on 7. A separate check on 2,000 randomised collisions found no case where a change was lost, overwritten or invented.

**Where it fits in `decider`, and the honest limitation.** `decider` deliberately keeps pipeline *structure* in Python, for a recorded reason: in its predecessor, structure lived in both code and configuration and the two disagreed — a flow declaring 51 steps, a generated file with 120, and a real flow with 192. That is sound, and it means this recommendation can only apply to the parts of a flow that are already data: its parameter documents and its configurable trees, decision tables and scorecards under `configs/`. That is, in practice, where most day-to-day policy change happens. Structure defined in Python would remain reviewed as code.
>>> i think we can in steps have an id that we auto assign somehow in a determanistic way and allow the user to add a custom step id if they want maybe even s form of code gen where we automatically assign step ids or names on the first run or with use of a decider command <<< - Might be usefult for both debugging and parameters

