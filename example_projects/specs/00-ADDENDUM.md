# 00 — Addendum: what the consumers actually expect

Read alongside 00. It covers where specs 01–11 expect more from `credit-core`, or something different. Where they conflict, **this addendum wins**.

## A. Capabilities and names consumers rely on that 00 does not define

1. **Decision identity.** Publish `decision_id`: globally unique, assigned before any logic runs, never reused (09 §5.15 item 1; 10 §5.31.2). Every capability must emit its evidence keyed by it. Publish generic evidence names for *table version* and *cell read*. Today 00 has these only for the rate card and the norm table (`rate_card_version`, `rate_cell_id`, `norm_table_version`), but every table needs them (09 §5.15 item 5; 03 §10 item 14; 05 §10 item 14).
2. **Segment.** 00 keys calibration, grade boundaries and appetite on "segment" (00 §6.11, §6.12, §6.17) but publishes no segment name. Publish `segment_code`. Consumers use it (10 §4.3, §5.21) or invent their own (05 §5.7, `sme-people`).
3. **Consent names.** 08 §4.1 says "the consent vocabulary come[s] from the library unchanged", but 00 §4 has no consent vocabulary. Publish a consent state with per-channel permissions (SMS, push, voice, email; 01 §5.2), `consent_record_id` (07 §4.3) and a verdict (03 §5.2 `consent_verdict`; 10 §5.21 `consent_state`). A regulated notice must not be suppressible by a marketing preference (07 §5.9).
4. **Shared identifiers that consumers already contradict each other on.** The library must publish or reserve these names:
   - `account_id`: "one revolving account, no other project may redefine" (07 §4.3) against "one credit agreement" (08 §4.1).
   - `matrix_cell_id`: declared in both 07 §4.3 and 08 §4.1.
   - `assessment_mode_code`: four affordability modes (02 §4.3) against four consolidation modes (06 §4.7).
   - `authority_level_code`: 06 §4.7 against 11 §4.8.
   - `event_id` / `event_type_code`: a payment event, uint64/int16 (01 §4.4), against an adverse event, int64/int8 (05 §4.4).
5. **Fraud verdict.** 03 §4.4 and 10 §5.21 consume `fraud_verdict_code` (1 approve, 2 refer, 3 decline, 4 unavailable) and `fraud_reason_codes`. No project publishes them, because 01 is transaction fraud and excludes credit (01 §12). Publish the verdict contract. "Unavailable" must stay distinct from "approve" in the record (03 §5.2).
6. **More than one score per decision.** 00 §4 allows one `score`, one `probability_of_default` and one `risk_grade`. Consumers carry several at once:
   - behaviour score and grade, "distinct from `risk_grade`" (07 §4.3);
   - a collections roll/cure score (08 §5.3);
   - propensity scores (04 §4.3);
   - one score per entity (05 §5.7);
   - nine scorecards (10 §6.3).

   `core.scorecard`, `core.calibration` and `core.risk_grade`, and `core.adjustments` over them, must tell scores apart by scorecard and role.
7. **Whose value it is, and a fourth kind of null.** A value must say *whose* it is (applicant, surety, director, entity *n*), without renaming it at every use (11 §5.17.1: 31 names are used by role). Add "not applicable to this role" as a fourth null alongside the three in 00 §7.4 (11 §5.17.3, §5.17.6 item 7).
8. **More than one date.**
   - `core.dates` must resolve against dates other than `decision_date`: a knowledge date (11 §4.8), and the version agreed when an instance was created (11 §5.17.4 item 3).
   - A consumer must be able to choose a capability's major version **per assessment**, not only per consumer (11 §5.17.4 item 1; this extends 00 §7.1).
   - `core.exposure` must aggregate *as at* a date, including contingent exposure at declared conversion factors (11 §5.17.2, §5.17.6 item 6).
9. **Rate cards do not always return a single rate** (06 §5.6.4–5.6.7, §13 Q6). Product 20 needs a promotional rate, a reversion rate and a duration. Product 40 needs a margin over a reference rate. Product 11 has its own card (63 072 cells), and so do business products 52–58 (11 §4.2). None of these is in 00 §8.
10. **Obligations.** `core.obligations` must:
    - take **two** account lists (bureau and internal) and de-duplicate them (02 §5.5.1; 10 §5.7(c));
    - support **ten** treatment behaviours (02 §5.5.2), not five (00 §6.4), with a behaviour and its coefficients in each cell;
    - accept 0..95 accounts (10 §5.7(c)).

    Separately, `court_ordered_deductions` is subtracted outside `statutory_deductions` (02 §4.3, §5.3), so `core.affordability` must accept it as an input.
11. **Affordability modes and answer shapes.**
    - `core.affordability` has no mode input, yet consumers need **four** modes (02 §5.8) and 11 needs a fifth (11 §5.17.2).
    - It must return three answer shapes: pass/fail, maximum instalment, and the facts 03 needs to solve for a maximum amount (02 §5.7.2).
    - It must be monotone in the proposed instalment (02 §5.7.2(c)).
    - 00 §6.5 counts "three consumers". There are five (02 §1).
12. **Adverse events and bureau.**
    - `core.adverse_events` must cover **14** event types (05 §5.5), not 7 (00 §6.14).
    - Its amount thresholds must be supplied by the caller, for example per criticality class (05 §5.5, §13 Q3).
    - It must support three different roll-ups by its consumers (11 §5.17.3).
    - It must be callable for a single event (11 §5.17.6 item 1).
    - `core.bureau` must normalise **commercial** bureau responses as well as consumer ones (05 §4.6).
13. **Business extensions.**
    - `core.appetite` needs a sector dimension (05 §5.11) and a facility-type dimension (11 §6.1 B item 18).
    - `core.fees` needs the unregulated business fee schedule (05 §5.2).
    - 11 §6.1 A item 2 consumes "Sector ratio benchmarks" from the library, but 00 §8 has no such table.

## B. Places where 00 and a consumer disagree

1. **Overlay identifiers.** 00 §4 uses `adjustment_set_id` (int16) and `adjustments_applied` (list[int16]). The consumers use:
   - `adjustment_stack_version` (int32) and `applied_adjustment_ids` (list[string]) (01 §4.4);
   - `overlay_stack_id` (int32) (04 §4.4);
   - `adjustment_set_version` (int32) and `overlay_id` (int16) (08 §4.1).

   Choose one pair and use int32 or wider, since the register is versioned weekly (00 §8).
2. **What an overlay may do.**
   - **Expiry.** 00 §6.22 property 5 only requires that an expired overlay "surface". 01 §6.5 property 5 says it "lapses automatically". Require both: at effective-to the overlay lapses and the lapse is recorded; at the review date the overlay is surfaced.
   - **Direction.** Consumers require tighten-only overlays, **rejected when defined** (02 §5.6.2 item 1; 03 §10 item 19; 04 §5.5 item 6; 09 §5.14.6 and §10 item 22). Also, some artefacts are overlay-exempt (01 §6.5). 00 says nothing about either.
   - **Kinds.** Beyond 00's nine numeric kinds, consumers need non-numeric ones: severity shift, action escalation, scope restriction and queue reroute (01 §6.5). They also need dials:
     - volume dial (04 §5.3.4);
     - cycle dial and cycle cap (07 §6.6);
     - band boundary shift (08 §5.3);
     - objective re-weight (06 §6.3).
   - **Targets.** 00 limits overlays to "values produced elsewhere in the library". Consumers point them at *their own* artefacts (07 §4.3; 04 §5.3.4; 05 §5.5). Scope must include rule family, rule id and event type (01 §6.5), sector and nesting level (05 §2 H7).
   - **Comparison.** Two overlay stacks from different dates must be comparable (11 §5.17.2).
3. **Who owns parameters.** 00 §7.2 classes the affordability buffer as library-policy that consumers cannot set, but:
   - 07 supplies its own buffer (18% against 12%) as "a parameter of the assessment, supplied by this project" (07 §5.6);
   - 08 supplies its own distressed thresholds (08 §5.6);
   - 10 replaces whole library tables: tax 9×4, norms 15×7, treatment matrix 52×7, a 412-code registry (10 §1.1, §5.7(b)–(c), §6.1).

   **Requirement:** a consumer may substitute a *whole, versioned* table family or parameter set that it owns and that is recorded as such. It may never override individual cells of a library table.
4. **Expense norm floor in arrangement mode.** 02 §5.8 says "the norm floor still binds". 08 §5.6 says the floor is only "a plausibility floor for recording purposes rather than as a disqualifier". The owner of `core.expense_norms` must decide which applies and state it.
5. **`total_exposure` has two producers**, `core.obligations` (00 §6.4; 02 §5.5) and `core.exposure` (00 §6.16; 11 §5.17.1). Rename one of them. Also, 03 §4.2 reads `group_exposure_limit`, which 00 does not publish (00 publishes `exposure_headroom`).
6. **Performance envelope.** 00 §9 allows "15 ms per application". 06 §5.6.2 needs 3 200–5 200 invocations in 900 ms, about 200 µs each. 02 §8 needs under 1.5 ms per repeated call, and 11 §4.9 makes 200–600 pricing calls per assessment. State a **per-invocation** budget for repeated calls.
7. **Table owners.** 00 §8 names Credit Risk Policy as owner of the sector risk table and the grade boundaries. 11 §6.1 A names Sector Analytics and Credit Risk Modelling.

## C. Stale, obsolete or unused in 00

1. **Counts.**
   - Capabilities: "twenty-two" (00 §6), "twenty-one" (00 §13 Q1; 06 §4.7; 09 §3, §4.3).
   - Consumers: "six product teams" (00 §1, §10 item 1) and "all ten specs" / "other nine specs" (00 §5, §2). There are 11 consuming specs, and 11 §4.2 adds products 52–58 to the catalogue in 00 §5.
   - Table families: 09 §4.2 says 15, but 00 §8 lists 16.
   - 03 §2 says "ten capabilities" and lists 13.
2. **Consumer lists that are out of date.**
   - §6.22 omits 01, 02, 10 and 11.
   - §6.21 omits 01, 06 and 11.
   - §6.4 omits 10 (P14 needs the per-account detail).
   - §6.5 says "three consumers".
3. **Dead references.** "doc 01 §5.1", "doc 04 §6" and `decider2` (00 §1, §13 Q9) point to design documents that no longer exist. Treat them as background only.
4. **Unused by any consumer:**
   - the inverse direction of `core.calibration` (00 §6.11);
   - `exposure_headroom`, `cover_type_code`, `credit_life_cap_applied` and `total_interest`;
   - the sector risk table (00 §8), which no 00 capability reads. Only 05 and 11 read it directly.

   Keep them, but do not treat them as acceptance-critical. Conversely, the inverse of `core.instalment` *is* needed (03 §13 Q10; 06 §5.6.2).
