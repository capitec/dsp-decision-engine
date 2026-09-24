# Business-Nested-Entities (Project 05) — Sonnet Implementation

## Decision & Business Purpose
Determines credit granting eligibility and pricing for SME business facilities. Takes a business application with nested ownership structure (owners, directors, shareholders as natural persons) and decides: approve with offer terms, refer for manual review, or decline. Outputs a single risk grade (1–12) and, if approved, a loan offer with interest rate, fees, and installment amount.

## Inputs & Outputs

**Inputs:**
- Application metadata (ID, date, product code, sector)
- Requested amount, term, security type  
- Business financials (EBITDA, current assets/liabilities, interest-bearing debt, net worth)
- Nested entities: list of natural persons with ownership %, control flags, control relationships, bureau score, credit history, and nested adverse events (judgments, defaults) per person

**Outputs:**
- Outcome: approve/refer/decline with reason codes
- Risk grade (1–12), probability of default
- If approved: offered amount, nominal annual rate, fees (initiation + monthly service), installment, total cost, effective annual rate
- Attribution: which entity/events drove decline decision

## Main Steps (In Order)

1. **Structure resolution** – Flatten nested entities + adverse events into parallel long-form lists; merge duplicate entities (same person reached via two ownership paths); assign criticality class (critical/significant/peripheral) based on ownership %, control, and surety status

2. **Adverse event classification** – Severity-code each event based on type, amount, and age using configurable thresholds and overlays

3. **Roll-up** – Aggregate event severities per entity into entity verdict (pass/refer/decline)

4. **Entity scoring** – Scorecard per natural person (bureau score, months on record, worst delinquency, ownership stake) → score → PD → grade

5. **People blend** – Select included entities (owners ≥5%, controllers, certain roles), weight them (ownership % or control points), blend their PD via log-odds, cap rules on grades 11–12

6. **Financial analysis** – Three ratios: interest cover, current ratio, gearing; assign financial PD and grade

7. **Combined business grade** – Blend financial PD (40% weight) + people PD (60%) via log-odds; apply one business-level overlay (sector multiplier); map to risk grade 1–12 using product-specific PD boundaries

8. **Pricing** – Single lookup: grade → appetite ceiling; cap offered amount; grade + security type → rate card; compute fees and installment

9. **Sole proprietor check** – If single natural person owns 100%, run statutory affordability test

10. **Outcome** – Hierarchy: structure unresolved → refer; entity declined → decline; insufficient people coverage → refer; sole proprietor failed → decline; grade ≥11 → decline; grade ≥9 → refer; else approve

## Hard to Understand

- **Framework mismatch**: decider cannot emit nested columns (e.g., list[struct]), so the nested input (entities → adverse_events) must be unpacked immediately into three parallel lists (entities, events, attributions) joined by ID. This normalisation happens in stage 1, not naturally in the input/output layer.
- **Nested scorecard**: Each entity runs a small scorecard (structure → severity → scoring), but decider has no "loop once per row element" construct. Workaround: build a second Engine at import time and call `Engine.run()` inside a `frame_step`, against a per-row slice of the entity lists.
- **Date handling**: `datetime.date` columns written by one `frame_step` are silently downcast to epoch integers when read by a downstream `frame_step`, causing `.year` calls to fail. Dates must be converted to computed values (e.g., age_months) inside the same step that reads them.
- **Type coherence**: Ownership reconciliation (sum of ownership % should be 90–110%) and path deduplication (same person reached twice) are both resolved in stage 1, but the logic depends on entity_id and entity_key (both from caller) being stable and unique.
- **Decline attribution**: Reasons are ranked and the primary reason + attributing entity/event IDs are pulled from a flat attribution table built during roll-up. The link between entity verdict, triggering event, and reason code is implicit in the rule that fired.
