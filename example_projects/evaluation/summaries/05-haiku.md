# Project 05: Business Credit Assessment with Nested Entities (Haiku)

## Business Decision

Approves, refers, or declines a business credit facility (loan) request based on the credit history and structure of the applicant and controlling entities.

## Inputs & Outputs

**Inputs:**
- Loan request (amount, term, facility purpose)
- Business profile (registration, tax compliance, turnover, sector)
- Entity graph: 1–40 people/companies with ownership percentages, control flags, and roles
- Adverse events: legal judgments, payment defaults, fraud flags linked by entity key

**Outputs:**
- Verdict per entity (Clear, Minor, Material, or Disqualifying)
- Entity risk grades and probability-of-default estimates
- People blend: aggregated PD and risk grade across owners
- Application verdict: Approve, Refer, or Decline (with decline event/rule if applicable)

## Main Steps

1. **Structure resolution (5.1):** Count entities, sum ownership percentages
2. **Criticality classification (5.4):** Assign each entity to Critical/Significant/Peripheral based on control, ownership ≥25%, required surety, or sole-role status
3. **Adverse event classification (5.5):** Assign event severity (Immaterial/Minor/Material/Disqualifying) using amount thresholds tuned by criticality; disputed events are downgraded one class
4. **Entity verdict roll-up (5.6):** Apply 12 rules to combine events into one entity verdict (e.g., any disqualifying event → disqualifying verdict; ≥3 recent minors → material)
5. **Entity scoring (5.7):** Map verdict to risk grade (4–12) and PD estimate
6. **People blend (5.8):** Verify ownership coverage (≥75% rule) and average entity PDs
7. **Combined business grade (5.10):** Decide verdict (Decline if any entity disqualifying; Refer if coverage <75%; else Approve)

## Ambiguities in Code

- **Effective vs. direct ownership:** `direct_ownership_pct` and `effective_ownership_pct` are stored but the code only uses effective; how they differ (e.g., through indirection chains) is not shown
- **Structure resolution stubbed:** Comments say "assume entities are already resolved"—the ownership path resolution logic is absent
- **Roll-up rules incomplete:** Code references "12 rules" but implements only 6 (AE-R-01, 02, 03, 06, 08, 12); others are not defined
- **Overlay thresholds never applied:** `threshold_material_overlaid` and `threshold_disqualifying_overlaid` fields exist but are never modified (comment cites "stack position 1")
- **People blend simplified:** Uses simple average PD across owners; no weighting by ownership %
- **Business grade stub:** Stage 5.10 just copies people grade; no business financials (turnover, sector risk) feed into the final verdict
- **Deceased flag unused:** `Entity.deceased_flag` field is defined but never referenced
