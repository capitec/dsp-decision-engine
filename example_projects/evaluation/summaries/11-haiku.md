# Project 11 (Business Credit E2E) - Haiku Implementation

## What It Decides (Business Terms)

This system makes three types of credit decisions for business lending:

1. **EP-1 Origination**: Approve/refer/decline new business loan applications with a business grade (1-12 scale)
2. **EP-3 Annual Review (L1)**: Re-assess existing facilities annually, compare grade changes to prior decision, and explain the migration
3. **L2 Covenant Test**: Monitor a single covenant (DSCR ≥ 1.25) after funding, classify breaches as material/severe

## Inputs and Outputs

**Inputs:**
- Application/facility ID, product code (50=term, 51=revolving), loan amount, term
- Entity list (directors, ownership percentages, criticality, adverse events)
- Financial metrics (EBITDA, annual debt service)
- Prior grade and master scale version (for annual review)

**Outputs:**
- Business grade, verdict code (1=clear, 2=refer, 3=decline)
- For review: grade migration (current−prior), reason codes (data change, model change, scale change)
- For covenant: measured DSCR, breach class code, headroom percentage

## Main Steps

1. **Origination**: Transparent pass-through to project 05's `assess_business_credit()` with entity and event data
2. **Annual Review**: Re-assess via origination, then compare current grade against prior grade; decompose movement into data/model/overlay/scale components
3. **Covenant Test**: Calculate DSCR ratio, classify breach severity (0=none, 2=material, 4=severe), return headroom as percentage above/below threshold

## Hard to Understand from Code Alone

- **Full assessment logic is stubbed**: The actual business credit grading lives in external project 05. This implementation reuses it but the scoring rules (1 280 of 1 900 decision points) are not visible here.
- **Grade decomposition is simplified**: Code mentions four causes of migration (data, model, overlay, scale) but only checks master scale version and defaults to "data or model change"—the full comparison logic referenced in §5.10 is not implemented.
- **Master scale versioning purpose**: Why versions are tracked and how version changes affect comparability isn't explained in the code.
- **Entity structure resolution (O2)**: Mentioned as reusing 05's component but no visible logic for resolving entity hierarchies or dependencies.
