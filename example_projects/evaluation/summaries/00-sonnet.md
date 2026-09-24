# Shared Credit Core (Sonnet) — Project 00

## Business Decision

Determines creditworthiness and pricing for a Flex Loan product. Outputs: approve/decline/refer decision, interest rate, fees, and instalment amount. Combines eligibility gating, income assessment, affordability analysis, and risk-based pricing from a 360-cell rate grid.

## Inputs and Outputs

**Inputs:**
- Applicant demographics: age, dependants, employment (type, tenure)
- Legal/compliance flags: residency, contractual capacity, deceased, exclusion list, sanctions, debt review
- Income: payslip, variable pay history (3 months)
- Obligations: court-ordered deductions, declared living expenses, bureau accounts (external), internal accounts
- Bureau score (0–1000)
- Loan request: amount, term (months)

**Outputs:**
- Gross/net monthly income, income source & verification tier, variability ratio
- Living expenses, norm version, expense basis
- Existing obligations by type (internal/external), worst arrears
- Affordability verdict (Pass/Marginal/Fail), max affordable instalment, discretionary income
- Raw and adjusted risk scores, probability of default, risk grade
- Nominal annual rate (from rate card cell lookup), offered amount, out-of-range flag
- Initiation fee, monthly service fee
- Instalment (pre-fees, post-fees), total interest, effective annual rate
- Decline reason codes (ranked by registry), final outcome (Approve/Decline/Refer/Approve-with-Conditions)

## Main Steps

1. **Eligibility gate**: Verify age, residency, capacity, product availability, exclusion/sanction/death/debt-review status
2. **Income**: Gross monthly income, source code, variability ratio, haircut applied per source type
3. **Deductions**: Tax tables, statutory deductions, net income
4. **Expenses**: Norm tables (statutory/internal), living expense basis, final living expense amount
5. **Obligations**: Sum bureau and internal account instalments; track worst arrears month
6. **Affordability**: Discretionary income = net − expenses − obligations − court orders; max instalment = discretionary × (1 − buffer)
7. **Risk scoring**: Raw scorecard + overlays (segment/channel tightening), calibration table, probability of default
8. **Risk grade**: Map PD to grade bucket
9. **Rate card**: Lookup 360-cell grid by amount/term bins and risk grade
10. **Fees**: Initiation fee (base + marginal, capped), monthly service fee
11. **Instalment**: Pre-fees amortisation → add fees → round to cent; total interest, EAR
12. **Reasons**: Combine eligibility decline codes with affordability fail code, rank via registry
13. **Outcome**: Approve if eligible AND affordability pass; Decline if either fails; Refer if marginal/indeterminate

## Hard to Understand from Code Alone

- **Score adjustments registry**: Two overlays (segment-based, channel-based) both tighten-only, stack position determines order when both apply; unclear how collisions resolve or why stack order matters without seeing scorecard internals.
- **Rate card grid structure**: Parameterized as 360-cell JSON (loaded at serve time), but the amount/term bin boundaries and grade mapping are opaque without inspecting the full file; unclear how out-of-range detection works relative to grid boundaries.
- **Affordability buffer semantics**: Why 12%? Is it risk-appetite driven or regulatory? Code shows it's overrideable per project (07 uses 18%) but no guidance on tuning.
- **Income haircuts**: Five different haircut rates applied per source (payslip 5%, statement 10–25%, declared 20%, bureau 30%), but no rationale visible—appears to reflect data-quality assumptions not explained in code.
- **Reason code ranking**: Registry resolves codes by custom rank; logic for which reason to surface as "primary" is hidden in the registry resolver, not in pipeline.

