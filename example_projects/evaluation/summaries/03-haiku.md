# Loan Granting & Pricing (Haiku)

## Business Decision
This system decides whether to approve unsecured "Flex Loan" applications and, if approved, generates loan offers with specific amounts, terms, and pricing. For each applicant, it produces an approval/decline outcome plus a set of term-priced offers (if approved).

## Inputs & Outputs

**Inputs** (from request JSON):
- Applicant: age, employment type, tenure months, residency code
- Credit/risk: fraud verdict, bureau score (via stubs), debt review status, arrears counts, credit enquiries
- Financial: net monthly income, living expenses, existing obligations, discretionary income, affordability verdict
- Application: requested amount, requested term, product/channel codes, campaign ID
- Exposure: internal and group limits (determines cap)
- Legal/compliance flags: deceased, estate, admin order, insolvency status, exclusion list hits, in-flight applications

**Outputs**:
- decision_id (UUID)
- outcome ("approve" or "decline")
- validated_offers (list with amount, term, rate, fees, instalment)
- reason_codes (if decline)

## Main Steps

1. **Intake & Parse**: Extract decision date.
2. **Eligibility Gates** (14 gates): Product-channel, min/max age, capacity, residency, employment type, debt review status, admin orders, insolvency, deceased/estate, exclusion lists, duplicates, in-flight applications.
3. **Scorecard & Grading**: Evaluate credit score and assign risk grade (1–12); gated applicants get worst-case.
4. **Cap Waterfall** (52 rules): Determine hard ceilings for amount, term, and acceptable risk grade based on employment tenure, arrears, credit enquiries, employer watchlist, exposure limits, campaign authority.
5. **Affordability Assessment**: Verify applicant can afford any offer based on max monthly instalment.
6. **Solve & Pricing**: Run bounded search (max 24 evals per term) to find largest amount per term that stays within affordability and cap limits; price each using rate card, fees, and credit life.
7. **Construct Offers**: Build full offer set from term results with suppression rules.
8. **Final Validation**: Ensure each offer meets affordability, cap, and grade constraints.

## Hard to Understand

- **Bounded solve algorithm**: Non-monotonic affordability (band-edge rate-card inversions) requires scanning downward from requested amount; eval ceiling prevents exhaustive search. Constraint binding logic opaque without test cases.
- **52-rule cap waterfall**: Rules themselves not visible in code; only witness verdicts and final ceilings. Relationship between rules and chains unclear.
- **Eligibility gate chaining**: All 14 gates evaluated even after failures (per spec §5.1); flow from verdicts to final is_eligible boolean is straightforward, but business significance of each gate requires external spec.
- **Pricing model**: Stub implementations (rate card interpolation, fee schedule, credit life calculation) are simplified; real model likely more complex. How rate card handles non-existent cells not explained.
