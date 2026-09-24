# Flex Loan Granting and Pricing (Project 03)

## Business Decision
Decides whether to approve, approve-with-conditions, refer, or decline a personal flex loan application. If approved, determines the interest rate and monthly instalment offered.

## Inputs and Outputs

**Inputs:** Applicant demographics (age, employment status, tenure), loan request (amount, term, purpose), credit bureau data (credit score, account history, arrears), internal bank exposure, fraud verdict, and compliance consent flags.

**Outputs:** Final outcome code (approve/approve-with-conditions/refer/decline), referral queue if referring, one recommended offer (amount, rate, instalment, total cost, effective annual rate) plus multiple alternative offers per term, decline/suppression reason codes, risk grade and score.

## Process (11 Stages)

1. **Eligibility gates** (14 tests): product/channel validation, age, employment type, debt review status, insolvency, exclusion lists, etc.
2. **Consent verdict**: regulatory consent check
3. **Fraud handling**: fraud verdict with small-loan/long-tenure bypass
4. **Bureau data quality**: verdict on data completeness and staleness
5. **Scoring**: credit scorecard with segment selection (thin-file/new-to-bank/existing), calibration, and risk grading
6. **Waterfall (52 rules)**: applies amount/term/grade ceilings per risk appetite rules and regulatory constraints
7. **Affordability**: consumed from project 02; calculates max affordable monthly instalment
8. **Bounded solve**: runs for each of 9 permitted terms; finds maximum affordable loan amount via banded top-down bisection
9. **Offer set construction**: filters each term's solution through 5 minimum viability rules, deduplicates, ranks by objective (largest amount/lowest cost/best expected value)
10. **Final validation**: re-checks the recommended offer
11. **Outcome resolution**: combines all verdicts into one final outcome and reason codes

## Hard to Understand

- **Bounded solve algorithm (solve.py):** Uses banded bisection not simple binary search because instalment affordability is non-monotone *across* rate-card bands (band edges have 41 documented inversions). Within one band monotonicity holds, so bisection is valid per band. Algorithm walks bands top-down, evaluates cheapest point per band before deciding whether to bisect it.
- **No short-circuit:** Framework runs all 11 stages for every application regardless of early failure (eligibility, fraud) unlike spec's language. Outcome resolution stage resolves precedence.
- **Offer set logic:** Each term's solve result is independently evaluated, filtered, and ranked. Multiple offers (up to 9 terms × offers-per-term) feed into recommendation selection by configured objective.
