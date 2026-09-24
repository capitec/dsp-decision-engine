# Retail Credit End-to-End Decision Engine (Entry Point 1 - Haiku)

## What This Project Decides

A new credit application approval engine for South African retail credit. Given a loan request (product 10: Flex Loan), it evaluates whether to approve or decline, and if approved, what terms to offer (amount, term, instalment, interest rate). Decision hinges on eligibility, fraud risk, credit score, risk grade, affordability, policy constraints, and consolidation eligibility.

## Inputs and Outputs

**Inputs:**
- Identity number, product code (10), requested amount (2k–500k ZAR), term (6–84 months), channel
- Gross monthly income, dependants, employment type, bureau account count, internal tenure
- Consent flags (bureau enquiry, data sharing), consolidation eligibility

**Outputs:**
- Decision record: outcome code (1=approved, 2=decline, 3=refer), reason codes, decision ID
- If approved: offer with proposed amount, term, monthly instalment, annual rate (5–30%), risk grade (1–12)
- Shared state: client ID, score, segment, affordability verdict, loop pass count

## Main Steps (Entry Point 1, 17 phases)

1. **P01** – Validate request (structure, domains, cross-fields); assign phase set
2. **P02** – Resolve client identity (confidence scoring, degradation handling)
3. **P03** – Verify consent and hard eligibility (must have bureau consent)
4. **P04** – Orchestrate external data calls (bureau, fraud, statements; 600ms budget total)
5. **P05** – Fraud assessment (5 rule families, 620+ rules; verdict: pass/soft/hard/refer)
6. **P06** – Derive features: net income, living expenses, obligations, segment (thin/thick file)
7. **P07** – Score using scorecard A3 (300–850; income-derived)
8. **P08** – Convert PD to risk grade (1–12) with segment calibration
9. **P09** – Apply policy gates and caps (amount cap, term cap reduce with grade)
10. **Loop L1** – Affordability re-run (up to 4 passes):
    - **P10** – Affordability: does discretionary income (net – living – obligations – buffer) cover instalment?
    - If fails and consolidation-eligible: **P14** (search consolidation scenarios; stub in this version)
    - Loop until passes or max iterations
11. **P11** – Product routing (EP1 forced to product 10)
12. **P12** – Pricing: rate lookup by grade, compute initiation fee, monthly service fee, instalment
13. **P13** – The solve: constrain amount/term to satisfy affordability + caps (finds binding constraint)
14. **P15** – Skip (limit assignment is for products 20/21 only)
15. **P16** – Assemble offers (typically 1 for EP1; more for multi-product entry points)
16. **P17** – Final validation (14 assertions for product 10: instalment, amount, term, rate, grade ranges)
17. **P18** – Emit decision record with outcome codes and metadata

## Parts Hard to Understand from Code Alone

1. **Loop L1 consolidation mechanism** – Code shows the loop structure but doesn't actually simulate different consolidation scenarios or update existing_obligations between iterations. Comments reference §5.21.1 "hard case: two simultaneous versions of existing_obligations" but the implementation is stubbed.

2. **P14 consolidation search** – Mentions up to 250 scenarios (real) vs. stub showing 0. Unclear what a scenario is operationally or how chosen_scenario_ref actually gets used in the loop.

3. **Binding constraint resolution in P13** – Code always returns "affordability" but real version iteratively re-solves amount/term/rate to find which constraint (affordability | amount_cap | term_cap | grade_cap) actually limits the offer.

4. **Entry points 2–8 and phase set routing** – Only EP1 is implemented; others are stubs returning status="stub". No visibility into which phases run for each entry point (§5.20) without reading the spec.

5. **Degradation modes** – Code marks degradation_mode and identity_degraded flags but doesn't show how external service failures (bureau_down, identity_verification_service_down) cascade into rejected decisions.
