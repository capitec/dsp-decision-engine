# Project 10: Retail Credit E2E (Sonnet)

## Business Decision

Approves or declines unsecured retail credit (personal loans) for product 10. If approved, determines the offered amount, term, interest rate, and instalment amount. Entry point 1 covers new credit applications from both new and existing clients.

## Inputs and Outputs

**Inputs:** Applicant demographics (age, dependants, employment), identity/compliance flags (residency, sanctions, debt review), consent state, credit history (bureau accounts, internal holdings, payment arrears), fraud risk signals (device age, address changes, synthetic identity score), income (payslip + variable pay), expenses, existing obligations, campaign/channel context.

**Outputs:** A decision record containing: approval verdict, offered amount/term/instalment/rate, risk grade, decline reasons (ranked, with primary reason), scoring details (unadjusted score, adjustments applied), affordability verdict, validation status, loop pass count (if consolidation considered), and audit trail of table versions/overlays used.

## Main Steps in Order

1. **P01** – Validate request format
2. **P02** – Resolve identity (matching confidence)
3. **P03** – Hard eligibility gates & consent verdict (all evaluated, none short-circuited)
4. **P04** – Acquisition checks (bureau down fallback, exclusion list)
5. **P05** – Fraud evaluation
6. **P06** – Derive income, deductions, living expenses norms, obligations; assign segment
7. **P08** – Score applicant, calibrate to probability of default, assign risk grade, apply overlays
8. **P09a** – Evaluate amount/term/grade caps
9. **P10** – Affordability: discretionary income → capacity → ratio ceiling → buffer → max affordable instalment
10. **Solve** – Optimize offered amount/term within affordability constraint (R250 grid-aligned)
11. **L1 Loop** – If affordability failed AND consolidation-eligible: iterate with hypothetical obligations
12. **P16** – Pick offer (loop result if fired; else solve result)
13. **P12** – Price offer: rate card lookup, fee, credit life premium, instalment calculation, rounding
14. **Validation** – Revalidate offer against recalculated instalment
15. **P18** – Rank all decline reasons, assign outcome code

## Unclear From Code Alone

- **Solve mechanics**: The code references `solve.solve_step_wired` but does not show what constraints it applies or its algorithm beyond "R250 grid-aligned."
- **Overlay resolution timing**: Why overlays must be resolved once at P08 (O-05/O-21) and reused later by reference, not re-resolved — likely to ensure consistency, but the code doesn't explain when/why re-resolving would break things.
- **Entry points 2-8**: Declared as a full matrix but only EP1 has real logic; entry points 2-8 are routing only (proven via test, not execution). The boundary between "stub" and "real" is clear but the business reason for supporting eight entry points when only one is demonstrated is not apparent from the code.
- **L1 loop convergence**: The seed function shows a boolean `loop_converged` that starts `True`, but the loop module itself is not read, so the exit condition is opaque.
