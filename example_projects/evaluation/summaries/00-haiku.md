# Shared Credit Core (Haiku)

## Business Purpose

Automated credit approval decision engine for retail installment lending. Evaluates applicant creditworthiness, affordability, and assigns risk-based pricing to determine whether to approve a loan application and at what terms.

## Inputs & Outputs

**Inputs**: Client/application IDs, decision date; income (declared and payslip); employment tenure and type; dependants; living expenses; existing bureau obligations; proposed instalment; loan amount and term.

**Outputs**: Decision metadata (ID, date); assessed income, deductions, expenses; affordability verdict; credit score and calibrated probability of default; risk grade (1–12); pricing (rate, fees, monthly instalment); gates (eligibility, appetite, exposure, consent).

## Main Steps (11 stages)

1. Generate unique decision ID  
2. Assess gross monthly income (declared vs. payslip with tenure haircut)  
3. Calculate statutory deductions → net income  
4. Apply expense norms (based on income, dependants, declared expenses)  
5. Aggregate existing bureau payment obligations  
6. Assess affordability: discretionary income vs. proposed instalment (pass/fail/indeterminate)  
7. Score applicant via scorecard (bins income, tenure, adverse events)  
8. Calibrate raw score → probability of default  
9. Map PD to risk grade (1=best, 12=worst; prime/near-prime/subprime/deep-subprime)  
10. Price product: lookup rate by grade/term, add fees, calculate monthly instalment  
11. Check gates: eligibility, appetite limits, total exposure, consent consent

## Hard to Understand from Code Alone

**Affordability modes**: Code names four modes (standard, limit_increase, arrangement, restructure) tied to regulatory guides ("00-ADDENDUM A11") but their practical behavior differences are not evident—both arrangement and standard apply the same buffer logic.

**Scorecard characteristics**: Pipeline assumes income_level, employment_tenure, adverse_events but passes hardcoded tenure (24.0) and adverse_events (0.0) regardless of input; unclear how adverse events are derived from bureau data.

**Calibration**: Score is passed to `calibrate_score()` but the calibration curve (score → PD mapping) is opaque; version and segment logic not shown.

**Risk grade boundaries**: Hardcoded in pipeline per segment rather than externalised; unclear if production values or placeholders.

**Decision outcome**: Affordability verdict (codes 1–4) is returned but final approval/decline logic is never produced—no explicit "approved" or "declined" field in outputs.
