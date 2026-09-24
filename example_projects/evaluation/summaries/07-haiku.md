# Project 07: Credit Limit Management — Haiku Summary

## Business Decision
Determines whether to increase a customer's credit limit on a revolving credit product (Card or Facility) and by how much, balancing risk, affordability, and portfolio budget constraints.

## Inputs & Outputs

**Inputs** (20 account fields + 5 policy parameters):
- Account state: current limit, balance, tenure, utilization, delinquency history, cash withdrawal ratio
- Customer consent and treatment status
- Income and existing obligations
- Policy controls: dial multiplier, income cap factor, spend cap, portfolio budget, over-allocation factor

**Outputs**:
- `is_excluded`: hard rejection flag (16 codes, 9 implemented)
- `behaviour_grade`: 1–12 credit risk grade
- `probability_of_default`: scaled to [0.1%, 15%] range
- `proposed_limit`: final capped limit recommendation
- `binding_cap_code`: which constraint (product max / income / spend) limited the offer
- `affordability_verdict_code`: pass/fail on discretionary income
- `ranking_value`: economic value (revenue minus expected loss per unit)
- `funded`: whether account received allocation from portfolio budget

## Main Steps (Stages 5.1–5.8)

1. **Capture account state** — snapshot Account dataclass
2. **Hard exclusions** — 16 rules: arrears, no consent, tenure <6m, at product max, treatment code set
3. **Behaviour score** — synthetic scorecard (600–850 range) from delinquency, utilization, tenure, over-limit, cash ratio; map to PD via logistic; grade by PD bucket
4. **Matrix lookup** — 4D cell (grade × util_band × mob_band × product) returns limit multiplier (1.0–1.75), apply cycle dial overlay to attenuate the excess
5. **Apply caps** — waterfall: product maximum, income multiple, observed spend; take lowest
6. **Affordability** — instalment vs discretionary income (product-specific min payment rate)
7. **Ranking** — net economic value: (CCF × incremental draw × margin − PD × LGD × proposed_limit) / proposed_increase
8. **Allocation** — fund in ranking order until budget envelope (with 1.38× over-allocation factor) exhausted

## Hard to Understand from Code Alone

- **Synthetic scoring to PD mapping** (limit_management.py:199–203): applies logistic curve to a 600–850 behaviour score, then scales to [0.1%, 15%]. The constants (100 in the exponent, 0.001 baseline, 0.15 scale) lack calibration context and appear conservative/arbitrary.
- **Cycle dial overlay** (line 279–294): multiplies the *excess above 1.0* by the dial factor (50–100%), not the whole multiplier. This is unintuitive from the code alone; the docstring clarifies the intent.
- **CCF and revenue calculation** (pipeline.py:112–114): Credit Conversion Factor (42% for Card, 55% for Facility) and margin (9% NIM + 1% fees) are hardcoded without explanation of their source or validation approach.
- **Ranking formula denominator** (pipeline.py:117): dividing by `proposed_increase` creates degeneracy when the proposed limit equals current limit (increase = 0); the code guards with a conditional but the economic logic would benefit from documentation.
- **Matrix stub** (pipeline.py:159–189): real matrix would be external and multi-dimensional; this is a synthetic generation function with no visibility into why specific grade/band combinations get specific multipliers.
