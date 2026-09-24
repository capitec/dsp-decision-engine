# Project 06: Debt Consolidation Decision Engine

## What it Decides

Whether to approve a client's request to consolidate multiple debts into a single new loan (Flex Loan Consolidation or Everyday Card balance transfer), and if approved, which accounts to consolidate, at what terms and rate, to minimize their financial burden.

## Inputs and Outputs

**Inputs:** Client profile, accounts with settlement quotations, income/expenses, bureau data, objective weights

**Outputs:** Outcome (approve/decline/refer), recommended product/terms/rate, and detailed scenario scores

## Main Steps in Order

1. **Eligibility gates**: Check debt review/administration status, reject obvious failures
2. **Settleability classification**: Flag which accounts can be consolidated (internal, quoted, or quotable)
3. **Settlement derivation**: Calculate payoff amounts (interest, fees, security release costs)
4. **Baseline assessment**: Current obligations and debt-service ratio; short-circuit test (if achievable rate is good, defer to project 03)
5. **Scenario generation**: Create ~400 candidate consolidations within 900ms budget
6. **Scenario evaluation**: Price each on products 11/20, apply affordability rules, reject non-viable
7. **Objective scoring**: Rank scenarios by weighted objective (cost, instalment relief, bank value)
8. **Outcome routing**: Output best scenario or decline with reason codes

## Hardest to Understand from Code Alone

**Rejection reason codes (CON-INT-01..14)**: These encode business constraints (rate ceilings, debt-service limits, new-money caps) that aren't obviously linked to code—you need the spec to know why scenarios fail. Reasons emerge implicitly from numeric checks, not explicit rule objects.

**Objective scoring**: Five measures (cost, instalment, new money, bank value, client outcome) are emitted as parallel primitive lists rather than structured objects, making the scoring logic harder to follow.
