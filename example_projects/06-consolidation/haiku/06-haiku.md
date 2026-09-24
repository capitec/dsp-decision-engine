# Project 06: Consolidation and Restructure (Haiku)

## What It Decides

Whether to consolidate a client's multiple loans/debts into a single product and which settlement set + product combination best serves them. Targets clients in financial distress or seeking new money.

## Inputs

- **Client metadata**: ID, application ID, decision date, channel/mode codes
- **Financial position**: Gross/net monthly income, living expenses, requested advance amount
- **Obligations inventory**: List of accounts with balance, instalment, rate, term, status, provider, open date
- **Policy thresholds**: 10 constraint parameters (max accounts, term extension limits, DSR ceiling, etc.)
- **Evaluation config**: Scenario budget (how many to evaluate) and objective ID

## Outputs

- **Eligibility verdict**: Boolean decision
- **Winner scenario**: Settlement set, product code, term, instalment relief, cost delta
- **Top 3 alternatives**: Ranked viable scenarios
- **Analysis**: Settleable account count, baseline instalment/rate/cost, scenarios evaluated

## Main Steps

1. **Parse request** into Account and policy objects
2. **Classify settleability**: Determine which accounts can legally consolidate (blocked by status, provider, policy, or security)
3. **Calculate settlement amounts**: Balance + buffer + fees per account
4. **Baseline assessment**: Current position (instalment, rate, DSR, arrears)
5. **Generate candidates**: Create ~400 scenarios via heuristics (by rate, product, term combinations)
6. **Evaluate scenarios**: Check product constraints (e.g., flex consolidation needs ≥2 accounts, ≥60% external), compute affordability
7. **Select winner**: Pick top scenario under objective (currently hardcoded: minimize monthly instalment)

## Hard to Understand from Code Alone

- **Settleability codes**: Domain logic (account types 20/21=revolving, status codes 3–7=blocked, provider >200=blocked) requires financial/regulatory context
- **Product constraints**: Why flex consolidation (11) requires ≥60% external vs. balance transfer (20) requires all-revolving is implicit business rule
- **Pricing**: Entirely stubbed (new instalment=300, total cost=20000)—real logic would call external pricing engine
- **Objective weights**: Only objective_id=2 (minimize instalment) is coded; others return first viable scenario
- **Settlement amount formula**: 1.5% buffer capped R2500 + 2% charge for high-rate accounts—calibration ratios unexplained

