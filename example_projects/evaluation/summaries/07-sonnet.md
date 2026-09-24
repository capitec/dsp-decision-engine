# Project 07: Credit Limit Management

## Business Decision

Per-account credit limit adjustment for revolving products (everyday card, access facility). For each existing customer account, decide whether to increase, decrease, or leave the limit unchanged; classify the change type (automatic offer vs. conditional/consent-required vs. none); assign a notice class (immediate action, requires consent, or N/A); and provide decline reasons if the customer is excluded or fails affordability.

## Inputs and Outputs

**Inputs** (from `sample_request.json`):
- Account state: `current_limit`, `cycle_balances`, `cycle_limits`, `months_on_book`, `product_code`
- Behavior: arrears history, payment/balance ratios, utilization over 6 months, over-limit cycles, cash withdrawal ratio
- Income: declared gross/net, verified salary deposit, salary variability, income age, expenses (declared + statement)
- Credit bureau: external accounts, inquiries, arrears on other obligations, bureau as-of date
- Life-stage: fraud flag, deceased flag, debt review status, treatment suspension, consent state

**Outputs**:
- `final_proposed_limit`: the new credit limit (equals current limit if excluded/affordability fails)
- `change_type_code`: 1=automatic increase, 2=conditional increase, 3=decrease, 4=no change
- `notice_class_code`: communication type (1=immediate, 2=consent required, 3=N/A)
- `decline_reason_codes`: list of exclusion or policy-block reason codes
- Supporting: behavior score, probability of default, behavior grade, all intermediate caps and assessment tiers

## Main Steps

1. **Population**: Derive utilization/tenure bands, mean utilization across 6 months, P90 observed spend
2. **Exclusions**: Evaluate 8 hard exclusion codes (arrears, fraud, debt review, cooling-off, treatment suspension, etc.)
3. **Scoring**: Behavior scorecard (8 characteristics per product), PD calibration table, behavior grade lookup
4. **Matrix**: Look up 1,152-cell assignment matrix (grade × utilization band × tenure band × product) → base multiplier, max increase, min increment
5. **Caps**: Compute 5 policy caps (product max, income multiple, total exposure, observed spend, cycle cap); lowest binding cap wins
6. **Affordability**: Classify evidence tier, map to income shapes, compute affordability verdict, apply staleness rules to produce automatic/conditional/fail classification
7. **Decrease**: Check if decrease trigger (D03) fires; compute decrease target and notice class if triggered
8. **Outcome**: Assemble final proposed limit, change type, notice class, decline reasons

## Hard to Understand

1. **Matrix cell logic**: The 1,152 cells (12 grades × 8 util bands × 6 tenure bands × 2 products) are indexed synthetically in code but the policy intent behind grade/utilization/tenure interactions and why certain cells return 1.00 (no increase) requires understanding credit risk modeling and the spec's rationale.

2. **Affordability evidence bridge**: Project 07 reuses project 02's affordability reassessment without forking it. The mapping between 07's own evidence tiers (verified salary, statement confidence, declared haircuts) and 02's income-evidence shape requires tracing across two pipelines; the "notional instalment" construct as a proxy for the proposed instalment is non-obvious.

3. **Circular dependency avoidance**: Caps use `declared_*_income_on_file` (origination state) while affordability uses freshly-assessed income. The reason is that caps (§5.5) precede affordability (§5.6) in the spec's stage order; using fresh affordability output in caps would create a cycle. The code comment points this out but only in the caps module.

4. **One implementation, three consumers**: The same `Engine`/pipeline serves real-time single-account scoring (`.score()`), batch scoring over the book (`.run()`), and simulation over candidate snapshots. The composition with population-level budget allocation (§5.8, handled by `allocation.py` outside the decider pipeline) adds coupling that is explained in pipeline.py's docstring but not in the code's flow.
