# Affordability Assessment (Project 02) - Sonnet Implementation

## Business Decision
Determines whether a borrower can afford a proposed loan instalment by assessing discretionary income against household obligations. Produces three answer types: affordability pass/fail/marginal against a specific instalment, capacity-only (max affordable amount), or a decline reason if evidence is insufficient.

## Inputs & Outputs

**Inputs:**
- Applicant(s) income evidence: payslip, variable pay history, declared and bank statement expenses
- Credit exposure: bureau accounts (balances, instalments, arrears) and internal accounts
- Application metadata: product/risk grade, assessment mode (new/limit increase/arrangement/scenario), dependants, proposed instalment (optional)
- Data freshness: bureau as-of date

**Outputs:**
- Gross/net household income with verification tier and haircut ratios applied
- Living expenses (statutory norm or statement-derived)
- Total existing obligations and exposure metrics (worst arrears, revolving utilisation)
- Discretionary income before and after proposed instalment
- **Verdict codes:** PASS (1), MARGINAL (2), FAIL (3), or INDETERMINATE (4)
- Evidence sufficiency code (why assessment is indeterminate, if applicable)
- Decline reason codes from a registry (e.g. income unestablished, bureau stale, instalment exceeds capacity)

## Main Steps (in order)

1. **Household Framing:** Consolidate two applicants' expenses (shared categories take max, personal sum); extract dependants count (higher of pair).

2. **Income Assessment:** Run `credit_core.income` unit twice (once per applicant), apply mode-specific verification tier thresholds (new app: tier 4, arrangement: tier 6), calculate haircuts for confidence levels, derive household gross/net income.

3. **Deductions:** Statutory deductions (tax, social), consolidate across applicants; merge with court-ordered deductions.

4. **Living Expenses:** Look up parametric norms (tables by income band, dependants, expense basis); optionally override with bank statement figures where verified.

5. **Obligations:** Consolidate bureau and internal accounts; calculate total monthly obligations, arrears profiles, revolving utilisation.

6. **Discretionary Income:** Net income minus living expenses minus obligations; constrained by two caps: (a) proportional buffer (10–35% by risk grade, per-product loadings), (b) absolute residual floor (800–2,300 per dependants count). Binding constraint tracked.

7. **Overlays & Verdict:** Apply tighten-only adjustment stack to max affordable instalment; check evidence gates (income gaps, stale bureau, refer accounts, tier below minimum, zero net income) in ombudsman priority order; derive verdict—INDETERMINATE if any gate fails, else PASS if no instalment proposed or proposed ≤ max, MARGINAL if within 5% band, FAIL if above.

## Hard-to-Understand Parts

- **Household compositing:** The `.relabel()` and `.named()` pattern that runs income/deductions units twice (once per applicant) then combines outputs under generic household field names without explicit branch logic. Only evident by tracing the DAG step-by-step; feels implicit rather than declarative.

- **Evidence indeterminacy ordering:** Why checks run in this order (income gaps, bureau staleness, refer accounts, tier, net income > zero) and why INDETERMINATE is conceptually distinct from FAIL. Requires understanding the spec's intent (ombudsman scrutiny priority) and reflects regulatory precedent, not obvious from arithmetic alone.

- **Shape (a) vs. (b) verdict dispatch:** The verdict function's behaviour changes entirely based on whether `proposed_instalment` is None, but this conditional switching of "answer shape" (affordability pass/fail/marginal vs. capacity-only pass/indeterminate) is not reflected in a separate code path or named mode—only implicit in the function signature and a docstring note. Makes the verdict logic less obvious than it first appears.
