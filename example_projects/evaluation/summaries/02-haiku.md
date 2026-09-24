# Project 02: Affordability Assessment (Haiku)

## What It Decides

This system determines whether an applicant can afford a proposed credit product (loan/credit line) based on a seven-stage financial assessment. It outputs a verdict (pass/fail/marginal/indeterminate) and quantifies the maximum affordable monthly instalment.

## Inputs and Outputs

**Inputs:** Applicant demographics (age, dependants, joint status) • Income sources (primary/secondary, employment type, tenure) • Deductions (statutory, court-ordered) • Declared living expenses • Existing credit accounts from bureau and internal systems (balances, payments, arrears) • Risk grade and proposed monthly instalment.

**Outputs:** Gross and net monthly income • Living expenses and statutory deductions • Total existing obligations (internal, external, exposure) • Discretionary income after all expenses and obligations • Maximum affordable instalment • Affordability verdict and evidence sufficiency code.

## Main Steps

1. **Household framing:** Establish applicant structure (single/joint, dependants count) as context.
2. **Income determination:** Calculate gross income from primary and secondary sources, apply income haircuts based on employment type and tenure, determine verification tier.
3. **Statutory deductions:** Compute PAYE and other mandatory deductions from gross income.
4. **Living expenses:** Apply standardized expense norms based on income level and dependant count; can adjust based on declared or statement-derived expenses.
5. **Existing obligations:** Sum monthly payments and total exposure across bureau and internal accounts; flag arrears severity.
6. **Discretionary income and capacity:** Calculate available income (net income minus all expenses, deductions, and obligations), apply risk-based buffer (proportional + floor-based minimum).
7. **Verdict:** Compare proposed instalment against maximum affordable; adjust verdict based on evidence quality (weak evidence in new-application mode yields indeterminate).

## Hard-to-Understand Parts

- **Evidence sufficiency logic:** Verdict depends on income tier (declared/bureau-estimated = weak, payslip/bank statement = strong). Weak evidence in new-application mode forces "indeterminate" regardless of affordability, while other modes allow pass/fail. The verification tier itself is determined in income calculation but used only at verdict time.
- **Buffer calculation:** Uses max(proportional 12% of discretionary, floor of 2000 + 500 per dependant). The floor basis and percentage are hardcoded, and it's unclear when proportional vs. floor dominates in practice.
- **Secondary income treatment:** Takes the weakest (highest) tier between primary and secondary income sources, but doesn't explicitly show how secondary employment type affects the combined haircut.
