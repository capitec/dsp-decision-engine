# Affordability assessment

Before a bank lends, it must show on evidence that the applicant can afford the instalment, and be able to
re-derive that years later for a regulator, an ombud or a court. Lending too much is reckless lending: the
agreement is set aside and the bank is penalised. Lending too little turns away people who could have paid.
This example is that calculation, built from the shared credit library so that every product uses the same one.

## The business problem

One regulated calculation has many callers: a new application wants a pass or fail against a known instalment;
a limit increase wants the largest instalment the client could carry; a consolidation search wants it hundreds
of times with different debts removed; a collections arrangement wants it for a client already in arrears. The
temptation is for each caller to copy the calculation and tune it, and copies drift: one gets the tax tables
fixed and three do not, and nobody notices until the ombud asks. So there is one calculation, parameterised
by *mode* (which evidence it accepts) and by *params* (policy numbers), and never forked.

Three rules shape it:

- **The evidence is graded, not just counted.** Income comes from the best available evidence tier (employer
  letter, payslip, bank statement, declared, bureau estimate), each with a haircut. A mode says how weak a tier
  it will accept.
- **"Could not establish" is not "cannot afford".** When the evidence is not good enough (an applicant's income
  is missing, the bureau view is stale, an account type nobody can cost) the verdict is *indeterminate*, with the
  reason. The remedy is more evidence, not a decline, and the client is told differently.
- **The statutory answer stays visible.** The bank may only tighten, through approved, scoped, expiring overlays.
  The capacity before the overlays is recorded beside the capacity after, and an overlay that would raise
  capacity is rejected when it is defined, not when it runs.

## What it decides

[`assessment.pipeline:build`](assessment/pipeline.py) takes one or two applicants (a joint application is
assessed as one household), their income evidence, declared and statement expenses, bureau and internal account
lists, and optionally a proposed instalment, and returns:

| Result | Fields |
|---|---|
| Income | `gross_monthly_income`, `income_source_code` and `income_verification_tier` (1 employer letter, 2 payslip, 3 bank statement, 4 declared, 5 bureau estimate, 0 none), `income_haircut_applied`, `income_variability_ratio` |
| Deductions | `statutory_deductions` (tax, per applicant), `net_monthly_income`, `court_ordered_deductions` |
| Expenses | `living_expenses` (the higher of the claim and the norm floor), `expense_basis_code`, `norm_table_version` |
| Debts | `existing_obligations`, `obligations_internal`, `obligations_external`, `total_exposure`, `revolving_utilisation`, `worst_arrears_months`, `accounts_in_arrears_count`, and per account `obligation_account_type_codes`, `obligation_treatment_codes`, `obligation_monthly_amounts`, `obligation_is_internal` |
| Capacity | `discretionary_income` (may be negative), `affordability_buffer_applied`, `residual_floor_amount`, `affordability_binding_constraint_code` (1 buffer, 2 floor), `max_affordable_instalment_unadjusted`, `max_affordable_instalment`, `adjustment_set_id`, `adjustments_applied` |
| Verdict | `affordability_verdict_code` (1 pass, 2 marginal, 3 fail, 4 indeterminate), `evidence_sufficiency_code` (0, or why indeterminate), `discretionary_income_after`, `decline_reason_codes`, `primary_reason_code` |
| Evidence | the version and cell of every table read: `applicant1_tax_table_cell_id`, `statutory_norm_cell_id`, `internal_norm_cell_id`, `buffer_grid_cell_id`, `residual_floor_cell_id`, `reason_registry_version` |

The arithmetic, in the order a regulator reads it:

1. **Household.** The higher dependants count wins. Expenses are consolidated: the higher declaration for shared
   costs (accommodation, water and electricity, food, insurance), the sum for everything else. Each applicant's
   income and tax is worked out on their own, then added; the expense norm is looked up on the combined income.
2. **Income.** Per applicant, the best tier with evidence wins and carries its haircut. The household tier is the
   weakest among applicants contributing more than 10% of the total, so a small side income does not drag it down.
3. **Tax.** Bracket table in force on the decision date, per applicant.
4. **Living expenses.** The higher of what the applicant claims (statement over declared) and the norm floor, which
   is the higher of the statutory norm and the bank's stricter internal norm, both effective-dated.
5. **Debts.** Both applicants' bureau and internal lists are merged (an account on several profiles counts once; a
   bureau account the bank also holds counts as the internal record) and each account is costed by its type.
6. **Discretionary income** = net income - living expenses - court-ordered deductions - existing obligations.
7. **Capacity.** The lower of two limits: discretionary income less a *buffer* that grows with risk grade, and
   discretionary income less an absolute *residual floor* that grows with dependants. Overlays then tighten it.
8. **Verdict.** Indeterminate if the evidence is short; otherwise the proposed instalment against the capacity:
   pass up to it, marginal up to 5% over, fail beyond. With no proposal the caller wants the capacity, and the
   verdict is pass (the evidence stands) or indeterminate, never fail.

## Features it shows

| decider feature | Where |
|---|---|
| Reusing another project's steps on other columns: `.relabel` and `.named` run `credit_core`'s income and tax once per applicant | [household.py](assessment/household.py) |
| Multi-output steps (`outputs=`) that fold several columns into household figures | [household.py](assessment/household.py), [debts.py](assessment/debts.py) |
| `dag` of `dag`s with `.emit`: the same step objects wired as one pipeline and as two halves | [pipeline.py](assessment/pipeline.py) |
| `flow` for a value written twice (reason codes fired, then ranked; the capacity rounded after the overlay) | [verdict.py](assessment/verdict.py), [capacity.py](assessment/capacity.py) |
| `DecisionTableConfig` with `eq` keys (grade by product) and an open-ended band ladder (dependants) | [capacity.py](assessment/capacity.py) |
| Tighten-only overlays from `credit_core`, switched off with a param through the same code | [capacity.py](assessment/capacity.py), [tests/test_capacity.py](tests/test_capacity.py) |
| `param` for policy (staleness windows, minimum tier per mode, marginal band), a generated params document | [modes.py](assessment/modes.py), [verdict.py](assessment/verdict.py), [`configs/0.1.0/params.json`](configs/0.1.0/params.json) |
| `missing_as` for an absent second applicant, `NaN` for "not supplied" which is not a zero | [household.py](assessment/household.py), [verdict.py](assessment/verdict.py) |
| `TypedDict` account lists: JSON date strings arrive as `date`s | [debts.py](assessment/debts.py) |
| `@allow_fallback` for steps that are Python on purpose (dicts, lists of accounts, dates), checked by `Engine(strict_compile=True)` | [tests/test_serving.py](tests/test_serving.py) |
| Warm-up from `sample_request.json`, and serving the raw request bytes | [tests/test_serving.py](tests/test_serving.py) |

## Run it

From the repository root, put every example on the path once (this one imports `credit_core`):

```bash
export PYTHONPATH=$(printf '%s:' "$PWD"/examples/*/)
cd examples/affordability
export DECIDER_API__CODE_PATH="$PWD" DECIDER_API__PIPELINE="assessment.pipeline:build" DECIDER_CONFIG__BASEPATH="$PWD/configs"
```

Build (validates `configs/0.1.0` and compiles and warms every kernel with `sample_request.json`; the first build
in a fresh checkout takes a minute while numba compiles):

```bash
uv run decider build
# built config version 0.1.0 (pipeline assessment.pipeline:build, mode fused)
```

Serve, and assess the sample application. Its dates are plain JSON strings:

```bash
uv run decider serve --workers 1 &
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations |
  jq '{gross_monthly_income, net_monthly_income, living_expenses, existing_obligations, discretionary_income,
       max_affordable_instalment_unadjusted, max_affordable_instalment, adjustments_applied,
       affordability_verdict_code, discretionary_income_after}'
```

```json
{
  "gross_monthly_income": 17575.0,
  "net_monthly_income": 16013.5,
  "living_expenses": 6200.0,
  "existing_obligations": 1600.0,
  "discretionary_income": 8213.5,
  "max_affordable_instalment_unadjusted": 6645.54,
  "max_affordable_instalment": 5980.99,
  "adjustments_applied": ["ADJ-02-2026-001"],
  "affordability_verdict_code": 1,
  "discretionary_income_after": 5713.5
}
```

The response carries every input and intermediate too. Now the same applicant with a bureau view from last month,
and a proposal of R1 000 000. The bank could not establish the debts, so the answer is indeterminate, not fail:

```bash
jq '.bureau_as_of_date="2026-08-01" | .proposed_instalment=1000000' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq '{affordability_verdict_code, evidence_sufficiency_code, decline_reason_codes, primary_reason_code}'
# {"affordability_verdict_code": 4, "evidence_sufficiency_code": 4, "decline_reason_codes": [204], "primary_reason_code": 204}
```

A joint application adds the second applicant's fields (R9 000 payslip, three dependants). Income adds up, tax
stays individual, the norm is looked up on the combined income:

```bash
jq '.is_joint_application=true | .applicant2_payslip_income=9000 | .applicant2_employment_type_code=2
    | .applicant2_dependants_count=3' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{gross_monthly_income, statutory_deductions, dependants_count, discretionary_income, max_affordable_instalment}'
# {"gross_monthly_income":26125.0,"statutory_deductions":1561.5,"dependants_count":3.0,"discretionary_income":16763.5,"max_affordable_instalment":12207.02}
```

To ask for capacity alone, drop `proposed_instalment` (`max_affordable_instalment` is the answer, the verdict is
1 and `discretionary_income_after` is `null`); a proposal of R9 000 fails (`affordability_verdict_code` 3,
`decline_reason_codes` `[208]`). To assess another way, set `assessment_mode_code`
(1 new application, 2 limit increase, 3 arrangement, 4 scenario).

Tests (run from the repository root, with `PYTHONPATH` set as above):

```bash
uv run pytest examples/affordability -q
```

To regenerate `configs/0.1.0/params.json`: `uv run python scripts/generate_configs.py`.

## Layout

```
assessment/             the package
  pipeline.py           build(), evidence_unit(), capacity_unit()
  modes.py              the four assessment modes and the weakest evidence each accepts
  household.py          applicants as one household: income, tax, expenses, dependants
  debts.py              the household's account lists, costed by credit_core.obligations
  capacity.py           buffer grid, residual floor, the overlay stack
  verdict.py            evidence sufficiency, the verdict, reason codes
configs/0.1.0/          params.json
scripts/                generate_configs.py
sample_request.json
tests/
```

## What it publishes

Later examples import `assessment` by package name. These names are stable:

| Name | Use |
|---|---|
| `assessment.pipeline.build()` | The whole assessment as one step. Embed it in another `dag`/`flow` (its params appear under `affordability_assessment/...`), and rename its inputs with `.relabel(reads=...)` |
| `assessment.pipeline.evidence_unit()`, `capacity_unit()` | The two halves. A search that varies only the debts scores `evidence_unit()` once, then `capacity_unit()` per candidate, feeding it the record plus the evidence half's outputs. `build()` is exactly the two wired together, so they cannot drift |
| `assessment.pipeline.EVIDENCE_OUTPUTS`, `CAPACITY_OUTPUTS` | The columns each half emits |
| `assessment.modes` | `NEW_APPLICATION`, `LIMIT_INCREASE`, `ARRANGEMENT`, `SCENARIO`; `minimum_income_tier` |
| `assessment.verdict` | `PASS`, `MARGINAL`, `FAIL`, `INDETERMINATE`; the `evidence_sufficiency_code` constants; `REASON_REGISTRY` |
| `assessment.capacity` | `CAPACITY_ADJUSTMENTS`, `ADJUSTMENT_SET_ID`, `build_buffer_grid()`, `build_residual_floor_table()`, `BUFFER_BOUND`, `RESIDUAL_FLOOR_BOUND` |
| `assessment.household`, `assessment.debts` | `applicant_income(prefix)`, `applicant_tax(prefix)` and `household_obligations` for a caller that frames its own household |

Inputs it reads that the caller owns: `decision_date`, `product_code`, `risk_grade`, `assessment_mode_code`,
`is_joint_application`, `bureau_as_of_date`, `court_ordered_deductions`, `proposed_instalment`, and per applicant
`applicantN_...` for the income evidence (`payslip_income`, `declared_income`, `statement_income` with
`statement_confidence`, `employer_confirmed_income`, `bureau_estimated_income`, `variable_pay_history`),
`employment_type_code`, `dependants_count`, `declared_expenses`, `statement_expenses`, `bureau_accounts` and
`internal_accounts`.

## Things worth knowing

- **Income tiers are `credit_core`'s five**, strongest first, so a *lower* number is stronger evidence. A mode's
  minimum is the weakest tier it accepts: a new application needs 3 (bank statement) or better, so a declared
  income alone is indeterminate; a limit increase accepts 4 (declared); an arrangement accepts 5.
- **A mode never changes the arithmetic.** It picks the minimum tier, and a scenario treats an account with a
  settlement quote as settled (it leaves the list) where every other mode costs it at its stated instalment.
  [`test_the_mode_changes_the_evidence_rule_but_never_the_arithmetic`](tests/test_modes.py) pins it.
- **The proposal cannot move the capacity.** [`binding_capacity`](assessment/capacity.py) never reads the proposed
  instalment, so a larger proposal can only worsen the verdict. A search that grows an instalment relies on
  this, and [tests/test_verdict.py](tests/test_verdict.py) sweeps the proposal across tax and norm band edges, grades,
  dependants and a floor-bound applicant.
- **Each applicant has their own haircut params**, because each runs its own copy of the income steps:
  `affordability_assessment/evidence/applicant1_income/...` and `applicant2_income`. Change both to change the policy.
- **Two things are constants in code, not params**: which expense categories are shared
  (`household.SHARED_EXPENSES`) and which account types need a manual look (`debts.REFER_ACCOUNT_TYPES`).
- **Left out.** Court-ordered deductions are one monthly total, so they are not matched against bureau accounts;
  there is no expense-stringency overlay (so no `living_expenses_unadjusted`); the arrangement mode uses the same
  buffer grid as the others; account matching is exact on type, opened date and balance, not the fuzzy match a bank
  would use.
- **Synthetic data.** Every table, threshold and rate here is invented for illustration.

## Reuse

Built on [`shared-credit-core`](../shared-credit-core/README.md) (`credit_core`): `income.income_unit`,
`deductions`, `expense_norms`, `obligations.assess`, `affordability`, `adjustments`, `reason_codes`, `rounding`,
`evidence.cell_id`. The assessment adds the household, the modes, the buffer-and-floor capacity, the evidence checks
and the verdict, and never re-derives what the library already computes.
