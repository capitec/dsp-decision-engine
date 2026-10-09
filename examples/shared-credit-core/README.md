# Shared credit core

A library of the credit calculations every lending product needs, plus one pipeline, the Flex Loan quote,
that wires a representative slice of them together. It is the foundation the other examples build on.

## The business problem

A bank sells many credit products: personal loans, cards, vehicle finance, business facilities. Each product
team needs to answer the same questions:

- What does this person really earn, and how much of it can they commit to a new loan?
- What do they already owe each month?
- What does a score mean as a probability of default, and which risk grade is that?
- What rate, fees and instalment follow, and why was an application declined?

If every team answers these separately, they drift apart, and an auditor asking in three years "what did the
bank do on 24 September 2026?" gets different answers from different products. So the bank keeps one shared
library. Teams may *parameterise* it (a stricter affordability buffer, their own thresholds) but not fork it.
Anything that comes from a table is looked up by the decision's own date, never by today's, and the decision
records which version of which table, and which cell, it read.

## What the quote pipeline decides

[`credit_core.pipeline:build`](credit_core/pipeline.py) takes three config documents (the rate card, the score overlays and the reason codes;
see "The documents" below) and returns a pipeline. It takes a Flex Loan application (applicant facts,
income evidence, expenses, the bureau and internal account lists, a requested amount and term) and returns:

| Result | Fields |
|---|---|
| Outcome | `outcome_code` (1 approve, 2 approve with conditions, 3 refer, 4 decline), `decline_reason_codes` (most severe first), `primary_reason_code` |
| Affordability | `gross_monthly_income`, `net_monthly_income`, `living_expenses`, `existing_obligations`, `discretionary_income`, `max_affordable_instalment`, `affordability_verdict_code` |
| Risk | `score`, `score_unadjusted`, `adjustments_applied`, `probability_of_default`, `risk_grade` |
| Price | `offered_amount`, `nominal_annual_rate`, `initiation_fee`, `monthly_service_fee`, `instalment`, `total_cost_of_credit`, `total_interest`, `effective_annual_rate` |
| Evidence | `decision_id`, and for every table read its version and cell: `tax_table_cell_id`, `statutory_norm_cell_id`, `internal_norm_cell_id`, `norm_table_version`, `calibration_cell_id`, `risk_grade_cell_id`, `rate_cell_id`, `rate_card_version`, `fee_caps_cell_id`, `reason_registry_version` |

The rules, in the order the pipeline applies them:

1. **Eligibility.** Every hard gate (age, residency, capacity to contract, exclusion lists, sanctions, debt
   review...) is checked, so the reason list is complete. Ineligible means decline.
2. **Income.** The best evidence tier wins (employer confirmation, payslip, bank statement, declared, bureau
   estimate) and carries a haircut. Tax comes off using the bracket table in force on the decision date.
3. **Living expenses.** The applicant's claim is floored by the higher of a statutory norm and the bank's
   stricter internal norm, both effective-dated.
4. **Obligations.** The bureau and internal account lists are merged (an account on both counts once) and each
   account is costed by its type: stated instalment, a share of the limit or balance, a settlement quote, or excluded.
5. **Affordability.** Discretionary income less a buffer is the most the applicant may commit. The quote's own
   instalment is compared with it: pass, marginal (up to 5% over, so *refer*), or fail (*decline*).
6. **Risk.** A scorecard gives points per characteristic. Tighten-only overlays then adjust the score, and the
   unadjusted score is kept beside it. A per-segment curve turns the score into a probability of default, and
   per-product, per-segment boundaries turn that into a grade from 1 (best) to 12.
7. **Price.** The grade, amount and term pick a cell of a 63 360-cell rate card. Fees are capped by the caps in
   force on the decision date. The instalment is the annuity payment plus fees, rounded to the cent.

## Features it shows

| decider feature | Where |
|---|---|
| Plain functions as steps; `param` for policy, `missing_as` for optional inputs | [affordability.py](credit_core/affordability.py), [income.py](credit_core/income.py), [fees.py](credit_core/fees.py) |
| `dag` (dependency order) beside `flow` (written order, for a value written twice) | [pipeline.py](credit_core/pipeline.py): the instalment waterfall and the reason-code ranking are flows inside the dag |
| `DecisionTableConfig` with inline rows, banded ranges and `eq` keys | [deductions.py](credit_core/deductions.py), [expense_norms.py](credit_core/expense_norms.py), [calibration.py](credit_core/calibration.py), [risk_grade.py](credit_core/risk_grade.py), [appetite.py](credit_core/appetite.py), [credit_life.py](credit_core/credit_life.py) |
| A 63 360-row table kept as a config document, handed to `build` by name | [rate_card.py](credit_core/rate_card.py), [`configs/0.1.0/rate_card_flex_loan.json`](configs/0.1.0/rate_card_flex_loan.json), `build(rate_card_flex_loan)` in [pipeline.py](credit_core/pipeline.py) |
| Effective dating: an `EffectiveVersions` step resolves a version id from the record's date, then it is ANDed into the table match | [dates.py](credit_core/dates.py), used by deductions, expense norms and fees |
| `ConfigurableStep` types whose fields are the data, loaded from a config document and compared by `decider.lab.diff` | [`AdjustmentStack`](credit_core/adjustments.py), [`ReasonCodes`](credit_core/reason_codes.py), [`EffectiveVersions`](credit_core/dates.py) |
| `ScorecardConfig` with bin edges that are params | [scorecard.py](credit_core/scorecard.py) |
| Multi-output steps (`outputs=`) | [adjustments.py](credit_core/adjustments.py), [reason_codes.py](credit_core/reason_codes.py), [bureau.py](credit_core/bureau.py), [obligations.py](credit_core/obligations.py) |
| `TypedDict` inputs: JSON date strings arrive as `date`s | [obligations.py](credit_core/obligations.py), [bureau.py](credit_core/bureau.py), [exposure.py](credit_core/exposure.py) |
| `@allow_fallback` for steps that are Python on purpose (dates, lists of dicts, strings), checked by `Engine(strict_compile=True)` | [tests/test_serving.py](tests/test_serving.py) |
| `.relabel` and `.named`: one step reused on other column names or twice with different settings | [pipeline.py](credit_core/pipeline.py), [tests/test_affordability.py](tests/test_affordability.py) |
| `decider.ops.decimal.round` for half-up money rounding that runs the same in Python and in a kernel | [rounding.py](credit_core/rounding.py) |
| The params document and a switch that turns a whole overlay stack off | [`configs/0.1.0/params.json`](configs/0.1.0/params.json), [tests/test_quote_pipeline.py](tests/test_quote_pipeline.py) |
| A release check that sees the overlays, reason codes and cut-overs, and no `HiddenStateWarning` | [tests/test_release_checks.py](tests/test_release_checks.py) |
| Warm-up from `sample_request.json`, and serving the raw request bytes | [sample_request.json](sample_request.json), [tests/test_serving.py](tests/test_serving.py) |

## Run it

From the repository root, put every example on the path once (the other examples import `credit_core` by name):

```bash
export PYTHONPATH=$(printf '%s:' "$PWD"/examples/*/)
cd examples/shared-credit-core
export DECIDER_API__CODE_PATH="$PWD" DECIDER_API__PIPELINE="credit_core.pipeline:build" DECIDER_CONFIG__BASEPATH="$PWD/configs"
```

Build (validates `configs/0.1.0` and compiles and warms every kernel with `sample_request.json`; the first
build in a fresh checkout takes a minute while numba compiles):

```bash
uv run decider build
# built config version 0.1.0 (pipeline credit_core.pipeline:build, mode fused)
```

Serve, and quote the sample application. Its dates are plain JSON strings:

```bash
uv run decider serve --workers 1 &
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations |
  jq '{outcome_code, instalment, nominal_annual_rate, score, score_unadjusted, risk_grade, max_affordable_instalment, rate_cell_id, adjustments_applied}'
```

```json
{
  "outcome_code": 1,
  "instalment": 2355.49,
  "nominal_annual_rate": 0.1604,
  "score": 642.0,
  "score_unadjusted": 665.0,
  "risk_grade": 8,
  "max_affordable_instalment": 7227.88,
  "rate_cell_id": "rate_card.flex_loan@rc-2026.09#12|21|8",
  "adjustments_applied": ["ADJ-2026-001", "ADJ-2026-002"]
}
```

The response carries every input and intermediate value too. Try a sanctioned applicant asking for R500 000 over six months:

```bash
jq '.requested_amount=500000 | .term_months=6 | .is_sanctioned=true' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq '{outcome_code, instalment, max_affordable_instalment, decline_reason_codes, primary_reason_code}'
# outcome_code 4, instalment 87445.96 against 7227.88 affordable, decline_reason_codes [1006, 4001], primary 1006
```

Tests (run from the repository root, with `PYTHONPATH` set as above):

```bash
uv run pytest examples/shared-credit-core -q
```

To regenerate `configs/0.1.0` (the synthetic rate card, the score overlays, the reason codes and the params document):
`uv run python scripts/generate_configs.py`.

## The documents

`build(rate_card_flex_loan, score_adjustments, reason_codes)` receives one document per argument, each a file in
`configs/<version>/`. The two below are what a risk or policy owner edits; `rate_card_flex_loan.json` is the 63 360-row table.
A change to any of them is a new config version, and `decider.lab.diff.structure` reports it: the overlays and the codes are fields
of a step, not objects a step closes over.

[`score_adjustments.json`](configs/0.1.0/score_adjustments.json) is an `adjustment_stack`: the overlays on `score`, in stack order.
The governance fields (`owner`, `approval_reference`, `rationale`, `review_date`) are for people and the ageing report; the diff
ignores them. Everything else is behaviour. Loading rejects a duplicate id, an overlay for another target, and a `tighten_only`
overlay whose effect loosens.

```json
{
  "type": "adjustment_stack", "name": "apply_score_adjustments", "target": "score", "set_id": "AS-2026.09",
  "base_field": "score_raw",
  "adjustments": [
    {"adjustment_id": "ADJ-2026-001", "kind": "score_shift", "target": "score",
     "effect": {"op": "add", "value": -18.0}, "scope": {"segment_code": 3}, "stack_position": 1,
     "owner": "Credit Risk Policy", "approval_reference": "CRC-2026-014",
     "rationale": "New-to-bank segment defaulting above model prediction for two quarters",
     "effective_from": "2026-01-01", "effective_to": null, "review_date": "2026-12-31", "tighten_only": true}
  ]
}
```

[`reason_codes.json`](configs/0.1.0/reason_codes.json) is a `reason_codes` set: `version` and the `codes` with their severity rank
(1 is the most severe). A code outside the set raises; a duplicate code is rejected when the document loads. A description is for
people and is not behaviour.

```json
{"type": "reason_codes", "name": "rank_reasons", "version": "reasons-2026.09",
 "codes": [{"code": 1006, "severity_rank": 1, "description": "Sanctioned party", "is_regulatory": true}]}
```

The effective-dated versions (tax tables, expense norms, fee caps) are `EffectiveVersions` steps declared in
[deductions.py](credit_core/deductions.py), [expense_norms.py](credit_core/expense_norms.py) and [fees.py](credit_core/fees.py), because
every product that uses a table shares its history. Each has the same fields as a document would (`family`, `date_field`, `versions`);
a gap or an overlap between versions is rejected when it loads. Moving a cut-over date shows in the diff.

The step names and the params paths did not change when these stopped being objects, so a params document and stored evidence from
before still apply: `flex_loan_quote/apply_score_adjustments/adjustment_stack_enabled` is still the switch.

## Layout

```
credit_core/            the library; one module per capability
  pipeline.py           the Flex Loan quote: build(rate_card_flex_loan)
  vocab.py              codes and names every consumer shares
  dates.py evidence.py rounding.py reason_codes.py adjustments.py        mechanisms
  instalment.py fees.py rate_card.py credit_life.py                      pricing
  scorecard.py calibration.py risk_grade.py                              risk
  income.py deductions.py expense_norms.py obligations.py affordability.py   capacity
  bureau.py eligibility.py appetite.py exposure.py consent.py adverse_events.py
configs/0.1.0/          params.json, the rate card, the score overlays and the reason codes
scripts/                generate_configs.py
sample_request.json
tests/
```

## The capabilities

Every module is usable alone: `from credit_core import rate_card`. Table-backed modules expose a `build_*_table()`
that returns a step writing the looked-up values and a `*_cell_id`.

| Module | Publishes |
|---|---|
| `dates` | `EffectiveVersions`, `Version`: the step that writes the version id in force on `date_field` (default `decision_date`) to `<family>_version`; `.resolve(on)` for code outside a flow (`.versions` keeps the order given, not sorted) |
| `evidence` | `new_decision_id()`, `cell_id(table, version, *keys)` |
| `reason_codes` | `ReasonCodes`, `Code`: the step that ranks fired codes in place and picks the primary; an unregistered code raises; `.rank(fired)` and `in` for code outside a flow |
| `adjustments` | `AdjustmentStack`, `Overlay`, `Effect`: the step that applies scoped, dated, stack-ordered overlays to `target`; tighten-only is enforced when the document loads; it records `<target>_unadjusted`, `adjustment_set_id`, `adjustments_applied`; `.apply(...)` and `.due_for_review(on)` for code outside a flow |
| `rounding` | `round_advance` (R100, half up), `round_instalment` (cent, half up), `round_rate` |
| `instalment` | annuity instalment, total cost, total interest, effective rate, and `solve_advance_for_instalment`, the inverse |
| `fees` | initiation and service fee, capped by the caps in force on the decision date |
| `rate_card` | `generate_flex_loan_card`, `lookup(card)` (rate, `rate_cell_id`, `rate_card_version`), `out_of_range`, `diff_cards` |
| `scorecard` | `build_scorecard()`: points per characteristic, `adverse_action_codes` |
| `calibration`, `risk_grade` | score to probability of default and back, per segment; PD to grade 1..12 with the boundaries that produced it |
| `income` | `income_unit`: evidence tier, haircut, gross income, variability |
| `deductions`, `expense_norms` | tax by bracket and date; the statutory and internal living-expense floor |
| `obligations` | `assess(bureau_accounts, internal_accounts)`: totals plus index-aligned per-account lists, ten treatments |
| `affordability` | discretionary income, maximum instalment, verdict on a proposed instalment (monotone in it) |
| `bureau`, `eligibility`, `appetite`, `exposure`, `consent`, `credit_life`, `adverse_events` | the smaller capabilities, each with its own tests |

`vocab` holds the product and segment codes, outcome codes, the four kinds of null (`NullKind`), roles, consent and
fraud-verdict enums, and the names reserved for consumers (`account_id`, `event_id`, ...).

## Things worth knowing

- **Column names.** A step's arguments are the columns it reads and its name is the column it writes, so
  consumers rename with `.relabel(reads=, writes=)` instead of wrapping steps. Tables used in the quote write
  namespaced names (`tax_rate`, `statutory_norm_amount`, `risk_grade_cell_id`) so they can sit in one `dag`.
- **One type per input column.** `employment_type_code`, `product_code` and `segment_code` are `float` wherever
  they are read, because decision tables and the scorecard read them as `float` and an input has one type.
- **Multi-dimensional grids.** A band ladder may leave its outer edges open (`None`) once per group of rows that
  share their `eq` keys ([risk_grade.py](credit_core/risk_grade.py)). A grid with two band dimensions repeats a
  ladder inside one group, so those tables (norms, credit life, the rate card) give every band explicit
  infinite edges and set `allow_gaps`.
- **What runs in Python.** Steps that read dates, lists of dicts or strings are Python on purpose and marked
  `@allow_fallback`; everything else, including all the tables, runs in the numba kernel.
- **Data in fields, not objects.** A step never closes over a register or a module-level `dict`. The overlays, reason codes and version
  histories are fields of their step; the account-type treatments and the scorecard's reason codes are tuples of pairs. A step that
  read an object would raise `HiddenStateWarning` when it binds, could not be compared by `decider.lab.diff` and could not compile.
  Dates inside a step's constants (overlay windows, version cut-overs) are ordinals (`date.toordinal()`), because the diff hashes
  numbers and strings by content but not a `date`.
- **The demo scores everyone.** An ineligible applicant is still scored; `outcome_code` is what gates the result.
- **Rates in the card are percentages**, as Treasury would author them. `rate_card_rate` converts to the fraction
  `nominal_annual_rate` that the instalment maths uses.
- **Synthetic data.** Every table, threshold and rate here is invented for illustration.

## Reuse

This example reuses no other example. Later examples import it by package name (`from credit_core import ...`):
the quote pipeline's capabilities are meant to be assembled into other flows, not copied. They rely on the names in
the tables above staying stable, in particular `AdjustmentStack`, `ReasonCodes`,
`EffectiveVersions`, `generate_flex_loan_card`, `obligations.assess`, the `build_*_table()` functions, the
`*_version_step` objects and the `credit_core.vocab` codes.

### Migrating from the registers

`AdjustmentRegister`, `ReasonCodeRegistry` and `EffectiveDatedSet` are being replaced and still work, unchanged, until the examples
that import them move. A step built from one closes over an object: it warns, and a release check cannot compare it. The new classes
accept the old row types (`Adjustment`, `ReasonCode`, `EffectiveVersion`), so a migration changes the constructor and the call sites
below and leaves the data alone.

| Old | New |
|---|---|
| `AdjustmentRegister([Adjustment(...), ...])` | one `AdjustmentStack(target=..., set_id=..., adjustments=[Adjustment(...), ...])` per target (a register for several targets becomes one stack each), or a `configs/<version>/*.json` document loaded with `ConfigurableStep.load` and passed to `build(...)` |
| `register.apply_stack_step(target, set_id, base_field=, adjusted_output=, unadjusted_output=)` | the stack is the step: `AdjustmentStack(target=..., set_id=..., base_field=, adjusted_output=, unadjusted_output=, ...)`; same name (`apply_<target>_adjustments`), outputs and `adjustment_stack_enabled` param; `.relabel(...)` still works |
| `register.apply_stack(target, base, provenance, on, set_id, stack_enabled=)` | `stack.apply(base, provenance, on, stack_enabled=)`, an `AdjustmentResult` as before; it matches any provenance field, while the step matches `product_code`, `segment_code`, `channel_code` and `scorecard_id` only |
| `register.for_target(t)`, `register._all` | `stack.adjustments` |
| `register.due_for_review(on)` | `stack.due_for_review(on)` |
| `ReasonCodeRegistry(version, [ReasonCode(...), ...])` | `ReasonCodes(version=..., codes=[ReasonCode(...), ...])`, or a `reason_codes` document |
| `registry.resolve_step()` | the `ReasonCodes` object is the step (name `rank_reasons`) |
| `registry.rank(fired)`, `code in registry`, `registry.version` | unchanged on `ReasonCodes`; `registry._by_code[c]` becomes `{x.code: x for x in codes.codes}[c]` |
| `EffectiveDatedSet(family, [EffectiveVersion(...), ...])` | `EffectiveVersions(family=..., versions=[EffectiveVersion(...), ...])` |
| `dated.resolver_step(date_field=, output=)` | the `EffectiveVersions` object is the step; `date_field=` and `output=` are fields; name `resolve_<family>_version` and output `<family>_version` as before |
| `dated.resolve(on)` | unchanged on `EffectiveVersions`; the result has `version_id`, `effective_from` and `effective_to` |

One difference: the stack and the reason codes are frozen objects, so a test that builds a register and then edits it builds a new
stack with `model_copy(update=...)` instead.
