# Consolidation

A client of eleven years walks into a branch and asks for R20 000 she cannot comfortably afford. She is servicing nine
credit accounts, two of them with the bank, at a balance-weighted 19.6% interest, for R12 123 a month. One is a month
in arrears. Declining her is correct and useless: her problem is not the R20 000, it is the shape of what she already
owes. This example finds which of her accounts the bank should settle, prices the settlement across the products that
could carry it, shows that the one chosen beats the ones that did not, and hands the back office the instructions to
execute it. Or it declines, with a reason that survives being read aloud two years later.

## The business problem

Somewhere among the 511 non-empty subsets of her nine accounts there may be a set the bank can settle such that her
instalment falls, the exposure improves, and the total she will pay before she is debt-free does not rise so far that
she has been harmed by being helped. Three things make it hard:

- **The search space is exponential and the answer is due while the client sits at the desk.** Eighteen settleable
  accounts are 262 143 sets; with four products and nine terms each that is 9 million priced scenarios. A few hundred are
  ever evaluated. *Which* few hundred is a business decision (Credit Risk Policy writes the ordering rules), and it must
  be the same few hundred every time the same client is assessed: the budget is a count of scenarios, never a clock.
- **Helping is not lowering the instalment.** Stretching a term always lowers an instalment and almost always raises
  what the client pays in total. A flow that optimises the instalment alone manufactures ombud complaints, so an
  **anti-harm** intervention rejects a scenario whose total cost rises too far, and every rejection says by how much.
- **Everyone will ask about the decisions that were not taken.** "Why did you not settle my store card?" has four
  different answers (it could not be settled; it was never generated; it was generated and failed an intervention; it
  was viable and lost by a stated margin on a stated component). The record distinguishes them.

Around the search sit the usual duties: debt-review clients are routed to a restructure path, never declined; income
and expenses are worked out once, so two scenarios cannot disagree about the client; every table read is recorded with
its version and cell; and approved, expiring *overlays* may tighten a threshold or re-weight the objective without a
release, with their ids on the record and a switch that runs the same code without them.

## What it decides

[`consolidation.pipeline:build`](consolidation/pipeline.py) takes the client's accounts, income evidence and request
(see [`sample_request.json`](sample_request.json)) and returns:

| Result | Fields |
|---|---|
| Outcome | `outcome_code` (1 approve, 2 approve with conditions, 3 refer, 4 decline), `decline_reason_codes` (most severe first), `primary_reason_code`, `conditions` |
| The offer | `chosen_offer` (the winning scenario: settlement set, product, term, amount, rate, instalment, total cost, fees, interventions, measures), `runners_up` (up to two, distinct), `offer_valid_until`, `before_and_after`, `execution_package` |
| Why it won | `objective_weights` and `objective_overlay_ids`, `scored_scenarios` (every viable scenario with its components), `winner_vs_runners_up` (the component that explains each gap), `indifference_band_triggered`, `shadow_best_scenario` (the best outcome for the client whatever the objective) |
| The search | `scenarios` (every one evaluated, viable or rejected, each with all fourteen interventions and the actual and threshold values), `search_record` (budget, consumption, why it stopped, rules in force, products unavailable, rejections by reason) |
| The inventory | `classified_accounts` (each with its `settleability_code`, the rule and the table cell behind it), `settleable_account_refs`, `mandatory_account_refs`, `excluded_account_refs`, `nomination_conflicts`, `settlements` (every settleable account's amount with its components) |
| The baseline | `baseline` (obligations, capacity, average rate, remaining cost, accounts, providers), `invariants` (the income and expense determination), `plain_offer`, `short_circuit_verdicts`, `short_circuit_applies`, `eligibility_gate_verdicts` |
| Evidence | `fee_caps_cell_id`, `statutory_norm_cell_id`, `applicant1_tax_table_cell_id`, `buffer_grid_cell_id`, `residual_floor_cell_id`, `reason_registry_version`, `product_policies`, and on every scenario its `rate_cell_id` and `rate_card_version` |

The stages, in the order the decision takes them:

1. **Inventory.** Each account gets one `settleability_code` from the first rule that matches: unknown (7), blocked by
   status (6), by policy (5), by provider (4), settleable with a security release (3), partly settleable revolving (8),
   quotable (2), quoted (1), the bank's own (0). The provider facts come from the `settleability_rules` table. A client
   exclusion removes an account from the search; a nominated account is in every set, and one that cannot be settled
   is a conflict for the consultant, not silently dropped.
2. **Settlement amounts.** Never the balance: capital and accrued interest to the assumed settlement date, unpaid fees,
   any early settlement charge and security release cost, less rebates. A held, unexpired quotation is the provider's
   firm figure. Each amount carries its components, its basis (quoted, derived, estimated) and a tolerance.
3. **Evidence.** The affordability assessment's income half runs once.
4. **Eligibility.** Seven gates, all recorded. Debt review routes to the debt-counsellor path (a referral); administration,
   no verified income or no bureau view decline; a consolidation in the last six months, a reckless lending allegation or
   a nomination conflict refer; fewer than two settleable accounts means consolidation does not apply.
5. **Baseline and short-circuit.** The do-nothing position, and the plain Flex Loan the client could have. If the
   request is affordable as made, debt service stays under 40%, nothing is in arrears, the average rate is within 300
   basis points of the consolidation rate, there are fewer than six accounts and none at 28% or more, the flow completes
   as plain granting. Otherwise the search runs, even when the plain offer would pass.
6. **Candidates.** Eight ordering rules (highest rate first, most relief per rand, shortest term last, secured last,
   provider in arrears first, quotation held first, whole provider relationship first, highest re-accumulation risk first)
   each contribute the prefixes of their order, beside the nominated set and the full set. Sets are routed to the products
   that can carry them: the consolidation loan needs two or more accounts, the balance transfer card only cards.
7. **Evaluation.** For every scenario, up to the budget: the obligations are re-derived without the settled accounts,
   the loan or card is priced from its own rate card and policy, affordability is tested on what remains, and the
   fourteen interventions are applied. A violated intervention removes the scenario; it is not scored down.
8. **Selection.** A blend of five objectives (new money, instalment, total cost, the bank's value, client outcome) chosen
   per channel. Scores are ratios to the baseline, so the budget cannot change the winner. Ties go to the lowest total
   cost, then the fewest accounts, then the lowest scenario id. Scores within 2% are flagged as equivalent. The three
   shown are distinct by settlement set and product.
9. **Outcome.** The route outranks the search. An approval resting on an estimate is *with conditions*, and lists them
   (quotations to obtain, cards to close, security to release). The before-and-after comparison states a rise in total
   cost in rands and percent, with the reason.

## Features it shows

| decider feature | Where |
|---|---|
| Reusing another example's halves: `assessment.evidence_unit()` once in the flow, `capacity_unit()` compiled and scored per candidate debt set, equal to the whole assessment | [pipeline.py](consolidation/pipeline.py), [capacity.py](consolidation/capacity.py), [tests/test_assessment.py](tests/test_assessment.py) |
| `each`: per-account rules as ordinary steps, with a `DecisionTableConfig` lookup inside the child | [inventory.py](consolidation/inventory.py) |
| A search in plain Python called from a step, bounded by a count, with a recorded cause of termination and a test that a truncated run is a prefix of the full one | [search.py](consolidation/search.py), [candidates.py](consolidation/candidates.py), [tests/test_search.py](tests/test_search.py) |
| `param_table`: the ordering rules in force and their sequence, the objective weights per channel, each product's candidate terms and rate margins, and the rate cards and credit life rates the search prices from, retuned like params | [candidates.py](consolidation/candidates.py), [objective.py](consolidation/objective.py), [flex_loan.py](consolidation/flex_loan.py), [rate_tables.py](consolidation/rate_tables.py) |
| One steps-and-params section per product team (`flex_loan_policy`, `everyday_card_policy`), gathered into one record, so a team changes its own limits and thresholds without touching another's | [products.py](consolidation/products.py), [flex_loan.py](consolidation/flex_loan.py), [everyday_card.py](consolidation/everyday_card.py), [tests/test_products.py](tests/test_products.py) |
| Rate cards kept as config documents and handed to `build` by name: an absolute-rate grid, and a promotional card and a reversion card read together. Each becomes the default rows of a table param, so a params document that sets the rows prices from its own | [pipeline.py](consolidation/pipeline.py), [rate_tables.py](consolidation/rate_tables.py), [`configs/0.1.0/`](configs/0.1.0), [scripts/generate_configs.py](scripts/generate_configs.py) |
| Pricing through `loan_granting`'s `PriceEvaluator` over a card it was never written for, reading its `RateCardTable` and `CreditLifeTable` over table params, and its `solve_term` for the plain offer | [flex_loan.py](consolidation/flex_loan.py), [baseline.py](consolidation/baseline.py), [rate_tables.py](consolidation/rate_tables.py) |
| Overlays as documents (`overlay_set`, built on `credit_core`'s `Overlay` and `evaluate`): the anti-harm ceiling tightened on one product and the objective re-weighted on one channel, both switched off by one shared param through the same code, with the base value kept beside the overlaid one. The search and the selection close over the overlays as tuples, so a release check sees an edit and not a reworded rationale | [overlays.py](consolidation/overlays.py), [interventions.py](consolidation/interventions.py), [objective.py](consolidation/objective.py), [tests/test_interventions.py](tests/test_interventions.py), [tests/test_documents.py](tests/test_documents.py) |
| `reason_codes` documents twice: why a scenario was rejected (read by the search) and why a client was declined or referred (a step in the flow) | [reasons.py](consolidation/reasons.py), [`configs/0.1.0/`](configs/0.1.0) |
| `each` over the products the search can price, reading the assessment's own buffer grid | [capacity.py](consolidation/capacity.py) |
| `TypedDict` account lists: JSON date strings arrive as `date`s | [inventory.py](consolidation/inventory.py) |
| `@allow_fallback` for steps that are Python on purpose, checked by `Engine(strict_compile=True)` | [tests/test_serving.py](tests/test_serving.py) |
| Warm-up from `sample_request.json`, and serving the raw request bytes | [sample_request.json](sample_request.json), [tests/test_serving.py](tests/test_serving.py) |

## Run it

From the repository root, put every example on the path once (this one imports `credit_core`, `assessment` and
`loan_granting`):

```bash
export PYTHONPATH=$(printf '%s:' "$PWD"/examples/*/)
cd examples/consolidation
export DECIDER_API__CODE_PATH="$PWD" DECIDER_API__PIPELINE="consolidation.pipeline:build" DECIDER_CONFIG__BASEPATH="$PWD/configs"
```

Build (validates `configs/0.1.0`, compiles and warms every kernel with `sample_request.json`; the first build in a fresh
checkout takes a minute while numba compiles):

```bash
uv run decider build
# built config version 0.1.0 (pipeline consolidation.pipeline:build, mode fused)
```

Serve, and assess the sample client. Her dates are plain JSON strings. The 178 scenarios the search evaluates take a
tenth of a second:

```bash
uv run decider serve --workers 1 &
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations |
  jq '{outcome_code, offer_valid_until,
       chosen: (.chosen_offer | {settlement_set, product_code, term_months, offered_amount, nominal_annual_rate, instalment,
                                 instalment_relief, total_cost_delta}),
       search: (.search_record | {budget, consumption, termination_cause}),
       before_and_after: {before: .before_and_after.before.monthly_instalment, after: .before_and_after.after.monthly_instalment,
                          note: .before_and_after.total_cost_note}}'
```

```json
{
  "outcome_code": 2,
  "offer_valid_until": "2026-10-15",
  "chosen": {
    "settlement_set": [1, 2, 3, 4, 5, 6, 7],
    "product_code": 11,
    "term_months": 72,
    "offered_amount": 211103.11,
    "nominal_annual_rate": 0.1582,
    "instalment": 5117.67,
    "instalment_relief": 1665.33,
    "total_cost_delta": 35692.28
  },
  "search": {"budget": 400, "consumption": 178, "termination_cause": "candidates_exhausted"},
  "before_and_after": {
    "before": 12123.0,
    "after": 10457.67,
    "note": "you will pay R35,692 more in total (6.8%) because the new facility runs 72 months against 60 on the accounts it settles"
  }
}
```

Read it as the business would. The bank settles seven of her nine accounts (the eighth is with a provider that does not
quote third parties, the ninth is the car) into one R211 103 loan over 72 months, which includes the R20 000 she asked
for. Her monthly payments fall from R12 123 to R10 458, and the comparison says plainly that she pays R35 692 more in
total. The outcome is *with conditions* (2) because most of the settlement amounts are estimates until quotations arrive
(`conditions` lists them), and the offer lapses on 15 October. The response also carries all 178 scenarios, 126 of them
failing the instalment-reduction test, 65 the anti-harm ceiling, and so on (`search_record.rejections_by_reason`).

Try the variations. A client under debt review is referred, not declined, and the search does not run:

```bash
jq '.client_under_debt_review=true' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{outcome_code, decline_reason_codes, eligibility_route, termination: .search_record.termination_cause}'
# {"outcome_code":3,"decline_reason_codes":[2001],"eligibility_route":"debt_review_path","termination":"not_run"}
```

Why was account 8 never considered? The record says which rule and which table cell decided it:

```bash
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '.classified_accounts[] | select(.account_ref == 8) | {account_ref, settleability_code, settleability_rule_id, settleability_cell_id}'
# {"account_ref":8,"settleability_code":4,"settleability_rule_id":"CON-SETTLE-PROVIDER","settleability_cell_id":"settleability@settleability-2026.09#901"}
```

The client insists on settling account 8. It cannot be, so the consultant must resolve the conflict:

```bash
jq '.client_nominated_settle=[8]' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{outcome_code, decline_reason_codes, nomination_conflicts}'
# {"outcome_code":3,"decline_reason_codes":[2007],"nomination_conflicts":[8]}
```

A client with one account and a small request does not need consolidation; she is granted plainly, with the Flex Loan:

```bash
jq '.accounts |= [.[0]] | .requested_amount=3000' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{outcome_code, route: .eligibility_route, offer: .chosen_offer}'
# outcome 1, route not_applicable, R3 000 over 36 months at 12.92%, instalment 224.34
```

A client who took a balance transfer in March cannot use the card again; the record says so, and no card scenario is
priced (`search_record.products_unavailable`):

```bash
jq '.last_balance_transfer_date="2026-03-01"' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{outcome_code, unavailable: .search_record.products_unavailable, consumption: .search_record.consumption}'
# {"outcome_code":2,"unavailable":{"20":"a balance transfer within the last 12 months"},"consumption":143}
```

Tests (run from the repository root, with `PYTHONPATH` set as above):

```bash
uv run pytest examples/consolidation -q
```

To regenerate `configs/0.1.0` (the synthetic rate cards, the settleability rules, the reason codes, the overlays and the params document):
`uv run python scripts/generate_configs.py`.

## Layout

```
consolidation/          the package
  pipeline.py           build(...): every stage wired into one flow
  inventory.py          the accounts, classified by the first matching rule (an `each` over the accounts)
  settlement.py         what it costs to close each settleable account, and the buffer
  eligibility.py        the seven gates and the route
  capacity.py           the assessment's capacity half, scored once per candidate debt set, and the buffer of each product
  rate_tables.py        product 11's card, the Everyday Card's two rate tables and the credit life rates as table params
  overlays.py           the `overlay_set` document: overlays on the values the search and the selection read
  documents.py          the documents in `configs/0.1.0/`, for a caller of `build` that passes none
  baseline.py           the do-nothing position, the invariants, the plain offer and the short-circuit
  candidates.py         the eight ordering rules and the sets they generate
  search.py             the bounded search and its record
  scenario.py           one scenario: obligations, pricing, interventions, measures
  interventions.py      the fourteen interventions and the anti-harm overlay
  objective.py          the weights per channel, the re-weight overlay, ranking and selection
  outcome.py            the decision, conditions, validity, before and after, execution package
  products.py           the Quote every product returns, and the gathered product policies
  flex_loan.py          product 11: a consolidation loan priced from its own card
  everyday_card.py      product 20: a balance transfer with a promotional rate that reverts
  reasons.py vocab.py   the codes this example declares
configs/0.1.0/          params.json, the product 11 card, the two card rate documents, the settleability rules, the two
                        reason code sets and the two overlay sets
scripts/                generate_configs.py
sample_request.json
tests/
```

## What it publishes

Later examples import `consolidation` by package name. These names are stable:

| Name | Use |
|---|---|
| `consolidation.inventory.classify_unit(settleability_rules)` | The classification of a client's accounts as one step to embed in another flow: reads `accounts`, `decision_date` and optional nominations and exclusions, writes `classified_accounts` and the settleable, mandatory and excluded refs. Collections and limit management want exactly this |
| `consolidation.inventory.Account` | The `TypedDict` of an account in the inventory |
| `consolidation.settlement.settlements`, `derive_settlement(...)`, `settlement_buffer(total, pct, cap)` | What it costs to close each settleable account, with components, and the cushion added to an advance |
| `consolidation.candidates.candidate_sets(...)`, `RULES`, `rule(number)` | The ordered, deduplicated candidate sets for any list of settleable accounts; `RULES` is a tuple of `(number, name, sort key)` |
| `consolidation.scenario.ScenarioEvaluator` | Evaluates one scenario for one assessment: `evaluate(scenario_id, refs, rule, product_code, term_months, promotional_months)` |
| `consolidation.capacity.Capacity(*capacity_documents())`, `case_record(...)` | The assessment's capacity half, scored per candidate debt set against fixed invariants; built from the capacity overlay and reason code documents the first time they are used |
| `consolidation.objective.rank(...)`, `weights_in_force(table, channel, date, enabled, overlays)` | Ranking viable scenarios by a weight vector: ties, distinct top three, indifference band, gaps, shadow result |
| `consolidation.products.Quote` and the product classes | What a product returns; a fifth product is a class with `unavailable`, `carries`, `variants`, `quote` and `bind`, a policy step, and a line in `product_policies` |

Inputs it reads that the caller owns: `decision_id`, `decision_date`, `channel_code`, `segment_code`, `risk_grade`,
`applicant_age_years`, `requested_amount` (may be absent), `accounts`, `client_nominated_settle` and
`client_excluded_settle`, `client_under_debt_review`, `client_under_administration`, `bureau_unobtainable`,
`active_reckless_lending_allegation`, `last_consolidation_date`, `consolidations_last_24_months`,
`last_balance_transfer_date`, and the assessment's `applicant1_...` income and expense evidence with
`bureau_as_of_date`.

## Things worth knowing

- **One truth for affordability.** The search never re-derives income. [`tests/test_assessment.py`](tests/test_assessment.py)
  checks that the baseline and every priced scenario agree, to the cent, with the *whole* assessment run on the accounts
  that remain. The only figure the product changes is the buffer, which is read from the assessment's own grid.
- **The assessment's own params stay at their defaults for the scenarios.** The evidence half is in the flow, so its
  params (haircuts, staleness window, minimum tier) sit under `consolidation/evidence/...` and are tuned there. The capacity
  half is scored inside the search and uses the assessment's defaults, except for the overlay switch, which follows
  `shared.overlay_stack_enabled`.
- **Two notions of the monthly cost.** `existing_obligations` is what the assessment costs the accounts at (a card is a
  share of its limit); `total_instalment` is what the client actually pays. Affordability uses the first, instalment
  relief and the comparison use the second.
- **The budget is a count.** The interactive target also has a clock, but a time cut-off is not reproducible, so this
  example bounds scenarios and leaves the clock to the serving layer.
- **A truncated search is a prefix.** Scenario ids follow generation order, so replaying with a larger budget extends the
  list and never reorders it.
- **No circular solve.** The consolidation loan's initiation fee is charged on the amount lent and capitalised on top of
  the proceeds, so pricing it once is exact; there is no fee that depends on itself to iterate over.
- **New money is clamped, not searched.** Each scenario carries the amount asked for, capped at a quarter of the settlement
  and R50 000 (the product's own thresholds, in `params.json`). A card carries none.
- **Params, per owner.** `flex_loan_policy` and `everyday_card_policy` hold each product team's limits, thresholds and
  term ladders (every one with the range the policy allows); `selection` holds the weights per channel; `search` the
  budget and the rules in force; `settlements` the buffer and the early settlement rules. The overlays are
  `overlay_set` documents (`intervention_adjustments.json`, `objective_adjustments.json`) with their approval, window and
  review date, and one shared param, `shared.overlay_stack_enabled`, switches them all off.
  A params document is read once: pass a new dict for each change.
- **Overlays.** The anti-harm ceiling is tightened to 80% of its value on product 11, and the branch channel's weights
  become 0.6 instalment and 0.4 new money. Ask for no new money and switch the stack off in `params.json`: the winner
  moves from 72 months to 84, because the 84-month scenario costs 14.7% more than the accounts it settles, which the base
  ceiling of 15% admits and the overlaid 12% rejects. The rejected scenario's record names `ADJ-CON-INT-2026-001` and both thresholds
  ([`tests/test_interventions.py`](tests/test_interventions.py)).
- **Left out.** Products 30 and 40 (they need collateral data), the restructure variant and batch identification, the
  consolidation-specific scorecard (the grade comes in with the request), joint applicants, a time budget, rate add-on
  overlays, and the persistence and replay of the record. `hardship_declared` and an assessment mode are not read.
- **One workaround for what decider lacks**, marked `# workaround`: the product policies are keyed by text because a `dict`
  with integer keys becomes a column that a JSON or CSV batch response cannot write, and `product_policies` is one of
  the pipeline's outputs.
- **The Flex Loan card is regenerated** in `build` from `credit_core`, not read from a config document, to avoid a third
  13 MB copy of it; the plain offer is the only thing that uses it. It is made once per process, and its rows are the
  default of the table param `plain_granting/plain_rate_card`, which `params.json` leaves out.
- **What a release check sees.** No step closes over an object, so `decider.lab.diff` compares everything the
  decision reads: editing a reason code's severity, an overlay's effect, scope or window, a rate card or the capacity
  overlay shows as the step that reads it, and a reworded rationale or description does not
  ([`tests/test_documents.py`](tests/test_documents.py), [`tests/test_hidden_state.py`](tests/test_hidden_state.py)).
  `baseline_measures`, `search` and `selection` stay Python steps on purpose (`@allow_fallback`): the search builds an
  evaluator, the products and the compiled capacity unit for each request, from what it is handed as inputs, params and
  tuples. The capacity unit is the one thing that cannot be data: it is `assessment.capacity_unit()` compiled, an
  executable. It is built the first time its two documents (`capacity_adjustments` and `affordability_reason_codes`,
  the assessment's own, passed to `build` or defaulting to the ones `assessment` ships, without their governance fields)
  are used, and kept for the life of the process under their text, so the build and serve warm-up from
  `sample_request.json` pay for it and an edited document is a different unit. Rewriting the search as an `each` over
  the scenarios with the capacity unit as a child flow was not done: the budget, the generation order and the prefix
  guarantee are properties of one loop, and the unit takes a list of accounts per scenario.
- **Table sizes.** The product 11 card is 12 amount bands by 12 terms by 12 grades; a production card would be larger and
  would be handed in the same way.
- **Synthetic data.** Every table, threshold and rate here is invented for illustration.

## Reuse

Built on [`shared-credit-core`](../shared-credit-core/README.md) (`credit_core`): `adjustments`, `fees`, `instalment`,
`rate_card`, `reason_codes`, `rounding`, `evidence`, `vocab`; on [`affordability`](../affordability/README.md)
(`assessment`): `evidence_unit()` embedded, `capacity_unit()` scored per candidate, `build_buffer_grid()`; and on
[`loan-granting-pricing`](../loan-granting-pricing/README.md) (`loan_granting`): `PriceEvaluator`, `RateCardTable`,
`CreditLifeTable` and `rate_card_rows` for the consolidation loan, and `solve_term` for the plain offer. Nothing in them is forked or
patched; the consolidation loan's card, the card product, the settleability rules and the search are this example's own.
