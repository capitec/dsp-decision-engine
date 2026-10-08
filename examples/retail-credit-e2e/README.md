# Retail credit, end to end

A client applies for a personal loan. Before the Bank says yes it has asked sixteen questions in a fixed order: is
the application well formed, who is this, may we lend to them at all, is it fraud, what do they earn and owe, how
risky are they, what do our own caps allow, what can they afford, what does that buy, and is the offer we are about to
make one we are willing to be bound by. This example is that whole chain for one product (the Flex Loan, an unsecured
loan of R2 000 to R500 000) and one way in (a new application), as one decider pipeline you can read from top to
bottom.

## The business problem

Retail credit decisioning is not one calculation. It is many, each owned by a different team, run in an order that
matters: a value computed early (income, grade) is read late (caps, pricing, validation), two rules that cap the same
amount give a different answer if their order changes, and the figure priced into the offer has to be the figure the
final check tests. What goes wrong in practice is not a single rule. It is the joins: a fee capitalised in one place
and counted again in another, an overlay applied twice, an offer affordable when the search looked at it and not when
it was written down, a decline that cannot say which of ten rules caused it.

So this example keeps the joins visible. Every stage is a small function or table with its own tests; every cap
keeps the chain of rules that produced it; every overlay is recorded beside the value it moved; every table cell that
priced the offer is named in the result. Other examples take one stage and go deep (affordability, collections, fraud).
This one has all of them at working depth, with this product's own thresholds, tables and band edges. It uses only the
shared `credit_core` library, deliberately: nothing from the other examples' packages.

## What it decides

[`retail_credit.pipeline:build`](retail_credit/pipeline.py) takes one application (the applicant, their income evidence,
declared expenses, bureau and internal account lists, fraud signals, exposure, a requested amount and term) and returns:

| Result | Fields |
|---|---|
| Outcome | `outcome_code` (0 rejected, a malformed request and not a decision; 1 approve; 2 approve with conditions; 3 refer; 4 decline), `decline_reason_codes` (most severe first), `primary_reason_code` |
| Admission | `p01_valid`, `p01_rejection_reasons`, `identity_resolution_path`, `identity_confidence`, `degraded_mode_code` |
| Fraud | `fraud_verdict_code` (1 proceed, 2 refer, 3 decline, 4 unavailable), `fraud_reason_codes`, `fraud_weighted_score`, `fraud_rules_fired_count` |
| Income and debt | `gross_monthly_income`, `net_monthly_income`, `living_expenses`, `existing_obligations`, `income_evidence_tier`, `segment_code` |
| Risk | `score`, `probability_of_default_unadjusted` and `probability_of_default`, `risk_grade` (1 best to 12) |
| Caps | `amount_cap`, `amount_cap_binding_rule`, `amount_cap_chain` (every rule, what it did, why), `term_cap`, `worst_acceptable_grade` |
| Affordability | `discretionary_income`, `max_affordable_instalment`, `instalment_cap`, `affordability_verdict_code` (1 pass, 2 marginal, 3 fail, 4 indeterminate), the same figures on a hypothetical basis, `offer_basis_code` |
| Offer | `has_offer`, `offer_amount`, `offer_term_months`, `offer_binding_constraint` (`BIND-AFF`, `BIND-REQ`, `BIND-CAP`, `BIND-MAX`, `BIND-MIN`) |
| Price | `nominal_annual_rate` and its `_unadjusted` form, `initiation_fee`, `amount_financed`, `credit_life_premium`, `monthly_service_fee`, `instalment`, `total_cost_of_credit`, `total_interest`, `effective_annual_rate` |
| Evidence | `rate_cell_id`, `credit_life_cell_id`, `risk_grade_cell_id`, `tax_table_cell_id`, `norm_table_version`, `reason_registry_version`, and the overlays applied to each overlaid value (`pd_adjustments_applied`, `amount_cap_adjustments_applied`, `buffer_adjustments_applied`, `rate_addon_adjustments_applied`) |

The response also carries every input and intermediate value, so a decision can be explained from the response alone.
When `has_offer` is false the price fields are zero and `rate_cell_id` is empty.

## The stages

Each stage reads what earlier stages wrote and decides one thing. `decider` orders them by what they read, so the order
below is the order a reader follows, not a call sequence anyone maintains.

| # | Stage | What it decides | Where |
|---|---|---|---|
| 1 | Admission | Whether the request is well formed: positive amount and term, this product, a known channel, a new application. Every failed check is listed. A rejection is outcome 0 and never a decline. | [intake.py](retail_credit/intake.py) |
| 2 | Identity | Who the client is. The first path that holds wins, each with its own confidence: authenticated session (1.00), identity number (0.97), verification service (0.92), probabilistic match (its own confidence). Below 0.88 the match is not trusted and the outcome is refer. | [intake.py](retail_credit/intake.py) |
| 3 | Eligibility | Whether we may lend at all. `credit_core`'s nine gates (age, residency, capacity, exclusion list, sanctions, debt review...) plus three of this product's: age at maturity over 75, a duplicate application in flight, no bureau-enquiry consent. All are checked, so the reason list is complete. | [eligibility.py](retail_credit/eligibility.py) |
| 4 | Bureau state | Whether the bureau is down (degraded mode 11). If it is, an approval is allowed only inside a reduced envelope (R25 000, 36 months, segments 3 to 5, 24 months' tenure, no arrears); everything else refers. Degradation narrows what can be approved, it never invents evidence. | [intake.py](retail_credit/intake.py) |
| 5 | Fraud | A verdict from 13 rules in five families. Every rule is checked and every firing recorded. A severity-4 watchlist hit declines; two serious identity hits refer; otherwise the weighted score of everything that fired refers (0.71) or declines (0.88). A small request from a long-standing client skips the consortium families, which are then *not evaluated*, a different fact from *did not fire*. | [fraud.py](retail_credit/fraud.py) |
| 6 | Income, tax, expenses, obligations, segment | The determinations everything later reads, made once. Income takes the best evidence tier and its haircut; tax comes from the bracket table in force on the decision date; living expenses are the higher of the claim and the norm floor; obligations merge the bureau and internal lists. Then this product's own two: the evidence tier on its seven-step ladder, and the client's segment (the first that fits: joint, staff, non-resident, employment type, existing client, new). | `credit_core`, [features.py](retail_credit/features.py) |
| 7 | Score | Points for eight characteristics on a base of 600. | [scoring.py](retail_credit/scoring.py) |
| 8 | Probability of default and grade | The score becomes a probability on the segment's logistic curve. The channel overlay (a tighten-only odds multiplier) moves the probability before grading, so a tightening changes the grade. The probability becomes a grade 1 to 12 on the segment's cut points. | [grading.py](retail_credit/grading.py) |
| 9 | Policy caps | Three ceilings from an ordered register: the amount, the term, and the worst acceptable grade. Order is part of the policy, so each ceiling keeps its chain: every rule, its before and after, and whether it *bound*, was *evaluated and did not bind*, or was *not applicable*. A campaign may raise the amount by a fifth up to R200 000, the only rule that can raise. A channel overlay then reduces the amount, and the result rounds down to the R250 grid. | [caps.py](retail_credit/caps.py) |
| 10 | Affordability | What the client can commit each month: discretionary income, less a residual floor for dependants, the lower of that and the grade's share of net income, less a buffer by grade and channel (which an overlay can raise). The verdict: pass from R185 of capacity, marginal within 8% below, otherwise fail; weak income evidence (bureau estimate or declaration) turns a pass into a marginal and never helps a fail; no income established at all is indeterminate. | [affordability.py](retail_credit/affordability.py) |
| 11 | Product routing | Whether this product may carry the request: amount and term in range, and a grade no worse than the caps allow. | [routing.py](retail_credit/routing.py) |
| 12 | Consolidation | If an eligible client fails affordability (and fraud has not stopped them), the affordability steps run again on a hypothetical figure, the external debts settled. Both results are kept. The offer is solved on the hypothetical basis only if it passes. | [consolidation.py](retail_credit/consolidation.py) |
| 13 | Pricing and the solve | The largest amount on the R250 grid whose instalment fits under the instalment ceiling. Each candidate is priced by the same steps that will price the offer: rate card cell, long-term loading, the rate add-on, initiation fee (capitalised), credit life on the amount financed, annuity plus service fee. | [pricing.py](retail_credit/pricing.py), [flex_card.py](retail_credit/flex_card.py), [solve.py](retail_credit/solve.py) |
| 14 | Offer | Whether there is an offer at all (the decision is still open, the basis is affordable, the solve found an amount), and the winning candidate's figures with the total cost and interest. | [offer.py](retail_credit/offer.py) |
| 15 | Final validation | Whether the Bank is willing to be bound: amount on the grid and within the cap, term within the cap, rate under the statutory ceiling, instalment within the ceiling of the basis it was priced on. Every assertion is checked. | [validation.py](retail_credit/validation.py) |
| 16 | Reasons and outcome | Every reason any stage raised, ranked by severity with a primary one, and the outcome. Refer means a person must look: the bureau is down outside the envelope, fraud or identity is unsure, income is unestablished, or there is no offer or it failed validation. An offer priced on settling the client's other debts, or on marginal affordability, carries conditions. | [disclosure.py](retail_credit/disclosure.py) |

## Features it shows

| decider feature | Where |
|---|---|
| Plain functions as steps; `param` for policy, `missing_as` for optional inputs | every module; the policy numbers are in [`configs/0.1.0/params.json`](configs/0.1.0/params.json) |
| One `dag` of a few dozen steps and units, with `flow` for a value written twice (`amount_cap` overlaid then rounded, the instalment, the reasons fired then ranked) | [pipeline.py](retail_credit/pipeline.py), [caps.py](retail_credit/caps.py), [pricing.py](retail_credit/pricing.py), [disclosure.py](retail_credit/disclosure.py) |
| `optimise` with `record=`: try every candidate, keep the best, and get the winner's whole price back from the same steps that scored it | [solve.py](retail_credit/solve.py), [pricing.py](retail_credit/pricing.py) |
| `param_table`: a 34 560-cell rate card in the params document, replaceable without a rebuild | [flex_card.py](retail_credit/flex_card.py) |
| `DecisionTableConfig` with `eq` keys and banded ranges | [grading.py](retail_credit/grading.py), [affordability.py](retail_credit/affordability.py) |
| `ScorecardConfig` | [scoring.py](retail_credit/scoring.py) |
| Reusing a unit twice with `.relabel` and `.named`: the affordability steps run on actual and on hypothetical obligations, one `capacity_unit` with its reads and writes renamed | [affordability.py](retail_credit/affordability.py), [consolidation.py](retail_credit/consolidation.py) |
| `credit_core`'s steps and tables reused on this flow's columns | [pipeline.py](retail_credit/pipeline.py), [pricing.py](retail_credit/pricing.py) |
| One overlay register applied to four values, each resolved once per decision, recorded beside the value, switchable with a param | [overlays.py](retail_credit/overlays.py), [tests/test_affordability.py](tests/test_affordability.py) |
| `Struct[...]` and `list[dict]` outputs: the winning quote and the cap chain | [pricing.py](retail_credit/pricing.py), [caps.py](retail_credit/caps.py) |
| Steps that are Python on purpose, marked `@allow_fallback` | [intake.py](retail_credit/intake.py), [caps.py](retail_credit/caps.py), [tests/test_serving.py](tests/test_serving.py) |
| Warm-up from `sample_request.json`, and serving the raw request bytes with JSON date strings | [sample_request.json](sample_request.json), [tests/test_serving.py](tests/test_serving.py) |

## Run it

From the repository root, put every example on the path once (this one imports `credit_core`):

```bash
export PYTHONPATH=$(printf '%s:' "$PWD"/examples/*/)
cd examples/retail-credit-e2e
export DECIDER_API__CODE_PATH="$PWD" DECIDER_API__PIPELINE="retail_credit.pipeline:build" DECIDER_CONFIG__BASEPATH="$PWD/configs"
```

Build (validates `configs/0.1.0` and compiles and warms every kernel with `sample_request.json`; the first build in a
fresh checkout takes a minute or two while numba compiles):

```bash
uv run decider build
# built config version 0.1.0 (pipeline retail_credit.pipeline:build, mode fused)
```

Serve, and decide the sample application. Its dates are plain JSON strings. A decision takes a few milliseconds:

```bash
uv run decider serve --workers 1 &
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations |
  jq '{outcome_code, offer_amount, offer_binding_constraint, instalment, nominal_annual_rate, risk_grade,
       amount_cap, amount_cap_binding_rule, max_affordable_instalment, rate_cell_id}'
```

```json
{
  "outcome_code": 1,
  "offer_amount": 76250.0,
  "offer_binding_constraint": "BIND-AFF",
  "instalment": 2578.72,
  "nominal_annual_rate": 0.19745,
  "risk_grade": 7,
  "amount_cap": 100750.0,
  "amount_cap_binding_rule": "CAP-0301",
  "max_affordable_instalment": 2580.09,
  "rate_cell_id": "rate_card.flex_loan@rc-2026.09#37|39|7"
}
```

The client asked for R95 000. The group-exposure headroom (R260 000 limit less R148 000 exposure) capped the amount at
R112 000, the channel overlay took 10% off that and the grid rounded it to R100 750, and the instalment ceiling of
R2 580.09 is what stops the offer at R76 250 (R76 500 would cost more than the ceiling). Why the cap is what it is:

```bash
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '.amount_cap_chain[] | select(.status == "bound")'
# {"rule_id":"CAP-0104","before":500000.0,"after":180000.0,"status":"bound"}
# {"rule_id":"CAP-0301","before":180000.0,"after":112000.0,"status":"bound"}
```

A sanctioned applicant, a campaign client, and a client whose debts make the request unaffordable until they settle the
external ones:

```bash
jq '.is_sanctioned=true' sample_request.json | curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{outcome_code, has_offer, decline_reason_codes, primary_reason_code}'
# {"outcome_code":4,"has_offer":false,"decline_reason_codes":[1006],"primary_reason_code":1006}

jq '.campaign_id="7712"' sample_request.json | curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{amount_cap_unadjusted, amount_cap, amount_cap_binding_rule}'
# {"amount_cap_unadjusted":134400.0,"amount_cap":120750.0,"amount_cap_binding_rule":"CAP-0118"}

jq '.bureau_accounts=[{"account_type_code":10,"balance":380000.0,"limit":400000.0,"instalment":4800.0,"months_in_arrears":0,"opened_date":"2020-01-01","closed":false,"is_internal":false},
                      {"account_type_code":20,"balance":60000.0,"instalment":5500.0,"months_in_arrears":0,"opened_date":"2019-05-01","closed":false,"is_internal":false}]
    | .requested_amount=30000 | .term_months=36 | .is_consolidation_eligible=true' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{outcome_code, affordability_verdict_code, affordability_verdict_code_hypothetical, offer_basis_code, offer_amount, existing_obligations, existing_obligations_hypothetical}'
# {"outcome_code":2,"affordability_verdict_code":3,"affordability_verdict_code_hypothetical":1,"offer_basis_code":2,"offer_amount":30000.0,"existing_obligations":18650.0,"existing_obligations_hypothetical":1150.0}
```

The last is outcome 2, approve with conditions: the offer is priced on R1 150 of obligations and holds only if the
client settles the rest. `affordability_verdict_code` stays 3 beside it, because the actual basis failed.

Tests (run from the repository root, with `PYTHONPATH` set as above):

```bash
uv run pytest examples/retail-credit-e2e -q
```

To regenerate `configs/0.1.0/params.json` (every policy number at its default, this product's fee and premium terms,
and the synthetic rate card): `uv run python scripts/generate_configs.py`.

## Layout

```
retail_credit/          one module per capability
  pipeline.py           build(): the whole decision
  intake.py eligibility.py fraud.py features.py                      admission to segment
  scoring.py grading.py caps.py overlays.py                          risk and policy
  affordability.py routing.py consolidation.py                       capacity and routing
  pricing.py flex_card.py solve.py offer.py                          the offer
  validation.py disclosure.py vocab.py                               checks, reasons, codes
configs/0.1.0/params.json   policy numbers and the rate card
scripts/generate_configs.py
sample_request.json
tests/                  behaviour tests; reference.py prices an offer by hand to check the pipeline
```

## Things worth knowing

- **One product, one way in.** The flow decides a new application for the Flex Loan. Other entry points (limit changes,
  campaigns, batch programmes) and other products would reuse most of these stages with different phase sets; they are
  not built here.
- **The solve tries every candidate.** The rate is a step function of the amount, so affordability is not monotone:
  the card drops about 155 basis points at R60 000 and an amount just above it can cost less per month than one just
  below. The solve prices every R250 step up to the lowest ceiling (373 for the sample) and keeps the largest that
  fits, so it is correct by construction and the tests check it against a hand-priced scan. A production search with a
  tight latency budget would bound the evaluations per request; this one takes a few milliseconds.
- **One pricing path.** The winner of `optimise` brings back its own price (`record=`), so the offer is never priced
  a second time and cannot disagree with the search. Everything that does not depend on the amount (credit life rate,
  fee terms, long-term loading) is looked up once before the search.
- **The rate card is a `param_table`** and not a `DecisionTableConfig` (marked `# workaround`): a table of 34 560 rows
  with two `between` columns is scanned linearly, about 200 microseconds per lookup, and the solve prices hundreds of
  candidates per request (373 for the sample). A `param_table` and an index cost about half a microsecond. The cost is
  that the card lives in the params document and its cells are found by position.
- **Consolidation is one scenario.** Settle every external debt, keep the internal ones. A search over subsets of debts
  proposes a new scenario each pass and is where `loop` would belong; with one scenario there is nothing to iterate.
- **Judgement calls, all in code and tests.** The segment order when a client fits several is this flow's choice. Income
  evidence that is only a bureau estimate or a declaration approves with conditions and caps the amount at R80 000.
  A campaign's uplift ceiling is an absolute R200 000. A fraud refer suppresses consolidation and refers the
  outcome, with an indicative offer attached. A grade worse than the caps allow is declined by routing.
- **One type per input column.** `employment_type_code`, `product_code` and `segment_code` are `float` wherever they are
  read, as `credit_core` requires. `channel_code` is read as `int` by the overlay step and as a number by a table, so
  [affordability.py](retail_credit/affordability.py) converts it for the table (marked `# workaround`).
- **What runs in Python.** Steps that read dates, lists of dicts or strings, or build lists and strings, are Python on
  purpose and marked `@allow_fallback`; the tables, the scorecard, the capacity arithmetic and the whole search compile.
- **Synthetic data.** Every table, threshold, rate and cap here is invented for illustration.

## Reuse

This example takes only `credit_core`, by package name: the income, deductions, expense-norm and obligation steps,
`affordability`'s discretionary income and capacity arithmetic, `eligibility`, `instalment`, `credit_life`, `fees`,
`rounding`, `adjustments.AdjustmentRegister`, `reason_codes.ReasonCodeRegistry`, `evidence` and the codes in `vocab`.
It does not import the affordability example's `assessment` package, although it covers the same ground: this flow
carries its own thresholds (a R185 minimum viable instalment, an 8% marginal band, its own buffer and ratio grids) and
restates the verdict rather than inheriting another product's policy.

It publishes no library. `retail_credit.pipeline:build` and its outputs are the artefact: a later example rebuilds this
flow from the other examples' packages and compares its decisions and figures with this one's.
