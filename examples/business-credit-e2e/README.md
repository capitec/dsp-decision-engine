# Business credit, end to end

A business asks the Bank for a term loan or a revolving facility. The Bank decides, prices it and binds a covenant to
it. Then the facility lives for years: every year the Bank reviews it against the decision before, it tests the
covenant, and it has to be able to say what it knew about the business's owners on the day it decided. This example is
that life for two products (50, a term loan, and 51, a revolving facility), built on the nested-entity assessment, the
affordability assessment and the limit programme from the other examples.

It is the business-side counterpart of [`retail-credit-e2e`](../retail-credit-e2e/README.md), and the two differ in what
makes them hard. See "Compared with the retail flow" below.

## The business problem

A retail loan is decided once and then serviced. A business facility is decided again and again: at origination, at
each annual review, at each covenant test. Three things follow.

- **A decision has to be comparable with the one before it.** "The grade went from 7 to 9" means nothing unless the
  Bank can say why: the business's figures, who owns it now, what the owners' records say, a change of policy, an overlay
  that lapsed. Each is a different conversation with the client, and the causes must add up to the movement.
- **A covenant is a promise on the terms of the day.** Debt service cover (DSCR) is defined in a library whose wording
  changes over the years. A facility approved in 2026 is tested on the 2026 definition in 2029, even though a 2028
  definition is by then in force. And a test has three dates that must never be mixed up: the date the ratio is as at,
  the date the certificate arrived, and the date the Bank determined the result.
- **Evidence gets old.** Many reviews run on financial statements that are late or stale. The review must still happen,
  and what it concludes must be restricted by how weak its evidence was, never improved by it.

## What it decides

[`business_credit_e2e.pipeline:build`](business_credit_e2e/pipeline.py) is the new application (EP-1). It takes the
application [`business-nested`](../business-nested/README.md) takes, plus `facility_id` and the business's
`existing_accounts` (its current debt), and returns everything that example returns (outcome, reasons and the entity
that caused a decline, grade, offer, price, evidence) with these added:

| Result | Fields |
|---|---|
| Facility cost | `new_facility_instalment`: the term loan's instalment (product 50), or 5% of the limit with a floor of R50, the minimum payment (product 51; the assessment prices a term loan only) |
| Master scale | `master_scale_version`: the version of the grade ladder, read out of `risk_grade_cell_id`; two grades are comparable only on the same version |
| Covenant | `covenant_instance_id`, `covenant_definition_version` (resolved from `decision_date`), `covenant_threshold`, `covenant_cure_days` |
| DSCR at approval | `dscr_existing_debt_service` (each existing account costed by `credit_core`'s obligation treatments), `dscr_total_debt_service` (plus the new instalment), `measured_dscr` (EBITDA over twelve, over the debt service), `dscr_headroom_ratio`, `dscr_breach_class_code` (0 met, 1 technical, 2 material, 3 severe) |

The later decisions are plain functions over the decision records the first one leaves behind:

- [`review.annual_review`](business_credit_e2e/review.py) (EP-3) scores a facility again and returns a decision of
  record with: `review_basis_code` (1 complete, 2 late, 3 stale, 4 turnover only, 5 not performed), `risk_grade`
  (restricted by the basis), `grade_migration` (the causes of the movement in probability of default), `reprice` (the
  contractual rate, the rate the Bank would charge today, the shortfall in basis points and rand per year, and the
  action), `limit_decision` (revolving facilities), `exit_triggers`, and `next_review_date`.
- [`covenant.evaluate`](business_credit_e2e/covenant.py) tests a bound covenant; `covenant.certificate_not_received`
  records the fourth state, *not tested*.
- [`history`](business_credit_e2e/history.py) holds the decision store and the two questions the Bank asks about
  entity facts: what was true then (`query_effective`), and what the Bank knew then (`query_knowledge`).

## The annual review

A review is the same assessment on refreshed figures, compared with the previous decision of record.

| Basis | When | What it does |
|---|---|---|
| 1 complete | statements up to 9 months old | no restriction; next review in 12 months |
| 2 late | up to 15 months | 5% off EBITDA; 12 months |
| 3 stale | up to 21 months, with management accounts | 15% off EBITDA; grade no better than 7; 6 months |
| 4 turnover only | older or no statements, 6 months or more of turnover through the Bank | 15% off EBITDA; grade no better than 6; 3 months |
| 5 not performed | nothing | the grade stays as it was and is flagged `grade_stale`; 2 months |

A weaker basis restricts: the grade is the worse of the scored grade and the floor. (Grade 1 is best, 12 worst. A client
scored at 10 on a stale review stays at 10; one scored at 4 is held at 7.)

**Why the grade moved.** The movement in probability of default is split into five causes that add up to it exactly, and
each is measured by scoring again with one thing changed, always on today's decision date and policy:

1. `model`: the previous request, unchanged, scored under today's policy and tables, against what was recorded then.
2. `business_data`: the current financial statements added.
3. `entity_structure`: the current owners and roles added (events and bureau data still the previous ones).
4. `entity_data`: what is left to reach today's score, the entities' own records.
5. `overlay`: the change in what the overlays add (the difference between the probability with and without them).

The scale is not a cause of probability movement. It decides whether two grades can be compared at all:
`master_scale_changed` makes `comparison_basis_code` "not comparable".

## Compared with the retail flow

| | [`retail-credit-e2e`](../retail-credit-e2e/README.md) | this example |
|---|---|---|
| Hard because of | the joins between sixteen stages that run once, in order | time: the same facility decided again and again, and each decision comparable with the last |
| Built from | `credit_core` only, deliberately | the other examples, by package name (below) |
| Served | the whole decision | the new application; the review, covenant test and history are functions called by a service that keeps the store |
| State | none: a request in, a decision out | a store of decisions of record per facility, and a store of entity facts with two dates each |
| Dates | one: `decision_date` | `decision_date` picks every artefact; `knowledge_date` says what the Bank knew; a covenant has `test_date`, `delivery_date` and `determination_date` |
| Search | `optimise` for the largest affordable amount | none: the offer is a lookup (the assessment caps the amount) |
| Reason for a decline | a ranked list of rule codes | the same, plus the entity and the events behind it (from `business-nested`) |

## Features it shows

| decider feature | Where |
|---|---|
| Another example's `build()` embedded whole as the first step of a `flow`, its params nested under the outer flow's name | [pipeline.py](business_credit_e2e/pipeline.py), [`configs/0.1.0/params.json`](configs/0.1.0/params.json) |
| `param` for the policy a facility adds (the minimum payment), next to the params of the embedded assessment | [pipeline.py](business_credit_e2e/pipeline.py) |
| An effective-dated library (`EffectiveDatedSet.resolver_step`) that picks a version by `decision_date` once, and a pinned instance that never resolves again | [covenant.py](business_credit_e2e/covenant.py), [tests/test_covenant.py](tests/test_covenant.py) |
| `@helper` functions called from compiled steps, and `@allow_fallback` where a step is Python on purpose (dates, strings, lists of dicts) | [covenant.py](business_credit_e2e/covenant.py), [pipeline.py](business_credit_e2e/pipeline.py), [tests/test_serving.py](tests/test_serving.py) |
| A typed list input (`list[Account]`), so the `opened_date`s in the request arrive as dates from raw JSON | [covenant.py](business_credit_e2e/covenant.py), [tests/test_serving.py](tests/test_serving.py) |
| One compiled pipeline scored many times with changed records and params: the counterfactual runs behind a cause | [review.py](business_credit_e2e/review.py), [tests/test_review.py](tests/test_review.py) |
| Another example's pipeline (the limit programme) bound to an `Engine` and scored for one record | [review.py](business_credit_e2e/review.py) (`limit_decision`) |
| A partial re-assessment (one entity) that equals the full run | [tests/test_daily_pass.py](tests/test_daily_pass.py) |
| `assert_equivalent`: every mode and a single record agree; warm-up from `sample_request.json` and raw request bytes | [tests/test_serving.py](tests/test_serving.py) |

## Run it

From the repository root, put every example on the path once (this one imports `credit_core`, `business_nested`,
`assessment` and `limit_mgmt`):

```bash
export PYTHONPATH=$(printf '%s:' "$PWD"/examples/*/)
cd examples/business-credit-e2e
export DECIDER_API__CODE_PATH="$PWD" DECIDER_API__PIPELINE="business_credit_e2e.pipeline:build" DECIDER_CONFIG__BASEPATH="$PWD/configs"
```

Build (validates `configs/0.1.0`, compiles and warms every kernel with `sample_request.json`; the first build in a
fresh checkout takes a minute while numba compiles):

```bash
uv run decider build
# built config version 0.1.0 (pipeline business_credit_e2e.pipeline:build, mode fused)
```

Serve, and decide the sample application. Its dates are plain JSON strings, inside the lists too:

```bash
uv run decider serve --workers 1 &
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations |
  jq '{outcome_code, risk_grade, offered_amount, nominal_annual_rate, instalment, master_scale_version,
       covenant_instance_id, covenant_definition_version, covenant_threshold, dscr_existing_debt_service,
       dscr_total_debt_service, measured_dscr, dscr_headroom_ratio, dscr_breach_class_code}'
```

```json
{
  "outcome_code": 1,
  "risk_grade": 7,
  "offered_amount": 1000000.0,
  "nominal_annual_rate": 0.198,
  "instalment": 30392.91,
  "master_scale_version": "biz-rg-2026.09",
  "covenant_instance_id": "COV-DSCR-500001",
  "covenant_definition_version": "dscr-2026.03",
  "covenant_threshold": 1.25,
  "dscr_existing_debt_service": 35500.0,
  "dscr_total_debt_service": 65892.91,
  "measured_dscr": 1.3911,
  "dscr_headroom_ratio": 0.1411,
  "dscr_breach_class_code": 0
}
```

The sample is a construction company (sector 412) asking for R1m over 48 months. It already services a R28 000 loan and
holds a card with a R250 000 limit, which `credit_core` costs at 3% of the limit (R7 500), so R35 500 a month. The new
instalment brings debt service to R65 892.91, against EBITDA of R1.1m a year (R91 667 a month): a cover of 1.39 against
the 1.25 the covenant requires. The covenant is the 2026.03 definition because that is in force on the decision date.

Weaker earnings breach the covenant on day one, and the response says how badly. A revolving facility is tested on the
minimum payment on its whole limit. A decision date in 2029 binds the 2028 definition. A disqualifying judgment on a
required surety still declines, and the covenant fields are still there:

```bash
jq '.ebitda=400000' sample_request.json | curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{outcome_code, measured_dscr, dscr_headroom_ratio, dscr_breach_class_code}'
# {"outcome_code":1,"measured_dscr":0.5059,"dscr_headroom_ratio":-0.7441,"dscr_breach_class_code":3}

jq '.product_code=51 | .requested_amount=400000' sample_request.json | curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{outcome_code, offered_amount, instalment, new_facility_instalment, measured_dscr}'
# {"outcome_code":1,"offered_amount":400000.0,"instalment":12198.56,"new_facility_instalment":20000.0,"measured_dscr":1.6517}

jq '.decision_date="2029-01-15" | .knowledge_date="2029-01-15"' sample_request.json | curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{outcome_code, risk_grade, covenant_definition_version, nominal_annual_rate}'
# {"outcome_code":1,"risk_grade":6,"covenant_definition_version":"dscr-2028.09","nominal_annual_rate":0.18}

jq '.entities[0].adverse_events += [{event_id:41,event_type_code:1,amount:184000,event_date:"2025-03-14",status_is_active:true,is_disputed:false,is_satisfied:false}]' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{outcome_code, decline_reason_codes, attributing_entity_id, attributing_event_ids, attributing_rule_id, covenant_definition_version}'
# {"outcome_code":4,"decline_reason_codes":[5402],"attributing_entity_id":1,"attributing_event_ids":[41],"attributing_rule_id":"AE-R-01","covenant_definition_version":"dscr-2026.03"}
```

The review is a function, not a served route: it needs the store of earlier decisions. File the origination, then review
the facility a year later. The term-loan overlay that priced the first decision lapsed at the end of 2026, which is the
only thing that moved:

```python
import json
from decider import Engine
from business_credit_e2e import history, review, vocab
from business_credit_e2e.pipeline import build
from business_credit_e2e.records import parse_dates

params = json.load(open("configs/0.1.0/params.json"))
executable = Engine().bind(build(), mode="interpreted")
sample = json.load(open("sample_request.json"))

store = history.DecisionStore()
request = parse_dates(sample)
store.append(history.decision_of_record(
    request, executable.score(request, params),
    assessment_kind_code=vocab.EP1_NEW_TO_BANK, comparison_basis_code=vocab.COMPARISON_ORIGINATION))

next_year = parse_dates(sample | {"decision_date": "2027-09-24", "knowledge_date": "2027-09-24"})
reviewed = review.annual_review(executable, params, store, next_year, statement_age_months=6.0)
reviewed["risk_grade"], reviewed["next_review_date"]   # 6, datetime.date(2028, 9, 24)
reviewed["grade_migration"]["causes"]
# {'model': 0.0, 'business_data': 0.0, 'entity_structure': 0.0, 'entity_data': 0.0, 'overlay': -0.0225...}
reviewed["reprice"]
# {'contractual_rate': 0.198, 'indicated_rate': 0.18, 'margin_shortfall_bps': -180.0,
#  'margin_shortfall_per_year': -18000.0, 'action': 'none'}
```

Tests (run from the repository root, with `PYTHONPATH` set as above):

```bash
uv run pytest examples/business-credit-e2e -q
```

To regenerate `configs/0.1.0/params.json`: `uv run python scripts/generate_configs.py`.

## Layout

```
business_credit_e2e/
  pipeline.py           build(): the assessment, the facility's instalment, the master scale, the DSCR covenant
  covenant.py           the DSCR library and its steps; bind_covenant, evaluate, certificate_not_received
  review.py             annual_review, review_basis_code, next_review_date, limit_decision
  history.py            DecisionStore, decision_of_record, grade_migration, entity facts and their two queries
  records.py            parse_dates: what the handler does to a JSON request, for in-process callers
  vocab.py              the codes this example declares
configs/0.1.0/params.json
scripts/generate_configs.py
sample_request.json
tests/
```

## What it reuses

Measured by [tests/test_reuse.py](tests/test_reuse.py), which imports the pipeline in a fresh process, scores the
sample and lists what was loaded:

| Example | What this one takes | Where |
|---|---|---|
| [`business-nested`](../business-nested/README.md) (`business_nested`) | The whole decision: `pipeline.build()` embedded as the first step, so structure, entities, events, people, financials, grade, price and outcome are its own. `counterfactual.reassess_entity` is tested against the full run | [pipeline.py](business_credit_e2e/pipeline.py), [tests/test_daily_pass.py](tests/test_daily_pass.py) |
| [`shared-credit-core`](../shared-credit-core/README.md) (`credit_core`) | 15 of its 25 modules are imported by a decision, almost all through `business_nested` and `assessment`: adjustments, adverse_events, affordability, calibration, dates, deductions, evidence, expense_norms, fees, income, instalment, obligations, reason_codes, rounding, vocab. Directly, this example calls `dates.EffectiveDatedSet` (the covenant versions) and `obligations.assess` (the DSCR's existing debt service) | [covenant.py](business_credit_e2e/covenant.py) |
| [`affordability`](../affordability/README.md) (`assessment`) | Imported and built inside `business_nested`, but it *runs* only for sole proprietors (`legal_form_code` 1). The sample is a company, so no test here runs it | via `business_nested.sole_proprietor` |
| [`credit-limit-management`](../credit-limit-management/README.md) (`limit_mgmt`) | The per-account decision, for the limit decision in a revolving facility's review. Not imported by a new application | [review.py](business_credit_e2e/review.py) (`limit_decision`), [tests/test_review.py](tests/test_review.py) |
| [`consolidation`](../consolidation/README.md) | Nothing. A consolidation or restructure of a business facility would use its inventory and settlement, and this example has no such decision | |

Not imported by a decision: `appetite`, `bureau`, `consent`, `credit_life`, `eligibility`, `exposure`, `rate_card`,
`risk_grade`, `scorecard` and `pipeline` of `credit_core`.

## What it publishes

Nothing is built on this one yet. These are stable if it is:

| Name | Use |
|---|---|
| `pipeline.build()`, `pipeline.OUTPUTS` | The new application as one step; its params appear under `business_credit_e2e/...` |
| `covenant.DSCR_HISTORY`, `bind_covenant`, `evaluate`, `certificate_not_received` | A covenant library resolved once and a test of a bound instance |
| `history.DecisionStore`, `decision_of_record`, `grade_migration`, `entity_fact`, `query_effective`, `query_knowledge` | The decision chain and the two views of what the Bank knew |
| `review.annual_review`, `review_basis_code`, `next_review_date`, `limit_decision` | The review and its parts |
| `records.parse_dates` | A JSON request with its dates parsed |

## Things worth knowing

- **The limit programme is built for retail accounts.** It reads a behaviour score, income on file and the client's
  debts as a natural person's, and caps a facility of this kind at R150 000. A review of a revolving facility passes it
  a `limit_account` the caller assembles (the tests use a R15 000 facility, where an increase is possible). For a
  facility above the cap it can only hold the limit or decrease it. A business limit decision would need the DSCR in the
  programme, which is not there. The review refuses a revolving facility with no `limit_account`.
- **The DSCR is a covenant here, not a gate.** It is measured at approval and the response carries a day-one breach, but
  the outcome does not read it: a serviceability decision on the DSCR belongs in the assessment.
- **Product 51's cost is a stand-in.** The assessment prices a term loan only, so a revolving facility is tested on the
  minimum payment on its limit, a param. Its offered amount is still the assessment's single lookup.
- **A review needs an approved predecessor.** A declined application has no facility, so a review of one raises
  `KeyError`; an approval or a referral (a person's approval) has.
- **The contract carries forward.** A review record keeps the facility's contracted amount and rate, and reports the
  rate it would price today in `reprice`, so a second review compares with the contract and not with the first review's
  idea of today's price. An action follows only a shortfall (the Bank earning less than today's rate): on a revolving
  facility the margin is reset on notice, on a term loan it is flagged for renegotiation.
- **The turnover-only basis haircuts EBITDA by 15% and floors the grade at 6.** The floor is the specified one; the
  15% is this example's choice, the same as for a stale basis.
- **Basis 5 still scores.** It scores the refreshed request and then keeps the previous grade, so the movement of
  probability of default is reported next to a grade that did not move.
- **Statements and management accounts are two arguments.** Complete and late bases look at the statements' age only;
  the three-month management accounts the complete basis asks for are not checked.
- **The history is in memory.** `DecisionStore` and the entity facts are plain Python; a service keeps its own. The
  bi-temporal queries are not fed into the structure step: a caller reconstructs the entities as of a date and
  puts them in the request.
- **What runs in Python.** The covenant version, the covenant terms, the master scale and the existing debt service read
  dates, strings or lists of dicts and are marked `@allow_fallback`. The arithmetic of the DSCR compiles.
- **Left out.** The other seven products, entry points and phases of a business facility's life (additional
  facilities, early warning, amendment, restructure, exit), collateral, the authority levels, and the rest of the exit
  triggers (one of eighteen is implemented).
- **Synthetic data.** Every table, threshold, rate and covenant here is invented for illustration.
