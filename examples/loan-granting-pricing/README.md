# Loan granting and pricing

A client asks the bank for an unsecured loan, naming an amount and a term. The bank answers within a fifth of
a second, and the answer is one of two things: a decline that carries every reason that applied, or a set of
offers (up to nine, one of them recommended), each stating an amount, a term, an instalment, a total cost and
an effective rate the client can be held to. This example is that decision for the bank's **Flex Loan**
(R2 000 to R500 000 over 6 to 84 months), built on the shared credit library and the affordability assessment.

## The business problem

The amount asked for is often not the amount the client can have. Policy may cap it at their risk grade, the
bank's appetite may cap it, their group exposure may leave no room for it, and most often what they can afford
does. So the real question is not "may they have R120 000?" but **"what is the largest amount they may have, at
each term the bank will write?"**

That has no formula, for three reasons you can see in [`pricing.py`](loan_granting/pricing.py):

- **The rate is a step function of the amount.** The rate card is a grid of 96 amount bands, 55 terms and
  12 grades (63 360 cells). Lending a little less can land in a dearer band, so the instalment goes *up* when
  the amount goes down. Affordability is not monotone in the amount.
- **The initiation fee is capitalised.** The client receives the amount and finances the amount plus the fee,
  and the fee itself depends on the amount (R1 050 plus 10% over R10 000, up to a statutory cap).
- **The credit life premium is charged on the amount financed**, so it inherits the fee's shape, and the
  instalment adds a service fee on top.

So the largest affordable amount is *found* by a search, and an approved offer is a contract: a mispriced one
is a regulatory finding and a direct loss. The search has to be bounded, deterministic, and **correct**: its
answer must equal pricing every R100 candidate and taking the largest that fits.

Around the search sits the rest of a granting flow: fourteen eligibility gates, a fraud hand-off, a bureau
data-quality check, a scorecard with approved policy **overlays** on its answers, a **waterfall of policy caps**
owned by different teams, the affordability assessment, offer-set rules, and a final validation that
re-derives the recommended offer from nothing and refuses to ship it if anything disagrees.

## What it decides

[`loan_granting.pipeline:build`](loan_granting/pipeline.py) takes the application (see
[`sample_request.json`](sample_request.json)) and returns:

| Result | Fields |
|---|---|
| Outcome | `outcome_code` (1 approve, 2 approve with conditions, 3 refer, 4 decline), `referral_queue_code`, `decline_reason_codes` (most severe first), `primary_reason_code` |
| Offers | `offers` (0 to 9, each with amount, term, rate, fees, premium, instalment, total cost, effective rate, `binding_constraint_code`, `is_recommended`), `recommended_offer`, `suppressed_terms` (every term without an offer, and why) |
| The search | `term_solutions` (per term: amount, evaluations used, what bound it), `solve_total_evaluations` |
| Ceilings | `amount_cap`, `term_cap`, `worst_acceptable_grade`, the rule that set each, each ceiling's chain of values, and one entry per rule (`waterfall_rules`) |
| Risk | `scorecard_id`, `segment_code`, `score`, `score_unadjusted`, `score_reason_codes`, `probability_of_default` and `_unadjusted`, `risk_grade`, `adjustments_applied` |
| Affordability | `max_affordable_instalment` and the assessment's other figures (`discretionary_income`, `living_expenses`, `existing_obligations`, `affordability_verdict_code`, ...) |
| Checks | `eligibility_gate_verdicts` (all 14, by name), `data_quality_verdict`, `fraud_handling_code`, `final_validation_passed`, `final_validation_failures` |
| Evidence | the cell or version of every table read: `rate_cell_id` on each offer, `risk_grade_cell_id`, `appetite_cell_id`, `fee_caps_cell_id`, `norm_table_version`, `reason_registry_version`, `recommendation_objective`, `statutory_ceiling` |

The stages, in the order the decision takes them:

1. **Eligibility.** Fourteen gates (age, age at maturity, residency, employment type, capacity to contract, debt review,
   administration order, insolvency and its seasoning, deceased or estate, exclusion lists, duplicate and in-flight
   applications, channel). Every gate runs even after one fails, so a client who fails four is told about all four. A gate
   whose input is not there yet (no term asked for) is `not_evaluated`: neither a pass nor a fail.
2. **Consent and fraud.** Bureau-enquiry consent must be present and unexpired, or the application is referred back to the
   channel. The fraud verdict (see below) declines, refers, or, if the fraud service timed out, refers unless the loan is small
   and the client long-standing with a clean internal history, which bypasses the check and is recorded.
3. **Bureau.** The enquiry is clean, has minor defects, has material defects (several subjects, a truncated list) or is
   unusable. Material and unusable **refer**: a record that could not be parsed has not shown the client to be a bad risk.
   A no-hit with no internal history goes to manual underwriting. Staleness uses the assessment's window, set to 45 days.
4. **Score, probability of default, grade.** `credit_core`'s scorecard, calibration and grade table, with four approved
   overlays applied in a declared order (score shift, scaling change, odds multiplier, boundary shift). The overlays are
   four `AdjustmentStack` documents in `configs/0.1.0`. The unadjusted value survives beside each adjusted one, and each
   overlay point has a switch that runs the same code with the stack off.
5. **Cap waterfall.** Twelve policy rules, each with an owner, narrow three ceilings (amount, term, worst acceptable grade)
   in register order, or decline outright. One rule, a pre-approved campaign's uplift, may *raise* the amount, within an
   authority ceiling. A grade worse than the worst acceptable grade declines.
6. **Affordability.** The assessment from [`affordability`](../affordability/README.md), asked for capacity alone.
7. **Offers.** Each permitted term is solved for its largest affordable amount. The minimum viable offer rules (amount,
   instalment, total cost ratio, in duplum, effective rate), a de-duplication, and the recommendation objective produce the
   offer set.
8. **Final validation.** The recommended offer is priced again from its amount and term alone and fourteen checks must
   agree to the cent. Any failure withdraws the offer and refers.
9. **Outcome.** The first stage that decides wins; the reasons are those of that stage.

## Features it shows

| decider feature | Where |
|---|---|
| A search in plain Python, called from a step, with a correctness argument and a test against brute force | [solve.py](loan_granting/solve.py), [tests/test_solve.py](tests/test_solve.py) |
| A 63 360-row rate card kept as a config document, handed to `build` by name and becoming the default rows of a `param_table` that the search reads with `bisect` | [pipeline.py](loan_granting/pipeline.py), [offers.py](loan_granting/offers.py), [rate_tables.py](loan_granting/rate_tables.py) |
| Reusing another example's whole pipeline as one step: `.relabel(reads=, writes=)` on `assessment.pipeline.build()` | [pipeline.py](loan_granting/pipeline.py) |
| `param_table`: policy numbers a step loops over, retuned like a param; one table `shared_key`-shared by two steps | [waterfall.py](loan_granting/waterfall.py) (`cap_bands`, the watchlist), [offers.py](loan_granting/offers.py) (permitted terms) |
| Small steps with `param` thresholds, one per gate, gathered by a final step | [eligibility.py](loan_granting/eligibility.py) |
| `AdjustmentStack` and `ReasonCodes` from `credit_core`, loaded from documents: the overlays and the reason ranking are in the diff, each overlay point switched off through the same code, and the unadjusted value kept | [scoring.py](loan_granting/scoring.py), [outcome.py](loan_granting/outcome.py), [`configs/0.1.0`](configs/0.1.0) |
| `DecisionTableConfig` tables from `credit_core` (appetite, grade boundaries, calibration, fee caps) renamed with `.relabel` | [pipeline.py](loan_granting/pipeline.py), [scoring.py](loan_granting/scoring.py) |
| `dag` of units with `.emit`, and `flow` for a value written twice (reasons fired, then ranked) | [pipeline.py](loan_granting/pipeline.py), [outcome.py](loan_granting/outcome.py) |
| `Text` inputs that stay in the kernel; `list[dict]` outputs (the offers) from steps that are Python on purpose | [bureau.py](loan_granting/bureau.py), [offers.py](loan_granting/offers.py) |
| `@allow_fallback` for those steps, checked by `Engine(strict_compile=True)` | [tests/test_serving.py](tests/test_serving.py) |
| Warm-up from `sample_request.json`; JSON date strings arrive as `date`s | [tests/test_serving.py](tests/test_serving.py) |

## Run it

From the repository root, put every example on the path once (this one imports `credit_core` and `assessment`):

```bash
export PYTHONPATH=$(printf '%s:' "$PWD"/examples/*/)
cd examples/loan-granting-pricing
export DECIDER_API__CODE_PATH="$PWD" DECIDER_API__PIPELINE="loan_granting.pipeline:build" DECIDER_CONFIG__BASEPATH="$PWD/configs"
```

Build (validates `configs/0.1.0`, compiles and warms every kernel with `sample_request.json`; the first build in a fresh
checkout takes a minute while numba compiles):

```bash
uv run decider build
# built config version 0.1.0 (pipeline loan_granting.pipeline:build, mode fused)
```

Serve, and decide the sample application: a client asking for R120 000 over 60 months. Its dates are plain JSON strings:

```bash
uv run decider serve --workers 1 &
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations |
  jq '{outcome_code, risk_grade, max_affordable_instalment, amount_cap, amount_cap_binding_rule_id,
       recommended: (.recommended_offer | {term_months, offered_amount, nominal_annual_rate, instalment, binding_constraint_code}),
       offers: [.offers[] | [.term_months, .offered_amount, .instalment]],
       suppressed: [.suppressed_terms[] | [.term_months, .reason_codes]]}'
```

```json
{
  "outcome_code": 1,
  "risk_grade": 9,
  "max_affordable_instalment": 5309.05,
  "amount_cap": 150000.0,
  "amount_cap_binding_rule_id": "CAP-0420",
  "recommended": {"term_months": 36, "offered_amount": 120000.0, "nominal_annual_rate": 0.1764, "instalment": 4969.47, "binding_constraint_code": "BIND-REQ"},
  "offers": [[6, 26500.0, 5297.18], [12, 49900.0, 5303.98], [18, 71100.0, 5304.5], [24, 92000.0, 5308.49], [36, 120000.0, 4969.47], [48, 120000.0, 4121.73]],
  "suppressed": [[60, [1603]], [72, [1603, 1604]], [84, [1603, 1604]]]
}
```

Read it as the business would. The client gets the R120 000 they asked for at 36 or 48 months (`BIND-REQ`). At 6 to 24
months the same R5 300 or so a month buys less, so each of those offers is the largest amount that fits the
instalment (`BIND-AFF`). At 60 months and longer the total cost is more than 1.85 times the advance (reason 1603), and
at 72 and 84 the charges exceed the advance (1604, the in duplum rule), so those terms are withheld. The recommended
offer is the largest amount, and among equals the cheapest. The response also carries every input and intermediate value.

Ask what the client qualifies for, with no amount and no term:

```bash
jq 'del(.requested_amount, .requested_term_months)' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{outcome_code, recommended: (.recommended_offer | [.term_months, .offered_amount, .binding_constraint_code]),
          offers: [.offers[] | [.term_months, .offered_amount, .binding_constraint_code]]}'
# 36 months now offers R129 400 (BIND-AFF); 48 and 60 months offer R150 000 (BIND-CAP), and 48 is recommended
```

A weaker client through the dearer channel is declined, with the reason and the characteristics behind it:

```bash
jq '.channel_code=4 | .bureau_score=480 | .revolving_utilisation=0.8' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{outcome_code, risk_grade, worst_acceptable_grade, decline_reason_codes, primary_reason_code, adjustments_applied}'
# outcome 4, grade 12 against a worst acceptable 9, reasons [1183, 4101, 4105], overlays ADJ-03-002, 003, 004
```

Tests (run from the repository root, with `PYTHONPATH` set as above):

```bash
uv run pytest examples/loan-granting-pricing -q
```

To regenerate `configs/0.1.0` (the synthetic rate card, the overlay stacks, the reason codes and the params document):
`uv run python scripts/generate_configs.py`.

## Layout

```
loan_granting/          the package
  pipeline.py           build(rate_card_flex_loan, the four *_adjustments, reason_codes): every stage wired into one dag
  pricing.py            RateCardIndex, CreditLifeIndex, PriceEvaluator: one candidate's price
  rate_tables.py        the rate card and credit life as param tables, and the lookups over them (what the steps use)
  solve.py              solve_term: the largest affordable amount at one term; exhaustive_search to check it
  waterfall.py          the cap register, run_waterfall, the cap_waterfall step, the cap_bands params table
  offers.py             solve every term, build the offer set, recommend, final validation
  scoring.py            segment, scorecard, calibration, grade, and the four overlays
  eligibility.py        the 14 gates and the step that gathers them
  fraud.py bureau.py    what the fraud verdict means here; consent and bureau data quality
  outcome.py reasons.py the outcome and queue; the reason codes' numbers (their ranking is a document)
  shipped.py            loads a shipped document by name
configs/0.1.0/          params.json, the rate card, the four overlay stacks and the reason codes
scripts/                generate_configs.py
sample_request.json
tests/
```

## What it publishes

Later examples import `loan_granting` by package name. These names are stable:

| Name | Use |
|---|---|
| `loan_granting.solve.solve_term(evaluator, term_months, risk_grade, applicant_age_years, is_joint, amount_cap, requested_amount, max_affordable_instalment, evaluation_ceiling=24)` | The largest affordable R100 amount at one term. Returns a `SolveResult` (`amount`, `binding_constraint_code`, `evaluation_count`, `pricing`). At most `evaluation_ceiling` candidates are priced; if that cannot prove the maximum it returns no amount and `BIND-EXH`, never a guess |
| `loan_granting.solve.exhaustive_search(...)` | The same answer by pricing every candidate. Slow; run it against each new rate card |
| `loan_granting.solve.BIND_AFF`, `BIND_CAP`, `BIND_REQ`, `BIND_MIN`, `BIND_MAX`, `BIND_CEIL`, `BIND_EXH`; `GRID`, `PRODUCT_MIN_AMOUNT`, `PRODUCT_MAX_AMOUNT` | What bound the amount; the grid and the product limits |
| `loan_granting.pricing.PriceEvaluator(rate_card_index, credit_life_index, statutory_ceiling, credit_life_substitution_declared, initiation_fee_cap=6500, monthly_service_fee_cap=69)` | `.evaluate(amount, term_months, risk_grade, applicant_age_years, is_joint, rate_addon_bps=0)` returns a `PricingResult`: rate and cell, fees, premium, instalment, total cost, effective rate, `priced` |
| `loan_granting.pricing.RateCardIndex.from_card(card)`, `RateCardIndex(rows)`, `CreditLifeIndex()` | The card as sorted bands: `.lookup(amount, term, grade)`, `.amount_bands(term, grade)` |
| `loan_granting.waterfall.cap_waterfall` | The waterfall as a step to embed in another `dag`. Its policy numbers are the `cap_bands` params table |
| `loan_granting.waterfall.run_waterfall(app, bands=None, ..., register=REGISTER)` | The same, as a function of a dict. `REGISTER` is the rules, `Rule` and `Effect` define one; `DEFAULT_BANDS` are the shipped numbers |
| `loan_granting.offers.build_offer_set(term_solutions, objective, probability_of_default, rules)` | The minimum viable offer rules, de-duplication and recommendation as a function; `steps(rate_card)` takes the rate card document and returns the three steps `solve_terms`, `offer_set`, `final_validation` |
| `loan_granting.scoring.unit(score_adjustments, calibration_scale_adjustments, probability_of_default_adjustments, probability_of_default_for_grading_adjustments)` | The segment-to-grade unit with its four overlay stacks (`AdjustmentStack` documents) |
| `loan_granting.outcome.unit(reason_codes)` | The outcome, then the `ReasonCodes` document ranking the reasons |

Inputs it reads that the caller owns, beyond the bureau and internal facts in `sample_request.json`: `decision_date`,
`product_code`, `channel_code`, `requested_amount` and `requested_term_months` (both may be absent), `is_joint_application`,
the fraud service's `fraud_verdict_code`, `fraud_response_ms` and `fraud_reason_codes`, `bureau_consent_given`
(required) and `bureau_consent_expires_on`, and the assessment's `applicant1_...` and `applicant2_...` income, expense
and account fields.

## Things worth knowing

- **The search is exact, and bounded.** While the rate never falls as the amount rises, the instalment never falls
  either (the fee, the premium and the annuity are all non-decreasing), so affordability is a prefix there. The search
  splits the amount range at every place the card's rate *drops* and bisects each stretch, highest first. The first
  stretch with an affordable amount holds the true maximum. A term costs at most 15 pricings plus one per
  stretch beyond the first, against a ceiling of 24. A card whose rate falls at *every* band edge can exhaust the ceiling, and then the term is withheld and
  the application is referred (queue 6, reason 1420): [`tests/test_solve.py`](tests/test_solve.py) checks 300 random cards
  full of drops against pricing every candidate, and shows that a bisection which assumes monotone affordability gets
  some of them wrong.
- **Affordability is asked for capacity alone.** The assessment is not given a proposed instalment, because the search
  tries many. Its verdict is therefore `pass` or `indeterminate` (short evidence: a stale bureau view, missing income,
  an account type nobody can cost), never `marginal` or `fail`. A client who cannot afford any instalment gets no offers,
  and is declined with reason 1310.
- **The fraud verdict arrives with the request.** The fraud service is a separate example: this flow takes its three fields
  (`fraud_verdict_code`, `fraud_reason_codes`, `fraud_response_ms`) as plain inputs and decides what they mean.
- **The steps read no Python objects.** The rate card is the default rows of the table param `shared.rate_card` (with
  `shared.rate_card_version`), the credit life rates are `shared.credit_life` in
  [`params.json`](configs/0.1.0/params.json), and the overlays and reason codes are documents, so the release diff sees
  each of them. `params.json` leaves the rate card out on purpose: the document `rate_card_flex_loan.json` is the one place
  Treasury edits it. A params document that does set `shared.rate_card` and `shared.rate_card_version` prices from those
  rows instead, sorted by grade, amount band and term band, with `cell_id`s of the form
  `rate_card.flex_loan@<version>#<amt_band>|<term_band>|<grade>` (`rate_card_rows` refuses a card whose ids are not).
  The credit life rows carry a `cell` number, so the cell ids of the premium come from the table too.
- **Policy lives in params, mechanism in code.** The cap bands, the permitted terms, the exposure watchlist, the offer
  thresholds, the statutory margin over repo and the recommendation objective are all in
  [`params.json`](configs/0.1.0/params.json), so their owners change them without a release. Which rules exist, and in
  which order, is code. The assessment's own params appear under `affordability_assessment/...`.
- **A params document is read once.** Pass a new dict for each change: editing one in place after scoring with it is not
  noticed.
- **Names the embedded assessment would overwrite are moved aside** (`affordability_decline_reason_codes`,
  `affordability_adjustments_applied`, ...), and the scorecard reads `applicant1_dependants_count` because the
  assessment's own `dependants_count` is downstream of the risk grade.
- **Two workarounds for what `credit_core` lacks**: it has no boundary-shift overlay (so the grade boundary tightening,
  `ADJ-03-004`, is an odds multiplier on the probability of default that is graded; the grade table is never edited),
  and its appetite grid has no new-to-bank rows (so they use the mass segment's, `# workaround` in `pipeline.py`).
- **Left out.** The challenger scorecard and the 45-characteristic scorecards (it uses `credit_core`'s eight), the long-term
  loading table (the shipped card prices all 6 to 84 months directly), the rate add-on and cap overlays (`PriceEvaluator`
  takes a rate add-on, nothing sets one), a fees-inclusive effective rate (the library's is the nominal rate compounded
  monthly, so the 60% effective-rate rule never fires on this card), the unadjusted grade, and the disclosure documents.
- **Synthetic data.** Every table, threshold and rate here is invented for illustration.

## Reuse

Built on [`shared-credit-core`](../shared-credit-core/README.md) (`credit_core`): `adjustments` and `reason_codes` (the
`AdjustmentStack` and `ReasonCodes` step types), `appetite`,
`calibration`, `credit_life`, `fees`, `instalment`, `rate_card`, `risk_grade`, `rounding`, `scorecard`,
`vocab`; and on [`affordability`](../affordability/README.md) (`assessment`), embedded whole. It does not reuse
`credit_core.consent` (that decides whether a message may go out; here consent gates the enquiry) or
`credit_core.eligibility` (it has no per-gate verdicts).
