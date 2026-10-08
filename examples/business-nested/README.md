# Business credit with nested entities

A small business applies for a term or revolving facility. The bank must decide whether to lend, how much, at what
price, and, when the answer is no, say *which person and which event* said no. This example is that decision,
built from the shared credit library and the affordability assessment.

## The business problem

A business is not one thing. It is a registered entity plus the people and companies standing behind it:
directors, members, trustees, shareholders, the sureties the bank will require. Each of them carries a credit
history, and inside that history is a list of adverse events: judgments, default listings, tax notices.

That is two levels of nesting, both of variable length: an application has 1 to 40 entities and each entity has 0 to 60
events. Everything the bank decides sits on top of that shape, and three things make it hard:

- **One person, many paths.** A director who also holds shares is reached twice. Counted twice, their judgments and
  their ownership double (a total of 143% of the equity was once seen). They must be one entity, with the ownership
  added up, the most senior role kept and every event counted once.
- **An event means different things to different entities.** The same R80 000 unsatisfied judgment is material on a
  minor shareholder and disqualifying on the person in control. The thresholds depend on how critical the entity is,
  and policy may tighten them (halve the judgment threshold for construction until year end) *before* any event is
  classified, so the overlay changes the classification itself.
- **A verdict is not the worst event and not an average.** Three minor events in a year are material, and so are six
  unsatisfied judgments that are each too small to matter alone. So every verdict names the rule that decided it and
  the whole set of events that satisfied the rule.

A decline that says "adverse credit history" is not an answer. This one says: entity 1, Ms M. Dlamini, a required
surety, event 41, a R184 000 unsatisfied judgment, classified disqualifying under rule AE-R-01.

## What it decides

[`business_nested.pipeline:build`](business_nested/pipeline.py) takes the application (the business's figures, its
financial statements, the list of entities with their adverse events, the facility requested) and returns:

| Result | Fields |
|---|---|
| Outcome | `outcome_code` (1 approve, 3 refer, 4 decline), `decline_reason_codes` (every reason that fired, most severe first; they explain a referral too), `primary_reason_code`, `reason_registry_version` |
| Attribution | `attributing_entity_id`, `attributing_event_ids`, `attributing_rule_id`: the entity, the events and the rule a decline or a disputed referral rests on |
| Structure | `resolved_entity_count`, `ownership_reconciliation_pct`, `structure_unresolved` |
| Every entity | `assessed_entities`: one record per person with the criticality class, the adverse verdict (`adverse_verdict_code`, `adverse_binding_rule`, `adverse_event_ids`, the verdict `_unadjusted` and `_undisputed`), the score, probability of default and grade (each also unadjusted) and, per event, the age, the thresholds used and their base values, the overlays that moved them, and both severities |
| People | `people_pd`, `people_pd_unadjusted`, `people_weights` (aligned with `assessed_entities`), `people_grade_blended`, `people_grade`, `people_cap_rule`, `insufficient_people_coverage`, `no_scoreable_natural_person`, `business_decline_from_entity`, `business_decline_entity_id` |
| Financial | `interest_cover`, `current_ratio`, `gearing`, `financial_score`, `financial_pd`, `financial_grade`, `financial_confidence_code` |
| Business grade | `probability_of_default_before_overlay`, `probability_of_default`, `probability_of_default_unadjusted`, `risk_grade`, `risk_grade_unadjusted`, `adjustments_applied` |
| Offer | `offered_amount`, `max_amount`, `binding_constraint_code` (1 the request, 2 appetite), `term_months`, `nominal_annual_rate`, `initiation_fee`, `monthly_service_fee`, `instalment`, `total_cost_of_credit`, `effective_annual_rate` |
| Sole proprietor | `sole_proprietor_assessment_ran`, `sole_proprietor_affordability_verdict_code`, `sole_proprietor_max_affordable_instalment` |
| Evidence | the cell of every table read: `threshold_cell_id` per event, `calibration_cell_id` and `entity_grade_cell_id` per entity, `risk_grade_cell_id`, `appetite_cell_id`, `rate_card_cell_id`, and the id of each overlay set |

The rules, in the order the pipeline applies them:

1. **Structure.** Paths are merged into one entity per `entity_key`: ownership adds up, the role is the most senior,
   control is any path's, an event on two paths counts once. Ownership that does not come to about 100%, or more than
   40 entities, makes the structure *unresolved*: a referral, never a decline and never an approval.
2. **Criticality.** Control, a required surety, a sole principal or a 25% stake makes an entity critical; a 10%
   stake, or a director or trustee, significant; the rest peripheral.
3. **Events.** Each event gets the thresholds of its type and its entity's criticality, then the overlays in force on
   the decision date, then a severity: immaterial, minor, material or disqualifying. A disputed event is classified
   one class lower (never below minor); an event of unknown amount counts as exactly material; a satisfied one is
   immaterial.
4. **Verdict.** An entity's events roll up by rule: any disqualifying event (AE-R-01); three or more minor events in
   12 months (AE-R-02); unsatisfied amounts adding up past a material or a disqualifying limit that scales with the
   amount requested (AE-R-06, AE-R-07); a couple of immaterial events (AE-R-11); a peripheral entity capped at material
   (AE-R-12). The most severe verdict wins; each names its events.
5. **Entity score.** A scorecard, a tighten-only overlay (a natural person behind a new-to-bank business scores 12
   points lower), calibration to a probability of default and a grade 1 to 12 on the ladder of the entity's segment.
6. **People.** Owners are weighted by stake, directors and trustees by a notional share, the person in control by at
   least 60%; sureties and guarantors are outside the blend. The probabilities of default are blended on the log-odds
   scale. A weak holder of a fifth of the business, or a critical entity with a material record, keeps the people grade
   from being better than 10 or 5. Any disqualifying entity, in the blend or not, declines the business.
7. **Business grade.** The financial and people components are blended on log-odds (40/60), a business-level overlay
   tightens term facilities, and the product's ladder gives the grade.
8. **Offer.** Appetite caps the amount by grade, the rate card gives the rate by grade and security type, fees are
   capped, and the instalment is the annuity payment plus fees.
9. **Sole proprietor.** A sole proprietor lends in their own name, so the affordability example is run on them
   against the quoted instalment.
10. **Outcome.** Decline for a disqualifying entity, a failed statutory affordability or a grade of 11 or worse; refer
    for an unresolved structure, thin coverage, no natural person, an affordability that cannot be established or a
    grade of 9 or 10. A decline that rests *only* on disputed events becomes a referral that names the dispute.

## Features it shows

| decider feature | Where |
|---|---|
| `each` over a list column, and an `each` inside an `each`: per-item rules as ordinary steps at both levels of nesting | [entity.py](business_nested/entity.py) |
| A child flow sees only its own item, so the parent's context is copied onto it; cross-item work (merging paths) is one step over the whole list | [structure.py](business_nested/structure.py) |
| Nested `TypedDict` inputs: the `event_date`s inside the entities arrive as `date`s from raw JSON | [structure.py](business_nested/structure.py), [tests/test_serving.py](tests/test_serving.py) |
| The same child flow run on one item alone or on every item, so a partial re-assessment cannot drift from the full run | [counterfactual.py](business_nested/counterfactual.py) |
| `DecisionTableConfig` keyed on two columns (event type by criticality), and a banded ladder with open edges and a default | [events.py](business_nested/events.py), [scoring.py](business_nested/scoring.py) (`grade_lookup`), [pricing.py](business_nested/pricing.py) |
| `ScorecardConfig` with bin edges that are params | [scoring.py](business_nested/scoring.py), [financial.py](business_nested/financial.py) |
| Tighten-only overlays from `credit_core` at three levels: on thresholds before classification, on an entity's score, on the business probability | [events.py](business_nested/events.py), [scoring.py](business_nested/scoring.py), [grade.py](business_nested/grade.py) |
| One step run twice on other columns (`.relabel`, `.named`): the severity against base and overlaid thresholds, the grade of each probability | [events.py](business_nested/events.py), [scoring.py](business_nested/scoring.py), [blend.py](business_nested/blend.py) |
| An example embedded whole and run for some rows only: `branch` around the affordability example, its columns relabelled | [sole_proprietor.py](business_nested/sole_proprietor.py) |
| `param` for every policy number, nested under the `each` that runs the step; a generated params document | [rollup.py](business_nested/rollup.py), [blend.py](business_nested/blend.py), [`configs/0.1.0/params.json`](configs/0.1.0/params.json) |
| `flow` for a value written twice (reasons fired, then ranked; the instalment, then rounded) | [outcome.py](business_nested/outcome.py), [pricing.py](business_nested/pricing.py) |
| `@allow_fallback` for steps that are Python on purpose (lists of dicts, dates, library calls), checked by `Engine(strict_compile=True)` | [tests/test_serving.py](tests/test_serving.py) |
| `assert_equivalent`: every mode and a single record agree; warm-up from `sample_request.json` and raw request bytes | [tests/test_serving.py](tests/test_serving.py) |

## Run it

From the repository root, put every example on the path once (this one imports `credit_core` and `assessment`):

```bash
export PYTHONPATH=$(printf '%s:' "$PWD"/examples/*/)
cd examples/business-nested
export DECIDER_API__CODE_PATH="$PWD" DECIDER_API__PIPELINE="business_nested.pipeline:build" DECIDER_CONFIG__BASEPATH="$PWD/configs"
```

Build (validates `configs/0.1.0`, compiles and warms every kernel with `sample_request.json`; the first build in a
fresh checkout takes a minute while numba compiles):

```bash
uv run decider build
# built config version 0.1.0 (pipeline business_nested.pipeline:build, mode fused)
```

Serve, and decide the sample application. Its dates are plain JSON strings, inside the nested lists too:

```bash
uv run decider serve --workers 1 &
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations |
  jq '{outcome_code, decline_reason_codes, attributing_entity_id, attributing_event_ids, attributing_rule_id,
       risk_grade, risk_grade_unadjusted, offered_amount, instalment}'
```

```json
{
  "outcome_code": 4,
  "decline_reason_codes": [5402],
  "attributing_entity_id": 1,
  "attributing_event_ids": [41],
  "attributing_rule_id": "AE-R-01",
  "risk_grade": 7,
  "risk_grade_unadjusted": 6,
  "offered_amount": 3500000.0,
  "instalment": 106202.67
}
```

The sample is a construction company (sector 412) asking for R3.5m. Its first shareholder is a required surety with a
R184 000 unsatisfied judgment (event 41), which is disqualifying on a critical entity, so it is declined, and the offer
that would have been made is still priced. Each entity and each event is in the result, with the thresholds used:

```bash
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '.assessed_entities[] | {entity_id, criticality_class, adverse_verdict_code, adverse_binding_rule, adverse_event_ids, entity_grade}'
# {"entity_id":1,"criticality_class":1,"adverse_verdict_code":4,"adverse_binding_rule":"AE-R-01","adverse_event_ids":[41],"entity_grade":8}
# {"entity_id":2,"criticality_class":1,"adverse_verdict_code":1,"adverse_binding_rule":"AE-R-11","adverse_event_ids":[],"entity_grade":5}
# {"entity_id":3,"criticality_class":2,"adverse_verdict_code":3,"adverse_binding_rule":"AE-R-02","adverse_event_ids":[51,52,53],"entity_grade":9}
```

Entity 2 appears twice in the request (as a director and as a 55% shareholder) and once here. Entity 3's three small
default listings are individually minor; together they are material, and the verdict names all three. Event 42, a R7 000
judgment on entity 1, is material only because the construction overlay halves the threshold:

```bash
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '.assessed_entities[0].adverse_events[1] | {event_id, event_severity_code, event_severity_code_unadjusted, material_threshold_base, material_threshold, threshold_overlay_ids}'
# {"event_id":42,"event_severity_code":3,"event_severity_code_unadjusted":2,"material_threshold_base":10000.0,"material_threshold":5000.0,"threshold_overlay_ids":["ADJ-05-014"]}
```

Without the judgment the same business is approved:

```bash
jq '.entities[0].adverse_events |= map(select(.event_id != 41))' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{outcome_code, decline_reason_codes, risk_grade, nominal_annual_rate, offered_amount, instalment}'
# {"outcome_code":1,"decline_reason_codes":[],"risk_grade":7,"nominal_annual_rate":0.198,"offered_amount":3500000.0,"instalment":106202.67}
```

A sole proprietor is a natural person lending in their own name. The same request from a sole proprietor is also
tested against what they can afford (the business's profit on its declared turnover, tested against the quoted
instalment), and fails:

```bash
jq '.legal_form_code=1 | .entities=[{entity_id:1, entity_key:"PRINCIPAL", is_natural_person:true,
      relationship_type_code:11, effective_ownership_pct:100, is_controlling:true, is_sole_principal:true,
      bureau_score:705, months_on_record:100, worst_delinquency_months:0, adverse_events:[]}]' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{outcome_code, decline_reason_codes, sole_proprietor_affordability_verdict_code, sole_proprietor_max_affordable_instalment, instalment}'
# {"outcome_code":4,"decline_reason_codes":[5601],"sole_proprietor_affordability_verdict_code":3,"sole_proprietor_max_affordable_instalment":65390.0,"instalment":102881.5}
```

(Ask for R1 000 000 instead, with `.requested_amount=1000000`, and it is approved.) A decline that rests only on events
the person disputes is not a decline. Here a holder of 15% has six R90 000 judgments, two of them disputed; with
all six they add up to a disqualifying R540 000 against an 800 000 request, without the two they do not:

```bash
jq '.requested_amount=800000 | .entities = [
      {entity_id:1, entity_key:"OWNER", is_natural_person:true, relationship_type_code:5, effective_ownership_pct:85,
       bureau_score:705, months_on_record:100, worst_delinquency_months:0, adverse_events:[]},
      {entity_id:2, entity_key:"HOLDER", is_natural_person:true, relationship_type_code:5, effective_ownership_pct:15,
       bureau_score:705, months_on_record:100, worst_delinquency_months:0,
       adverse_events:[range(1;7) as $i | {event_id:$i, event_type_code:1, amount:90000, event_date:"2024-0\($i)-01", is_disputed:($i<=2)}]}]' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{outcome_code, decline_reason_codes, attributing_entity_id, attributing_event_ids, attributing_rule_id}'
# {"outcome_code":3,"decline_reason_codes":[5404],"attributing_entity_id":2,"attributing_event_ids":[1,2,3,4,5,6],"attributing_rule_id":"AE-R-07"}
```

Tests (run from the repository root, with `PYTHONPATH` set as above):

```bash
uv run pytest examples/business-nested -q
```

To regenerate `configs/0.1.0/params.json`: `uv run python scripts/generate_configs.py`.

## Layout

```
business_nested/        the package; one module per capability
  pipeline.py           build(), decision_unit(), DECISION_OUTPUTS
  entity.py             entity_unit() (events, verdict, score for one entity) and entities_unit() (all of them)
  structure.py          merge paths into entities, criticality, the structure bounds
  events.py             event thresholds, the construction overlay, severities
  rollup.py             an entity's verdict from its events, with attribution
  scoring.py            entity scorecard, overlay, calibration; the grade ladder
  blend.py              the people component: weights, log-odds blend, caps, the disqualifying entity
  financial.py          three ratios, scorecard, grade
  grade.py              the business grade and its overlay
  pricing.py            appetite, rate card, fees, instalment
  sole_proprietor.py    the affordability example, embedded
  outcome.py            outcome, reasons, attribution
  counterfactual.py     one event removed; one entity re-assessed
  vocab.py              roles, criticality and verdict codes
  records.py            builders for requests: entity(), event(), parse_dates()
configs/0.1.0/          params.json
scripts/                generate_configs.py
sample_request.json
tests/
```

## What it publishes

Later examples import `business_nested` by package name. These names are stable:

| Name | Use |
|---|---|
| `pipeline.build()` | The whole decision as one step. Embed it in another `flow`, and rename its inputs with `.relabel(reads=...)` |
| `pipeline.decision_unit()`, `DECISION_OUTPUTS` | Everything after the entities: reads `assessed_entities` and the business's fields. Run it again over changed entities |
| `entity.entity_unit()`, `entity.entities_unit()` | The flow for one resolved entity (a child of `each`, or run alone), and the structure step plus that `each` over all of them |
| `structure.resolved_entities`, `structure.structure_check`, `structure.criticality_class`, `Entity`, `AdverseEvent` | The nested request schema, the merge of paths, and the structure bounds |
| `events.event_unit()`, `events.build_threshold_table()`, `events.THRESHOLD_OVERLAYS` | Classify one event; the threshold table; the overlay register |
| `rollup.rollup()`, `rollup.adverse_verdict`, `rollup.AE_R_01` ... | A verdict from classified events, the step that writes it, and the rule ids |
| `scoring.entity_scoring_unit()`, `scoring.grade_lookup()`, `scoring.ENTITY_OVERLAYS` | Score one entity; a grade ladder on any probability column and key column |
| `blend.people_unit()`, `blend.blend_weights()`, `blend.log_odds_blend()` | The people component and its parts |
| `financial.financial_unit()`, `grade.business_grade_unit()`, `grade.BUSINESS_OVERLAYS`, `pricing.pricing_unit()` | The business-level units |
| `sole_proprietor.sole_proprietor_unit()` | The statutory affordability on a sole proprietor, gated on `legal_form_code` |
| `outcome.REASON_REGISTRY`, the `R_*` codes, `outcome.outcome_unit` | The reasons and the outcome step |
| `counterfactual.single_event_counterfactual()`, `without_event()`, `reassess_entity()` | What a decision would have been without one event; a decision with one entity corrected, assessing only that entity |
| `vocab` | Relationship roles, criticality classes, verdict codes |
| `records.entity()`, `records.event()`, `records.parse_dates()` | Request builders for tests and callers: an entity, an adverse event, a JSON request with its dates parsed |

Inputs it reads that the caller owns: `decision_date`, `product_code` (50 term, 51 revolving), `requested_amount`,
`requested_term_months` (default 36), `legal_form_code`, `sector_code`, `declared_annual_turnover`,
`security_type_code`, the statement lines `ebitda`, `finance_charges`, `current_assets`, `current_liabilities`,
`interest_bearing_debt` and `tangible_net_worth`, and `entities` (each with its `adverse_events`, see
[structure.py](business_nested/structure.py)). For a sole proprietor, optionally `principal_bureau_accounts`,
`principal_internal_accounts`, `principal_declared_expenses` and `principal_dependants_count`.

## Things worth knowing

- **The result is the audit record.** `assessed_entities` is the nested record: everything an auditor asks about an
  entity or an event is in it. `people_weights` is index-aligned with it; sorting is by `entity_id`, so the order of the
  request never changes an answer (tested by shuffling).
- **Event types are `credit_core`'s fourteen**, and garnishee orders stand in for the municipal and rental bucket of
  the threshold table. The thresholds are inclusive: an amount equal to a threshold is at that severity.
- **Events are not aged.** `credit_core` classifies on type, amount, status and satisfaction, so a five-year-old
  unsatisfied judgment weighs what a new one does.
- **A material verdict stands in for a recent adverse event** when capping the people grade of a critical entity. The
  caps change `people_grade`, which is reported; the business grade is blended from probabilities of default and
  does not read it.
- **What runs in Python.** Steps that read dates, lists of dicts or call library functions are Python on purpose and
  marked `@allow_fallback`; the scorecards, tables and ratios run in the numba kernel. Per-event work in an `each`
  child costs a Python call per event.
- **Params live under the `each` that runs them.** The verdict's rule numbers are at
  `business_credit/entities/entities/...` because `each` hoists its child's params to its own path.
- **Left out.** The graph collapse (each path arrives with its effective ownership already worked out), the
  business and entity disqualification matrices, severity decay by age, the remaining roll-up rules, thin-file and
  no-hit scoring, sureties and guarantors as cover, the financial spread beyond three ratios, a pricing search, the
  override, conditions precedent, the committee pack, and the communicable (client-facing) reason set.
- **The sole proprietor's income is a stand-in:** the business's profit on its declared turnover at an assumed margin
  (a param), evidenced as bank statements.
- **Synthetic data.** Every table, threshold and rate here is invented for illustration.

## Reuse

Built on [`shared-credit-core`](../shared-credit-core/README.md) (`credit_core`): `adverse_events` (the severity of one
event), `adjustments` (the overlays), `calibration`, `fees`, `instalment`, `rounding`, `reason_codes`, `evidence.cell_id`
and the vocabulary; and on [`affordability`](../affordability/README.md) (`assessment`), whose whole `build()` is
embedded for sole proprietors. It adds the nesting: merging paths, per-entity and per-event rules, the roll-up and the
people blend.
