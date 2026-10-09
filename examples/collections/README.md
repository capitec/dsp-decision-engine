# Collections treatment

Every weekday morning, before the contact centre opens, a bank has to decide what happens today to each of its
delinquent accounts: a message, a call, a visit, a notice, a hand-off to an agency, a legal instruction, or nothing.
This example is that decision for one account, plus the step that rations it against the people and the postage
the bank actually has.

## The business problem

The decision is hard for two reasons that most credit decisions do not have.

- **Time.** Almost every rule is about elapsed days or a count in a window: days past due, days since the last
  treatment, contacts in the last seven days, business days since a prescribed notice was delivered. Nothing is a
  snapshot.
- **History.** Today's treatment is the next position in a sequence that began when the account fell into arrears.
  An account that got an SMS yesterday and did not respond must escalate, not repeat. The bank's own past actions
  are inputs, and years later someone will ask why a client was telephoned eleven times in a fortnight.

Three things follow, and the example is built around them:

- **The law outranks the strategy.** A client under debt review, in an insolvency, deceased, in a hardship
  arrangement or in a dispute may not be contacted the way the matrix says. Every suspension is checked every day,
  all of them are recorded (not the first found), and each blocks a stated set of treatments. A blocked treatment
  is not swapped for a cheaper one: the account gets no action.
- **Strategy moves faster than releases.** Collections Strategy owns a 5 376-cell treatment matrix and adjusts it
  within the month with overlays (an intensity dial, a treatment suppression, score and probability overlays).
  Each overlay is scoped, dated and has a review date, and the unadjusted answer is kept beside the adjusted one. An
  overlay may make the bank do less than the matrix says, never more than the law allows: one that names a statutory
  parameter is refused when its document loads.
- **Supply is finite.** There are far more recommended calls than agent hours. Ranking and rationing are
  population-level facts, and each unselected account still needs a reason ("you were 91 412th of 186 000 and we
  stopped at 27 000").

## What it decides

[`collections_treatment.pipeline:build`](collections_treatment/pipeline.py) takes one account (its arrears, scoring
characteristics, status feeds, yesterday's path state, today's contact counts and, optionally, a proposed
arrangement with the client's income and debts) and returns:

| Result | Fields |
|---|---|
| State | `arrears_bucket_code` (1 to 8), `balance_band_code`, `contact_band_code` (1 responsive to 4 unreachable) |
| Suspensions | `suspension_codes`, `suspension_scopes`, `suspension_sources`, `suspension_expiries` (a date, or the event that lifts it), `permitted_treatment_codes`, `pre_prescription_flag` |
| Score | `collections_score` and `collections_score_unadjusted`, `roll_probability` and `roll_probability_unadjusted`, `collections_band_code` (1 most likely to cure to 6 most likely to roll) and `collections_band_code_unadjusted`, the overlays that applied to each, `recovery_estimate`, `cost_to_collect` |
| Treatment | `matrix_treatment_code` (as the matrix says), `overlaid_treatment_code` (after the overlays), `treatment_code` (what is done today), `suspension_blocked`, `treatment_intensity`, `matrix_cell_id` |
| Path | `path_position`, `episode_open`, `intensity_ceiling`, `channel_attempt_count`, `reset_event`, `escalation_or_hold`, each again with the suffix `_as_at_now`; `path_floored_intensity`, `must_escalate`, `interval_or_cap_blocked`, `interval_or_cap_detail` |
| Arrangement | `max_affordable_instalment`, `discretionary_income`, `affordability_verdict_code`, `arrangement_sustainable`, `arrangement_residual`, `arrangement_failure_detail` |
| Capacity | `pool_code`, `allocated`, `allocation_rank`, `allocation_cutoff_rank`, `allocation_ranking_basis`, `non_selection_reason_code` |

Treatments are 0 no action, 1 SMS, 2 email, 3 in-app message, 4 automated voice message, 5 to 7 agent call (low,
standard, high intensity), 8 field visit, 9 agency handover, 10 pre-legal notice, 11 legal handover, 12 settlement
offer, 13 write-off recommendation. Non-selection reasons: 200 matrix said no action, 205 an overlay suppressed it,
210 suspended, 220 cooling-off not over, 230 an interval or cap would be breached, 240 promise in force, 250 below the
capacity cut-off.

In the order the pipeline applies them:

1. **State.** The arrears bucket comes from the days past due, but an account in a performing arrangement is held at
   the bucket it entered in, and one cured twice in the year is floored at bucket 3.
2. **Suspensions.** All twenty are evaluated. Each fired one records its scope, source and how it lifts; the
   permitted set is every treatment no suspension blocks.
3. **Score.** A points scorecard gives `collections_score`; an overlay can shift it. A logistic curve per product
   family turns it into a roll probability (an overlay can stretch the curve's scale, and another can multiply the
   probability). Six bands cut the probability; an overlay can move the edge between bands 3 and 4 only. Unadjusted
   score, probability and band are computed from the unadjusted inputs and kept.
4. **Matrix.** Bucket, band, balance band, contact band and product family pick one of 5 376 cells: a treatment, an
   intensity, a number of retries and a cooling-off period. Overlays then dial the intensity or suppress a
   treatment.
5. **Suspension check.** The suspensions are checked against the *overlaid* recommendation. If it is blocked the
   account gets no action today.
6. **Path.** Yesterday's position, ceiling and attempt count are inputs. Today's events (a payment, a promise, a
   cure, a dispute, an arrangement) hold or reset the sequence, a roll to a worse bucket restarts the position but
   keeps the attempt count and the ceiling, and the per-client contact caps are checked. The path is worked out
   twice, from the events known on the day and from the events known now.
7. **Arrangement.** The affordability assessment runs in arrangement mode, then a sustainability test of its own:
   what is left after the instalment must be at least R350 and 5% of net income, and the instalment at most 85% of
   discretionary income.
8. **Capacity.** Accounts that survive are ranked within their pool and allocated down to the pool's supply, after
   a share reserved for high balances.

## Features it shows

| decider feature | Where |
|---|---|
| A config document handed to `build` by name: the 5 376-row matrix, changed with no release | [pipeline.py](collections_treatment/pipeline.py), [matrix.py](collections_treatment/matrix.py), [`configs/0.1.0/treatment_matrix.json`](configs/0.1.0/treatment_matrix.json) |
| `DecisionTableConfig` with inline rows, banded ranges and `eq` keys | [state.py](collections_treatment/state.py), [scoring.py](collections_treatment/scoring.py) |
| A decision table whose rows live in the params document, so a revised daily supply needs no rebuild | `build_pool_table` in [capacity.py](collections_treatment/capacity.py), [tests/test_capacity.py](tests/test_capacity.py) |
| `ScorecardConfig` with a default bin for each missing characteristic | `build_scorecard` in [scoring.py](collections_treatment/scoring.py) |
| `param_table`, shared by two steps, for the band edges | `collections_bands` in [scoring.py](collections_treatment/scoring.py) |
| `frame_step`: a population-level step (ranking and rationing) inside a per-record pipeline, with its own `param` | `allocate_capacity` in [capacity.py](collections_treatment/capacity.py) |
| `param` for policy, `missing_as` for optional inputs, request data never a `param` | [state.py](collections_treatment/state.py), [arrangements.py](collections_treatment/arrangements.py) |
| Multi-output steps (`outputs=`) returning index-aligned lists, one entry per suspension | [suspensions.py](collections_treatment/suspensions.py) |
| `.relabel` and `.named`: one step on other column names, or twice with different settings | `resolve_path_position_as_at_now` in [path.py](collections_treatment/path.py), the two roll probabilities in [scoring.py](collections_treatment/scoring.py) |
| Overlays as config documents, one `adjustment_stack` document per target, switched off with one param through the same code | [overlays.py](collections_treatment/overlays.py), [`configs/0.1.0/score_shift.json`](configs/0.1.0/score_shift.json) and its five siblings |
| Another example's pipeline halves, reused in arrangement mode | `assessment_unit` in [arrangements.py](collections_treatment/arrangements.py) |
| `@allow_fallback` for steps that are Python on purpose (dates, lists, strings), checked by `Engine(strict_compile=True)` | [tests/test_serving.py](tests/test_serving.py) |
| Warm-up from `sample_request.json`, and serving the raw request bytes | [sample_request.json](sample_request.json), [tests/test_serving.py](tests/test_serving.py) |

## Run it

From the repository root, put every example on the path once (this one imports `credit_core` and `assessment`):

```bash
export PYTHONPATH=$(printf '%s:' "$PWD"/examples/*/)
cd examples/collections
export DECIDER_API__CODE_PATH="$PWD" DECIDER_API__PIPELINE="collections_treatment.pipeline:build" DECIDER_CONFIG__BASEPATH="$PWD/configs"
```

Build (validates `configs/0.1.0`, compiles and warms every kernel with `sample_request.json`; the first build in a
fresh checkout takes a minute while numba compiles):

```bash
uv run decider build
# built config version 0.1.0 (pipeline collections_treatment.pipeline:build, mode fused)
```

Serve, and decide the sample account. Its dates are plain JSON strings:

```bash
uv run decider serve --workers 1 &
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations |
  jq '{arrears_bucket_code, collections_score, collections_band_code, matrix_cell_id, matrix_treatment_code,
       treatment_code, suspension_codes, pool_code, allocated, allocation_rank, non_selection_reason_code}'
```

```json
{
  "arrears_bucket_code": 3,
  "collections_score": 640.0,
  "collections_band_code": 3,
  "matrix_cell_id": "treatment_matrix@matrix-2026.09#3|3|3|2|1",
  "matrix_treatment_code": 4,
  "treatment_code": 4,
  "suspension_codes": [116],
  "pool_code": 4,
  "allocated": true,
  "allocation_rank": 1,
  "non_selection_reason_code": null
}
```

A voice message, ranked first of one in the voice-message pool. The one suspension is the client's withdrawn email
consent (116), which blocks email and nothing else.

The response carries every input and intermediate too. Now the same account under debt review. The matrix still
recommends the voice message, but a client under a debt review application may only receive statutory notices:

```bash
jq '.debt_review_stage_code=1 | .debt_review_case_number="DR-9001"' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{overlaid_treatment_code, treatment_code, suspension_blocked, suspension_codes, permitted_treatment_codes, allocated, non_selection_reason_code}'
```

```json
{"overlaid_treatment_code":4,"treatment_code":0,"suspension_blocked":true,"suspension_codes":[101,116],"permitted_treatment_codes":[0,10,13],"allocated":false,"non_selection_reason_code":210}
```

An account 150 days past due is recommended a field visit, but the in-force overlay suppresses field visits:

```bash
jq '.days_past_due=150' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{matrix_treatment_code, overlaid_treatment_code, suppression_adjustments_applied, treatment_code, non_selection_reason_code}'
```

```json
{"matrix_treatment_code":8,"overlaid_treatment_code":0,"suppression_adjustments_applied":["ADJ-COLL-MATRIX-002"],"treatment_code":0,"non_selection_reason_code":205}
```

Tests (run from the repository root, with `PYTHONPATH` set as above):

```bash
uv run pytest examples/collections -q
```

To regenerate `configs/0.1.0` (the synthetic matrix, the overlay stacks and the params document):
`uv run python scripts/generate_configs.py`.

## Capacity is a population fact

`decider serve` decides one account, which is a population of one: it ranks first of one in its pool. The ranking
and rationing step is written for the whole day. Run the pipeline over a frame of accounts and it ranks them
together (see `test_a_revised_pool_supply_in_the_params_document_changes_who_is_allocated`):

```python
out = pipeline.run(day, params=params)          # day: a polars frame, one row per account
out.select("account_id", "pool_code", "allocated", "allocation_rank", "non_selection_reason_code")
```

Two things change the allocation without a release. The day's supply is in the params document, in the rows of
`collections_treatment/capacity/pools` (revise a row's `pool_supply`). The ranking basis is
`collections_treatment/allocate_capacity/ranking_basis`: `"value"` ranks by expected recovery net of cost,
`"priority"` by policy (deeper arrears first, pre-prescription and new bucket-3 entrants above all). Ties break on
`account_id`, so a re-run gives the same allocation account for account.

## Layout

```
collections_treatment/    the package
  pipeline.py             build(treatment_matrix, and the six overlay stacks)
  vocab.py                treatment, pool, non-selection and reset codes
  state.py                arrears bucket, balance band
  suspensions.py          the twenty suspensions and what each blocks
  scoring.py              scorecard, calibration, bands, contact band, recovery, the score's tiny post-steps
  matrix.py               the matrix document, its lookup, the intensity and treatment post-steps
  overlays.py             `in_dag`: renames a stack's output columns and refuses statutory targets
  path.py                 the escalation path, as known and as at now; intervals and caps
  arrangements.py         the affordability assessment in arrangement mode, sustainability
  capacity.py             pools, blockers before ranking, ranking and rationing
configs/0.1.0/            params.json, treatment_matrix.json and the six overlay stacks
scripts/                  generate_configs.py
sample_request.json
tests/
```

## What it publishes

Later examples import `collections_treatment` by package name. These names are stable:

| Name | Use |
|---|---|
| `collections_treatment.pipeline.build(treatment_matrix, score_shift, scaling_change, odds_multiplier, band_boundary_shift, intensity_dial, treatment_suppression)` | The whole daily decision as one step. Embed it in another `dag`/`flow` (its params appear under `collections_treatment/...`) and rename its inputs with `.relabel(reads=...)` |
| `collections_treatment.pipeline.EMIT` | The columns it emits |
| `collections_treatment.vocab` | `NO_ACTION` to `WRITE_OFF_RECOMMENDATION`, `TREATMENT_NAMES`, `ALL_TREATMENTS`, `TREATMENT_POOL`, the pool codes, the `NS_*` non-selection reasons, the `RESET_*` events |
| `collections_treatment.suspensions` | `RULES` (a tuple of code, scope, source, how it lifts, treatments blocked), `evaluate_suspensions`, `treatment_after_suspensions`, `pre_prescription_flag`, `add_business_days` |
| `collections_treatment.overlays` | `in_dag(stack, applied)`: renames an `AdjustmentStack`'s output columns for the shared `dag` and refuses a stack that overlays a statutory target; `STATUTORY_TARGETS` |
| `collections_treatment.matrix` | `generate_matrix()`, `lookup(document)`, `matrix_cell_id`, `MATRIX_VERSION` |
| `collections_treatment.capacity` | `allocate_capacity` (the `frame_step`), `build_pool_table()`, `default_pool_rows()`, `BASIS_VALUE`, `BASIS_PRIORITY` |
| `collections_treatment.path` | `resolve_path_position_step`, `PATH_FLAGS`, `PATH_OUTPUTS`, `interval_and_cap_check_step` |
| `collections_treatment.scoring`, `state` | the scorecard, band and recovery tables, `arrears_bucket_code`, the balance-band table |

Inputs it reads that the caller owns: `account_id`, `decision_date`, `days_past_due`, `arrears_amount`,
`outstanding_balance`, `product_family_code`, the scorecard characteristics (`payment_ratio_last`,
`days_since_last_payment`, `right_party_contact_rate_90d`, `promise_kept_rate`, `broken_promises_90d`,
`account_age_months`, `other_accounts_in_arrears`, `times_cured_12m`), the status fields in
[`suspensions.evaluate_suspensions`](collections_treatment/suspensions.py), yesterday's `previous_*` path state, the
event flags (`qualifying_payment`, `promise_captured`, `cured`, `dispute_raised`, `arrangement_activated` and their
`_as_at_now` twins), the contact counts, `days_since_last_treatment`, and for the arrangement
`proposed_instalment` plus the applicant fields described in the
[affordability example](../affordability/README.md).

## Things worth knowing

- **Overlays are documents.** Each of the six overlay targets is an `adjustment_stack` document in
  `configs/0.1.0/` (`score_shift`, `scaling_change`, `odds_multiplier`, `band_boundary_shift`, `intensity_dial`,
  `treatment_suppression`), handed to `build` by name like the matrix. Owner, approval reference, rationale and review
  date sit beside the effect and scope, and editing them is not a change of behaviour. The step names and params paths are
  unchanged, so `collections_treatment/score_shift/adjustment_stack_enabled` still switches a stack off. A stack
  reads the columns its overlays scope on (any names); the few steps that need the overlaid value as an integer or a probability
  are small post-steps (`matrix.treatment_intensity`, `scoring.clamp_probability`).

```json
{"type": "adjustment_stack", "name": "score_shift", "target": "collections_score", "set_id": "AS-COLL-SCORE-2026.09",
 "base_field": "collections_score_unadjusted", "adjusted_output": "collections_score",
 "adjustments": [{"adjustment_id": "ADJ-COLL-SCORE-001", "kind": "score_shift", "target": "collections_score",
   "effect": {"op": "add", "value": -22.0}, "scope": {"agency_placed_12m": true}, "stack_position": 1,
   "owner": "Collections Strategy", "approval_reference": "MRV-2026-014", "rationale": "...",
   "effective_from": "2026-01-01", "effective_to": "2026-12-31", "review_date": "2026-10-01", "tighten_only": true}]}
```

- **Each suspension blocks a stated set of treatments.** [`suspensions.RULES`](collections_treatment/suspensions.py)
  is the table. A debt review application (101) leaves only the statutory notice; a hardship arrangement in good
  standing (111) leaves nothing; withdrawn consent blocks only the treatments that use that channel; a promise in
  force leaves the SMS reminder. The sets are illustrative readings of the scopes, not legal advice.
- **The suspension check reads the overlaid recommendation.** An overlay that suppressed a treatment shows as 205;
  one that left a blocked treatment standing shows as 210. Nothing downstream can un-suspend an account.
- **The band overlay moves one edge.** It shifts only the edge between bands 3 and 4; the unadjusted band uses the
  edges as authored and the unadjusted probability, which comes from the unadjusted score and the base scale.
- **One cure count.** `times_cured_12m` drives both the bucket-3 floor and the re-entry position. The policy text
  says "twice in six months" in one place and "twice in 12 months" in another; this example uses 12.
- **Request data is never a param.** `under_performing_arrangement`, `bucket_at_arrangement_entry` and
  `times_cured_12m` are inputs. The only param in the arrangement unit is the assessment mode, and a request that
  supplies `assessment_mode_code` is refused, because the pipeline writes it.
- **A rationed pool, not a rationed treatment.** Treatments 12 and 13 draw on no pool and are never cut off.
- **Left out.** The escalation path is evidence only: `path_floored_intensity` and `must_escalate` are reported but
  do not change `treatment_code`. There is no allocation-weighting overlay, no fairness floors beyond the reserved
  share, no pacing of agency and legal volumes against the month, no cohort neutrality, and no sibling-account
  coverage (reasons 260 and 270). Business days are Monday to Friday with no public holidays. The scorecard has ten
  characteristics, not the policy's twenty-eight.
- **Synthetic data.** Every table, threshold and rate here is invented for illustration.

## Reuse

Built on [`shared-credit-core`](../shared-credit-core/README.md) (`credit_core`): `adjustments`, `calibration`,
`evidence.cell_id`, `vocab.ConsentChannel`; and on the [affordability example](../affordability/README.md)
(`assessment`): `pipeline.evidence_unit()`, `capacity_unit()` and `modes.ARRANGEMENT`, run in arrangement mode
with no change to its arithmetic.
