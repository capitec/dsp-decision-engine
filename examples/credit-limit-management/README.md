# Credit limit management

Every month a bank decides, for each of its revolving accounts (here a card and an overdraft-style facility),
whether the limit should go up, go down or stay. A client who has run a R15 000 card cleanly for three years is worth
more at R28 000, and will move if the bank does not offer it. An account about to default is R60 000 of exposure the
bank chose to keep. Both directions are regulated: an increase needs an affordability assessment and the client's
agreement, a reduction needs a reason and, in most cases, notice.

## The business problem

Two things make this unlike the other examples.

- **The increase is not free.** All the increases the programme would like to make, added up, land on the balance
  sheet, and the Asset and Liability Committee (ALCO) sets a monthly ceiling on new limit. Typically only about half of
  the qualifying accounts can be funded, and *which* half is a decision about the whole book that no amount of
  per-account reasoning produces. So the example has two halves: a per-account pipeline that `decikit` serves, and a
  book-wide allocation that runs over the pipeline's results ([why it is not a step](#things-worth-knowing)).
- **Everything the bank does to the numbers must stay visible.** Policy moves the programme between quarterly matrix
  releases with *overlays*: "run at 80% of the matrix this month", "nothing above R5 000 this cycle", "-15 points on
  the partner-originated channel". A credit committee asks at every sitting what would have happened without them. So
  every adjusted value is recorded beside its unadjusted twin, and the whole cycle can be rerun with the overlays off
  through the same code.

## What it decides

[`limit_mgmt.pipeline:build`](limit_mgmt/pipeline.py) takes one account's state (cycle history, arrears and consent
markers, behaviour characteristics, income evidence, the client's bureau and internal debts) and returns:

| Result | Fields |
|---|---|
| Account state | `revolving_utilisation_6m`, `observed_spend_p90`, `cycle_bucket_count`, `utilisation_band` (1 is exactly 0%, 8 is 90% and over), `mob_band` (1 is under a year, 6 is 60 months and over) |
| Exclusions | `exclusion_codes` (every one that applies, not the first), `is_excluded` |
| Behaviour | `behaviour_score_raw`, `behaviour_score`, `behaviour_score_unadjusted`, `probability_of_default` (and `_raw`, `_unadjusted`), `behaviour_grade` 1 (best) to 12, `*_adjustments_applied` |
| Matrix | `matrix_cell_id`, the cell as authored (`matrix_multiplier_unadjusted`, `matrix_max_increase_unadjusted`, `matrix_min_increment_unadjusted`) beside the cell after the dial and the cycle cap (`matrix_multiplier`, `matrix_max_increase`), `uncapped_target_limit` and `uncapped_target_limit_unadjusted` |
| Caps | `cap_product_max`, `cap_income_multiple`, `cap_total_exposure`, `cap_observed_spend`, `policy_proposed_limit` and `policy_proposed_limit_unadjusted`, `policy_binding_cap_code` |
| Affordability | `evidence_tier_code` (1 verified salary, 2 irregular deposits, 3 declared and refreshed, 4 declared and stale, 5 none), `income_staleness_days`, `notional_instalment`, the assessment's own figures (`gross_monthly_income`, `discretionary_income`, `max_affordable_instalment`, `affordability_buffer_applied` beside `affordability_buffer_unadjusted`, ...), `cap_affordability`, `binding_cap_code` (0 matrix target, 1 product max, 2 income multiple, 3 total exposure, 5 observed spend, 6 matrix ceiling, 7 affordability) |
| Path | `increase_path_code`: 1 automatic, 2 conditional on the client confirming income, 3 not offered |
| Decrease | `decrease_trigger_fires`, `decrease_trigger_codes`, `decrease_target_limit`, `decrease_notice_class_code` |
| Answer | `final_proposed_limit`, `change_type_code` (1 automatic increase, 2 conditional increase, 3 decrease, 4 no change), `notice_class_code` (1 immediate, 2 prescribed notice, 3 consent required, 4 none), `decline_reason_codes` and `primary_reason_code` |

The rules, in the order the pipeline applies them:

1. **Population.** The matrix keys off the *6-month mean* utilisation, not the spot balance, so a client who settles
   the card the day before month end is not read as idle. Observed spend is the 90th percentile of the last six
   months' purchases. A young account has fewer than six cycles; it is scored on what it has.
2. **Exclusions.** Arrears now or in the last six months, debt review, deceased, fraud, a recent limit change
   (cooling off), no consent for automatic increases, under six months on book, and a suspension from collections. All
   are evaluated, so the reason set is complete. An excluded account is still scored: "we did consider your account"
   is itself an answer.
3. **Behaviour score.** A points scorecard on eight characteristics, a score-to-PD curve per product, and a PD ladder
   to a grade from 1 to 12. A score shift and a PD multiplier are overlays between the stages.
4. **The matrix.** 12 grades x 8 utilisation bands x 6 tenure bands x 2 products = 1 152 cells, each with a multiplier,
   a ceiling on the rand increase and a minimum increment. A grade-9 account gets 1.00 everywhere. The cycle dial
   applies to the multiplier's *excess over 1.00*: a 1.50 cell at an 80% dial is 1.40, not 1.20.
5. **Caps.** The lowest of the product maximum, an income multiple, the client's total exposure room, observed spend
   and the cell's ceiling. The proposal is rounded *down* to R500: a cap that rounding can breach is not a cap. The
   binding cap is always named.
6. **Affordability.** The shared assessment, in limit-increase mode, run against a notional instalment (the minimum
   payment on the new limit: 3.5% on the card, 5% on the facility). The programme layers a wider buffer on it and its
   own freshness rule on the evidence: an increase is *automatic* only on salary evidence within 45 days, a bureau view
   within 35, refreshed expenses and debts that have not grown over 25%. Anything else the evidence still supports is
   *conditional* on the client confirming income. The limit the affordable instalment implies caps the proposal, and a
   client who cannot carry any of it is not offered an increase.
7. **Decrease.** One trigger is built: a cycle past due this month. The new limit may not fall below the money already
   owed (statement balance, unsettled authorisations and accrued interest). It is immediate-class.

Then [`allocation.allocate`](limit_mgmt/allocation.py) takes the whole book:

8. **Allocation.** Qualifying increases are ranked by return per rand of limit (or by expected value, or by a policy
   priority, as ALCO chooses), then funded down the ranking while three envelopes hold: new limit, risk-weighted assets
   and expected loss. *An increase is funded in full or not at all.* One that does not fit is skipped and the walk
   continues, up to a cap on skips, after which the cycle stops. 15% of the offer envelope is held back to lift
   segments (product x grade x tenure) that the ranking starves; a segment it cannot lift is flagged. An immediate
   decrease on any of a client's accounts suppresses every increase for that client. Every account that is not funded
   carries the reason: below the funding line, skipped, fairness-capped, suppressed, excluded, not affordable, or
   "reduced below the minimum meaningful increase by an overlay" (which is reversible by withdrawing the overlay, and
   which a client and a credit committee need told apart from "ranked too low").

## Features it shows

| Decider toolkit feature | Where |
|---|---|
| A 1 152-cell `DecisionTableConfig` kept as a config document and handed to `build(matrix)` by name, so policy can replace it without a deployment | [matrix.py](limit_mgmt/matrix.py), [`configs/0.1.0/matrix.json`](configs/0.1.0/matrix.json), [pipeline.py](limit_mgmt/pipeline.py) |
| `ScorecardConfig`, a `DecisionTableConfig` with open-ended band ladders per product, and `credit_core`'s PD step reused with `.relabel` | [scoring.py](limit_mgmt/scoring.py) |
| Another example's whole pipeline embedded in a `flow`: `assessment.pipeline.evidence_unit()` and `capacity_unit()` with `.relabel(reads=, writes=)`, a programme overlay slotted between them | [affordability.py](limit_mgmt/affordability.py) |
| Overlays as `adjustment_stack` documents (`credit_core.adjustments.AdjustmentStack`) with the unadjusted value kept beside the adjusted one, switched off by one param through the same code; the reason codes as a `reason_codes` document; a ceiling that the overlay vocabulary cannot express as a plain param-switched step | [`configs/0.1.0/*_adjustments.json`](configs/0.1.0), [`reason_codes.json`](configs/0.1.0/reason_codes.json), [overlays.py](limit_mgmt/overlays.py), [tests/test_pipeline.py](tests/test_pipeline.py) |
| `flow`s composed into one `dag` with `.emit`, and step paths that become the params document | [pipeline.py](limit_mgmt/pipeline.py), [`configs/0.1.0/params.json`](configs/0.1.0/params.json) |
| Multi-output steps, `param` with bounds, `missing_as`, `@helper` for a function a compiled step calls | [caps.py](limit_mgmt/caps.py), [exclusions.py](limit_mgmt/exclusions.py) |
| Steps whose only input is a param, so a constant (the assessment mode) shows in the params document and the evidence record | [affordability.py](limit_mgmt/affordability.py) |
| `@allow_fallback` for steps that are Python on purpose (dates, ragged lists), checked by `FusedEngine(strict_compile=True)` | [population.py](limit_mgmt/population.py), [tests/test_serving.py](tests/test_serving.py) |
| One pipeline, three consumers: `decikit serve` for one account, `.compile("fused").run(frame)` over the whole book, and the same run with a candidate params document for simulation | [simulation.py](limit_mgmt/simulation.py), [tests/test_cycle.py](tests/test_cycle.py) |
| A book-wide decision outside the engine, as a plain polars function over the engine's frame | [allocation.py](limit_mgmt/allocation.py), [tests/test_allocation.py](tests/test_allocation.py) |
| Warm-up from `sample_request.json`, and serving the raw request bytes | [tests/test_serving.py](tests/test_serving.py) |

## Run it

From the repository root, put every example on the path once (this one imports `credit_core` and `assessment`):

```bash
export PYTHONPATH=$(printf '%s:' "$PWD"/examples/*/)
cd examples/credit-limit-management
export DECIKIT_API__CODE_PATH="$PWD" DECIKIT_API__PIPELINE="limit_mgmt.pipeline:build" DECIKIT_CONFIG__BASEPATH="$PWD/configs"
```

Build (validates `configs/0.1.0`, loads the matrix, and compiles and warms every kernel with `sample_request.json`; the
first build in a fresh checkout takes a minute while numba compiles):

```bash
uv run decikit build
# built config version 0.1.0 (pipeline limit_mgmt.pipeline:build, engine fused)
```

Serve, and decide the sample account: a card at R15 000, grade 2, a steady salary credited nine days ago.
Its dates are plain JSON strings:

```bash
uv run decikit serve --workers 1 &
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations |
  jq '{change_type_code, notice_class_code, matrix_cell_id, matrix_multiplier_unadjusted, matrix_multiplier,
       matrix_adjustments_applied, policy_proposed_limit_unadjusted, policy_proposed_limit, final_proposed_limit}'
```

```json
{
  "change_type_code": 1,
  "notice_class_code": 2,
  "matrix_cell_id": "limit_matrix@limit-matrix-2026.09#20|2|5|5",
  "matrix_multiplier_unadjusted": 1.4331,
  "matrix_multiplier": 1.3465,
  "matrix_adjustments_applied": ["ADJ-07-DIAL-2026-03"],
  "policy_proposed_limit_unadjusted": 21000.0,
  "policy_proposed_limit": 20000.0,
  "final_proposed_limit": 20000.0
}
```

The matrix said R21 000; the 80% dial took it to R20 000; both are in the response. The response carries every input
and intermediate too. Now the same client with the salary credit gone stale (last seen in June), and then with arrears:

```bash
jq '.salary_deposit_as_of_date="2026-06-01"' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{income_staleness_days, increase_path_code, change_type_code, notice_class_code, final_proposed_limit}'
# {"income_staleness_days":115,"increase_path_code":2,"change_type_code":2,"notice_class_code":3,"final_proposed_limit":20000}

jq '.worst_arrears_months_now=1' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{exclusion_codes, decrease_trigger_codes, change_type_code, notice_class_code, final_proposed_limit}'
# {"exclusion_codes":[1],"decrease_trigger_codes":["D03"],"change_type_code":3,"notice_class_code":1,"final_proposed_limit":8445}
```

The stale salary does not cost the client the increase; it makes it *conditional* on confirming income (consent
required, notice class 3). Arrears exclude the account from increases and propose a decrease to what is owed (R8 200
balance, R150 authorised, R95 interest), effective at once.

To see the affordability cap, give the account a R11 000 salary: the client can carry a R19 000 limit but not R20 000,
so `final_proposed_limit` is 19 000 and `binding_cap_code` is 7.

### The monthly cycle

The book-wide half is a Python call, not a request. `tests/population.py` perturbs the sample into a book of accounts;
a real run would read the month-end snapshot:

```python
from limit_mgmt.allocation import BudgetInstruction
from limit_mgmt.simulation import run_cycle, swap_set
from population import make_book  # tests/population.py

book = make_book(2_000)
budget = BudgetInstruction(applied_limit_budget=250_000.0)   # 269 accounts qualify for R586 500 of increases
production = run_cycle(book, budget=budget)
production.summary   # total_ranked 269, funded_count 162, envelope_consumed 344 000.0, which_envelope_bound "limit", tail_skips 128

# Simulate a candidate: nothing above R3 000 this cycle. Same code, a different params document.
cap = {"limit_mgmt": {"matrix": {"apply_cycle_cap": {"cycle_cap_enabled": True, "cycle_cap_amount": 3_000.0}}}}
capped = run_cycle(book, params=cap, budget=budget)
capped.summary["funded_count"]                          # 181: smaller increases reach 19 more clients for the same money
swap_set(production, capped)["swap_class"].value_counts()   # unaffected 1950, loser 29, gainer 21
```

`production.book` is the book with the decision on every account: `allocation_rank`, `ranking_value`,
`allocation_outcome_code`, `funded_amount`, `granted_limit` and `allocation_reason_code`.

Tests (run from the repository root, with `PYTHONPATH` set as above):

```bash
uv run pytest examples/credit-limit-management -q
```

To regenerate `configs/0.1.0` (the matrix and the params document):
`uv run python examples/credit-limit-management/scripts/generate_configs.py`.

## Layout

```
limit_mgmt/               the package
  pipeline.py             build(matrix): the per-account decision; OUTPUTS
  population.py           utilisation, observed spend, bands
  exclusions.py           the hard exclusions
  scoring.py              behaviour scorecard, score to PD, PD to grade
  matrix.py               the 1 152-cell matrix and its lookup
  overlays.py             the dial's arithmetic, the cycle cap, `due_for_review`
  documents.py            loads the documents in configs/0.1.0/ for callers of build() that pass none
  caps.py                 the caps and the rounding
  affordability.py        the bridge to the shared assessment, the evidence tiers, the automatic/conditional rule
  decrease.py             the decrease path
  outcome.py              final limit, change type, notice class, decline reasons
  allocation.py           allocate(): the budget over the whole book
  simulation.py           run_cycle(), swap_set()
  reasons.py vocab.py     reason codes; product, exclusion, cap, outcome codes
configs/0.1.0/            params.json, matrix.json, the four overlay documents and reason_codes.json
scripts/                  generate_configs.py
sample_request.json
tests/
```

## What it publishes

Later examples import `limit_mgmt` by package name. These names are stable:

| Name | Use |
|---|---|
| `limit_mgmt.pipeline.build(matrix, ...)` | The per-account decision as one `dag` named `limit_mgmt`. Embed it, or run it over a frame. Its params appear under `limit_mgmt/...`. Each argument is the config document of that name; all but `matrix` default to the documents shipped in `configs/0.1.0/`, so `build(matrix)` still works |
| `limit_mgmt.pipeline.OUTPUTS` | The columns it emits |
| `limit_mgmt.matrix` | `build_matrix_table(rows=None)` (a candidate matrix in code), `matrix_document()`, `lookup(matrix)`, `MATRIX_VERSION` |
| `limit_mgmt.allocation` | `allocate(scored, BudgetInstruction())` and `BudgetInstruction` |
| `limit_mgmt.simulation` | `run_cycle(snapshot, matrix=, params=, budget=)`, `swap_set(baseline, candidate)`, `Cycle` |
| `limit_mgmt.overlays` | `due_for_review(on, stacks=None)`, the ids of the shipped overlays (or of `stacks`) past their review date |
| `limit_mgmt.documents` | `shipped(name)`, a document of `configs/0.1.0/` loaded as the step it describes |
| `limit_mgmt.reasons` | The `R_*` codes, including the non-selection reasons; their severity ranking is `reason_codes.json` |
| `limit_mgmt.vocab` | `Exclusion`, `Cap`, `EvidenceTier`, `IncreasePath`, `ChangeType`, `NoticeClass`, `AllocationOutcome`, the product codes |
| `limit_mgmt.exclusions`, `limit_mgmt.decrease` | `exclusion_codes`, `decrease_trigger_fires`, ... for a caller that adds triggers or exclusions of its own |
| `limit_mgmt.affordability.unit()` | The programme's affordability bridge alone, for an event-driven request that has a requested amount |

Inputs it reads that the caller owns: `account_id`, `client_id`, `decision_date`, `product_code`, `current_limit`,
`cycle_balances`, `cycle_limits` and `cycle_purchase_values` (newest first), `months_on_book`, the arrears, debt-review,
deceased, fraud and consent markers, `last_change_date`, the eight behaviour characteristics, the income on file
(`declared_gross_income_on_file`, `declared_net_income_on_file`, `total_exposure_other`), the income evidence
(`verified_salary_present_3m`, `salary_deposit_variability_pct`, `salary_deposit_as_of_date`, `verified_salary_amount`,
`declared_income_raw`, `declared_income_age_days`), `bureau_as_of_date`, `expenses_refreshed_or_normed`,
`obligations_growth_pct`, `applicant1_*` debts and expenses as the assessment reads them, and for a decrease
`statement_balance`, `unsettled_authorisations` and `accrued_unbilled_interest`.

**Treatment state from collections is a data feed.** The pipeline reads `treatment_suspension_active` (true while an
account is under a payment arrangement or blocked) as a plain request input; whoever owns collections publishes it,
and the programme excludes the account for as long as it is true.

## Things worth knowing

- **The overlays and the reason codes are documents.** Credit Risk Policy edits `configs/0.1.0/matrix_dial_adjustments.json`
  (the 80% dial), `behaviour_score_adjustments.json` and `behaviour_pd_adjustments.json` (the partner-channel score shift,
  the Access Facility PD multiplier), `buffer_adjustments.json` (the programme's wider affordability buffer) and
  `reason_codes.json` (the severity order); `decikit build` hands each to `pipeline.build` by its name.
  Owner, approval reference, rationale and review date are in the same file but are not behaviour: editing them does not
  show in a `decikit.lab.diff`, changing an effect or a window does. Step names and params paths are unchanged.

  ```json
  {"type": "adjustment_stack", "name": "apply_matrix_multiplier_excess_adjustments", "target": "matrix_multiplier_excess",
   "set_id": "AS-07-2026.09", "adjusted_output": "matrix_multiplier_excess_adjusted", "adjustments": [
     {"adjustment_id": "ADJ-07-DIAL-2026-03", "kind": "cap_adjustment", "effect": {"op": "multiply", "value": 0.8},
      "scope": {}, "effective_from": "2026-09-01", "effective_to": "2026-11-01", "review_date": "2026-10-15", "...": "..."}]}
  ```

  `scripts/generate_configs.py` writes them. The affordability bridge still embeds `assessment`'s capacity unit, whose
  own overlay and reason codes are not yet documents, so a whole-pipeline `structure(a, a')` is not yet identical; that
  is the affordability example's to change.
- **Why the allocation is not a `decikit` step.** A rank and a funding line mean nothing for one row: they depend on
  where an account sits among 700 000 others and where the money runs out. An implementation that decides one account
  at a time ends up with a hard-coded score threshold standing in for the budget, wrong every month and silently. So
  `run_cycle` runs the pipeline compiled over the whole book and hands the frame to `allocate`.
- **One code path.** Production and simulation both call `run_cycle`; a candidate is just a different matrix, params
  document or budget. [`test_running_the_cycle_again_reproduces_every_rank_and_amount`](tests/test_cycle.py) pins that a
  rerun reproduces the cycle, and the same rerun with the overlays off (`switch_off_every_overlay` in
  [conftest.py](tests/conftest.py)) is the unadjusted answer for the whole book.
- **Overlays are checked before the cycle, not only applied.** `run_cycle` raises while any overlay is past its review
  date and not yet lapsed, so an unreviewed 70% dial cannot quietly run for three years. A single request served by
  `decikit serve` does not run that check.
- **Caps use the income on file.** C2 and C3 read `declared_gross_income_on_file` and `declared_net_income_on_file`, not
  the income the assessment works out: the assessment needs the proposed limit these caps produce, so keying them on
  its output would make the two depend on each other. C3 leaves this account only what the client's other accounts do
  not use (`total_exposure_other`).
- **Affordability is a cap and a gate.** The assessment's verdict on the notional instalment is recorded, but what
  decides the path is the evidence (`evidence_sufficiency_code`) and whether the client can carry *any* increase: the
  limit their capacity implies (C7) caps the proposal, and only a client who cannot carry even the first rand of it
  is `increase_path_code` 3. "We could not establish it" (reason 7010) is told apart from "you cannot carry it" (7003).
- **Two bureau windows.** The assessment treats a bureau view older than its `staleness_window_days` as no evidence,
  and `configs/0.1.0/params.json` sets that to 60 days; the programme's own 35 days decides automatic against
  conditional, between them. The params document also sets the programme's income haircuts: none on a steady salary
  or on declared income (which is indexed for inflation for up to a year instead), 15% on irregular deposits.
  [generate_configs.py](scripts/generate_configs.py) lists them.
- **A notice period, not an instant change.** An automatic increase runs on the client's standing agreement and takes
  effect after notice (class 2), during which they may decline. A conditional one waits for their confirmation
  (class 3). A decrease for arrears is immediate (class 1).
- **A funded increase is not yet a changed limit.** `granted_limit` is the limit the account ends the cycle with if the
  client does not decline; offers, consent records and expiry are out of scope.
- **The fairness floor is measured on the ranking alone.** A segment is starved if it is funded at under 40% of the
  book-wide rate the ranking produced; the reserve then funds its best-ranked unfunded accounts until it reaches the
  floor, or flags the segment (`starved_segments`) and marks its unfunded accounts fairness-capped. The reserve is
  spent after the ranking walk, not before, and only on the limit envelope.
- **Left out.** Thirteen of the fourteen decrease triggers; exclusions for insolvency, disputes, dormancy, in-flight
  applications, staff and product maximum; the group-exposure cap (C4); the hysteresis that keeps accounts at the funding
  line from flipping between cycles; the separate sub-envelope for the conditional list (conditional increases are not
  ranked); the short-circuit that skips scoring and affordability for an excluded account (it needs a `branch` around
  every step); the offer, consent and event-driven paths; cap overlays; a scorecard of 8 characteristics rather
  than dozens.
- **Synthetic data.** Every table, threshold, rate, curve and budget here is invented for illustration.

## Reuse

Built on [`shared-credit-core`](../shared-credit-core/README.md) (`credit_core`): `adjustments`, `reason_codes`,
`calibration`, `evidence`; and on [`affordability`](../affordability/README.md) (`assessment`): `pipeline.evidence_unit()`
and `capacity_unit()`, `modes`, `verdict`. The programme supplies what differs from origination (its evidence tiers, its
buffer, its freshness rule, the notional instalment) as inputs and an overlay, and never re-derives what the assessment
already computes.
