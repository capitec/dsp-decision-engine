# Transaction fraud interdiction

Decide, while an instant payment is in flight, whether to let it through, challenge the client, hold it for a
person, decline it, close the channel or freeze the account. It is a flat set of 635 rules that analysts own,
tuned between releases by approved overlays, and it records everything that fired, not just the winner.

## The business problem

An instant payment that clears is gone within seconds, so the assessment has to happen before it completes. The
bank runs a large flat rule set written by fraud analysts, not engineers:

- **521 live rules and 114 shadow rules** in five families: card fraud (`CF`), account takeover (`AT`),
  mule and scam (`MS`), first-party fraud (`FP`) and AML-adjacent (`AA`). A *shadow* rule is a candidate on live
  traffic: it is evaluated and recorded, and it must never change the outcome.
- **Rules overlap on purpose.** On a real mule payment nine rules may fire at once across three families with
  three owners. All nine are recorded, because the analyst tuning the family, the dispute pack and the monthly
  false-positive review each need the whole set. One rule *decides*.
- **Overlays tune the set without editing it.** During a scam wave the duty officer layers a "1.4x more
  sensitive" overlay on the mule family. The authored rule still reads R8 000; the effective threshold is lower
  while the overlay is in force. Each decision records what the rules would have done without the overlays.
- **Some things are not negotiable.** A sanctions match, a frozen account, a confirmed mule, a blocked device, a
  compromised card or a court order sets a minimum action that no rule or overlay can soften, yet every rule is
  still evaluated so the record is complete.
- **A fraudster must not learn the rule.** The client is told the same thing for every `decline`, whichever
  rule fired.

The enrichment feeding the rules (velocity counters, device and counterparty lookups, the model score) can be
stale or missing. The decision records how degraded it was, and stale or restricted inputs change which rules apply.

## What it decides

[`fraud_interdiction.pipeline:build`](fraud_interdiction/pipeline.py) takes an instant-payment event, already
enriched, and returns:

| Result | Fields |
|---|---|
| Decision | `action_code` (10 allow, 20 monitor, 30 step up, 40 hold for review, 50 decline, 60 block channel, 70 freeze account), `action_source_rule_id` (the rule, or `HARD_BLOCK:<gates>`), `governance_exception` |
| Reasons | `decline_reason_codes` (every reason, most severe first), `primary_reason_code` (the deciding rule's or gate's), `reason_registry_version` |
| What fired | `fired_rule_ids`, `unevaluable_rule_ids`, `fired_on_overlay_ids`, `shadow_fired_rule_ids` |
| What the overlays did | `mule_scam_amount_multiplier` and `_unadjusted`, `applied_adjustment_ids`, `adjustment_set_id`, `counterfactual_fired_rule_ids`, `counterfactual_action_code` |
| Enrichment | `velocity_completeness_band`, `enrichment_degradation_code` (a bit per source), `degraded_mode_code` (0 normal, 1 reduced, 2 restricted, 3 fail closed), `late_arrival_flag` |
| Hard blocks | `hard_block_code` (a bit per gate), `hard_block_forced_action` |
| Client | `client_visible`, `client_wording_key` |
| Evidence | `rule_set_version` |

The rules, in the order the pipeline applies them:

1. **Enrichment verdict.** Each velocity aggregate is fresh, stale or absent. A short window absent, or two
   aggregates absent, is *materially degraded*. The degraded sources become a bitset; two or more sources, or
   materially degraded velocity, is *restricted*; a stale sanctions or mule list is *fail closed*.
2. **Hard blocks.** Six gates, each forcing a minimum action. When two hold, both are recorded and the more severe
   action applies.
3. **Overlays.** The mule and scam amount thresholds are multiplied by the product of the overlays in force on the
   decision date (a festive x0.7, an attack-window x1/1.4), in their declared order. A separate overlay turns an
   account-takeover rule's `monitor` into a `step_up` on the web channel.
4. **Rules.** Every *applicable* rule is evaluated, with no early exit. A rule applies when its event type, effective
   dates and segments fit, and (in restricted mode) when it does not read velocity. A rule that reads a stale
   aggregate follows the behaviour it declares: be *unevaluable*, evaluate to false, or use the value as it is.
   Rules that read an overlay-adjusted threshold are evaluated twice, so the decision can say which rules fired
   only because of an overlay.
5. **Action.** The most severe action among the firings wins; ties go to higher severity, then lower priority
   number, then family seniority (`AA`, `MS`, `AT`, `CF`, `FP`), then rule id. A *critical* rule is binding: when
   one fires, only critical rules compete. Two critical rules that disagree are a *governance exception*. A hard
   block is a floor under all of it.
6. **Counterfactual.** The action the authored rules alone would have produced, without the overlays.
7. **Wording.** A generic wording per action; the key is replaced if the client has not consented to the
   channel (`credit_core.consent`).

Not built: the enrichment lookups themselves (the request carries resolved values), staleness for all 168 velocity
aggregates (three are tracked), compensating rules in restricted mode, the held-payment rule for fail-closed mode,
dispatch to queues, outcome feedback, languages, the runners-up record, and the latency target.

## Features it shows

| decider feature | Where |
|---|---|
| `TreeConfig` as a flat rule set: `prioritized_flat_rule`, `mode: "all"`, so every rule answers and none exits early | `_tree` in [rules.py](fraud_interdiction/rules.py) |
| A step type of your own, `ConfigurableStep`, whose document holds rule governance *and* conditions and which expands into trees and a frame step with `ctx.expand` | [rules.py](fraud_interdiction/rules.py), [`configs/0.1.0/rule_set.json`](configs/0.1.0/rule_set.json) |
| `frame_step` for work as wide as the rule set (635 columns read by name) | [firing.py](fraud_interdiction/firing.py) |
| Computed features that read a column an earlier step wrote (`amount - 8000 * mule_scam_amount_multiplier`) | [`rule_set.json`](configs/0.1.0/rule_set.json), [overlays.py](fraud_interdiction/overlays.py) |
| Steps that compile: `param`, `missing_as`, `IntEnum` constants, bitsets, multi-output `step(outputs=)` | [enrichment.py](fraud_interdiction/enrichment.py), [hard_blocks.py](fraud_interdiction/hard_blocks.py) |
| `@allow_fallback` for steps that are Python on purpose (dates, lists, strings), checked with `Engine(strict_compile=True)` | [action.py](fraud_interdiction/action.py), [overlays.py](fraud_interdiction/overlays.py), [tests/test_serving.py](tests/test_serving.py) |
| `DecisionTableConfig` in a config document, then `flow` so a later step may overwrite what the table wrote | [`client_wording.json`](configs/0.1.0/client_wording.json), [pipeline.py](fraud_interdiction/pipeline.py), [wording.py](fraud_interdiction/wording.py) |
| `dag` (dependency order) for the pipeline, `flow` (written order) for one value written twice | [pipeline.py](fraud_interdiction/pipeline.py) |
| A params document, including a switch that turns the whole overlay stack off | [`params.json`](configs/0.1.0/params.json), [backtest.py](fraud_interdiction/backtest.py) |
| One executable for a batch (`run`) and a record (`score`) with the same answers; a backtest is the batch run | [backtest.py](fraud_interdiction/backtest.py), [tests/test_serving.py](tests/test_serving.py) |
| JSON dates arriving as strings, warm-up from `sample_request.json`, serving the raw request bytes | [sample_request.json](sample_request.json), [tests/test_serving.py](tests/test_serving.py) |

## Run it

From the repository root, put every example on the path once (this project imports `credit_core`):

```bash
export PYTHONPATH=$(printf '%s:' "$PWD"/examples/*/)
cd examples/transaction-fraud
export DECIDER_API__CODE_PATH="$PWD" DECIDER_API__PIPELINE="fraud_interdiction.pipeline:build" DECIDER_CONFIG__BASEPATH="$PWD/configs"
```

Build (validates `configs/0.1.0`, compiles and warms every kernel with `sample_request.json`). The first build in
a fresh checkout takes several minutes, because numba compiles the 521-rule tree; later builds reuse the cache:

```bash
uv run decider build
# BUILD_LINE
```

Serve, and assess the sample payment. Its date is a plain JSON string:

```bash
uv run decider serve --workers 1 &
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations |
  jq '{CURL_FIELDS}'
```

```json
CURL_OUTPUT
```

The response carries every input too. Change a field and ask again. A sanctioned counterparty is blocked, and the
rules are still all evaluated:

```bash
jq '.sanctions_list_match=true' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq '{action_code, action_source_rule_id, primary_reason_code, hard_block_code, fired_rule_ids}'
SANCTION_OUTPUT
```

Tests (from the repository root, with `PYTHONPATH` set as above):

```bash
uv run pytest examples/transaction-fraud -q
```

To regenerate `configs/0.1.0` (the synthetic rules, the wording table and the params document):
`uv run python scripts/generate_configs.py`.

## Backtest

The rules that run in the payment path also run over stored events, so a threshold or an overlay can be costed
before it is deployed. `backtest` runs the events through the same executable with the overlay stack on and off:

```python
import polars as pl
from fraud_interdiction.backtest import backtest

events = pl.read_parquet("last_90_days.parquet")
backtest(executable, events, params)
# {"with_overlays": {"events": ..., "blocked": ..., "hit_rate": ..., "value_blocked": ...}, "without_overlays": {...}}
```

## Layout

```
fraud_interdiction/
  pipeline.py        build(rule_set, client_wording): the whole decision
  rules.py           Rule and RuleSet: the config document, expanded into trees plus the firing step
  firing.py          applicability, stale velocity and the overlay effect on each firing (a frame step)
  action.py          action resolution: precedence, critical rules, hard-block floor, reasons, counterfactual
  enrichment.py      velocity band, degradation bitset, degraded mode, late arrival
  hard_blocks.py     the six gates
  overlays.py        the overlay register (built on credit_core.adjustments) and its steps
  reasons.py         the fraud reason registry (credit_core.reason_codes)
  wording.py         consent-aware client wording
  backtest.py        batch costing with the overlays on and off
  vocab.py           actions, modes, gate and family codes
configs/0.1.0/       rule_set.json, client_wording.json, params.json
scripts/             generate_configs.py
sample_request.json
tests/
```

## Things worth knowing

- **Governance beside the condition.** A rule's owner-facing attributes (family, severity, action, priority,
  effective dates, segments, criticality, reason code, what to do when velocity is stale) sit in the same JSON
  object as its condition. A tree node has no field for them, and they do not vary per record, so `RuleSet` keeps
  them and generates the trees. Changing a rule's action is a config change, not a code change.
- **Overlay-adjusted rules carry two conditions.** `condition` reads the multiplier column; `base_condition` is
  the same rule at its authored thresholds. The base tree is evaluated only for those rules (18 to 19 of 635).
  An `overlay_exempt` rule may not have a `base_condition`; the document refuses to load otherwise.
- **Shadow isolation is structural.** `resolve_action` is not given `shadow_fired_rule_ids`, so a shadow rule has
  no way to reach the action.
- **Critical rules.** "Binding" is read as: any firing critical rule means only critical rules compete, so a
  critical `allow` also overrides ordinary firings of other families. The business description is not explicit
  about that second part.
- **Hard blocks floor the counterfactual too.** They are not overlays, so what the authored rules alone would have
  done still includes them.
- **What runs in Python.** Steps reading dates, lists or strings (`rule_set_version`, `amount_overlays`,
  `resolve_action`, the wording step) are Python on purpose and marked `@allow_fallback`; the enrichment verdict,
  the hard blocks and the rule trees run in the kernel.
- **Synthetic data.** Every rule, threshold and band is invented. The 635 rules come from a fixed seed; `MS-0208`
  is written by hand so the worked examples have a rule to point at.

## Reuse

Uses [`shared-credit-core`](../shared-credit-core/README.md) by package name:

| From `credit_core` | Used for |
|---|---|
| `adjustments.AdjustmentRegister`, `Adjustment` | the overlay stack: scope, stack order, expiry and tighten-only checked when an overlay is defined; a switch that turns the stack off |
| `reason_codes.ReasonCodeRegistry` | ranking every fired reason; an unregistered code raises |
| `dates.EffectiveDatedSet` | the `rule_set_version` in force on the decision date |
| `consent.consent_verdict_step`, `channel_permitted_step` | whether the client may be told on the chosen channel |

It publishes nothing for other examples to import.
