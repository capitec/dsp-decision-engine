# Campaign targeting trees

Once a month a bank decides, for each of its clients, which of its live credit campaigns the client should be
offered, at what tier, for how much and through which channel. This example is that decision for one campaign,
the **Flex Loan pre-approved top-up** (campaign 23), and it is the example that shows the decider toolkit's **decision trees**:
the targeting policy is a tree that campaign analysts own, kept as a config document a business user can edit.

## The business problem

Campaign analysts fit a decision tree for each campaign against a history of who took what, argue about it in a
forum, and publish it. Dozens are live at once. Nobody in the business thinks about targeting in any other shape,
and three things follow from that:

- **The path is part of the answer.** Reporting is built on how many clients reach each node: node volumes,
  dead branches, the response rate of a leaf, "why did this client get this offer" two years later. A decision that
  names only its leaf is a failed decision. So every assignment carries the ordered list of nodes it walked.
- **A published tree is not edited in flight.** Policy still has to move: a funding constraint arrives, the contact
  centre can staff fewer calls, early arrears get worse. Policy layers an **overlay** over the tree: a named, approved,
  dated change to one threshold. The tree stays as validated, no node changes identity, and the answer *without*
  the overlay is kept beside the answer with it, so the tree's own performance can still be measured.
- **Not everyone in the tree is contacted.** Some clients must never be evaluated (deceased, fraud marker); some
  are evaluated but must not be contacted (opted out, contacted too often), because analytics reports who *would*
  have been targeted; some are deliberately held out as a control group so the campaign's effect can be measured.

## What it decides

[`campaign_trees.pipeline:build`](campaign_trees/pipeline.py) takes one client's evaluation for campaign 23 (see
[`sample_request.json`](sample_request.json): the client's facts from the feature mart, the cycle and the consent
flags) and returns the assignment record:

| Result | Fields |
|---|---|
| Outcome | `assignment_outcome_code` (0 do not target, 1 target, 2 held out, 3 suppressed), `advertised_amount` (non-zero only for 1) |
| The tree | `leaf`, `tree_path` (node ids, root first, joined with `>`), `leaf_outcome_code`, `offer_tier_code`, `tier_ceiling`, `channel_1`, `channel_2`, `priority_weight`, `reason_label` |
| The tree without overlays | the same fields prefixed `unadjusted_` (`unadjusted_leaf`, `unadjusted_tree_path`, ...) |
| Suppression | `suppression_codes` (every rule that applies), `suppressed_absolute`, `evaluated` |
| Eligibility and consent | `is_eligible`, `eligibility_decline_reasons`, `consent_verdict`, `channel_permitted` |
| The amount | `pre_assessed_amount`, `pre_assessed_term`, `binding_constraint_code`, `pre_assessment_valid_from` / `_to`, `appetite_max_amount`, `offer_amount` |
| Overlays | `overlay_stack_id`, `advertised_tier_ceiling`, `tier_ceiling_before_cap_overlay`, `cap_adjustments_applied` |
| Groups | `is_control`, `is_universal_holdout` |

In the order the pipeline applies them:

1. **Eligibility and consent**, from `credit_core`.
2. **Suppressions.** Every rule that applies is recorded, not only the first. Four are *absolute* (deceased, debt
   review, fraud marker, do-not-target list); the rest only forbid the contact.
3. **Evaluation**, only for a client who is eligible and not absolutely suppressed. Anyone else has these fields null,
   which is how the record says "not evaluated":
   1. the **pre-assessed amount**, the largest affordable Flex Loan at 36 months, from the loan-granting search;
   2. the **tree**, twice, with the overlays and without (it reads the pre-assessed amount at nodes 7 and 8);
   3. the **cap reduction** overlay on the tree's tier ceiling;
   4. the **offer amount**: the pre-assessed amount, but never more than the tier ceiling.
4. **Groups.** A stable hash of the client's id puts about 5% of clients in the campaign's control group and 1% in the
   universal holdout.
5. **The assignment.** A client the tree targets is `TARGET`, unless they are in a control group (`HOLD_OUT`) or a
   measurement-relevant suppression applies (`SUPPRESSED`); either way the tree and the amount are on the record.

[`arbitration.arbitrate`](campaign_trees/arbitration.py) is the step after, over a whole batch of assignments: which of
a client's qualifying campaigns become a contact, given the channels' capacity and a two-contact cap per cycle.

## Features it shows

| Decider toolkit feature | Where |
|---|---|
| A tree as a config document: `TreeConfig` in v3 form, `build(campaign_23, ...)` receives it by name | [`configs/0.1.0/campaign_23.json`](configs/0.1.0/campaign_23.json), [pipeline.py](campaign_trees/pipeline.py) |
| `trace_output`: the whole path as node ids, and `path_output`: the leaf | the first lines of [`campaign_23.json`](configs/0.1.0/campaign_23.json), [tests/test_tree.py](tests/test_tree.py) |
| `and`/`or` over several conditions in one node, `isin`, a computed feature comparing two columns (node 7) | [`campaign_23.json`](configs/0.1.0/campaign_23.json) |
| A node with two parents (node 6 is reached from nodes 3 and 4) keeps one id, so its volume is one figure | [tests/test_tree.py](tests/test_tree.py) |
| A threshold written `{"param": ..., "default": ...}`: an overlay moves it from the params document, with no new tree version | [`params.json`](configs/0.1.0/params.json), [overlays.py](campaign_trees/overlays.py) |
| One document run twice with `.named` and `.relabel`: the adjusted and the unadjusted answer | [pipeline.py](campaign_trees/pipeline.py) |
| A node whose branch is unknown (null feature) takes its otherwise branch instead of failing | [tests/test_tree.py](tests/test_tree.py) |
| A tree that reads a value an earlier step computed (`pre_assessed_amount`) | [pipeline.py](campaign_trees/pipeline.py) |
| `branch` with one arm: absolute suppression skips the whole evaluation, the tree and the solve included | [pipeline.py](campaign_trees/pipeline.py), [tests/test_pipeline.py](tests/test_pipeline.py) |
| Checking a tree document before it is published, and diffing two versions by node id | [trees.py](campaign_trees/trees.py), [tests/test_tree.py](tests/test_tree.py) |
| A 300-node tree loads, compiles and gives the same paths compiled and interpreted | [tests/test_scale.py](tests/test_scale.py) |
| Reusing other examples: the search (`loan_granting`), eligibility, consent, appetite and overlays (`credit_core`) | [preassessment.py](campaign_trees/preassessment.py), [overlays.py](campaign_trees/overlays.py) |
| Policy numbers as params, per cycle: `resolve` writes the overlays in force on the cycle date into the params document | [overlays.py](campaign_trees/overlays.py), [scripts/generate_configs.py](scripts/generate_configs.py) |
| The cap reduction is an `adjustment_stack` document (an `AdjustmentStack` scoped by campaign and tier), so a release diff sees its effect, scope and window | [`cap_reductions.json`](configs/0.1.0/cap_reductions.json), [overlays.py](campaign_trees/overlays.py), [tests/test_hidden_state.py](tests/test_hidden_state.py) |
| The rate card and the credit life rates are `param_table`s read through `loan_granting.rate_tables`, so no step closes over an object | [preassessment.py](campaign_trees/preassessment.py), [tests/test_hidden_state.py](tests/test_hidden_state.py) |
| `@allow_fallback` for steps that are Python on purpose, checked by `Engine(strict_compile=True)` | [tests/test_serving.py](tests/test_serving.py) |
| Warm-up from `sample_request.json`; the JSON date string arrives as a `date` | [tests/test_serving.py](tests/test_serving.py) |

## Run it

From the repository root, put every example on the path once (this one imports `credit_core` and `loan_granting`):

```bash
export PYTHONPATH=$(printf '%s:' "$PWD"/examples/*/)
cd examples/campaign-trees
export DECIKIT_API__CODE_PATH="$PWD" DECIKIT_API__PIPELINE="campaign_trees.pipeline:build" DECIKIT_CONFIG__BASEPATH="$PWD/configs"
```

Build (validates `configs/0.1.0`, compiles and warms every kernel with `sample_request.json`; the first build in a
fresh checkout takes a minute while numba compiles):

```bash
uv run decikit build
# built config version 0.1.0 (pipeline campaign_trees.pipeline:build, mode fused)
```

Serve, and decide the sample client. Its date is a plain JSON string:

```bash
uv run decikit serve --workers 1 &
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{assignment_outcome_code, advertised_amount, tree_path, offer_tier_code, reason_label, pre_assessed_amount, overlay_stack_id}'
```

```json
{"assignment_outcome_code":1,"advertised_amount":20800.0,"tree_path":"n1>n2>n4>n6>n8>n10>leaf_912","offer_tier_code":2,"reason_label":9103,"pre_assessed_amount":20800.0,"overlay_stack_id":"OS-ADJ-C23-VOL-001"}
```

The client failed node 2 (settlement ratio over 0.65), passed node 4, was let through node 6 by the discretionary-income
floor, cleared the R15 000 minimum at node 8, and node 10 found them responsive to SMS: leaf 912, tier B, SMS then
in-app. They are offered the R20 800 they can afford, under the tier's R120 000 ceiling. The response also carries every
input and intermediate value.

The volume dial in force raised node 6's floor from R2 200 to R2 600. A client with R2 400 of discretionary income
would have passed it unadjusted, and now does not. The tree without the overlay is on the record beside the one with it:

```bash
jq '.discretionary_income=2400' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{assignment_outcome_code, tree_path, unadjusted_tree_path, reason_label}'
# outcome 0, tree_path n1>n2>n4>n6>leaf_903, unadjusted_tree_path n1>n2>n4>n6>n8>n10>leaf_912
```

An absolutely suppressed client is not evaluated at all; one who merely cannot be contacted still walks the tree:

```bash
jq '.is_deceased=true' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{assignment_outcome_code, evaluated, suppression_codes, tree_path, advertised_amount}'
# outcome 0, evaluated false, ["S01"], tree_path null, 0.0

jq '.contacts_30d=4' sample_request.json |
  curl -s -d @- -H 'content-type: application/json' localhost:8080/invocations |
  jq -c '{assignment_outcome_code, evaluated, suppression_codes, tree_path, advertised_amount}'
# outcome 3, evaluated true, ["S30"], the full path to leaf_912, 0.0
```

Tests (run from the repository root, with `PYTHONPATH` set as above):

```bash
uv run pytest examples/campaign-trees -q
```

## The documents

`build(campaign_23, rate_card_flex_loan, cap_reductions)` receives one document per argument, each a file in `configs/<version>/`.
[`cap_reductions.json`](configs/0.1.0/cap_reductions.json) is what the credit policy owner edits: the overlays on the tier
ceiling, in stack order. The governance fields (`owner`, `approval_reference`, `rationale`, `review_date`) are for people
and the ageing report and do not count as behaviour in a diff; everything else does. Loading rejects a duplicate id and a
`tighten_only` overlay whose effect loosens.

```json
{"type": "adjustment_stack", "name": "cap_reduction", "target": "tier_ceiling", "set_id": "AS-CAMPAIGN23",
 "adjusted_output": "advertised_tier_ceiling", "unadjusted_output": "tier_ceiling_before_cap_overlay",
 "scope_types": {"campaign_id": "int", "offer_tier_code": "int"},
 "adjustments": [
   {"adjustment_id": "ADJ-C23-CAP-001", "kind": "cap_adjustment", "target": "tier_ceiling",
    "effect": {"op": "multiply", "value": 0.8}, "scope": {"campaign_id": 23, "offer_tier_code": 1}, "stack_position": 1,
    "owner": "Credit Risk Policy", "approval_reference": "CRC-2026-041",
    "rationale": "Tier A funding constrained for the October cycle; 20% off while reviewed",
    "effective_from": "2026-10-01", "effective_to": "2026-11-01", "review_date": "2026-10-25", "tighten_only": true}]}
```

The pre-assessment prices candidate loans from two tables that are table params of its step: `flex_loan_rate_card` (the rows of
`rate_card_flex_loan.json`, in the columns of `loan_granting.rate_tables.RATE_CARD_COLUMNS`) and `credit_life_rates` (the rows of
`credit_core`'s credit life table), each with a version param beside it. Their defaults are those rows, so `params.json` does not
repeat them; a params document may replace either. The lookups are `loan_granting`'s `RateCardTable` and `CreditLifeTable`, which
index a table once per params bundle. The step name and every params path that existed before are unchanged. A diff of two builds
compares the 63 360 rate card rows column by column, which takes about a tenth of a second.

## Edit a tree

The tree is [`configs/0.1.0/campaign_23.json`](configs/0.1.0/campaign_23.json). It has three parts, all plain JSON:
`nodes` (what each node tests), `edges` (which node each branch leads to: `sourceIndex` 0 is "the test holds", 1 is
"it does not"), and `output` (the table of leaf answers; a leaf's `result_idx` is a row of it, so `leaf_912` is
row 6 of `output.data`). Node 6 looks like this:

```json
{"id": "n6", "data": {"type": "composite", "op": "and", "conditions": [
  {"op": ">=", "feature": "discretionary_income", "threshold": {"param": "n6_discretionary_income_floor", "default": 2200}},
  {"op": "<=", "feature": "estimated_instalment_to_income", "threshold": 0.33},
  {"op": "isin", "feature": "employment_type_code", "values": [1, 2, 4]}]}}
```

Config versions are immutable and a build picks the latest, so every change below starts with
`cp -r configs/0.1.0 configs/0.2.0`, and ends with `uv run decikit build`.

**Change a published threshold** (the analysts re-fitted the tree). Edit the number, here `default` 2200 to 2500, or a
plain `0.33`. The node keeps its id, so its volume history continues; ask which nodes a new version touched:

```python
import json
from decikit import TreeConfig
from campaign_trees.trees import changed_nodes, problems

old, new = (TreeConfig.load(json.load(open(f"configs/{v}/campaign_23.json"))) for v in ("0.1.0", "0.2.0"))
changed_nodes(old, new)   # {'added': [], 'removed': [], 'changed': ['n6']}
problems(new)             # []
```

**Add a branch.** Give the new nodes ids nobody has used (`n11`, `leaf_914`), add a row to `output.data` for the new
leaf, and point an edge at them. Only the nodes you added or touched start a new history. `problems` lists a node
nothing reaches and a branch that leads nowhere, which a campaign tree must never have.

**A typo is caught at build,** not in production. Misspelling `pre_assessed_amount` in node 8 stops the build with
`input 'pre_assesed_amount' is not produced by any earlier step. Did you mean 'pre_assessed_amount'`.

**Move a threshold for a while, without a new tree.** That is an overlay, not an edit. Add a `TreeOverlay` to
`TREE_OVERLAYS` in [overlays.py](campaign_trees/overlays.py): which param, the published and the new value, owner,
approval reference, rationale, an effective window and a review date. An overlay can only tighten, and one past its
review date stops the cycle. Then write the params document for the cycle:

```bash
uv run python scripts/generate_configs.py 2026-10-05 0.2.0    # the overlays in force on that date
```

It resolves overlays by the cycle date given, never by today, so a re-run of an old cycle uses that cycle's overlays,
and it records the stack in `overlay_stack_id`. The October run adds the cut-off shift on node 3's behaviour-score
floor beside the volume dial: `OS-ADJ-C23-VOL-001+ADJ-C23-CUT-001`.

## Layout

```
campaign_trees/         the package
  pipeline.py           build(campaign_23, rate_card_flex_loan, cap_reductions): eligibility, suppressions, the evaluation, the groups, the outcome
  suppressions.py       the suppression rules, and which of them are absolute
  preassessment.py      the pre-assessed amount: affordable instalment, the rate card and credit life rates as table params, and the granting example's search
  overlays.py           tree overlays resolved by cycle date
  trees.py              problems(tree) and changed_nodes(old, new)
  holdout.py            control and holdout groups from a stable hash
  arbitration.py        which qualifying campaigns become a contact, over a batch
  vocab.py              the assignment outcome codes
configs/0.1.0/         campaign_23.json (hand-maintained); params.json, the rate card and cap_reductions.json (scripts/generate_configs.py)
scripts/                generate_configs.py
sample_request.json
tests/
```

## What it publishes

Nothing else imports this example. Its reusable parts are `campaign_trees.trees` (`problems`, `changed_nodes`, which take
any `TreeConfig`), `campaign_trees.overlays.resolve` and `campaign_trees.arbitration.arbitrate`.

## Things worth knowing

- **One campaign is served.** A bank runs 30 to 60; this pipeline is campaign 23's. A second campaign is a second tree
  document, a second `build` argument and its own overlays; [tests/test_scale.py](tests/test_scale.py) shows that a tree
  of real size (hundreds of nodes) loads, compiles and traces. Campaigns do not see each other until arbitration.
- **"Not evaluated" is null.** A client kept out of the evaluation has null `tree_path`, `pre_assessed_amount` and the other
  evaluation fields, rather than made-up zeros. The outcome is still `0` and the amount `0.0`.
- **Overlays have two clocks.** Threshold overlays are resolved once per cycle into the params document (a params document is
  per run, not per client). The cap reduction is resolved per client from `cycle_date`, from the overlays in [`cap_reductions.json`](configs/0.1.0/cap_reductions.json).
- **The params dict is read once.** Pass a new dict after editing one; changing it in place after scoring with it is
  not noticed.
- **Tree params are named in one place.** The thresholds an overlay may move are the `{"param": ...}` names in the tree
  document; `overlays.resolve` refuses an overlay naming one the tree does not have.
- **Arbitration is not a step.** It needs to see every client at once (channel capacity, the per-client cap), which a
  one-client pipeline cannot, so it is a function over a batch with its own tests.
- **The affordable instalment is an approximation.** It is 30% of the mart's discretionary-income estimate. The
  affordability example's assessment is the real one; its result would replace `max_affordable_instalment` in
  [preassessment.py](campaign_trees/preassessment.py).
- **Left out.** Most of the suppression registry (nine of about thirty-four rules are here), contradiction checks between
  a node and its ancestors, champion/challenger splits, the node-volume reporting and path store the analytics team
  builds on `tree_path`, fairness reporting in arbitration, and the check that a pre-assessment is still valid when
  an offer is shown (the dates are on the record).
- **Synthetic data.** Every threshold, tier and amount here is invented for illustration.

## Reuse

Built on [`shared-credit-core`](../shared-credit-core/README.md) (`credit_core`): `adjustments`, `appetite`, `consent`,
`eligibility`, `rate_card` (the script writes its synthetic card) and `vocab`; and on
[`loan-granting-pricing`](../loan-granting-pricing/README.md) (`loan_granting`): `solve.solve_term` and
`pricing.PriceEvaluator` for the pre-assessed amount. It does not reuse `affordability` (see above).
