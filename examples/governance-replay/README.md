# Decision governance and replay

A harness that sits beside the bank's decision flows and answers the questions asked after a decision was made:
*can you reproduce it, why did it come out that way, what changes if we change the policy, which overlays are still
in force, which rules never fire, and is this release safe to ship?* It decides nothing itself. It runs against three
other examples (transaction fraud, loan granting and business credit), which stand in for the bank's eight flows.

## The business problem

A credit decision is questioned months or years later by an ombud, an auditor, a client's lawyer or the Credit
Committee. "The system said no" is not an answer. The bank must be able to:

- **Replay** a decision from what was kept, with no live call, and say it comes out the same. If it does not, say
  which field differs first. If the policy was edited after the decision, say so rather than replay the wrong logic.
- **Explain** it differently for a consultant (the deciding reason, nothing internal), an analyst (every reason and how
  the flow decided) and an adjudicator (plain sentences with the policy version).
- **Compare** two versions of a flow: what changed in the logic, the params and each table, and who would have been
  decided differently (the *swap set*). A release usually holds several changes, so each needs its own swap set.
- **Govern the overlays**: one list of every approved overlay across the flows, which are overdue for review, and what
  the decisions would have been with all of them switched off.
- **Find dead logic**: rules that never bound, and rules that bind on so much of the traffic that they are policy in
  disguise.
- **Certify a release**: run a golden set of earlier decisions through the candidate, compare the movement with what the
  release declared, and block on any difference.

## What it provides

| Capability | Entry | Returns |
|---|---|---|
| Capture | [`evidence.capture(flow, request)`](governance/evidence.py) | `Evidence`: the request as received, the response, the config version, the digest of every config document, the params in force |
| Replay | [`replay.replay(evidence)`](governance/replay.py) | `Verdict`: `reproduced`, `reproduced_within_tolerance`, `not_reproduced` or `not_reproducible`, every divergence in execution order, the step that wrote each |
| Explain | [`explain.explain(evidence, audience)`](governance/explain.py) | `Explanation`: text and detail for one audience |
| Version diff | [`versions.version_diff(flow, old, new)`](governance/versions.py) | The decider toolkit's step diff and proof, the changed param leaves, a summary of each changed document |
| Swap set | [`swapset.swap_set`, `swapset.attribute`](governance/swapset.py) | who moved outcome, reasons appearing and disappearing, records that could not be compared; `attribute` gives n swap sets for a release of n changes |
| Overlays | [`overlays.register`, `ageing`, `unwind_estimate`](governance/overlays.py) | the combined register; overdue overlays by age; decisions that change with the overlays off |
| Coverage | [`coverage.coverage(flow, responses)`](governance/coverage.py) | hits per rule, dead and dominant rules, ids outside the rule list, static findings, cells read |
| Certification | [`certify.certify(golden, candidate, expected_to_move)`](governance/certify.py) | `Certificate`: what moved, what moved unexpectedly, what was expected to move and did not, the proof, the changed documents, approvals |
| Replay service | [`governance.pipeline:build`](governance/pipeline.py) | `replay_verdict`, `divergence_count`, `diverged_fields`, `first_divergence` for a flow and a decision id |

Only the replay service is a decider toolkit pipeline; the rest is a library you call from a notebook, a script or a CI job
(see [`scripts/tour.py`](scripts/tour.py)). Their results are nested documents with no business reason to be a flow,
and a CI job wants a Python object, not an HTTP call.

## Which decider toolkit features it uses, and what is its own

| Decider toolkit feature | Where it is used |
|---|---|
| `RequestHandler` and `JsonFileStore`: stage one immutable config version and score raw JSON request bytes exactly as `decikit serve` does, so dates, defaults and output formatting match production | [replayer.py](governance/replayer.py), [flows.py](governance/flows.py) |
| The handler's `module_fn` hook, to score the same version with other params (a what-if, the overlays off) | [replayer.py](governance/replayer.py), [overlays.py](governance/overlays.py) |
| Config versions as the identity of the logic (`store.latest_version()`, whatever it is for each flow), plus a digest of every document so an in-place edit is detected | [evidence.py](governance/evidence.py), [replay.py](governance/replay.py) |
| Debug `Session` (`executable.session(...)`) to order the divergences by the step that first wrote each | [replay.py](governance/replay.py) |
| `decikit.lab.diff.structure` and `prove` for the change record of a release | [versions.py](governance/versions.py) |
| `decikit.lab.checks.run` with `unreachable_branch` for logic that cannot run whatever the traffic | [coverage.py](governance/coverage.py) |
| `assert_equivalent`: the replay service gives the same verdicts in every mode and for a single record | [tests/test_pipeline.py](tests/test_pipeline.py) |
| `step(outputs=)`, `param`, `@allow_fallback`, warm-up from `sample_request.json` for the replay service | [pipeline.py](governance/pipeline.py) |
| From `credit_core`: the `Overlay` rows of the `adjustment_stack` documents (`due_for_review`) and the `reason_codes` document (`ReasonCodes`); `rate_card.diff_cards` for a rate-card refresh | [overlays.py](governance/overlays.py), [versions.py](governance/versions.py) |

What is the harness's own, because the decider toolkit has no equivalent:

- **The evidence contract.** What a decision must keep to be replayable (request as received, response, version,
  document digests, params) and a write-once store for it ([evidence.py](governance/evidence.py)).
- **Tolerance grading.** Exact for codes, ids, booleans and lists; to the cent for money; four places for rates; 1e-6
  for scores. `assert_equivalent` is exact only, which is right between modes and wrong between a decision made last
  year and a replay today ([replay.py](governance/replay.py)).
- **Swap sets and their attribution**, as outcome movement and reason movement, with n+1 runs for n changes
  ([swapset.py](governance/swapset.py)).
- **Certification**: the golden-set comparison against the declared expected movement, the three failure modes, the
  no-effect rule and the sign-off record ([certify.py](governance/certify.py)).
- **The overlay register view and ageing** (read from the flows' config documents), **rule hit counting** against each flow's rule list, and the
  three-audience **explanation**.

Not used, and why: `decikit.lab.diff.outputs` and `decikit.lab.experiments` run a typed *frame* through the flow, and
only the handler converts a request's JSON date strings. A frame of raw business requests keeps the dates inside the
`entities` list as strings and fails there (`'str' object has no attribute 'year'`). The harness therefore scores one
request at a time through the handler, which also keeps each result tied to a decision id.

## Run it

From the repository root, put every example on the path once (the harness imports the three flows by package name):

```bash
export PYTHONPATH=$(printf '%s:' "$PWD"/examples/*/)
cd examples/governance-replay
export DECIKIT_API__CODE_PATH="$PWD" DECIKIT_API__PIPELINE="governance.pipeline:build" DECIKIT_CONFIG__BASEPATH="$PWD/configs"
```

Build (validates this example's own `configs/0.1.0` and warms the replay service with `sample_request.json`, which replays a loan
decision, so the first build in a fresh checkout compiles the loan flow too):

```bash
uv run decikit build
# built config version 0.1.0 (pipeline governance.pipeline:build, engine fused)
```

Serve, and replay the loan decision in [`evidence/loan/`](evidence/loan). The first request for each flow stages that
flow (the fraud flow takes about a minute the first time numba compiles it):

```bash
uv run decikit serve --workers 1 &
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations
```

```json
{"flow_code":"loan","decision_id":"d5f5f6b0-3f8a-4f2a-9c4b-2f6e7a1b0003","replay_verdict":"reproduced","divergence_count":0,"diverged_fields":[],"first_divergence":""}
```

The same for the fraud flow:

```bash
curl -s -d '{"flow_code":"fraud","decision_id":"9f2b6f0e-2a5b-4b8a-9b0c-111111111111"}' \
  -H 'content-type: application/json' localhost:8080/invocations
# {"flow_code":"fraud","decision_id":"9f2b6f0e-...-111111111111","replay_verdict":"reproduced","divergence_count":0,"diverged_fields":[],"first_divergence":""}
```

and the business flow, whose documents are config version 0.2.0 of its own store:

```bash
curl -s -d '{"flow_code":"business","decision_id":"biz-2026-0900123"}' -H 'content-type: application/json' localhost:8080/invocations
# {"flow_code":"business","decision_id":"biz-2026-0900123","replay_verdict":"reproduced","divergence_count":0,"diverged_fields":[],"first_divergence":""}
```

A decision with no evidence is an error, not a verdict: the service answers with the missing file, never a guess.

Tests (from the repository root, with `PYTHONPATH` set as above):

```bash
uv run pytest examples/governance-replay -q
```

## The library, on the shipped evidence

[`evidence/`](evidence) holds 22 decisions, written by [`scripts/capture_demo_evidence.py`](scripts/capture_demo_evidence.py):
the sample request of each flow and variants that decide differently (a fraud allow, hold, step-up, decline and
block; loan approvals and declines; business approvals, a decline and a referral). `uv run python scripts/tour.py`
runs everything below:

```text
replay: reproduced
replay, request edited: ['requested_amount', 'solve_total_evaluations', 'term_solutions', 'has_any_offer']
explain consultant: An entity behind the business carries a disqualifying adverse record
overlays due on 2026-10-01: [('fraud', 'ADJ-2026-003', 0)]
unwind fraud: {'decisions': 9, 'changed': 1, 'moves': {'9f2b6f0e-...-111111111111-v2': (40, 30)}}
swap set: overlays off 0 moved {}
swap set: refer from grade 6 2 moved {5603: 5}
loan rules never bound: 9 of 12 ; dominant: ('CAP-0100', 'CAP-0210', 'CAP-0420')
certificate: True 1 moved, ('rate_card_flex_loan',)
```

Read it as the business would:

- **Replay.** The loan decision reproduces. Edit the requested amount in the evidence and it does not, and the first
  divergence is the edited input, then `solve_total_evaluations` and `term_solutions`, in the order the flow wrote
  them, not alphabetical order.
- **Overlays.** Of the fraud flow's nine decisions, one is only a hold (40) because of the festive overlay; with the
  overlays off it is a step-up (30).
- **Attribution.** Two changes make two swap sets. Switching the overlays off moves nobody in the business population;
  lowering the referral cut-off to grade 6 on top of that moves two approvals to referral and adds reason 5603 to five
  decisions. A single swap set for the release would not say which change did it.
- **Certification.** Raising grade 12 rates by one point is declared to move the grade-12 decisions. Exactly one moves,
  so the release passes; declare nothing and it is blocked with that decision listed as moved unexpectedly.

Replay a decision, or build a candidate version and compare it, in your own code:

```python
import tempfile

from governance.evidence import EvidenceStore
from governance.flows import LOAN
from governance.replay import replay
from governance.swapset import swap_set

store = EvidenceStore("evidence")
replay(store.load("loan", "d5f5f6b0-3f8a-4f2a-9c4b-2f6e7a1b0003")).verdict   # "reproduced"

def cheaper_grade_9(config):                       # the candidate: the latest config, edited, in a scratch store
    for row in config["rate_card_flex_loan"]["rows"]:
        if row["grade"] == 9:
            row["rate"] -= 0.5

requests = [e.request for e in store.all("loan")]
candidate = LOAN.candidate(cheaper_grade_9, tempfile.mkdtemp())
swap_set(LOAN, requests, LOAN.replayer(), candidate).moved   # (): a cheaper grade 9 changes prices, not outcomes
```

## Layout

```
governance/
  flows.py        the Flow of each decision flow: entry, outcome and reason fields, its overlay documents, rule lists; reads a version's documents
  replayer.py     RequestHandler subclass: `score` (typed) and `served` (JSON-safe response dict), optional params, the typed record kept
  evidence.py     Evidence, capture, EvidenceStore (write once)
  replay.py       tolerance grading and replay, divergences in execution order
  explain.py      three audiences
  versions.py     version_diff
  swapset.py      swap_set, attribute
  overlays.py     register, ageing, switched_off, unwind_estimate
  coverage.py     rule hits, dead and dominant rules, static findings, cells read
  certify.py      certify, Certificate
  pipeline.py     build(): the replay service
configs/0.1.0/    params.json
evidence/         22 captured decisions, by flow
scripts/          capture_demo_evidence.py, tour.py
sample_request.json
tests/
```

## Things worth knowing

- **Replay refuses rather than guesses.** A decision stores the digest of every config document of its version. If
  the rate card, rule set or params in `configs/<version>` have been edited since, `replay` answers
  `not_reproducible` and names the documents; it does not run the new logic and call it the old decision. Config
  versions are meant to be immutable: write a new one instead of editing.
- **What replay compares.** Every field of the response, as JSON, including the inputs echoed back. A field that
  differs because the *request* was altered comes first, since no step wrote it.
- **Swap sets are per record.** `swap_set` and `attribute` score each request through the handler, so one request that
  fails is listed under `uncomparable` and the rest are still compared.
- **A candidate is a version.** `Flow.candidate(edit, root)` copies the latest config, applies `edit`, writes it as a
  new version under `root` and stages it, so a rate-card refresh, a rule change and a params change are all the same
  kind of thing and attribution does not care which. A params-only run is `Flow.replayer(params=...)`.
- **What the decider toolkit's diff covers.** `version_diff` reports the decider toolkit's step-level `structure` and `prove`. The flows keep
  their overlays, reason codes and rule-set versions in config documents, so two pipelines built from the same
  documents are identical to the decider toolkit: `diff.structure(a, a)` reports `identical` with nothing added, removed, renamed
  or moved and no changed edges for each of the fraud, loan and business flows, and `prove` says `True`. A change in
  a document is a change the decider toolkit sees as well; the documents are also summarised here, because a 63,360-row rate card
  is better read as cells than as a step diff (rate cards cell by cell by key through
  `credit_core.rate_card.diff_cards`, rule sets by rule id), and a reordered rate card shows no change.
- **Where the overlays and reasons come from.** Nothing is imported from the flows' modules by object. `Flow.documents`
  reads the documents of a config version from the flow's own store (`configs/<version>/*.json`, the same files the
  flow is built from); `Flow.overlays` lists the `Overlay` rows of its overlay documents (`amount_overlays` and
  `action_escalation` for fraud; the four `*_adjustments` for loan; `event_threshold_overlays`,
  `entity_score_adjustments` and `business_pd_adjustments` for business), and `Flow.reason_codes` the codes of its
  `reason_codes` document, which is what `explain` reads the wording from. A new overlay document in a flow needs one
  name added to its `Flow`.
- **The evidence and the version rule.** Evidence is captured against a flow's latest config version, whatever that is
  (`business-nested` ships its documents as `0.2.0` while the others are at `0.1.0`), and keeps the digest of every
  document in that version. Replay reads `evidence.config_version` back, so a version bump needs no change here; but
  adding or editing any document of a version after capture makes its decisions `not_reproducible`. Write a new
  version instead and capture new evidence, or keep the old version directory untouched.
- **One declared Python boundary.** The replay service looks the flows up by code in `governance.flows.FLOWS`, a Python
  registry, so the decider toolkit reports it as hidden state when the service is built. The service is a harness, not a decision;
  there is no document to hold the flows. As a consequence `decikit.lab.diff.structure` of the replay service against
  itself is never identical (the registry is keyed by identity); the flows it replays diff normally.
- **Rule lists.** The fraud rule list is the live rules in `rule_set.json`; the loan list is the cap waterfall's
  `REGISTER`; the business list is the roll-up rule ids. The business flow also reports a binding rule
  `worst-event` that is in none of them; coverage lists it under `unknown`.
- **Not built.** Drift and outcome monitoring, fairness testing, the regulator pack, personal-information masking, the
  what-if tool with its own identifiers, attribution of a population or data-feed change (only logic, reference-data
  and overlay changes are attributable), expected *direction* of movement in certification (only which decisions
  move), and volumes: the demo populations are 5 to 9 decisions per flow.
- **Synthetic data.** The evidence is produced by the other examples' invented rules and tables.

## Reuse

Runs against [`transaction-fraud`](../transaction-fraud/README.md) (`fraud_interdiction`),
[`loan-granting-pricing`](../loan-granting-pricing/README.md) (`loan_granting`) and
[`business-nested`](../business-nested/README.md) (`business_nested`) by package name, and through them uses
[`shared-credit-core`](../shared-credit-core/README.md) (`credit_core`: `adjustments`, `reason_codes`, `rate_card`). It publishes
nothing for other examples to import.
