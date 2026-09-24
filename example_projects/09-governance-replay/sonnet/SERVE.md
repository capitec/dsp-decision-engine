# Serving the governance and replay harness

This project reads evidence produced by three flows (01, 03, 05) and, for 03
and 05, transitively needs 00 and 02 on `PYTHONPATH` (03 calls 02's
affordability assessment; 05 calls it too, for sole proprietors).

## Setup

```bash
cd example_projects/09-governance-replay/sonnet   # this directory, in the repo
export PYTHONPATH="$PWD/../../00-shared-credit-core/sonnet:$PWD/../../01-transaction-fraud/sonnet:$PWD/../../02-affordability/sonnet:$PWD/../../03-loan-granting-pricing/sonnet:$PWD/../../05-business-nested/sonnet"
export DECIDER_API__CODE_PATH="$PWD"
export DECIDER_API__PIPELINE="pipeline:build"
export DECIDER_CONFIG__BASEPATH="$PWD/configs"
export DECIDER_API__MODE=interpreted
```

**Never `import pipeline` for another flow.** This project, and every flow it
reads from, ships a top-level `pipeline.py` (the BRIEF's own required
filename). `governance/paths.py::load_pipeline` loads each flow's
`pipeline.py` under a distinct module name via `importlib` -- the same,
already-proven workaround 03's and 05's own NOTES.md document for the
identical collision. `DECIDER_API__CODE_PATH` still needs to win for *this*
project's own `pipeline.py` (`decider`'s own `sys.path.insert(0, code_path)`
at serve/build time gets this right automatically).

## Populate the evidence store

The harness has nothing to replay, explain or measure until some evidence
exists. `evidence_store/` is not checked in as generated data belongs to a
run, not a commit -- populate it once per checkout:

```bash
uv run --project <REPO> python capture_demo_evidence.py
```

This captures each of 01/03/05's own `sample_request.json`, plus a small
deterministic (not random -- 09 §5.15 item 3) population of variants per
flow, under `evidence_store/<flow_code>/<decision_id>.json`. `sample_request.json`
in this directory names one of them (`03`, `gr-2026-0000987654`) -- **run
this script before `decider build`**, or the warm-up (`inference.py`) and
the sample-request scoring below will both fail with a `FileNotFoundError`
for that decision.

## Build

```bash
uv run --project <REPO> decider build
```

Expect: `built config version 0.1.0 (pipeline pipeline:build, mode interpreted)`.

## Score the sample request

```bash
uv run --project <REPO> decider serve &
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations | python -m json.tool
kill %1
```

Or, without a server, exactly as `tests/test_pipeline.py` does it:

```python
from decider.serving.handler import construct_handler_from_settings
handler = construct_handler_from_settings()
handler.stage(); handler.activate()
live = handler.module_fn()
import json
record = json.load(open("sample_request.json"))
live.executable.score(record, live.params)
```

For the sample request (flow 03's own captured decision), expect
`replay_verdict: "reproduced"`, `divergence_count: 0`,
`diverged_fields: ""` -- the decision was captured with the same config
version and params this build uses, so it must replay bit-identically or
this project's own claim to "exact replay" is false.

## Why interpreted mode

`run_replay` calls into 01/03/05's own pipelines internally (via
`governance.replay`), which SERVE.md for 01 and 03 documents as needing
`interpreted` mode themselves (`str`-column comparisons compiled mode
rejects). This project's own single step has no such restriction, but it
would be misleading to build it in `fused` mode while the flows it drives
internally cannot be, so this project matches them.

## Run the tests

```bash
uv run --project <REPO> pytest example_projects/09-governance-replay/sonnet/tests -q
```

`tests/conftest.py` puts this project and every flow it consumes on
`sys.path` in the right order and captures a fresh `evidence_store/` into a
temporary directory per test session, so the tests do not depend on
`capture_demo_evidence.py` having been run first, and do not pollute this
directory's own `evidence_store/`.
