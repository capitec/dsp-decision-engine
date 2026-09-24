# Serving fraud_interdiction

This project depends on `00-shared-credit-core` for `core.reason_codes`,
`core.dates`, `core.consent`, `core.rounding` and `core.adjustments`
(DEPS.md), so its directory must be on `PYTHONPATH`.

## Setup

```bash
cd example_projects/01-transaction-fraud/sonnet   # this directory, in the repo
export PYTHONPATH="$PWD/../../00-shared-credit-core/sonnet"
export DECIDER_API__CODE_PATH="$PWD"
export DECIDER_API__PIPELINE="pipeline:build"
export DECIDER_CONFIG__BASEPATH="$PWD/configs"
export DECIDER_API__MODE=interpreted   # see "Why interpreted mode" below
```

**Path order matters.** Both this project and `00-shared-credit-core` ship a
top-level `pipeline.py` (the brief's required filename). If
`00-shared-credit-core`'s directory came first on `sys.path`, `import
pipeline` inside `decider` would resolve to *its* `pipeline.py`, not this
one -- confirmed with a minimal repro (see NOTES.md "Framework friction").
`decider` itself gets this right at serve time
(`decider/serving/handler.py` does `sys.path.insert(0, code_path)`, so
`DECIDER_API__CODE_PATH` always wins over `PYTHONPATH`), which is why the
commands below are safe as written; a script that manipulates `sys.path`
itself (see `tests/conftest.py`) has to reproduce that ordering by hand.

If `configs/0.1.0/{live_rules,shadow_rules,overlay_base_rules}.json` are
ever regenerated (a real deployment would push a new rule-set version the
same way -- no code deploy, §9's "minutes to live"):

```bash
uv run --project <REPO> python generate_configs.py 0.1.0
```

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
import json, datetime
record = json.load(open("sample_request.json"))
record["decision_date"] = datetime.date.fromisoformat(record["decision_date"])
live.executable.score(record, live.params)
```

For the sample request (a first payment to a beneficiary added 90 minutes
ago, above the mule/scam amount band, on a device changed 12 hours ago),
expect `action_code: 50` (`decline`), `action_source_rule_id: "AT-0109"`,
around 31 live rules firing (four of them, in the `MS` family, only
because the festive-period threshold-multiplier overlay is in force on
`decision_date` 2026-01-10 -- see `fired_on_overlay_ids`).

## Why interpreted mode

`fused` mode can't build this pipeline yet: compiling its largest decision
tree for the fused kernel fails with `RecursionError: maximum recursion depth
exceeded` inside decider's tree walker. Interpreted mode has no such limit.
Given this slice explicitly skips the latency target (SCOPE.md), interpreted
mode is the pragmatic choice.

## Run the tests

```bash
uv run --project <REPO> pytest example_projects/01-transaction-fraud/sonnet/tests -q
```

`tests/conftest.py` puts both this project and `00-shared-credit-core` on
`sys.path` in the right order (see "Path order matters" above); no
`PYTHONPATH` export is needed for pytest specifically. One test
(`test_decider_build_cli_succeeds`) shells out to the real `decider build`
CLI, mirroring this document exactly.
