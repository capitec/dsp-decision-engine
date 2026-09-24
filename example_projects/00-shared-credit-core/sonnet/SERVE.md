# Serving credit-core's demo pipeline

`credit-core` is a library, not a product flow -- see NOTES.md "What I
publish" for how another project imports its capabilities directly. This
document serves the one demo pipeline in this directory (`pipeline.py`), a
Flex Loan (product 10) affordability + pricing + risk quote, so
`decider build`/`decider serve` and the sample request can be verified.

## Setup

```bash
cd example_projects/00-shared-credit-core/sonnet   # this directory
export DECIDER_API__CODE_PATH="$PWD"
export DECIDER_API__PIPELINE="pipeline:build"
export DECIDER_CONFIG__BASEPATH="$PWD/configs"
export DECIDER_API__MODE=fused
```

No other project's `PYTHONPATH` entry is needed: this project consumes
nothing (spec 00 is wave 0, the root of the dependency graph).

## Build

```bash
uv run --project <REPO> decider build
```

Expect: `built config version 0.1.0 (pipeline pipeline:build, mode fused)`.
The first build takes a few seconds to load and validate the Flex Loan
rate card's 63 360 rows; `decider build` stages and warms it once so a
live request never pays that cost.

If `configs/0.1.0/rate_card_flex_loan.json` is ever regenerated (a real
Treasury refresh, in production, would replace its `rows` directly --
00 §9 "no code deployment"), rebuild it here with:

```bash
uv run --project <REPO> python generate_configs.py 0.1.0
```

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
for acc in record["bureau_accounts"] + record["internal_accounts"]:
    acc["opened_date"] = datetime.date.fromisoformat(acc["opened_date"])
live.executable.score(record, live.params)
```

## Mode

Served in `fused` mode: on `sample_request.json` its output equals
`interpreted` mode's exactly. Steps no kernel can run faithfully (for example
`credit_core.expense_norms.norm_table_version`, which compares two `str`
inputs) run in Python, row by row, with a warning naming each at build time.

## Run the tests

```bash
uv run --project <REPO> pytest example_projects/00-shared-credit-core/sonnet/tests -q
```

No `PYTHONPATH` needed beyond this project's own directory; `tests/conftest.py`
adds it. Runtime is dominated by `test_pipeline.py`'s `decider build` CLI
test and the rate-card fixtures (~2 minutes total, mostly building and
validating the 63 360-row table twice).
