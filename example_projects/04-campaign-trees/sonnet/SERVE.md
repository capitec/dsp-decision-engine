# Serving campaign 23 (Flex Loan pre-approved top-up)

This project serves one campaign end to end -- spec 04 §5.3.3's own worked example -- not
all sixty (see NOTES.md "What I built"). `generate_trees.py` and `tests/` prove the
framework holds at the spec's real scale (60 trees, ~9 200 nodes) without wiring all of them
into one served pipeline.

## Setup

```bash
cd example_projects/04-campaign-trees/sonnet   # this directory
export DECIDER_API__CODE_PATH="$PWD"
export DECIDER_API__PIPELINE="pipeline:build"
export DECIDER_CONFIG__BASEPATH="$PWD/configs"
export DECIDER_API__MODE=fused
export PYTHONPATH="<REPO>/example_projects/00-shared-credit-core/sonnet:<REPO>/example_projects/03-loan-granting-pricing/sonnet"
```

Both `credit_core` (project 00) and `loan_granting` (project 03) are consumed read-only via
`PYTHONPATH`; nothing here imports from anywhere but those two and `decider` itself.

## Build

```bash
uv run --project <REPO> decider build
```

Expect: `built config version 0.1.0 (pipeline pipeline:build, mode fused)`.

If `configs/0.1.0/` is ever regenerated -- the rate card (reused unmodified from
`credit_core`, exactly as 00/03 generate it) and `params.json` (which bakes in the overlay
stack resolved for one chosen `cycle_date`, §5.3.4 requirement 7 -- see `pipeline.py`'s
`overlay_stack_id_param` docstring for why that resolution happens here and not per record):

```bash
uv run --project <REPO> python generate_configs.py 0.1.0 configs 2026-09-24
```

## Score the sample request

```bash
uv run --project <REPO> decider serve &
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations | python -m json.tool
kill %1
```

Or, without a server, exactly as `tests/test_pipeline.py` does it:

```python
import datetime, json
from decider import Engine
from decider.steps.tables import DecisionTableConfig
import pipeline

rate_card = DecisionTableConfig.load("configs/0.1.0/rate_card_flex_loan.json")
params = json.load(open("configs/0.1.0/params.json"))
exe = Engine().bind(pipeline.build(rate_card), mode="interpreted")

record = json.load(open("sample_request.json"))
record["cycle_date"] = datetime.date.fromisoformat(record["cycle_date"])

out = exe.score(record, params=params)
print(out["leaf"], out["offer_tier_code"], out["reason_label"], out["advertised_amount"])
# -> 4e037203530de67b 2 9103 20800.0 -- tier B, "Standard top-up, SMS responsive",
# reproducing spec 04 §5.4.1(d)'s worked example leaf (912) exactly.
```

`inference.py`'s `Handler` is what `decider serve`/`decider build` actually use (it replaces
the framework's synthetic warm-up record with `sample_request.json` -- see NOTES.md
"Framework friction" #1, the same workaround 00 and 03 document); the snippet above binds
the pipeline directly, which is what the CLI does internally too, minus the warm-up
substitution.

## Mode

Served in `fused` mode: on `sample_request.json` its output equals
`interpreted` mode's exactly. Steps no kernel can run faithfully (for example
`credit_core.expense_norms.norm_table_version`, which compares two `str`
inputs) run in Python, row by row, with a warning naming each at build time.

## Run the tests

```bash
PYTHONPATH="<REPO>/example_projects/00-shared-credit-core/sonnet:<REPO>/example_projects/03-loan-granting-pricing/sonnet" \
  uv run --project <REPO> pytest example_projects/04-campaign-trees/sonnet/tests -q
```

`tests/conftest.py` adds this project's own directory to `sys.path`; `credit_core` and
`loan_granting` still need the `PYTHONPATH` entries above. 48 tests, well under a minute.
