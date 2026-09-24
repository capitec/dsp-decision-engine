# Serving retail_credit's entry-point-1 demo

`retail_credit` is project 10 (spec 10), built standalone: the only thing it
consumes from outside itself is project 00's `credit_core` library (spec 10
§1.1). This document serves the one path this slice flesh out end to end --
entry point 1 (new credit application), product 10 only -- so `decider
build`/`decider serve` and the sample request can be verified.

## Setup

```bash
cd example_projects/10-retail-credit-e2e/sonnet   # this directory
export DECIDER_API__CODE_PATH="$PWD"
export DECIDER_API__PIPELINE="pipeline:build"
export DECIDER_CONFIG__BASEPATH="$PWD/configs"
export DECIDER_API__MODE=interpreted   # see "Why interpreted mode" below

# PYTHONPATH: this project's own directory FIRST, then 00's, in that order.
export PYTHONPATH="$PWD:$PWD/../../00-shared-credit-core/sonnet"
```

**The PYTHONPATH order matters and is not optional.** Every example project in
this set has a top-level `pipeline.py` (`decider template`'s own convention).
Put project 00's directory *ahead* of this one on `PYTHONPATH` and `import
pipeline` resolves to **project 00's** demo pipeline instead of this one --
`decider build` then fails with `KeyError: "the config version has no
'rate_card_flex_loan' document for argument 'rate_card_flex_loan' of
'pipeline:build'"` (00's `build()` takes an argument this project's `build()`
does not), which is a confusing error for what is actually a `sys.path`
ordering mistake. See NOTES.md "Framework friction" for the full writeup --
this is not specific to project 00 and 10; it will happen to **any** two
example projects served from the same environment.

## Build

```bash
uv run --project <REPO> decider build
```

Expect: `built config version 0.1.0 (pipeline pipeline:build, mode
interpreted)`. The first build takes a few seconds to generate and validate
this project's own 34 560-cell Flex Loan rate card (`retail_credit/pricing.py`,
generated at import time, not loaded from a config document -- see NOTES.md
"How I organised a large project" for why this project keeps its tables in
code rather than externalising them the way project 00 does its one rate
card).

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

The sample request is an existing client (segment 4), channel 3, requesting
R95 000 over 60 months on product 10. It scores `outcome_code` 1 (approve),
with an offer at R76 250 (BIND-AFF, capacity-bound after the channel-3 rate
add-on and buffer overlay both apply) -- below the requested amount, because
the amount cap (bound by CAP-0301, group exposure headroom) and the
affordability-bound solve both narrow it. Every field of `pipeline.py`'s
`.emit(...)` list is present in the record; `tests/` exercises the decline,
refer, degraded-mode and consolidation-loop paths against the same fixture
with targeted overrides.

## Why interpreted mode

Same reason as project 00 (see its NOTES.md): `expense_norms.norm_table_version`
(`credit_core.expense_norms`, reused unchanged) selects between two `str`
inputs at runtime, which compiled (`fused`/`stepped`) mode rejects at bind
time. This project's own tables add a second instance of the same shape
(`grading.probability_of_default_adjustment_step`'s overlay resolution
compares `str` scope keys), so interpreted mode is required throughout, not
only for the one inherited table.

## Run the tests

```bash
uv run --project <REPO> pytest example_projects/10-retail-credit-e2e/sonnet/tests -q
```

`tests/conftest.py` adds this project's own directory *and* project 00's to
`sys.path`, in that order (see "PYTHONPATH order" above) -- no manual
`PYTHONPATH` needed beyond what `pytest`'s own working directory gives it.
Runtime is dominated by `test_pipeline.py`'s `decider build` CLI test and the
rate-card generation (~2 minutes total).
