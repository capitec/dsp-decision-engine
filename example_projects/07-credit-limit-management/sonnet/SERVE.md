# Serving credit limit management

This is the **per-account** pipeline (§5.1-§5.7, §5.9 minus notice/consent
mechanics) -- what `decider build`/`decider serve` serve one account at a
time, and what the batch/simulation callers run over the whole book. The
**population-level budget allocation** (§5.8) is a separate, plain-Python
stage (`limit_mgmt/allocation.py`) over that pipeline's *batch* output --
see NOTES.md "How I handled the portfolio-level budget" and
`simulation.py`.

## Setup

```bash
cd example_projects/07-credit-limit-management/sonnet   # this directory
export DECIDER_API__CODE_PATH="$PWD"
export DECIDER_API__PIPELINE="pipeline:build"
export DECIDER_CONFIG__BASEPATH="$PWD/configs"
export DECIDER_API__MODE=interpreted   # see "Why interpreted mode" below
export PYTHONPATH="<REPO>/example_projects/00-shared-credit-core/sonnet:<REPO>/example_projects/02-affordability/sonnet"
```

`credit_core` (project 00) and `assessment` (project 02) are consumed
read-only via `PYTHONPATH`; nothing here imports from anywhere else.

## Build

```bash
uv run --project <REPO> decider build
```

Expect: `built config version 0.1.0 (pipeline pipeline:build, mode interpreted)`.

If `configs/0.1.0/matrix.json` (the 1 152-cell limit assignment matrix)
is ever regenerated -- a real Credit Risk Policy spreadsheet refresh, in
production, would replace this file's `rows` directly (§6.3 item 1: "no
deployment") -- rebuild it here with:

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

import inference
record = inference._typed_sample_record()   # types the JSON's date fields
live.executable.score(record, live.params)
```

## Run the monthly cycle (batch + allocation)

The per-account pipeline in *batch* mode, then §5.8's allocation over the
whole result -- the two calls `simulation.py` and the (unbuilt) production
scheduler both make:

```python
import polars as pl
from decider import Engine
from limit_mgmt.matrix import build_matrix_table
from limit_mgmt.allocation import run_allocation, BudgetInstruction
import pipeline

exe = Engine().bind(pipeline.build(build_matrix_table()))
scored = exe.run(population_df, params={})               # one account per row
funded_df, cycle_summary = run_allocation(scored, BudgetInstruction())
```

## Why interpreted mode

Inherited from 00/02: `credit_core.expense_norms.norm_table_version`
(consumed unmodified through project 02) compares two `str` table-version
columns, which compiled (`fused`/`stepped`) mode rejects at bind time. See
00's SERVE.md/NOTES.md for the full writeup.

## Requests need every ragged/optional field populated explicitly

Inherited from 00 and 02's own documented gotchas, one level further in:
an entirely-absent `list`/`dict` column, an empty `[]` literal with no
non-empty sibling, and a lone `null` for an optional scalar all crash
request construction differently (`ShapeError`, `ArrowImportError`, or a
silent `Null`-dtype column decider then rejects). `sample_request.json`
follows the same conventions 02 established (placeholder applicant-2
ragged fields, real dates instead of `null`) -- see NOTES.md "Framework
friction" for the specific reproductions this project added.

## Run the tests

```bash
PYTHONPATH="<REPO>/example_projects/00-shared-credit-core/sonnet:<REPO>/example_projects/02-affordability/sonnet" \
  uv run --project <REPO> pytest example_projects/07-credit-limit-management/sonnet/tests -q
```

`tests/conftest.py` adds this project's own directory to `sys.path`;
`credit_core` and `assessment` still need the `PYTHONPATH` entry above.
`tests/test_allocation.py` runs a synthetic population through the batch
pipeline and the allocation stage together; its size is controlled by
`LIMIT_MGMT_TEST_POPULATION` (default 4 000, kept small for CI speed --
manually verified at 100 000 rows, see NOTES.md for the timing).
