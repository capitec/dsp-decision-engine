# Serving Flex Loan granting and pricing

## Setup

```bash
cd example_projects/03-loan-granting-pricing/sonnet   # this directory
export DECIDER_API__CODE_PATH="$PWD"
export DECIDER_API__PIPELINE="pipeline:build"
export DECIDER_CONFIG__BASEPATH="$PWD/configs"
export DECIDER_API__MODE=interpreted   # see "Why interpreted mode" below
export PYTHONPATH="<REPO>/example_projects/00-shared-credit-core/sonnet:<REPO>/example_projects/02-affordability/sonnet"
```

Both `credit_core` (project 00) and `assessment`/`pipeline.py` (project 02) are
consumed read-only via `PYTHONPATH`; nothing here imports from anywhere but
those two and `decider` itself.

## Build

```bash
uv run --project <REPO> decider build
```

Expect: `built config version 0.1.0 (pipeline pipeline:build, mode interpreted)`.
The first build takes a few seconds while the Flex Loan rate card's 63 360
rows are staged, indexed for the solve (`loan_granting/pricing.py`'s
`RateCardIndex`) and warmed (see `inference.py`).

If `configs/0.1.0/rate_card_flex_loan.json` is ever regenerated (a real
Treasury refresh would replace its `rows` directly, no redeploy -- 00 §9):

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
import datetime, json
from decider import Engine
from decider.steps.tables import DecisionTableConfig
import pipeline

rate_card = DecisionTableConfig.load("configs/0.1.0/rate_card_flex_loan.json")
exe = Engine().bind(pipeline.build(rate_card), mode="interpreted")

record = json.load(open("sample_request.json"))
record["decision_date"] = datetime.date.fromisoformat(record["decision_date"])
record["bureau_as_of_date"] = datetime.date.fromisoformat(record["bureau_as_of_date"])
for key in ("bureau_accounts", "internal_accounts", "applicant2_bureau_accounts", "applicant2_internal_accounts"):
    for a in record[key]:
        a["opened_date"] = datetime.date.fromisoformat(a["opened_date"])

out = exe.score(record)
print(out["outcome_code"], out["recommended_term"], out["recommended_amount"])
```

`inference.py`'s `Handler` is what `decider serve`/`decider build` actually
use (it replaces the framework's synthetic warm-up record with
`sample_request.json` -- see NOTES.md "Framework friction" #1); the snippet
above binds the pipeline directly, which is what the CLI does internally
too, minus the warm-up substitution.

## A request must populate every ragged field explicitly

Same gotcha 00 and 02 document, one level further: an **entirely absent**
`list` column, or a **present-but-empty** list literal (`[]`) with no
non-empty sibling, both crash single-record scoring at the arrow-import
boundary. `sample_request.json`'s own convention: `exclusion_list_hits`
sends `[0]` (a "no hit" sentinel this project's own code filters out --
`loan_granting/eligibility.py`), never `[]`. Any consumer building a request
by hand needs to follow the same pattern for every ragged field it might
otherwise send empty.

## Why interpreted mode

Inherited from 00/02: `credit_core.expense_norms.norm_table_version`
(reused unmodified inside the affordability call) compares two `str`
table-version columns, which compiled (`fused`/`stepped`) mode rejects at
bind time. This project also has its own reason to prefer interpreted mode
even setting that aside: the solve and the cap waterfall (`solve.py`,
`waterfall.py`) are plain Python functions inside `frame_step`s, not
compiled kernels, so `fused`/`stepped` buys nothing for them regardless.

## Run the tests

```bash
PYTHONPATH="<REPO>/example_projects/00-shared-credit-core/sonnet:<REPO>/example_projects/02-affordability/sonnet" \
  uv run --project <REPO> pytest example_projects/03-loan-granting-pricing/sonnet/tests -q
```

`tests/conftest.py` adds this project's own directory to `sys.path`;
`credit_core` and `assessment` still need the `PYTHONPATH` entries above.
