# Serving the affordability and obligations assessment

## Setup

```bash
cd example_projects/02-affordability/sonnet   # this directory
export DECIDER_API__CODE_PATH="$PWD"
export DECIDER_API__PIPELINE="pipeline:build"
export DECIDER_CONFIG__BASEPATH="$PWD/configs"
export DECIDER_API__MODE=interpreted   # see "Why interpreted mode" below
export PYTHONPATH="<REPO>/example_projects/00-shared-credit-core/sonnet"
```

`credit_core` (project 00) is consumed read-only via `PYTHONPATH`; nothing
here imports from anywhere but `credit_core` and `decider` itself.

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
import datetime, json
from decider.serving.handler import construct_handler_from_settings
handler = construct_handler_from_settings()
handler.stage(); handler.activate()
live = handler.module_fn()
record = json.load(open("sample_request.json"))
record["decision_date"] = datetime.date.fromisoformat(record["decision_date"])
record["bureau_as_of_date"] = datetime.date.fromisoformat(record["bureau_as_of_date"])
for key in ("applicant1_bureau_accounts", "applicant1_internal_accounts",
            "applicant2_bureau_accounts", "applicant2_internal_accounts"):
    for a in record.get(key, []):
        if a.get("opened_date"):
            a["opened_date"] = datetime.date.fromisoformat(a["opened_date"])
live.executable.score(record, live.params)
```

## Why interpreted mode

Inherited from 00: `credit_core.expense_norms.norm_table_version` (used
unmodified here) compares two `str` table-version columns, which compiled
(`fused`/`stepped`) mode rejects at bind time. See 00's SERVE.md /
NOTES.md for the full writeup.

## A request must populate every applicant's ragged fields explicitly

`decider`'s single-record `.score()` path infers each top-level column's
type from the one value it sees. An **entirely absent** `list`/`dict`
column combined with any `frame_step` in the pipeline crashes at frame
construction (`ShapeError: unable to add a column of length 0 to a
DataFrame of height 1`), and a **present-but-empty** `list` literal
(`[]`) with no non-empty sibling infers as `List(Null)` and crashes at the
arrow-import boundary (`ArrowImportError: Expected array with 0 buffer(s)
but found 1 buffer(s)`) -- 00 NOTES.md "Framework friction" 4.1/4.3, hit
again here one level deeper (two-applicant ragged fields, not one).

For a **solo** application, `sample_request.json` sends applicant 2's
ragged fields (`applicant2_variable_pay_history`,
`applicant2_bureau_accounts`, `applicant2_internal_accounts`) each with
one harmless placeholder element (`account_type_code: 31`, the `EXCLUDE`
treatment -- balance/limit/instalment all zero, `closed: true`) rather
than `[]` or an omitted key. This contributes exactly `0.0` to every
obligations figure, but does add one entry to the per-account annotation
lists (`obligation_account_type_codes` etc.) -- a real, documented
consumer trap for any caller of this pipeline sending a solo application.
See NOTES.md "Framework friction" for the full writeup, including a
scalar-field variant of the same bug (a lone `None` value for an
optional field also infers an ambiguous `Null` dtype).

## Run the tests

```bash
PYTHONPATH="<REPO>/example_projects/00-shared-credit-core/sonnet" \
  uv run --project <REPO> pytest example_projects/02-affordability/sonnet/tests -q
```

`tests/conftest.py` adds this project's own directory to `sys.path`;
`credit_core` still needs the `PYTHONPATH` entry above (00 is consumed
read-only, never copied in).
