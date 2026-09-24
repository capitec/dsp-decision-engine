# Serving business credit, end to end (EP-1)

This project serves **EP-1** (new-to-bank business application, products 50
and 51) -- SCOPE.md's "one real path". L1 (annual review), the DSCR covenant
test, the bi-temporal structure query and the daily-pass partial
re-assessment are all exercised directly in `tests/`, not through the CLI --
see NOTES.md and `pipeline.py`'s own docstring for why.

## Setup

```bash
cd example_projects/11-business-credit-e2e/sonnet   # this directory
export DECIDER_API__CODE_PATH="$PWD"
export DECIDER_API__PIPELINE="pipeline:build"
export DECIDER_CONFIG__BASEPATH="$PWD/configs"
export DECIDER_API__MODE=interpreted   # see "Why interpreted mode" below
export PYTHONPATH="<REPO>/example_projects/00-shared-credit-core/sonnet:<REPO>/example_projects/02-affordability/sonnet:<REPO>/example_projects/05-business-nested/sonnet:<REPO>/example_projects/07-credit-limit-management/sonnet"
```

`credit_core` (00), `assessment` (02), `business_nested` (05) and `limit_mgmt`
(07) are consumed **read-only** via `PYTHONPATH`; nothing here imports from
anywhere else. This project's own `pipeline.py` composes 05's `pipeline.py`
by file path (`business_credit_e2e/reuse.py`), not by a plain `import
pipeline` -- see NOTES.md "Framework friction" for why a plain import cannot
work once more than one sibling project sits on `sys.path`.

**06 is not on this list.** DEPS.md marks it hard, but §4.9 attributes 06's
components (obligation inventory/settleability, the concession catalogue) to
L5 only, and SCOPE.md explicitly skips L4-L6 for this slice. `reuse.
load_project06_pipeline()` exists and is tested to load
(`tests/test_reuse.py::test_project06_pipeline_loads`), but nothing in the
served path or the other tests calls it -- put 06's directory on
`PYTHONPATH` only if you run that one test in isolation; it is not needed for
`decider build`/`decider serve`. **06 itself hard-depends on project 03**
(`loan_granting`) -- if you do add 06 to `PYTHONPATH` for that test, add 03's
directory too, or the import chain fails with `ModuleNotFoundError: No
module named 'loan_granting'` three modules deep inside `consolidation/
pricing11.py`.

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

Or, without a server, exactly as `tests/test_origination.py` does it:

```python
from decider.serving.handler import construct_handler_from_settings
handler = construct_handler_from_settings()
handler.stage(); handler.activate()
live = handler.module_fn()

import inference
record = inference._typed_sample_record()
live.executable.score(record, live.params)
```

The sample request is a **decline** (project 05's own entity-disqualification
path, inherited unmodified -- entity 1 carries a disqualifying adverse
event). This is deliberate: it is the identical request shape 05's own
`sample_request.json` uses, plus this project's own facility/covenant
fields, and it proves the whole composed pipeline (05's assessment plus this
project's covenant binding and DSCR test) runs end to end regardless of the
final outcome. `tests/test_review.py` builds a clean (approved) applicant for
the L1 scenarios, since a review needs a predecessor decision to exist.

## Why interpreted mode

Inherited from 00/02/05: `credit_core.expense_norms.norm_table_version`
(consumed through project 02's sole-proprietor call) compares two `str`
table-version columns, which compiled (`fused`/`stepped`) mode rejects at
bind time; 05's `events.py`/`structure.py` also run a nested `Engine` call
per entity inside a `frame_step`, which only the interpreted runner reaches.
See 00's own SERVE.md/NOTES.md for the full write-up.

## Requests need every ragged/optional field populated explicitly

Inherited from 00/02/05's own documented gotchas: an entirely-absent
`list`/`dict` column, an empty `[]` literal with no non-empty sibling, and a
lone `null` for an optional scalar all crash request construction
differently. `sample_request.json` follows the same conventions 00/02/05
established. **A new instance of the same family, found while building this
project**: setting *every* entity's `adverse_events` to `[]` at once (to
build a "clean applicant" test fixture) reproduces the identical
`ArrowImportError: Expected array with 0 buffer(s) but found 1 buffer(s)` --
see NOTES.md "Framework friction" and `tests/test_review.py`'s own comment
for the fix (reduce event amounts, never clear every ragged list in the
record to `[]` at once).

## Run the tests

```bash
PYTHONPATH="<REPO>/example_projects/00-shared-credit-core/sonnet:<REPO>/example_projects/02-affordability/sonnet:<REPO>/example_projects/03-loan-granting-pricing/sonnet:<REPO>/example_projects/05-business-nested/sonnet:<REPO>/example_projects/06-consolidation/sonnet:<REPO>/example_projects/07-credit-limit-management/sonnet" \
  uv run --project <REPO> pytest example_projects/11-business-credit-e2e/sonnet/tests -q
```

`tests/conftest.py` adds this project's own directory to `sys.path`. The test
PYTHONPATH is wider than the serving one above: `tests/test_reuse.py` and
`tests/test_daily_pass.py` load 05, 06 and 07 (06 transitively needs 03 --
see "Setup" above), and `tests/test_review.py` calls into 07's per-account
pipeline for the revolving-facility limit decision. All 29 tests pass.
