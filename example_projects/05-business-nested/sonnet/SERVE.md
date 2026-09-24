# Serving business credit granting over nested entities

## Setup

```bash
cd example_projects/05-business-nested/sonnet   # this directory
export DECIDER_API__CODE_PATH="$PWD"
export DECIDER_API__PIPELINE="pipeline:build"
export DECIDER_CONFIG__BASEPATH="$PWD/configs"
export DECIDER_API__MODE=interpreted   # see "Why interpreted mode" below
export PYTHONPATH="<REPO>/example_projects/00-shared-credit-core/sonnet:<REPO>/example_projects/02-affordability/sonnet"
```

`credit_core` (project 00) is consumed read-only via `PYTHONPATH`, exactly as
project 02 does. Project 02's own directory is *also* on `PYTHONPATH` --
`business_nested/sole_proprietor.py` loads its `pipeline.py` by file path (see
that module's docstring for why not a plain `import pipeline`), found by
locating `assessment/household.py` on `sys.path`.

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
for entity in record["entities"]:
    for event in entity["adverse_events"]:
        event["event_date"] = datetime.date.fromisoformat(event["event_date"])
live.executable.score(record, live.params)
```

## Why interpreted mode

`events.py`'s `classify_events_step` and `structure.py`'s `resolve_structure`
run arbitrary Python per application (ragged entity/event loops, a nested
`Engine` call for the threshold table) inside a `frame_step` -- `frame_step`s
run under the interpreted runner regardless of the pipeline's declared mode
(they are already outside numba's compiled path), and `sole_proprietor.py`'s
dynamically-loaded nested pipeline call is Python-level I/O (importlib), not
something a compiled kernel can call into. Inherited from 00/02: the shared
`core.expense_norms` machinery 02 uses for the sole-proprietor regulated call
also compares two `str` table-version columns, which compiled (`fused`/
`stepped`) mode rejects at bind time. `interpreted` is therefore the only mode
this project (and everything downstream of it) can serve in unchanged.

## A request must populate every entity's ragged fields explicitly

Inherited from 00/02's documented trap (00 NOTES.md 4.1/4.3, 02 NOTES.md 4.1):
an **entirely absent** top-level `list`/`dict` column, or a **lone `None`**
top-level scalar field with no other value in the record to infer its type
from, both crash single-record scoring. `sample_request.json` gives
`client_id` a real integer rather than `null` for exactly this reason -- a
solo application (no client history yet) should send an out-of-band sentinel
(e.g. `0`) rather than `null`, not because `null` is semantically wrong, but
because `decider`'s single-record type inference cannot resolve it.

A **new, project-05-specific** instance of the same family of gaps, found
while building this project (see NOTES.md "Framework friction" for the full
write-up and a two-`frame_step` minimal reproduction): a `list[date]` column
written by one `frame_step` and read by a *second* `frame_step` downstream in
the same `dag` silently degrades to raw epoch-day integers at the boundary
between them -- no error, until something calls `.year` on an int.
`structure.py` works around this by deriving `ev_age_months` (an int) inside
the same `frame_step` invocation that still holds real `datetime.date`
objects, and never emitting a `list[date]` column for any other step to read.

## Run the tests

```bash
PYTHONPATH="<REPO>/example_projects/00-shared-credit-core/sonnet:<REPO>/example_projects/02-affordability/sonnet" \
  uv run --project <REPO> pytest example_projects/05-business-nested/sonnet/tests -q
```

`tests/conftest.py` adds this project's own directory to `sys.path`;
`credit_core` and project 02 still need the `PYTHONPATH` entries above (both
are consumed read-only, never copied in).
