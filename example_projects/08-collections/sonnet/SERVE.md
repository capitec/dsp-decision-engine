# Serving collections treatment assignment

## Setup

```bash
cd example_projects/08-collections/sonnet   # this directory
export DECIDER_API__CODE_PATH="$PWD"
export DECIDER_API__PIPELINE="pipeline:build"
export DECIDER_CONFIG__BASEPATH="$PWD/configs"
export DECIDER_API__MODE=interpreted   # see "Why interpreted mode" below
export PYTHONPATH="<REPO>/example_projects/00-shared-credit-core/sonnet:<REPO>/example_projects/02-affordability/sonnet"
```

Both 00 and 02 are consumed read-only via `PYTHONPATH`. `collections_treatment/arrangements.py`
loads 02's `pipeline.py` by file path under a private module name (never a bare
`import pipeline`) -- see that module's docstring for why a plain import would risk `decider`
serving the wrong project's pipeline.

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
from inference import _typed_sample_record
handler = construct_handler_from_settings()
handler.stage(); handler.activate()
live = handler.module_fn()
record = _typed_sample_record()
out = live.executable.score(record, live.params)
```

## Why interpreted mode

Inherited from 00 and 02: `credit_core.expense_norms.norm_table_version` (used
unmodified by 02's `arrangement_affordability_unit()`) compares two `str`
table-version columns, which compiled (`fused`/`stepped`) mode rejects at
bind time. See 00's / 02's own SERVE.md for the full writeup.

## Request shape gotchas (see NOTES.md "Framework friction" for the full writeup)

- Every `date | None` field this project declares (`debt_review_default_date`,
  `complaint_logged_date`, `notice_delivered_date`, `frequency_cap_window_reset_date`,
  `promise_date`) must be sent as a real, harmless placeholder date (e.g.
  `"2020-01-01"`) rather than JSON `null`, when nothing else in the request makes
  that suspension fire. A genuine JSON `null` infers as a `Null`-dtype polars column
  and crashes at the arrow-import boundary (`ArrowImportError: Expected array with
  0 buffer(s) but found 1 buffer(s)`) -- the same finding 02 NOTES.md 4.1 made for a
  lone `None` scalar, confirmed again here for every optional date this project adds.
- `consent_withdrawn_channels: list[int]` must never be sent as `[]` alone (00
  NOTES.md 4.3's empty-list-literal-needs-a-sibling finding); send a real value.
- The `applicant1_*`/`applicant2_*` fields (income, expenses, bureau/internal
  account lists) follow 02's own sample-request convention exactly: a solo
  application still populates every one of applicant 2's ragged fields with one
  harmless placeholder element, never `[]` or an omitted key. See 02's own
  SERVE.md/NOTES.md for why.

## Run the tests

```bash
export PYTHONPATH="<REPO>/example_projects/00-shared-credit-core/sonnet:<REPO>/example_projects/02-affordability/sonnet"
uv run --project <REPO> pytest example_projects/08-collections/sonnet/tests -q
```

**Do not add this project's own directory to `PYTHONPATH`** -- `tests/conftest.py`
already puts it on `sys.path` itself. Adding it to `PYTHONPATH` as well is not merely
redundant: it is actively dangerous, and reproduces this project's single worst piece
of framework friction (see NOTES.md). In short: `decider`'s handler only inserts
`DECIDER_API__CODE_PATH` at `sys.path[0]` when it *isn't already on `sys.path`*
(`decider/serving/handler.py`: `if code_path not in sys.path: sys.path.insert(0,
code_path)`). If this project's own directory is already on `sys.path` via
`PYTHONPATH` -- at whatever position `PYTHONPATH`'s ordering put it, which is *after*
project 00's and 02's directories in the export above -- that guard is skipped, this
project's own directory keeps its later position, and `import pipeline` (both
`decider`'s own resolution of `DECIDER_API__PIPELINE=pipeline:build` *and* this
project's own `test_pipeline.py`, which drives the real serving handler) silently
resolves to **project 00's** `pipeline.py` instead -- the first `pipeline.py` on
`sys.path`, not this project's. `decider build` then fails with a confusing, unrelated
error (`KeyError: "the config version has no 'rate_card_flex_loan' document for
argument 'rate_card_flex_loan' of 'pipeline:build'"` -- 00's own `build()` signature),
or worse, if 00's config version happens to also build cleanly, `decider serve` would
silently serve **the wrong project's decision logic** with no error at all. Reproduced
directly while writing this project's own SERVE.md; see NOTES.md "Framework friction"
for the full writeup and why `arrangements.py` loads project 02's `pipeline.py` by file
path rather than a bare `import pipeline`, for exactly this reason.
