# Serving consolidation and restructure

## Setup

```bash
cd example_projects/06-consolidation/sonnet   # this directory
export DECIDER_API__CODE_PATH="$PWD"
export DECIDER_API__PIPELINE="pipeline:build"
export DECIDER_CONFIG__BASEPATH="$PWD/configs"
export DECIDER_API__MODE=fused
export PYTHONPATH="<REPO>/example_projects/00-shared-credit-core/sonnet:<REPO>/example_projects/02-affordability/sonnet:<REPO>/example_projects/03-loan-granting-pricing/sonnet"
```

`credit_core` (00), `assessment`/`pipeline.py` (02) and `loan_granting`/`pipeline.py`
(03) are all consumed read-only via `PYTHONPATH`; nothing here imports from
anywhere but those three and `decider` itself.

## Build

```bash
uv run --project <REPO> decider build
```

Expect: `built config version 0.1.0 (pipeline pipeline:build, mode fused)`.
`pipeline.py`'s `build(rate_card_flex_loan, rate_card_product11)` takes two
`ConfigurableStep` arguments, auto-loaded by `decider` from
`configs/0.1.0/rate_card_flex_loan.json` (project 00's original product-10 card,
needed only for the short-circuit hand-off to project 03) and
`configs/0.1.0/rate_card_product11.json` (this project's own product-11 card).

If either is ever regenerated (a real Treasury refresh would replace its `rows`
directly, no redeploy -- 00 §9):

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
rate_card_p11 = DecisionTableConfig.load("configs/0.1.0/rate_card_product11.json")
exe = Engine().bind(pipeline.build(rate_card, rate_card_p11), mode="interpreted")

record = json.load(open("sample_request.json"))
record["decision_date"] = datetime.date.fromisoformat(record["decision_date"])
record["bureau_as_of_date"] = datetime.date.fromisoformat(record["bureau_as_of_date"])
record["last_consolidation_date"] = datetime.date.fromisoformat(record["last_consolidation_date"])
for a in record["accounts"]:
    a["opened_date"] = datetime.date.fromisoformat(a["opened_date"])
    a["quotation_expiry_date"] = datetime.date.fromisoformat(a["quotation_expiry_date"])

out = exe.score(record)
print(out["outcome_code"], out["chosen_product_code"], out["chosen_instalment"])
```

Expected result for `sample_request.json` as shipped: `outcome_code="approve"`,
`chosen_product_code=20` (Everyday Card balance transfer), settling accounts
2, 4, 5, 7, term (promotional duration) 24 months, `chosen_instalment≈1917.56`,
`chosen_instalment_relief≈242.44`. The objective in force is `OBJ-02` blended
0.6/0.4 with `OBJ-01` on the branch channel (`channel_code=1`) by the one
declared re-weight overlay (`objective.py`'s `OBJECTIVE_REWEIGHT` -- both
`ADJ-06-OBJ-2026-001`/`-002` should appear in `objective_overlay_ids`).

`inference.py`'s `Handler` is what `decider build`/`decider serve` actually use
(it replaces the framework's synthetic warm-up record with `sample_request.json`
-- see NOTES.md "Framework friction" #1, the same finding projects 00, 02 and 03
document); the snippet above binds the pipeline directly, which is what the CLI
does internally too, minus the warm-up substitution.

## Every account needs the same set of fields, every time, with no null-only column

Two related traps, both fatal at the arrow-import boundary and both silent
until they crash (see NOTES.md "Framework friction" #1 for the full story):

- **Same gotcha 00/02/03 document, one level further.** An entirely-absent
  `list` column, a present-but-empty list literal with no non-empty sibling,
  or a lone `None` scalar on a single-record request all crash single-record
  scoring. This project's convention: every optional field gets a **sentinel**
  (`"2000-01-01"` for "no such date", `0` for "no security type", `""` for "no
  quotation reference"), never `null`, and every account carries a real,
  non-null value for at least one row of any field that *can* be null across
  the population (e.g. `security_type_code`).
- **New in this project: every element of a `list[struct]` column must carry
  the exact same set of keys.** A single account dict missing one key that
  every other account (and every other request this project has ever scored
  in the same process) carries does not raise a clear "missing field" error --
  it silently routes the whole `accounts` column through a numpy fallback path
  that then crashes with `DTypePromotionError: The DType
  <numpy.dtypes.Float64DType> could not be promoted by
  <numpy.dtypes.DateTime64DType>` (a struct field mismatch masquerading as a
  numeric/date type conflict). `sample_request.json`'s `_make_sample.py` and
  `tests/test_pipeline.py`'s synthetic 18-account client both build every
  account through the exact same field set for this reason.

## Mode

Served in `fused` mode: on `sample_request.json` its output equals
`interpreted` mode's exactly. Steps no kernel can run faithfully (for example
`credit_core.expense_norms.norm_table_version`, which compares two `str`
inputs) run in Python, row by row, with a warning naming each at build time.

## Run the tests

```bash
export PYTHONPATH="<REPO>/example_projects/00-shared-credit-core/sonnet:<REPO>/example_projects/02-affordability/sonnet:<REPO>/example_projects/03-loan-granting-pricing/sonnet"
uv run --project <REPO> pytest example_projects/06-consolidation/sonnet/tests -q
```

`tests/conftest.py` adds this project's own directory to `sys.path`; `credit_core`,
`assessment` and `loan_granting` still need the `PYTHONPATH` entries above.
`tests/test_pipeline.py::test_budget_holds_at_18_settleable_accounts` is the
acceptance-criterion-1 check (spec 06 §10 item 1 / SCOPE.md "test the budget:
400 scenarios in 900 ms"): an 18-settleable-account client, asserting the
search never exceeds its declared 400-scenario budget and completes in under
900 ms. Measured on this machine: ~110 ms end to end (baseline call to project 02 plus
the full search, 120 scenarios) for the 9-account `sample_request.json`, and
~95-150 ms (occasionally ~250 ms under GC/JIT noise) for a synthetic
18-settleable-account client (153 scenarios, `candidates_exhausted` before the
400-scenario budget) -- both well inside the 900 ms budget. The test asserts
the bound directly rather than a fixed number, since wall-clock time is
legitimately hardware-dependent.
