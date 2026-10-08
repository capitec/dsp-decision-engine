# Testing, checks, runs and experiments

Use this to prove a change is right, to read what a pipeline does over data, and
to compare a proposed policy with the current one.

## Prove a change with a worked example

Pick records whose answer you computed by hand, at the point where the rule
flips, and assert the exact value through decider:

```python
from datetime import date

import polars as pl
import pytest

from decider import Engine
from decider.lab.testing import assert_equivalent

from credit_risk.pipeline import build

EXE = Engine().bind(build(), mode="fused")
PARAMS = {"credit_risk": {"approved": {"limit": 0.4, "month_end_limit": 0.3}}}


@pytest.mark.parametrize(
    "income, debt, applied_on, approved",
    [
        (1000.0, 400.0, date(2026, 1, 10), True),    # 0.40 meets the 0.4 limit exactly
        (1000.0, 401.0, date(2026, 1, 10), False),   # one rand over
        (1000.0, 300.0, date(2026, 1, 28), True),    # month end: 0.30 meets the lower limit
        (1000.0, 301.0, date(2026, 1, 28), False),
    ],
)
def test_approval_at_the_limits(income, debt, applied_on, approved):
    out = EXE.score({"income": income, "debt": debt, "applied_on": applied_on}, PARAMS)
    assert out["approved"] is approved


def test_every_mode_agrees_on_the_worked_rows():
    frame = pl.DataFrame({"income": [1000.0, 1000.0], "debt": [400.0, 401.0],
                          "applied_on": [date(2026, 1, 10)] * 2})
    assert assert_equivalent(build(), frame)["approved"].to_list() == [True, False]
```

Run `uv run pytest -q`. `score` takes one record as a dict and the params
document; `assert_equivalent` runs every mode, `run`, `score` per row and a
session, and fails when any disagree. Name tests for behaviour
(`test_missing_debt_counts_as_zero`). The template's own test scores the sample
request through the handler as `decider serve` does.

## Checks

`check_report(entry, config=None, data=None)` returns the report:
`ok`, `gate`, `counts` and `findings`, each with `code`, `check`, `severity`,
`step.path`, `file`, `line` and a one-line `fix`. `config` names a checks file
(default: the project's `checks.yaml`); `data` is a data file and enables the
checks listed under `skipped` that need data. `checks_list()` is the catalogue:
alias, code, family, `needs_data`, `executes`, severity, config schema.

```
check_report(entry="credit_risk.pipeline:build")
  ok: true   counts: {info: 1, warning: 5, error: 0}
  DCD104 division_by_variable  credit_risk/debt_ratio  guard it (`if b != 0`), use max(b, eps), or bound the param with gt=0
  DCD401 durable_ids           credit_risk/approved    run `decider ids`
```

A check with `executes: true` (`crash_probe`) runs the pipeline on generated edge
cases, so it needs the same go-ahead as `run_flow`. Listing
`{type: crash_probe}` under `checks:` in a checks file and passing it as
`config` reports `DCD501 ... ZeroDivisionError: division by zero`: the real
version of the `DCD104` warning above. It also reports `AttributeError: 'float'
object has no attribute 'day'` for a `date` input, because its synthetic values
are numbers; judge each finding by whether real data could reach it. Guard the
division, then re-run. `ok` stays true while findings are below `fail_on`; the user decides
what blocks a release. From the shell: `uv run decider check <entry> [--config
FILE] [--data FILE] [--format text|json|sarif] [--select DCD1]`; it exits 1 when
the gate fails.

## Run over data

`run_summary(entry, data)` runs the flow over a data file and returns the row
count and a value tally per output column, with no per-record data.
`run_flow(entry, data)` returns the same summary with `ran: true` and asks the
client for confirmation first.

```
run_summary(entry="credit_risk.pipeline:build", data="loans.parquet")
  rows: 3   outputs: {approved: [{value: false, count: 2}, {value: true, count: 1}]}
  params: {validated: ["credit_risk/approved"], invalid: [], warnings: []}
```

- `data` is a `.parquet`, `.csv` or `.json` path (one object per line, or an
  array). JSON gives dates as strings, so a `date` input fails with
  `'str' object has no attribute 'day'`; write dates as typed columns in a
  parquet file (`pl.DataFrame(...).write_parquet("loans.parquet")`).
- These tools run the params defaults from the code, not
  `configs/<version>/params.json`. Compare policies with an experiment.
- Ask the user before pointing them at real data.

## Experiments

An experiment runs the unchanged flow (the baseline) and each scenario over the
same records and measures the difference. `experiment.json`:

```json
{
  "name": "tighter_limit",
  "flow": {"entry": "credit_risk.pipeline:build"},
  "input": {"path": "loans.parquet"},
  "scenarios": [{"name": "tight", "params": {"credit_risk": {"approved": {"limit": 0.2}}}}],
  "metrics": [{"type": "divergence", "column": "approved"}]
}
```

- `experiment_definition(path)` reads and validates it (it fills the defaults).
- `run_experiment(definition, data)` runs it and returns the result; ask first.
  `summaries.divergence.tight` is `{compared: 3, changed: 1, first: 0}` here: one
  of three decisions flips at limit 0.2, and `samples` shows
  `{row: 0, expected: true, actual: false}`.
- `uv run decider experiment run experiment.json --out results` writes
  `results/result.json`; `experiment_results("results/result.json")` reads it
  back. The exit code is 0 when the run completed.
- A scenario can take `params`, `overrides` or another git `revision`.
  `status` reports a partial, cancelled or `NON_REPRODUCIBLE` run as such; read
  it before the numbers.

## Look at one record

`raw_record(entry, data, record)` returns each step's value for row `record`
(0-based); `raw_trace(entry, data, record)` returns the decision-trace events
(which step, branch arm, loop iteration). Both answer "raw data disabled" until
the server runs with `--raw` (setup-project.md); get the user's consent, since
they return per-record values.

```
raw_record(entry="credit_risk.pipeline:build", data="loans.parquet", record=1)
  credit_risk/debt_ratio: 0.35   credit_risk/month_end: true   credit_risk/approved: false
```

`start_debug(entry, data)` opens a debug session paused before anything runs and
returns `structure` (every step path and kind). It does not step; stepping and
breakpoints happen in VS Code (vscode.md). For a what-if in code, use
`pipeline.session(df)`: `break_at("credit_risk/approved")`, `resume()`,
`value("debt_ratio")`, `set("debt_ratio", 0.1)` (`uv run decider guide`,
"Debugging, testing and introspection").

## Durable ids

`generate_ids(path=".", check=True)` previews the ids `decider ids` would add and
returns `changes` as `[file, line, name, id]`. Pass `check=False` to write them;
that needs every affected file git-tracked and a clean tree, and the user's
go-ahead. See best-practices.md for when to commit ids.

## Tools

- `check_report(entry, config=None, data=None)`: the checks over a flow as a structured report; `config` names a checks file, `data` enables data checks.
- `checks_list()`: every registered check with alias, code, family, `needs_data`, `executes` and config schema.
- `run_summary(entry, data)`: row count and per-output value tallies for one run, no per-record data.
- `run_flow(entry, data)`: run a flow end to end and return the summary; needs client confirmation.
- `experiment_definition(path)`: an `experiment.json` read and validated.
- `experiment_results(path)`: a persisted experiment `result.json` read back.
- `run_experiment(definition, data)`: run an experiment definition over data; needs client confirmation.
- `raw_record(entry, data, record)`: per-step values for one record; off unless the server runs with `--raw`.
- `raw_trace(entry, data, record)`: the decision-trace events for one record; off unless the server runs with `--raw`.
- `start_debug(entry, data)`: open a debug session paused before anything runs and return its structure; needs client confirmation.
- `generate_ids(path=".", check=True)`: add durable ids to source under a path (`check=True` only reports); needs client confirmation.
