# decider: getting started

Don't read the repo's `docs/` folder: it describes an older version. This guide
and the docstrings (`help(decider)`, `help(decider.flow)`, ...) are current.
`decider guide` prints this guide.

decider builds decision pipelines (credit, pricing, fraud rules, campaigns)
from plain Python functions. A pipeline runs over a polars frame (`run`) or one
record (`score`, a dict), interpreted or compiled with numba, and serves over
HTTP with `decider serve`.

## Mental model

- **A step is a function.** Its arguments are the names it reads; its name is
  the name it writes. One step writes one value (several only with
  `@step(outputs=(...))` and a `tuple` return). Never return a `dict`.
- **Inputs** are plain arguments: request or row data (amounts, dates, ids).
  `missing_as(x)` fills a null input.
- **Params** are `param(default)` arguments: policy that is the same for every
  request (limits, cut-offs, weights). Values come from the **params
  document** (`configs/<version>/params.json`, keyed by step path), so they
  change without a code change, rebuild or recompile. **Request data is never
  a `param()`**: a param ignores the request.
- **Outputs**: `run` returns the input columns plus every value nothing reads;
  intermediates are dropped unless kept with `.emit("name")`.
- **Modes**: `Engine().bind(p, mode=...)`: `"interpreted"` (plain Python; what
  `p.run` uses), `"stepped"` or `"fused"` (numba; `decider serve`'s default).
  All modes give the same answers.

## Which construct for which need

| Need | Use |
|---|---|
| a calculation per record | a plain function |
| a policy number | `param(default, ge=, le=)` |
| a null input with a fallback | `missing_as(value)` |
| steps in written order; a waterfall where a later write wins | `flow(a, b, name=...)` |
| steps in dependency order | `dag(...)` |
| different logic per row (eligibility, suppression) | `branch(cond, if_true, if_false, modifies=[...], name=)` |
| iterate per row (a solve, a schedule) | `loop(cond, body, carries=[...], max_iterations=n, name=)` |
| whole-frame work (joins, group-bys, ragged lists) | `@frame_step(reads=, writes=)` |
| a decision tree, or a flat rule set | `TreeConfig` (`prioritized_flat_rule`, `mode: "all"`) |
| a grid (bands, rate cards) fixed in its config document | `DecisionTableConfig`, inline `"rows": [...]` |
| a grid retuned like a param (params document, per call) | `DecisionTableConfig`, `"rows": {"table": "name"}` |
| a points scorecard | `ScorecardConfig` |
| the same step twice, or on other column names | `.named("x")`, `.relabel(reads=, writes=)` |
| another project's pipeline | import its `build()`; put the step inside yours |
| immutable config versions | `decider.config.JsonFileStore` (`ConfigStore`) |
| why a record got its answer; what-if | `pipeline.session(df)` |
| modes, batch and single record agree | `decider.testing.assert_equivalent` |

## Steps, params, run and score

```python
from datetime import date

import polars as pl
from decider import Engine, flow, missing_as, param

def debt_ratio(income: float, debt: float = missing_as(0.0)) -> float:
    return debt / income

def month_end(applied_on: date) -> bool:
    return applied_on.day >= 25

def approved(debt_ratio: float, month_end: bool,
             limit: float = param(0.4, ge=0.0, le=1.0), month_end_limit: float = param(0.3)) -> bool:
    return debt_ratio <= (month_end_limit if month_end else limit)

pipeline = flow(debt_ratio, month_end, approved, name="credit")
df = pl.DataFrame({"income": [1000.0, 500.0], "debt": [350.0, None],
                   "applied_on": [date(2026, 1, 5), date(2026, 1, 28)]})
assert pipeline.run(df)["approved"].to_list() == [True, True]
assert "debt_ratio" in pipeline.emit("debt_ratio").run(df).columns

params = pipeline.parameters().defaults()        # the params document, with every default
assert params == {"credit": {"approved": {"limit": 0.4, "month_end_limit": 0.3}}}
params["credit"]["approved"]["limit"] = 0.3
assert pipeline.run(df, params=params)["approved"].to_list() == [False, True]

exe = Engine().bind(pipeline, mode="fused")      # bind once, call many times
assert exe.score({"income": 1000.0, "debt": 350.0, "applied_on": date(2026, 1, 5)}, params)["approved"] is False
```

## Branch, loop, frame steps

```python
import polars as pl
from decider import branch, flow, frame_step, loop, param, step

def eligible(age: int) -> bool:
    return age >= 18

@step(output="limit")
def full_limit(income: float, multiple: float = param(3.0)) -> float:
    return income * multiple

@step(output="limit")
def no_limit(income: float) -> float:
    return 0.0

def unpaid(balance: float) -> bool:
    return balance > 0.0

@step(output="balance")
def pay(balance: float, instalment: float, rate: float = param(0.01)) -> float:
    return balance * (1 + rate) - instalment

@step(output="months")
def tick(months: int) -> int:
    return months + 1

@frame_step(reads=["client_id"], writes=["prior_defaults"])
def join_history(df: pl.DataFrame) -> pl.DataFrame:
    return df.join(pl.DataFrame({"client_id": [1, 2], "prior_defaults": [2, 0]}), on="client_id", how="left")

pipeline = flow(
    join_history,
    branch(eligible, full_limit, no_limit, modifies=["limit"], name="by_eligibility"),
    loop(unpaid, flow(pay, tick, name="month"), carries=["balance", "months"], max_iterations=360, name="repay"),
    name="offer")
out = pipeline.run(pl.DataFrame({"client_id": [1, 2], "age": [30, 16], "income": [1000.0, 800.0],
                                 "balance": [1000.0, 500.0], "instalment": [100.0, 100.0], "months": [0, 0]}))
assert out["limit"].to_list() == [3000.0, 0.0] and out["months"].to_list() == [11, 6]
assert out["prior_defaults"].to_list() == [2, 0]
```

## Trees, rule sets, tables, scorecards

Each loads from a JSON document (`TreeConfig.load(doc)`, or
`ConfigurableStep.load(doc)` for any type). Any number in it can be
`{"param": "name", "default": x}`, which makes it a param. In a project,
`def build(fraud_rules)` receives `configs/<version>/fraud_rules.json` loaded.

```python
import polars as pl
from decider.steps.scorecard import ScorecardConfig
from decider.steps.tables import DecisionTableConfig
from decider.steps.trees import TreeConfig

def rule(name, feature, op, threshold):
    return {"meta": {"name": name}, "rule": {"type": "unary", "then": {"id": f"{name}_hit", "type": "leaf", "result_idx": 0},
            "condition": {"op": op, "feature": feature, "threshold": {"param": name, "default": threshold}}}}

rules = TreeConfig.load({"type": "tree", "name": "fraud_rules", "path_output": "leaf", "tree": {
    "type": "prioritized_flat_rule", "mode": "all",      # every rule answers, not just the first match
    "rules": [rule("big_amount", "amount", ">", 5000.0), rule("new_device", "device_age_days", "<", 30.0)],
    "output": {"data": [{"hit": 1}], "default": {"hit": 0}, "dtypes": [["hit", "Int64"]]}}})
out = rules.run(pl.DataFrame({"amount": [9000.0, 10.0], "device_age_days": [3.0, 400.0]}))
assert out["big_amount.hit"].to_list() == [1, 0] and out["new_device.leaf"].to_list() == ["new_device_hit", None]

bands = DecisionTableConfig.load({"type": "decision_table", "name": "band",
    "columns": {"lo": "Float64", "hi": "Float64", "band": "String"},
    "rows": [{"lo": None, "hi": 600.0, "band": "C"}, {"lo": 600.0, "hi": 700.0, "band": "B"},
             {"lo": 700.0, "hi": None, "band": "A"}],
    "expression": {"type": "between", "variable": "bureau_score", "lower_bound_column": "lo", "upper_bound_column": "hi"},
    "outputs": ["band"], "default": ["C"]})
assert bands.run(pl.DataFrame({"bureau_score": [550.0, 720.0]}))["band"].to_list() == ["C", "A"]

card = ScorecardConfig.load({"type": "scorecard", "name": "card", "variables": [{
    "type": "scored", "variable_name": "age", "default": {"value": 0},
    "bins": [{"value": 5, "upper_bound": {"param": "young", "default": 25}},
             {"value": 10, "lower_bound": {"param": "young", "default": 25}}]}]})
assert card.run(pl.DataFrame({"age": [20.0, 40.0]}))["score"].to_list() == [5.0, 10.0]
```

`path_output` names the leaf that answered (per rule in `mode: "all"`);
`trace_output` names every node id on the row's path, in order. `TreeConfig`
also takes a v3 node/edge tree:

```python
import polars as pl
from decider.steps.trees import TreeConfig

tree = TreeConfig.load({"type": "tree", "name": "risk", "trace_output": "risk_path", "tree": {
    "nodes": [{"id": "root", "data": {"type": "unary", "condition": {"op": ">", "feature": "ratio", "threshold": 0.7}}},
              {"id": "high", "data": {"type": "leaf", "result_idx": 0}}],
    "edges": [{"source": "root", "target": "high", "data": {"sourceIndex": 0}}],
    "output": {"data": [{"risk": 1}], "default": {"risk": 0}, "dtypes": [["risk", "Int64"]]}}})
assert tree.run(pl.DataFrame({"ratio": [0.9]}))["risk_path"].to_list()[0][-1] == "high"
```

## Parameter tables

Inline `"rows"` fix a table in its config document: changing them is a new
document and a rebuild (never a recompile). With `"rows": {"table": "prices"}`
the rows live in the params document under the step's path, are checked
against `columns` when the document arrives, and can be replaced (any number
of rows) per call with no rebuild or recompile. A String *output* of a
parameter table must be an `Enum` column, so its values are known up front.

```python
import polars as pl
from decider import Engine
from decider.steps.tables import DecisionTableConfig
from decider.testing import no_recompile

price = DecisionTableConfig.load({"type": "decision_table", "name": "price",
    "columns": {"product": "String", "rate": "Float64"}, "rows": {"table": "prices"},
    "expression": {"type": "eq", "variable": "product", "value_column": "product"},
    "outputs": ["rate"], "default": [0.0]})
assert price.parameters().defaults() == {}       # the rows are required: there is no default
assert price.parameters() == {"price": {"prices": {"type": "table", "schema": {"product": "String", "rate": "Float64"}}}}

params = {"price": {"prices": [{"product": "loan", "rate": 0.12}, {"product": "card", "rate": 0.2}]}}
exe = Engine().bind(price, mode="fused")
df = pl.DataFrame({"product": ["loan", "card", "car"]})
assert exe.run(df, params=params)["rate"].to_list() == [0.12, 0.2, 0.0]
with no_recompile():
    params = {"price": {"prices": [{"product": "car", "rate": 0.1}]}}
    assert exe.run(df, params=params)["rate"].to_list() == [0.0, 0.0, 0.1]
```

`{"table": "prices", "shared": true}` reads `shared.prices`, one table for
several steps. Table params for plain function steps are coming; until then,
use a `DecisionTableConfig`.

## A project: template, build, serve

```bash
decider template credit_risk && cd credit_risk
pytest -q                     # scores sample_request.json through the handler
decider build                 # validates configs/<latest>, warms every kernel with sample_request.json
decider serve --workers 1     # POST /invocations, GET /ping; needs decider[serve-starlette]
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations
```

The template writes a package named after the project, never a top-level
`pipeline.py` (two projects on one path would shadow each other):

- `credit_risk/pipeline.py`: `build()` returns the pipeline; each argument of
  `build` receives the config document of that name.
- `credit_risk/inference.py`: `Handler(RequestHandler)`; override `input_fn`,
  `output_fn`, ... to change request handling.
- `configs/0.0.0/params.json`, `sample_request.json`, `tests/`.
- `.env`: `DECIDER_API__PIPELINE=credit_risk.pipeline:build` and the other
  settings. Real environment variables win. `decider --help` lists every
  command and option; there are no others.

JSON dates (`"2026-01-15"`) arrive as the declared `date` inputs. Score through
the handler in-process, raw bytes in, exactly as `decider serve` does:

```python
import asyncio, json
from decider.config import JsonFileStore
from decider.serving import RequestHandler

handler = RequestHandler(JsonFileStore(basepath="configs"), "credit_risk.pipeline:build")
handler.stage()
handler.activate()
body = b'{"income": 1000, "debt": 250, "applied_on": "2026-01-15"}'
response = asyncio.run(handler.process_fn(body, "application/json", "application/json"))
assert json.loads(response.content)["approved"] is True
```

`store.create_version({...})` writes a new immutable version; replay a stored
decision against `store.read(version)`, never whatever `configs/` holds now.

## Reusing another project

Import its package (`credit_core.pipeline`), never a sibling file through
`importlib`. Put your own directory first on `PYTHONPATH`, consumed projects
after: `PYTHONPATH=$PWD:../credit_core`.

```python
import polars as pl
from decider import flow, param

def ratio(income: float, debt: float) -> float:
    return debt / income

def approved(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit

def build_core():               # stands in for `from credit_core.pipeline import build`
    return flow(ratio, approved, name="credit")

def offer(approved: bool, income: float) -> float:
    return income * 2 if approved else 0.0

app = flow(build_core().relabel(reads={"debt": "total_debt"}), offer, name="app")
assert app.parameters().defaults() == {"app": {"credit": {"approved": {"limit": 0.4}}}}
assert app.run(pl.DataFrame({"income": [1000.0], "total_debt": [100.0]}))["offer"].to_list() == [2000.0]
```

## Debugging, testing and introspection

```python
import polars as pl
from decider import Engine, flow, param
from decider.engine import step_map, to_ir
from decider.testing import assert_equivalent, corpus, no_recompile

def ratio(income: float, debt: float) -> float:
    return debt / income if income > 0 else 1.0

def approved(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit

pipeline = flow(ratio, approved, name="credit")
df = pl.DataFrame({"income": [1000.0, 800.0], "debt": [500.0, 100.0]})

s = pipeline.session(df)                 # a debug session: break, inspect, override, resume
s.break_at("credit/approved")
s.resume()                               # paused before credit/approved
assert s.value("ratio").to_list() == [0.5, 0.125]
s.set("ratio", 0.1)                      # what-if
s.resume()
assert s.output()["approved"].to_list() == [True, True]

out = assert_equivalent(pipeline, df)    # every mode; run == score per row == a session
assert out["approved"].to_list() == [False, True]
for frame in corpus(pipeline).values():  # zeros, negatives, empty and chunked frames
    assert_equivalent(pipeline, frame)

exe = Engine().bind(pipeline, mode="fused")
exe.run(df)
with no_recompile():                     # retuning a param never compiles
    exe.run(df, params={"credit": {"approved": {"limit": 0.9}}})

assert list(step_map(pipeline)) == ["credit/ratio", "credit/approved", "credit"]
ir = to_ir(pipeline)                     # the checked IR that Engine().bind runs
```

Sessions also `step()`, `step_into()` (branches, loops), `rewind(path)` and
break on a tree node (`"risk#high"`) or table row (`"band#2"`); see
`help(decider.engine.debug.Session)`. Test through decider
(`Engine().bind(build())` or the handler), not by calling the functions: only
that proves the pipeline wires, builds and serves.

## Common mistakes

- Request fields declared as `param()`: they ignore the request. Inputs are plain arguments.
- A step returning a `dict`: write one step per output, or `@step(outputs=(...))`.
- `from decider import RequestHandler`: it is `from decider.serving import RequestHandler`.
- Invented CLI flags (`--code-path`): settings are `DECIDER_*` variables or `.env`; see `decider --help`.
- A top-level `pipeline.py`: use `<package>/pipeline.py` and `DECIDER_API__PIPELINE=<package>.pipeline:build`.
- Tables, thresholds and weights in Python: put them in `configs/<version>/` documents or params.
- `date.today()`, `hash()` or `uuid4()` in a decision: pass the decision date in the request.
- Hand-written tree walkers, rule loops or if-chain scorecards: use `TreeConfig`, `DecisionTableConfig`, `ScorecardConfig`.
- Tests asserting `>= 0` or `in (1, 2, 3)`: assert the exact answer of a worked example.
- Reporting that a build works without running `decider build` and scoring `sample_request.json` over HTTP.
