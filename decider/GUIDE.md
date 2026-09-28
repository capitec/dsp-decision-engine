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
| rows a plain step loops over, retuned like a param | `param_table({"col": float}, default=[...])` |
| a `list[dict]` column, heavy per-item work in a kernel | `Columnar[Item]` (one array per field) |
| a `struct` column as one record per row, in a kernel | `Struct[Item]` |
| per-item rules over a list, as ordinary steps | `each(column, child_flow, name=)` |
| try every candidate, keep the best (best bundle, term, price) | `optimise(count, evaluate, score=, name=)` |
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

For one record, call `score(record)`, never `run(df)` on a one-row frame: `run`
builds a frame, exports it through Arrow and assembles an output frame, which
costs about five times what `score` does on the same record.

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
`trace_output` gives the row's whole path as node ids joined by `>`, root first
(e.g. `"1>2>4>912"`), so the decision record shows how it got there. `TreeConfig`
also takes a v3 node/edge tree:

```python
import polars as pl
from decider.steps.trees import TreeConfig

tree = TreeConfig.load({"type": "tree", "name": "risk", "trace_output": "risk_path", "tree": {
    "nodes": [{"id": "root", "data": {"type": "unary", "condition": {"op": ">", "feature": "ratio", "threshold": 0.7}}},
              {"id": "high", "data": {"type": "leaf", "result_idx": 0}}],
    "edges": [{"source": "root", "target": "high", "data": {"sourceIndex": 0}}],
    "output": {"data": [{"risk": 1}], "default": {"risk": 0}, "dtypes": [["risk", "Int64"]]}}})
assert tree.run(pl.DataFrame({"ratio": [0.9]}))["risk_path"].to_list()[0].split(">")[-1] == "high"
```

## Parameter tables

A table-valued param holds rows in the params document (rate ladders, fee
caps, band floors), retuned per call like any `param()`: editing the rows, or
how many there are, never rebuilds or recompiles; only changing the columns
does. A plain step declares one with `param_table({column: type}, default=[...])`
(or `required=True`, or `shared_key="name"` to read `shared.name`); columns
are `int`, `float` or `bool`. The step receives a namedtuple with one
read-only numpy array per column, the same in every mode, so loop over it:

```python
import polars as pl
from decider import Engine, Table, flow, param_table
from decider.testing import no_recompile

def rate(score: int, ladder: Table = param_table({"floor": int, "rate": float},
                                                  default=[{"floor": 0, "rate": 0.2}])) -> float:
    r = 0.0
    for i in range(len(ladder.floor)):
        if score >= ladder.floor[i]:
            r = ladder.rate[i]
    return r

pricing = flow(rate, name="pricing")
assert pricing.parameters() == {"pricing/rate": {"ladder": {
    "type": "table", "schema": {"floor": "int", "rate": "float"}, "default": [{"floor": 0, "rate": 0.2}]}}}
params = pricing.parameters().defaults()          # {"pricing": {"rate": {"ladder": [{"floor": 0, "rate": 0.2}]}}}
params["pricing"]["rate"]["ladder"] = [{"floor": 0, "rate": 0.2}, {"floor": 600, "rate": 0.1}]
df = pl.DataFrame({"score": [550, 720]})
assert pricing.run(df, params=params)["rate"].to_list() == [0.2, 0.1]     # interpreted

exe = Engine().bind(pricing, mode="fused")
assert exe.run(df, params=params)["rate"].to_list() == [0.2, 0.1]
with no_recompile():                              # more rows, new values: same kernel
    params = {"pricing": {"rate": {"ladder": [{"floor": 0, "rate": 0.25}, {"floor": 500, "rate": 0.15},
                                              {"floor": 700, "rate": 0.08}]}}}
    assert exe.run(df, params=params)["rate"].to_list() == [0.15, 0.08]
    assert exe.score({"score": 720}, params)["rate"] == 0.08
```

Rows are checked when the document arrives: a missing or undeclared column or
a value of the wrong type is a `ParamsError` naming the step, the param, the
row and the column. `defaults()` shows a required table as `[]`.

For a grid whose rows *match* records (bands, keys, string columns), use a
`DecisionTableConfig` instead: no loop to write, and it reads string columns.
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
assert price.parameters().defaults() == {"price": {"prices": []}}     # required: no default rows
assert price.parameters() == {"price": {"prices": {"type": "table", "schema": {"product": "String", "rate": "Float64"},
                                                   "required": True}}}

params = {"price": {"prices": [{"product": "loan", "rate": 0.12}, {"product": "card", "rate": 0.2}]}}
exe = Engine().bind(price, mode="fused")
df = pl.DataFrame({"product": ["loan", "card", "car"]})
assert exe.run(df, params=params)["rate"].to_list() == [0.12, 0.2, 0.0]
with no_recompile():
    params = {"price": {"prices": [{"product": "car", "rate": 0.1}]}}
    assert exe.run(df, params=params)["rate"].to_list() == [0.0, 0.0, 0.1]
```

`{"table": "prices", "shared": true}` reads `shared.prices`, one table for
several steps.

## A project: template, build, serve

```bash
decider template credit_risk && cd credit_risk
pytest -q                     # scores sample_request.json through the handler
decider build                 # validates configs/<latest>, warms every kernel with sample_request.json
decider serve --workers 1     # POST /invocations, GET /ping; needs decider[serve-starlette]
curl -s -d @sample_request.json -H 'content-type: application/json' localhost:8080/invocations
```

The template writes a project directory that is itself the package (two
projects on one path never shadow each other):

- `pipeline.py`: `build()` returns the pipeline; each argument of `build`
  receives the config document of that name.
- `inference.py`: `Handler(RequestHandler)`; override `input_fn`, `output_fn`,
  ... to change request handling.
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

## What compiles, and what falls back

**The plain type always works.** Annotate a step `str`, `date`, `list[dict]`,
`dict` or a TypedDict and it runs in every mode and gives the same answer. A
numba kernel holds only numbers (`float`, `int`, `bool`), so a step reading or
writing anything else runs on its own, row by row, in Python, and says so once
per step with a `FallbackWarning`. Only speed changes: such a step runs at
under 1M rows/s in a batch instead of a kernel's hundreds of millions, and
costs under 10 µs more on a single `score()`. Trees and decision tables read
their strings as byte spans inside their kernels, so a string rule set does not
pay this at all.

`@allow_fallback` says you accept that: the warning stops, and
`Engine(strict_compile=True)` — which otherwise raises on a step that runs in
Python — allows it. `fallbacks()` still reports it, marked `@allow_fallback`,
so it stays visible when you are chasing latency. Put it on a function a step
*calls* and the step falls back rather than compiling it: an intentional
Python boundary.

```python
import warnings

import polars as pl
from decider import Engine, FallbackWarning, allow_fallback, flow

def is_private(sector: str) -> float:        # a str never enters a kernel, so this runs in Python
    return 1.0 if sector == "private" else 0.0

@allow_fallback                              # ... and this one says so on purpose
def order_total(items: list[dict]) -> float:
    return sum(item["price"] for item in items)

pipeline = flow(is_private, order_total, name="p")
df = pl.DataFrame({"sector": ["private", "public"], "items": [[{"price": 2.0}], [{"price": 5.0}]]})

exe = Engine().bind(pipeline, mode="fused")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    out = exe.run(df)
assert out["is_private"].to_list() == [1.0, 0.0] and out["order_total"].to_list() == [2.0, 5.0]
assert len(caught) == 1 and issubclass(caught[0].category, FallbackWarning)
assert "p/is_private runs in Python, row by row" in str(caught[0].message)
assert exe.fallbacks() == {
    "p/is_private": "reads 'sector' as str, which no kernel holds",
    "p/order_total": "@allow_fallback: reads 'items' as list[dict], which no kernel takes",
}

strict = Engine(strict_compile=True).bind(flow(order_total, name="q"), mode="fused")
assert strict.score({"items": [{"price": 2.0}]})["order_total"] == 2.0
```

To silence the lot instead of step by step, filter the category once at
start-up: `warnings.filterwarnings("ignore", category=FallbackWarning)`.

`@allow_fallback` is for function steps. A tree, table, scorecard, branch or
loop falls back as a whole, so filter `FallbackWarning` for those instead;
putting the decorator on one raises `TypeError` rather than doing nothing.

For a batch big enough to care, `Raw[str]` compares strings as integer codes
inside the kernel (`==` and `!=` only) at 4-5x the throughput — but it is
*slower* on a single record, because the codes have to be built first. Since
`score()` is the usual workload, reach for it only when a batch is the point.

## Nested columns: `Struct[Item]` and `Columnar[Item]`

A list or struct column reaches a compiled kernel two ways, and reads the same
as its plain type in every mode:

- **`Struct[Item]`** — a `struct` column as one record per row. `Item` is a
  `TypedDict` naming the fields; the step reads `applicant["income"]`, compiled
  into the kernel.
- **`Columnar[Item]`** — a `list[dict]` column as one flat array per `Item`
  field, sliced per row. This is a columnar (struct-of-arrays) shape: `items.price`
  is one array of every item's `price`, read by position `items.price[j]`. It is
  *not* a list of per-item dicts — `for i in items` iterates the *fields*, not
  the items, so index each field by position. An `Item` field may be `float`,
  `int`, `bool`, `str`, or `float | None` / `str | None`; reach for it when the
  per-item work is heavy or the lists are long (it costs ~20 µs a `score()` call
  to build the arrays).

```python
from typing import TypedDict

import polars as pl
from decider import Columnar, Engine, flow

class Item(TypedDict):
    price: float
    qty: int

def total(items: Columnar[Item]) -> float:
    t = 0.0
    for j in range(len(items.price)):            # index by position, not `for i in items`
        t += items.price[j] * items.qty[j]
    return t

exe = Engine().bind(flow(total, name="p"), mode="fused")
df = pl.DataFrame({"items": [[{"price": 2.0, "qty": 3}, {"price": 5.0, "qty": 1}], []]})
assert exe.run(df)["total"].to_list() == [11.0, 0.0]
assert exe.fallbacks() == {}                     # compiled into the shared kernel
```

A field that is itself a list or struct is **nested** data, which neither shape
can hold: binding the step raises a `TypeError` that names the field. Read such
a column as its plain type instead — `list[dict]` for a list column, `dict` for
a struct column — which runs in Python and gives the same answers, and say so
with `@allow_fallback`.

### Per-item rules: `each`

`each(column, item, name=...)` runs a child flow on every element of a list
column and writes the enriched list back, so per-item rules are ordinary steps
with `missing_as` and `param` instead of polars expressions. The child's outputs
become new fields, which a later `Columnar[Item]` step reads in a kernel.

```python
from typing import TypedDict

import polars as pl
from decider import Columnar, Engine, each, flow, missing_as, param

def heavy(weight: float = missing_as(0.0), heavy_kg: float = param(20.0)) -> bool:
    return weight > heavy_kg

class Item(TypedDict):
    weight: float
    heavy: bool

def bundle_total(items: Columnar[Item]) -> float:
    t = 0.0
    for j in range(len(items.weight)):
        if items.heavy[j]:
            t += items.weight[j]
    return t

pipeline = flow(each("items", flow(heavy, name="item"), name="items"), bundle_total, name="order")
exe = Engine().bind(pipeline, mode="fused")
df = pl.DataFrame({"items": [[{"weight": 5.0}, {"weight": 25.0}], []]})
assert exe.run(df)["bundle_total"].to_list() == [25.0, 0.0]
assert pipeline.parameters().defaults() == {"order": {"items": {"item": {"heavy": {"heavy_kg": 20.0}}}}}
```

`execution_mode=EachMode.PER_ROW` (the default) runs the child once per parent
row in Python — fastest on a single `score()`, slower on a large batch.
`EachMode.BATCH` explodes the list into one child frame and runs the child once
over all items — faster on a large batch, slower on a single record. A null list
reads as a row with no items in both modes.

## Fan-out and reduce: `optimise`

"Try every candidate, keep the best" — best bundle, best term, best price point —
is `optimise`. It runs your `evaluate` flow once per candidate `index` (from 1 to
`count`) and keeps the one with the highest `score`. It lowers to a `loop`, so it
fuses into one kernel like any loop.

```python
from typing import TypedDict

import polars as pl
from decider import Columnar, flow, optimise, step

class Item(TypedDict):
    price: float
    weight: float

@step(output="count")
def bundles(items: Columnar[Item]) -> int:
    return (1 << len(items.price)) - 1

@step(output="score")
def bundle_total(index: int, items: Columnar[Item]) -> float:
    total = 0.0
    for j in range(len(items.price)):
        if (index >> j) & 1:
            total += items.price[j]
    return total

best = optimise(bundles, flow(bundle_total, name="evaluate"), max_candidates=1 << 10, name="best")
out = flow(best, name="order").run(pl.DataFrame({"items": [
    [{"price": 2.0, "weight": 1.0}, {"price": 5.0, "weight": 2.0}], []]}))
assert out["best_index"].to_list() == [3, -1] and out["best_score"].to_list() == [7.0, -1e300]
```

The winner is `best_index` (`-1` when nothing survives), its `best_score`, and
the counts `evaluated` / `disqualified`. `score=` names the evaluate output to
maximise (negate it to minimise); a tie keeps the earlier candidate. Pass a bool
`disqualify=` step to reject candidates after they are scored — e.g. a
post-pricing rule — and their count lands in `disqualified`. `max_candidates` is
the loop's bound; `count` per row may be less, and the loop stops there.

The winner's other evaluate outputs (the product, the rate) aren't carried
back yet: recompute them on the winner with
`evaluate.relabel(reads={"index": "best_index"})`, or wait for a struct output.

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

Say what a value measures with `Annotated` and the flow debugger shows it that
way (`R 4000.00`, `25%`, `36 months`); undeclared values are guessed from their
names. The engine runs the plain type.

```python
from typing import Annotated
import polars as pl
from decider import Duration, Money, Percent, flow, param

def instalment(loan: Annotated[float, Money()], term_months: Annotated[int, Duration("months")],
               rate: Annotated[float, Percent()] = param(0.25)) -> Annotated[float, Money()]:
    return loan * (1 + rate) / term_months

assert flow(instalment, name="loan").run(pl.DataFrame({"loan": [1200.0], "term_months": [12]}))["instalment"].to_list() == [125.0]
```

`Money(cents=True)` is an int of cents; subclass `decider.FieldMetadata` for another kind.

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
