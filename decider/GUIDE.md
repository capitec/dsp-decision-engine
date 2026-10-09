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
- **Modes**: `p.compile(mode)`: `"interpreted"` (plain Python; what
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
| apply logic only where a bool condition holds, else pass through | `branch(cond, if_true, modifies=[...], name=)` |
| iterate per row (a solve, a schedule) | `loop(cond, body, carries=[...], max_iterations=n, name=)` |
| whole-frame work (joins, group-bys, ragged lists) | `@frame_step(reads=, writes=)` |
| a decision tree, or a flat rule set | `TreeConfig` (`prioritized_flat_rule`, `mode: "all"`) |
| a grid (bands, rate cards) fixed in its config document | `DecisionTableConfig`, inline `"rows": [...]` |
| a grid retuned like a param (params document, per call) | `DecisionTableConfig`, `"rows": {"table": "name"}` |
| rows a plain step loops over, retuned like a param | `param_table({"col": float}, default=[...])` |
| string tests (`==`, `startswith`, `in`) kept in a kernel | `Text` (`decider.ops`) |
| a `list[dict]` column, heavy per-item work in a kernel | `Columnar[Item]` (one array per field) |
| a `struct` column as one record per row, in a kernel | `Struct[Item]` |
| per-item rules over a list, as ordinary steps | `each(column, child_flow, name=)` |
| try every candidate, keep the best (best bundle, term, price) | `optimise(count, evaluate, score=, name=)` |
| a points scorecard | `ScorecardConfig` |
| the same step twice, or on other column names | `.named("x")`, `.relabel(reads=, writes=)` |
| another project's pipeline | import its `build()`; put the step inside yours |
| structure in a YAML/JSON file, steps in Python | `load_flow`, `DECIDER_API__PIPELINE=flow.yaml` |
| immutable config versions | `decider.config.JsonFileStore` (`ConfigStore`) |
| why a record got its answer; what-if | `pipeline.session(df)` |
| modes, batch and single record agree | `decider.lab.testing.assert_equivalent` |

## Steps, params, run and score

```python
from datetime import date

import polars as pl
from decider import DeciderError, Executable, flow, missing_as, param
from decider.exceptions import ParamsError

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

pipeline.parameters().validate(params).check()   # a Validation; check() raises if the document is invalid
bad = {"credit": {"approved": {"limit": "high"}}}
assert pipeline.parameters().validate(bad).errors
try:
    pipeline.run(df, params=bad)
except DeciderError as error:        # the base of every error a user handles; ParamsError is one
    assert isinstance(error, ParamsError)

exe: Executable = pipeline.compile("fused")      # compile once, call many times
assert exe.score({"income": 1000.0, "debt": 350.0, "applied_on": date(2026, 1, 5)}, params)["approved"] is False
```

`run` and `score` treat a params document as immutable: the one you pass is
validated once and remembered by identity, so editing it in place and passing
it again keeps the old values, silently. Pass a new dict to change a value
(`copy.deepcopy(params)`, then edit the copy). In the example above the edit
comes before the first call, so it is seen.

An input with neither `missing_as(fill)` nor a `T | None` annotation is required:
a null value or an absent column raises `MissingInputError`. To default it for a
step you reuse as it is, put a step in front that writes the column
(`def income(income: float = missing_as(0.0)) -> float: return income`).

For one record, call `score(record)`, never `run(df)` on a one-row frame: `run`
builds a frame, exports it through Arrow and assembles an output frame, which
costs about five times what `score` does on the same record.

An `int` input or `-> int` output takes a whole float (`7.0`) but never truncates one: `7.5`, NaN
and infinity raise `NotIntegralError` naming the input or step, in `score` and `run`, interpreted
or compiled. Where you mean to drop the fraction, say so in the step (`int(x)`, `round(x)`).

### Rounding

`round(x, 2)` rounds the exact value of the double, half to even, like CPython.
For another mode, call `decimal.round` from `decider.ops`; it runs the same in
plain Python and inside a compiled step, and `Rounding` is `HALF_UP` (the
default, away from zero), `HALF_EVEN` or `DOWN` (toward zero).

```python
from decider import flow, param
from decider.ops import decimal
from decider.ops.decimal import Rounding

def instalment(amount: float, months: float = param(12.0)) -> float:
    return decimal.round(amount / months, 2, mode=Rounding.HALF_EVEN)

assert flow(instalment).compile("fused").score({"amount": 100.0})["instalment"] == 8.33
assert decimal.round(2.5) == 3.0 and decimal.round(2.5, mode=Rounding.HALF_EVEN) == 2.0
```

### Money and decimals

Money does not belong in a float (`0.1 + 0.2` is not `0.3`). Annotate it `Currency`, a
`Decimal` of four decimal places, or `Decimal[p, s]` for another scale (`p` is at most 19
digits: a Decimal is an int64 count of `10**-s`, so a `Currency` reaches 922 trillion).
A step reads and returns plain `Decimal` values, in every mode; a compiled step keeps the
int64, so a Decimal costs the same as an int. `Currency["R"]` sets display units.

```python
import polars as pl
from decimal import Decimal as StdDecimal
from decider import flow, param
from decider.ops import decimal
from decider.ops.decimal import Currency, Decimal, Rounding

def instalment(principal: Currency["R"], rate: float = param(0.1275), months: int = param(12)) -> Currency["R"]:
    return decimal.round(principal * rate / months, 2, mode=Rounding.HALF_EVEN)

def fee(instalment: Currency["R"], floor: Currency = param(Currency(5, 5000))) -> Currency["R"]:
    return max(instalment, floor) + Currency(1, 2500)       # Currency(major, minor): 1.25

exe = flow(instalment, fee, name="loan").compile("fused")
out = exe.score({"principal": "1000.50"})                   # a str, int, float, Decimal or decimal.Decimal
assert out["fee"] == Currency("11.88") and repr(out["fee"]) == "R 11.8800"
assert exe.score({"principal": 10}, {"loan": {"fee": {"floor": "20"}}})["fee"] == Currency("21.25")

frame = pl.DataFrame({"principal": [StdDecimal("1000.50"), StdDecimal("2.5")]}, schema={"principal": pl.Decimal(18, 2)})
result = exe.run(frame)                                      # a Decimal column in, Decimal(38, 4) out
assert result["fee"].dtype == pl.Decimal(38, 4) and result["fee"].to_list()[0] == StdDecimal("11.8800")
```

- `+ - * /` between decimals give the larger scale, rounded half-even once per operation
  (like polars); an int is whole units (`d + 1`), a float may multiply or divide (`d * 0.0175`,
  a Decimal at `d`'s scale), and `float / d`, `d + 1.5`, `//`, `%`, `**` and `round(d, n)` are
  errors: write `float(d)` for a ratio, `decimal.round(d, 2, mode=...)` to round (it keeps the
  scale). Comparisons are exact across scales and against ints; against a float they compare
  `float(d)`.
- A float rounds to the scale from its exact value (`Currency(2.675)` is 2.67), a str from its
  digits (`Currency("2.675")` is 2.68); `Currency(2, 6500)` is exact (`minor` in units of the
  scale, 0 to 9999). A Decimal output of a coarser scale rounds half-even, and an int return is
  whole units.
- A frame column is read at the annotation's scale (rescaled half-even; a value that does not
  fit int64 names its column); an integer column is whole units, a float column is refused.
  Two steps naming one value at different scales is a wiring error.
- A result outside int64 raises `OverflowError` in Python and wraps in a compiled step; the
  `int64_overflow` check (DCD101) flags the products that can reach it.

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

# One arm: false rows run nothing and keep their earlier `limit`.
only_eligible = branch(eligible, full_limit, modifies=["limit"], name="only_eligible")
out2 = only_eligible.run(pl.DataFrame({"age": [30, 16], "income": [1000.0, 800.0], "limit": [0.0, 500.0]}))
assert out2["limit"].to_list() == [3000.0, 500.0]

# With no earlier `limit` there is nothing to keep, so a skipped row's `limit` is null. A step after
# the branch reads it with `missing_as`, or it raises `MissingInputError`.
out3 = only_eligible.run(pl.DataFrame({"age": [30, 16], "income": [1000.0, 800.0]}))
assert out3["limit"].to_list() == [3000.0, None]
```

## Trees, rule sets, tables, scorecards

Each loads from a JSON document (`TreeConfig.load(doc)`, or
`ConfigurableStep.load(doc)` for any type). Any number in it can be
`{"param": "name", "default": x}`, which makes it a param. In a project,
`def build(fraud_rules)` receives `configs/<version>/fraud_rules.json` loaded.

```python
import polars as pl
from decider import DecisionTableConfig, ScorecardConfig, TreeConfig

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
(e.g. `"1>2>4>912"`), so the decision record shows how it got there. For a route
format of your own, `path_columns=fn` calls `fn(route, tree)` once per route when
the step is built and adds the dict it returns as output columns. `TreeConfig`
also takes a v3 node/edge tree:

```python
import polars as pl
from decider import TreeConfig

tree = TreeConfig.load({"type": "tree", "name": "risk", "trace_output": "risk_path", "tree": {
    "nodes": [{"id": "root", "data": {"type": "unary", "condition": {"op": ">", "feature": "ratio", "threshold": 0.7}}},
              {"id": "high", "data": {"type": "leaf", "result_idx": 0}}],
    "edges": [{"source": "root", "target": "high", "data": {"sourceIndex": 0}}],
    "output": {"data": [{"risk": 1}], "default": {"risk": 0}, "dtypes": [["risk", "Int64"]]}}})
assert tree.run(pl.DataFrame({"ratio": [0.9]}))["risk_path"].to_list()[0].split(">")[-1] == "high"
```

A decision table reads every numeric variable (a `between` input, or `eq` / `in` on a
number column) as a `float`, exact up to 2**53. An input column has one type, so a step
that reads the same column as `int` raises a `WiringError`. Declare the variable `int` in
`variable_types` and the table compares it as int64, exactly, and shares the `int` column:
its conditions must then use integer columns, and a step that writes it must write an
`int`. A null int is `-2**63`, so that value never matches. Otherwise add a step that writes
a float copy (`def amount_f(amount: int) -> float`) and point the table's `variable` at it.

```python
import polars as pl
from decider import DecisionTableConfig

by_channel = DecisionTableConfig.load({
    "type": "decision_table", "name": "by_channel", "variable_types": {"channel": "int"},
    "columns": {"code": "Int64", "pts": "Int64"}, "rows": [{"code": 2**60, "pts": 5}],
    "expression": {"type": "eq", "variable": "channel", "value_column": "code"}, "outputs": ["pts"], "default": [0]})
assert by_channel.run(pl.DataFrame({"channel": [2**60, 2**60 + 1]}))["pts"].to_list() == [5, 0]
```

A table has one band dimension with open (`None`) edges. A grid of two (amount by term)
gives every row both bounds, `-inf` and `inf` at the outer edges, and sets `allow_gaps`
on each `between`.

A step type of your own is a `ConfigurableStep` subclass whose `to_ir` calls
`ctx.call` on a plain function; `Value[float]` fields accept a param. Anything
past `ctx.call` (row kernels, packed tables) is the `decider.ir` kit
(`Input`, `Output`, `ParamDecl`, `NullPolicy`, `CallNode`, `IRContext`).

```python
from typing import Literal
import polars as pl
from decider import ConfigurableStep, Value, flow

def scale(x: float, factor: float) -> float:
    return x * factor

class Markup(ConfigurableStep):
    type: Literal["markup"] = "markup"
    column: str
    factor: Value[float] = 1.0

    def to_ir(self, ctx):
        return ctx.call(self, scale, inputs={"x": self.column}, values={"factor": self.factor})

step = ConfigurableStep.load({"type": "markup", "name": "doubled", "column": "income", "factor": 2.0})
assert flow(step, name="p").run(pl.DataFrame({"income": [10.0]}))["doubled"].to_list() == [20.0]
```

### A step type that is data plus one function: `CustomFunction`

When a step's data is a few fields and its logic is a pure function, subclass
`CustomFunction` instead of writing `to_ir`. Use it instead of a function that
reads a module-level object or a closure: those are invisible in the config,
can't be compiled and can't be diffed, while fields are all three. `impl` is a
`staticmethod` (no `self`, so numba compiles it as a plain function), and each
of its arguments has a field of the same name. A `str` field is the column to
read; a `Value[T]` field is a value that a param can also set. The step's
`name` is the column it writes. For several outputs, annotate `impl` with a
`tuple[...]` return and set `outputs=["a", "b"]`.

```python
from typing import Literal
import polars as pl
from decider import ConfigurableStep, CustomFunction, Value, flow

class Cap(CustomFunction):
    type: Literal["cap"] = "cap"
    x: str
    limit: Value[float] = 100.0

    @staticmethod
    def impl(x: float, limit: float) -> float:
        return min(x, limit)

cap = ConfigurableStep.load({"type": "cap", "name": "capped", "x": "amount", "limit": 60.0})
out = flow(cap, name="p").run(pl.DataFrame({"amount": [50.0, 150.0]}))
assert out["capped"].to_list() == [50.0, 60.0]
```

A mistake in the class (no `impl`, an `impl` argument without a field) raises
when the class is defined. A field other than `str` or `Value[T]` is rejected:
rows belong in a `param_table`, and anything else in a `ConfigurableStep` with
its own `to_ir`. `impl` cannot call other steps or read the other fields.

### Shipping a step type in a package

To make `"markup"` load without the user importing your package, declare it
as an entry point in the package's `pyproject.toml`, named by its tag:

```toml
[project.entry-points."decider.steps"]
markup = "credit_risk.steps:Markup"
```

`ConfigurableStep.load({"type": "markup", ...})` imports the package on the
first lookup. Only a registered `ConfigurableStep` is ever returned. Other
roots use `decider.config_stores`, `decider.metrics`, `decider.plots`,
`decider.blocks` and `decider.datasets`.

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
from decider import ParamTable, flow, param_table
from decider.lab.testing import no_recompile

def rate(score: int, ladder: ParamTable = param_table({"floor": int, "rate": float},
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

exe = pricing.compile("fused")
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

from decider import DecisionTableConfig
from decider.lab.testing import no_recompile

price = DecisionTableConfig.load({"type": "decision_table", "name": "price",
    "columns": {"product": "String", "rate": "Float64"}, "rows": {"table": "prices"},
    "expression": {"type": "eq", "variable": "product", "value_column": "product"},
    "outputs": ["rate"], "default": [0.0]})
assert price.parameters().defaults() == {"price": {"prices": []}}     # required: no default rows
assert price.parameters() == {"price": {"prices": {"type": "table", "schema": {"product": "String", "rate": "Float64"},
                                                   "required": True}}}

params = {"price": {"prices": [{"product": "loan", "rate": 0.12}, {"product": "card", "rate": 0.2}]}}
exe = price.compile("fused")
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

JSON dates (`"2026-01-15"`) arrive as the declared `date` inputs, but only through
the handler: `Executable.score` and `run` take typed values (`date` objects), and a
string date reaches the step as a string. In-process, `handler.score` takes the
parsed record (or a frame) and converts dates exactly as `decider serve` does;
`handler.process_fn` takes raw bytes. Both are plain calls, and the `*_fn` hooks are
synchronous: an `async def` override is rejected when the class is created.

```python
import json
from decider.config import JsonFileStore
from decider.serving import RequestHandler

handler = RequestHandler(JsonFileStore(basepath="configs"), "credit_risk.pipeline:build")
handler.stage()
handler.activate()
record = {"income": 1000, "debt": 250, "applied_on": "2026-01-15"}
assert handler.score(record)["approved"] is True
body = json.dumps(record).encode()
response = handler.process_fn(body, "application/json", "application/json")
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

## A pipeline from a flow file

`load_flow` builds the same `Step` tree from a YAML or JSON
file instead of Python: the structure is reviewable and diffable on its own,
while the steps stay plain Python functions.

```
myapp/
  flow.yaml              # the root: {flow, components, steps}
  flows/
    screen.yaml           # a `flow: screen` node with no `steps:` looks here
  components/
    __init__.py           # plain functions and Step instances, by name
  configs/
    0.0.0/
      params.json
      limit_band.json      # a scorecard/table/tree document; step: limit_band loads it
```

```yaml
flow: credit
components: myapp.components
steps:
  - flow: screen                                 # flows/screen.yaml
  - branch: affordable
    when: {expr: "score >= 600 and crc_model == \"CrC3.0\""}
    modifies: [limit]
    then: [{step: limit_band}]                   # a decision_table document writing limit
  - loop: payoff
    while: {expr: "balance > 0"}
    carries: [balance]
    max: 10
    steps: [{step: pay}]
```

```python
import json
import tempfile
import types
from pathlib import Path

import polars as pl

from decider import load_flow

def eligible(score: float) -> bool:
    return score > 50.0

def approved(eligible: bool) -> bool:
    return eligible

components = types.ModuleType(__name__)   # stands in for a `components` package
components.eligible = eligible
components.approved = approved

project = Path(tempfile.mkdtemp())
(project / "flow.json").write_text(json.dumps(
    {"flow": "credit", "steps": [{"step": "eligible"}, {"step": "approved"}]}))

pipeline = load_flow(project / "flow.json", components=components)
df = pl.DataFrame({"score": [60.0, 10.0]})
assert pipeline.run(df)["approved"].to_list() == [True, False]
```

A `.yaml`/`.yml` file loads the same way (`pip install 'decider[yaml]'` for
the YAML parser); serve one with `DECIDER_API__PIPELINE=flow.yaml`, exactly
like `DECIDER_API__PIPELINE=myapp.pipeline:build`.

- `modifies` on a `branch` and `carries` on a `loop` are required, as they are
  in Python.
- an omitted `else` passes false rows through unchanged.
- `dag: name` (with `steps`, optional `inputs`), `each: name` (with `column`,
  `steps`, optional `output`) and `optimise: name` (with `count`, `evaluate`,
  `max_candidates`, optional `score` and `disqualify`, each a list of nodes) take
  the arguments of the Python constructors; `optimise` has no `record`.
- a name defined in both `components` and `documents` is an error.
- `path_columns: fn` on a `step:` node naming a tree sets its `path_columns` to
  the function `fn` in `components`.
- an `expr` condition is false wherever an input it reads is null.
- flow files live with the code (reviewed and deployed with it); params and
  documents stay versioned in `configs/<version>/`.

## What compiles, and what falls back

**The plain type always works.** Annotate a step `str`, `date`, `list[dict]`,
`dict` or a TypedDict and it runs in every mode and gives the same answer. A
numba kernel holds only numbers (`float`, `int`, `bool`, `Decimal`) and `Text`, so a step
reading or writing anything else runs on its own, row by row, in Python, and says so once
per step with a `FallbackWarning`. Only speed changes: such a step runs at
under 1M rows/s in a batch instead of a kernel's hundreds of millions, and
costs under 10 µs more on a single `score()`. Trees and decision tables read
their strings as byte spans inside their kernels, so a string rule set does not
pay this at all.

A `dict`, TypedDict, `list[...]`, `Struct` or `Columnar` output is held to the type you declare. No
kernel stores a dict, so a step returning one runs in Python and is checked on every call with strict
pydantic validation (`@allow_fallback` steps too): `{"v": "high"}` for `v: float` raises a `ValueError`
naming the step, row, field and value. An int is a valid float, a str or bool is not. A `Struct` or
`Columnar` step returning tuples that can't be the declared record (wrong arity, a float for an `int`
field) is caught when it compiles and falls back with that reason.

`score` returns each value as the step made it. A plain `dict` or `list[dict]` output declares no
shape, so a frame from `run` takes one type per field: an int beside a float comes back as a
float, a number beside a string raises, and non-string keys of a `dict` become strings, so a
JSON, CSV or Arrow response can carry it. Declare a TypedDict to have the fields checked at the
step instead.

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

exe = pipeline.compile("fused")
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

A compiled step that calls another function of yours must say how it compiles:
`@helper(signatures=[((float,), float)])` compiles the callee with the step,
whether it is imported by name or called as `module.fn(...)`, and a plain
function makes the step fall back with `calls 'fn' without @helper or
@allow_fallback` in `fallbacks()`.

To silence the lot instead of step by step, filter the category once at
start-up: `warnings.filterwarnings("ignore", category=FallbackWarning)`.

`@allow_fallback` is for function steps. A tree, table, scorecard, branch or
loop falls back as a whole, so filter `FallbackWarning` for those instead;
putting the decorator on one raises `TypeError` rather than doing nothing.

## Strings: `str` and `Text`

A `str` input runs its step in Python. Annotate it `Text` instead and the step
stays in the kernel, reading the string where the frame holds it: `==`, `!=`,
`len`, `in`, `startswith` and `endswith` compile and answer as they would on a
`str`, against a literal, a module-level constant or a `str` `param()`. Anything
else a `str` can do (`.upper()`, slicing, f-strings) needs a `str` input. A null
`Text` equals nothing; declare it `Text | None` to accept one.

```python
import polars as pl
from decider import flow, param
from decider.ops import Text

def government(employer: Text, prefix: str = param("GOV")) -> bool:
    return employer.startswith(prefix) or "DEPT" in employer

def label(employer: str) -> str:                            # str: runs in Python
    return employer.title()

exe = flow(government, label, name="emp").compile("fused")
out = exe.run(pl.DataFrame({"employer": ["GOV HEALTH", "acme", "TRANSPORT DEPT"]}))
assert out["government"].to_list() == [True, False, True]
assert set(exe.fallbacks()) == {"emp/label"}
```

One column can be read as `str` by one step and as `Text` by another. A step
*returning* `Text` writes a `str` column, in Python, as a `str` output does.

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
from decider import flow
from decider.ops import Columnar

class Item(TypedDict):
    price: float
    qty: int

def total(items: Columnar[Item]) -> float:
    t = 0.0
    for j in range(len(items.price)):            # index by position, not `for i in items`
        t += items.price[j] * items.qty[j]
    return t

exe = flow(total, name="p").compile("fused")
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
from decider import each, flow, missing_as, param
from decider.ops import Columnar

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
exe = pipeline.compile("fused")
df = pl.DataFrame({"items": [[{"weight": 5.0}, {"weight": 25.0}], []]})
assert exe.run(df)["bundle_total"].to_list() == [25.0, 0.0]
assert pipeline.parameters().defaults() == {"order": {"items": {"heavy_kg": 20.0}}}
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
from decider import flow, optimise, step
from decider.ops import Columnar

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

To get the winner's other fields back (the product, the rate), pass `record=Item`
and have `evaluate` also write `record` as a `Struct[Item]`; or recompute them
on the winner with `evaluate.relabel(reads={"index": "best_index"})`.

## Debugging, testing and introspection

```python
import polars as pl
from decider import Engine, describe, flow, param
from decider.ir import step_map, to_ir
from decider.lab.testing import assert_equivalent, corpus, no_recompile

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

exe = pipeline.compile("fused")
exe.run(df)
with no_recompile():                     # retuning a param never compiles
    exe.run(df, params={"credit": {"approved": {"limit": 0.9}}})

assert list(step_map(pipeline)) == ["credit/ratio", "credit/approved", "credit"]
ir = to_ir(pipeline)                     # the checked IR that Engine().bind runs
assert {n.path for n in describe(pipeline).nodes} >= {"credit/ratio", "credit/approved"}   # nodes, edges, value slots
```

Say what a value measures and the flow debugger shows it that way (`R 4000.00`,
`25%`, `36 months`); undeclared values are guessed from their names. An amount
is a `Currency["R"]` (see Money and decimals); `Percent()` and `Duration()` go in
`Annotated`, and the engine runs the plain type.

```python
from typing import Annotated
from decider import flow, param
from decider.ops import Currency, Duration, Percent

def instalment(loan: Currency["R"], term_months: Annotated[int, Duration("months")],
               rate: Annotated[float, Percent()] = param(0.25)) -> Currency["R"]:
    return loan * (1 + rate) / term_months

assert flow(instalment, name="loan").compile("fused").score({"loan": 1200, "term_months": 12})["instalment"] == Currency(125)
```

Subclass `decider.ops.FieldMetadata` for another kind.

Sessions also `step()`, `step_into()` (branches, loops), `rewind(path)` and
break on a tree node (`"risk#high"`) or table row (`"band#2"`); see
`help(decider.debug.Session)`. `s.events` holds the `decider.debug` events
(`Paused`, `NodeFinished`, ...). Test through decider
(`Engine().bind(build())` or the handler), not by calling the functions: only
that proves the pipeline wires, builds and serves.

## Durable ids, tracing, checks, experiments and MCP

These compose over the same pipeline. Each is opt-in or additive; none changes
a pipeline's answers.

### Durable ids

`decider ids [PATH]` inserts an opaque `id="0123abcdef45"` into every step and
named flow that lacks one, and changes no other byte. It is a deliberate,
reviewable source edit: it refuses to run unless every affected file is
git-tracked and the tree is clean, and it aborts (before writing) on a syntax
error, malformed or duplicate id. It is a command only: there is no
Python API for it.

Ids are additive: with `id=None` (the default) name, path, `step_map` and
execution are unchanged. A project that never runs the command works exactly as
before — identity falls back to the derived `path`/`source`. Commit ids where a
trace, comparison or experiment needs a reference that survives rename, extract
or reorder.

### Decision tracing

A run captures compact decision evidence only when you pass a `TraceSink`
(off by default). Each event resolves to a step's durable `Origin` with its
branch-arm / loop-iteration context and record index; batch capture can be
bounded with `max_events` or sampled with `sample_rate`, and loss is reported in
`sink.status()`, never silent.

```python
import polars as pl
from decider import flow, param
from decider.trace import Kind, TraceSink

def ratio(income: float, debt: float) -> float:
    return debt / income

def approved(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit

pipeline = flow(ratio, approved, name="credit")
df = pl.DataFrame({"income": [1000.0], "debt": [200.0]})
sink = TraceSink()
pipeline.compile("fused").run(df, trace=sink)
assert [e.origin.path for e in sink.events() if e.kind is Kind.STEP] == ["credit/ratio", "credit/approved"]
assert sink.status()["dropped"] == 0
```

`decider` emits evidence; it does not retain, redact or deliver it. Hand the
events to an `Adapter` (`TraceSink(adapter)`) to transform, enqueue or export
them — that adapter owns retention and privacy policy.

### Checks

`decider.lab.checks.run(pipeline)` runs `checks.standard` (wall-clock reads,
shared-state writes, numeric risk, missing durable ids, unknown suppressions)
and returns a `Report` a CI job, notebook or editor can render. Every finding has
a stable `code` (`DCD401`), the check's `check` alias, the contract `step`
reference, `file` and `line`, and a one-line `fix`.

```python
import polars as pl
from decider import flow
from decider.lab import checks

def ratio(income: float, debt: float) -> float:
    return debt / income

pipeline = flow(ratio, name="credit")
report = checks.run(pipeline)                # checks.standard, or the project's checks.yaml
assert report.ok is True                     # warnings only; the client decides what blocks a release
assert any(f.code == "DCD401" for f in report.findings)
assert report.counts[checks.Severity.WARNING] >= 1
print(report.to_text())
report.model_dump_json()                     # machine-readable; report.to_sarif() for code scanning
```

Add checks with a flat list of check instances, aliases or sets. Checks that
need data (`needs_data`) are listed under `report.skipped` until you pass
`data=`, and the rest of the catalogue is `checks.catalogue()`.

```python
import polars as pl
from typing import Literal
from decider import flow
from decider.lab import checks
from decider.lab.checks import Check

def ratio(income: float, debt: float) -> float:
    return debt / income

class MinRows(Check):
    type: Literal["min_rows"] = "min_rows"
    code = "ACME003"
    needs_data = True
    rows: int = 100

    def run(self, ctx):
        if ctx.frame.height < self.rows:
            yield ctx.finding(ctx.ir, f"only {ctx.frame.height} rows", fix="use a bigger sample")

pipeline = flow(ratio, name="credit")
assert checks.run(pipeline, [MinRows()]).skipped[0].code == "ACME003"
frame = pl.DataFrame({"income": [1000.0], "debt": [350.0]})
assert checks.run(pipeline, [checks.standard, MinRows()], data=frame).findings[-1].code == "ACME003"
bad_doc = {"credit": {"ratioo": {"k": 2}}}                  # a typo in a params document
assert checks.run(pipeline, ["invalid_params_doc"], params=bad_doc).findings[0].code == "DCD304"
```

A custom check is a `Check` subclass like `MinRows` (give it a `type` alias and a
`code` with your own prefix; config names it by alias, or `"module:Class"`),
or a function decorated for a one-off rule. An installed package can ship
checks under the `decider.checks` entry-point group.

```python
from decider import flow, step
from decider.lab import checks

@checks.check(code="ACME002", severity="info")
def single_step(ctx):
    if len(ctx.calls()) == 1:
        yield ctx.finding(ctx.ir, "a one-step flow")

@step(output="y")
def double(x: float) -> float:
    return x * 2.0

assert checks.run(flow(double, name="d"), [single_step]).findings[0].code == "ACME002"
```

A step names the checks it suppresses (aliases or codes); `flow`, `dag`,
`branch`, `loop`, `each` and `optimise` take the same keyword, as does a
`suppress_checks:` key in a flow file or a `ConfigurableStep` field. A finding
inside a suppressing step or any step above it stays in the report, marked
`suppressed`, and doesn't count against `ok`. A name that matches no check is
reported as `DCD403`.

```python
from decider import flow, step
from decider.lab import checks

@step(output="cents", suppress_checks=["durable_ids"])      # or the code, "DCD401"
def capped(cents: int) -> int:
    return cents * 1000

report = checks.run(flow(capped, name="loan"), ["durable_ids"])
step_finding = next(f for f in report.findings if f.step.path == "loan/capped")
assert step_finding.suppressed.name == "durable_ids"
assert sum(f.suppressed is None for f in report.findings) == 1     # the flow's own missing id
```

The project's check policy is `checks.yaml` (or `checks.json`) beside `configs/`.
It sets which checks run, their severities and options, what is disabled, and
the default gate for `decider check`:

```yaml
extends: standard
checks:
  - {type: durable_ids, severity: error}
  - {type: "acme.checks:NoFloatMoney", allow: [fee]}
disable: [numeric]                  # alias or code
fail_on: error
```

```bash
decider check credit.pipeline:build [--config checks.yaml] [--data loans|file.parquet] \
    [--format text|json|sarif] [--fail-on error|warning] [--select DCD1] [--list]
decider check credit.pipeline:build --format sarif > checks.sarif   # exits 1 when the gate fails
```

A `ConfigurableStep` can report on its own document, which the source checks
can't see, by overriding `check(self, ctx)`; `checks.run` calls it for each
such step.

### Comparing versions

`decider.lab.diff` compares two versions of a pipeline. `structure` says what changed
without running anything (steps added, removed, renamed or changed, params,
wiring), `outputs` says which rows and columns change over a frame and by how
much, and `prove` returns `identical=True` only when both versions have the
same IR content hash, or, with the `decider[prove]` extra, when the only changed
steps are trees and decision tables that z3 shows agree on every input
(`tier="smt"`). It never claims it from sampled rows: otherwise `identical` is
`None`, with the structure diff as evidence and a `reason`. A step z3 finds
different comes back as `identical=None` with the input row in
`evidence.counterexample`: it judges the step alone, so the flow around it may
never produce that row.

```python
import polars as pl
from decider import flow, step
from decider.lab import diff

def ratio(income: float, debt: float) -> float:
    return debt / income

@step(output="ratio")
def ratio_fee(income: float, debt: float) -> float:
    return (debt + 50.0) / income

old = flow(ratio, name="credit")
new = flow(ratio_fee, name="credit")
frame = pl.DataFrame({"id": ["a", "b"], "income": [1000.0, 500.0], "debt": [350.0, 400.0]})

s = diff.structure(old, new)
assert not s.identical and s.added and s.removed
o = diff.outputs(old, new, frame, key="id")
assert o.rows_changed == 2 and o.columns["ratio"].max_abs > 0
assert o.table().columns == ["id", "column", "old", "new", "delta"]
renamed = flow(step(output="ratio", id="0123abcdef45", name="debt_to_income")(ratio), name="credit")
tagged = flow(step(output="ratio", id="0123abcdef45")(ratio), name="credit")
assert [(m.old.name, m.new.name) for m in diff.structure(tagged, renamed).renamed] == [("ratio", "debt_to_income")]   # steps match by durable id
assert diff.prove(old, flow(ratio, name="credit")).identical is True
assert diff.prove(old, new).identical is None

try:
    import z3  # noqa: F401
except ImportError:
    pass
else:
    from decider.steps.trees import TreeConfig

    def bands(limit):
        return TreeConfig.load({"type": "tree", "name": "band", "tree": {"nodes": [
            {"id": "root", "data": {"type": "unary", "condition": {"op": ">", "feature": "debt", "threshold": limit}}},
            {"id": "high", "data": {"type": "leaf", "result_idx": 0}}],
            "edges": [{"source": "root", "target": "high", "data": {"sourceIndex": 0}}],
            "output": {"data": [{"risky": 1}], "default": {"risky": 0}, "dtypes": [["risky", "Int64"]]}}})

    proof = diff.prove(bands(300.0), bands(350.0))
    assert (proof.tier, proof.identical) == ("smt", None)
    assert 300.0 < proof.evidence.counterexample["debt"] <= 350.0
```

Between git revisions, each loads in its own process:
`diff.revisions("main", "HEAD", entry="credit.pipeline:build").outputs(frame)`, or
`decider diff main HEAD credit.pipeline:build --data loans [--key id] [--format json]`.
The command exits 1 when anything differs.

### Experiments

An experiment runs the unchanged flow (the baseline) and each scenario —
other params, an overridden value, another git revision — over the same
records, one batch at a time, and measures them with metrics. The runner only
executes; a `Metric` collects through hooks (`on_batch_end` for the flow's
output, `on_node_end` for what nodes matching its `nodes` glob wrote) and
`summarise`s into a typed `Summary`, which a `Plot` requiring that type draws
with altair (`pip install 'decider[experiments]'`). The definition saves as
`experiment.json` (`Experiment.save` / `Experiment.load`); subclass `Metric`,
`Plot` (`from decider.lab.experiments.plots import Plot, require_altair`),
`Callback` or a report `Block` for your own. A metric reads the flow's
output, or steps by `nodes="<glob>"` or a durable `node_id`. `.report(...)`
lays out blocks (`Overview`, `Text`, `Table`, `Chart`, `Row`, `Section`) that
`result.show()` draws in a notebook and `result.save_html(path)` writes as a
page; without one it shows an overview and a section per metric. Metrics are in
`decider.lab.experiments.metrics`, plots in `.plots`, report blocks in `.report`
and callbacks in `.callbacks`.

```python
import polars as pl
from decider import flow, param
from decider.lab.experiments import Experiment, RunStatus
from decider.lab.experiments.metrics import Divergence, ValueCounts
from decider.lab.experiments.plots import Changes

def ratio(income: float, debt: float) -> float:
    return debt / income

def approved(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit

pipeline = flow(ratio, approved, name="credit")
df = pl.DataFrame({"income": [1000.0, 500.0], "debt": [350.0, 400.0]})
result = (Experiment("drift", pipeline, df)
          .scenario("limit_0.3", params={"credit": {"approved": {"limit": 0.3}}})
          .metric(Divergence(column="approved"), ValueCounts(name="ratios", column="ratio", nodes="credit/ratio"))
          .plot(Changes(metric="divergence"))
          .run())
assert result.status is RunStatus.COMPLETED
assert result.summaries["divergence"]["limit_0.3"].first == 0
assert result.plots["divergence"].grammar == "vega-lite"   # .show() draws it in a notebook
```

Run a saved `experiment.json` headlessly, from CI or a script; `input.path`
is relative to the file. `start()` runs it in a background thread and returns a
handle with `.status`, `.progress`, `.cancel()` and `.result()`.

```python
import polars as pl
from datetime import date
from pathlib import Path
from decider.lab.experiments import Experiment, ExperimentDef, RunStatus, run

pl.DataFrame({"income": [1000.0, 500.0], "debt": [350.0, 400.0],
              "applied_on": [date(2026, 1, 15)] * 2}).write_parquet("loans.parquet")
def_ = ExperimentDef(name="drift", flow={"entry": "credit_risk.pipeline:build"}, input={"path": "loans.parquet"},
                     scenarios=[{"name": "tight", "params": {"credit_risk": {"approved": {"limit": 0.2}}}}],
                     metrics=[{"type": "divergence", "column": "approved"}])
Path("experiment.json").write_text(def_.model_dump_json())

assert run("experiment.json", out_dir="results").status is RunStatus.COMPLETED
handle = Experiment.from_def(def_).start()
assert handle.result().status is RunStatus.COMPLETED and handle.progress == 1.0
```

```bash
decider experiment run experiment.json --out results   # exit 0 if completed, 1 otherwise
```

Nondeterminism is a first-class result (`NON_REPRODUCIBLE`), never silent; a
partial or cancelled run never looks like a completed one. A metric that reads
nodes runs the flow one compiled kernel per node instead of fused, still
compiled. Results write to a caller-chosen `out_dir`; `decider` does not own
their storage.

### Data loading

`decider.lab.data` reads Redshift-export JSON/CSV/Parquet (`read` for a frame,
`scan` for a lazy one); `checks.run(..., ["missing_required"], data=frame)` flags required
inputs the data lacks before anything executes.

```python
from decider import flow
from decider.lab import checks
from decider.lab.data import read

def ratio(income: float, debt: float) -> float:
    return debt / income

pipeline = flow(ratio, name="credit")
df = read(b'{"income": 1000.0, "debt": 200.0}\n{"income": 500.0, "debt": 400.0}\n', format="json")
assert df.height == 2
assert checks.run(pipeline, ["missing_required"], data=df).ok
```

### Datasets

`decider.lab.data.datasets()` lists the project's named datasets, read from
`datasets/` beside `configs/`. `name.csv|parquet|jsonl|ndjson|json` is a `file`
dataset; a `name/` folder of part files of one format is a `folder` dataset
(`region=za/` subfolders become a `region` column; mixed formats are an error).

```python
import polars as pl
from pathlib import Path
from decider.lab.data import datasets

Path("datasets/loans").mkdir(parents=True)
pl.DataFrame({"client_id": [1, 2], "income": [1000.0, 500.0]}).write_parquet("datasets/clients.parquet")
Path("datasets/loans/part-0.csv").write_text("client_id,debt\n1,350\n")
Path("datasets/loans/part-1.csv").write_text("client_id,debt\n2,400\n")
Path("datasets/config.json").write_text('{"loans": {"type": "folder", "path": "loans", "key": ["client_id"]}}')

found = datasets()
assert {n: p.type for n, p in found.items()} == {"clients": "file", "loans": "folder"}
loaded = found["loans"].load()
assert loaded.dataset == "loans" and loaded.suggested_key == ("client_id",)
assert loaded.frame.height == 2
```

`datasets/config.json` maps `name` to a provider document and replaces an
inferred dataset of that name; relative `path`s resolve against `datasets/`.
`key` is the record key: `load()` rejects null or duplicate keys. Other
members: `scan()` (lazy), `schema()`, `head(n=50)`, `fingerprint(frame)`.

For another source, subclass `DatasetProvider` and implement `scan()`.
Name it in `config.json` by its `type`, or by `"package.module:Class"`, which
is imported on demand. Credentials are the provider's business.

```python
import sqlite3
import typing as t
import polars as pl
from pathlib import Path
from decider.lab.data import DatasetProvider, datasets

class SqliteTable(DatasetProvider):
    type: t.Literal["sqlite"] = "sqlite"
    database: str
    table: str

    def scan(self) -> pl.LazyFrame:
        return pl.read_database(f"select * from {self.table}", sqlite3.connect(self.database)).lazy()

Path("datasets").mkdir()
with sqlite3.connect("clients.db") as db:
    db.execute("create table clients (client_id integer, income real)")
    db.execute("insert into clients values (1, 1000.0), (2, 500.0)")
Path("datasets/config.json").write_text(
    '{"clients": {"type": "sqlite", "database": "clients.db", "table": "clients", "key": ["client_id"]}}'
)

loaded = datasets()["clients"].load()
assert loaded.frame.height == 2 and loaded.suggested_key == ("client_id",)
```

Set a different folder with `DECIDER_DATASETS_PATH`; the default is
`${PROJECT_ROOT}/datasets`. In a `.env`, `decider` expands `${...}` itself
(python-dotenv), even in single quotes, so `${PROJECT_ROOT}` becomes empty:
use an absolute path, or a relative one (relative to the cwd).

An experiment can take `input={"dataset": "<name>"}` instead of `path`. The
VS Code extension's Datasets view lists, previews and debugs a flow on one.

### MCP

`decider mcp` (needs `pip install 'decider[mcp]'`) serves a FastMCP server over stdio for an agent: structural and
summarised reads are always available; per-record values and trace events are
opt-in (`decider mcp --raw`); running code and persisting are confirmation-gated.
For Claude Code, the repository's plugin
(`/plugin marketplace add capitec/dsp-decision-engine`, then
`/plugin install decider@decider`) installs this server together with a `decider`
skill that teaches an agent the project layout and best practices.

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
