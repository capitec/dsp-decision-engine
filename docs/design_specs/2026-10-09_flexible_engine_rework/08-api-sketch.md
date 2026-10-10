# API sketch

What users write before and after. Everything after is a proposal; names may change in review. The
examples that survive become `GUIDE.md` sections and so run in `tests/test_guide.py`.

## Running a pipeline

Before:

```python
pipeline.run(df)                                   # interpreted
exe = pipeline.compile("fused")
exe.run(df)
exe.score({"income": 1000.0, "debt": 200.0})
```

After:

```python
pipeline.run(df)                                   # the default engine: fused unless set otherwise
pipeline.run({"income": 1000.0, "debt": 200.0})    # a dict: the record path, dict out
pipeline.run(df.lazy())                            # a LazyFrame: the polars engine, LazyFrame out

exe = pipeline.compile("fused")                    # or compile(FusedEngine(strict_compile=True))
exe.run(df)
exe.run({"income": 1000.0, "debt": 200.0})         # the former score(); same cost
```

## Choosing an engine

```python
from decikit import FusedEngine, PolarsEngine, engine

pipeline.run(df, engine="interpreted")
pipeline.run(df.lazy(), engine=PolarsEngine(segments=FusedEngine(strict_compile=True)))

with engine("interpreted"):                        # the default for run and compile in this context
    pipeline.run(df)                               # instant; what you see is what runs

pipeline.session(df)                               # a session opens interpreted whatever the default
pipeline.session(df, engine="fused")               # pause at kernel boundaries, as production runs
```

Settings for serving (`DECIKIT_API__ENGINE__*`, or `decikit serve --engine fused`):

```toml
[api.engine]
type = "fused"
strict_compile = true
params_validation = "lazy"
```

## Streaming a large batch

```python
import polars as pl

out = pipeline.run(pl.scan_parquet("portfolio/*.parquet"))   # a LazyFrame
out.sink_parquet("scored.parquet")                            # streams; each batch runs the fused pipeline

exe = pipeline.compile("polars")
exe.segments
# [(("p/ratio", "p/affordable"), True),            streamable: runs per batch
#  (("p/join_bureau",), False),                    a frame step: runs once on the whole frame
#  (("p/flag",), True)]
exe.fallbacks()                                    # the numba fallbacks inside the segments, by step path
```

## Equivalence

```python
from decikit.lab.testing import assert_equivalent

assert_equivalent(pipeline, df)                               # built-in numba engines and polars
assert_equivalent(pipeline, df, engines=("fused", "polars"))
```

## A polars body for a step (only if phase P6 is built)

The Python body is required and is what serving runs. A polars body is an optional optimisation that only
the polars engine's native segments use.

```python
from datetime import date
import polars as pl
from decikit import flow, impl, step

@step
def tenure_months(start: date, as_of: date) -> int:
    return (as_of.year - start.year) * 12 + as_of.month - start.month

@tenure_months.impl("polars")
def _(start: pl.Expr, as_of: pl.Expr) -> pl.Expr:
    return (as_of.dt.year() - start.dt.year()) * 12 + as_of.dt.month() - start.dt.month()

def plain(code: str) -> bool:                      # a plain function works too
    return code.strip().upper().startswith("A")

@impl(plain, "polars")
def _(code: pl.Expr) -> pl.Expr:
    return code.str.strip_chars().str.to_uppercase().str.starts_with("A")

pipeline = flow(tenure_months, plain, name="p")
pipeline.compile("polars").fallbacks()
# {}                                               both steps run natively

pipeline.compile("fused").fallbacks()
# {"p/tenure_months": "reads 'start' as date, which no kernel takes",
#  "p/plain": "reads 'code' as str, which no kernel takes"}     the numba engine ignores polars bodies
```

A polars body on a step inside a `loop` is never used; the loop runs in the numba engine as one boundary.

## Spark (last)

```python
from decikit_spark import SparkEngine                         # or by tag once installed: engine="spark"

scored = pipeline.run(spark_df)                               # a Spark DataFrame
scored.write.parquet("s3://.../scored")
```

A `frame_step` in the pipeline raises: cross-row work is done in the job between pipelines.

## Writing an engine

```python
from decikit import Engine

class MyEngine(Engine):
    type: Literal["mine"] = "mine"
    some_setting: int = 3

    def bind(self, step_or_ir) -> Executable:
        ...
```

```toml
[project.entry-points."decikit.engines"]
mine = "my_package.engine:MyEngine"
```

`pipeline.compile("mine")` and `DECIKIT_API__ENGINE__TYPE=mine` then find it. An engine that reuses the numba
driver subclasses `NumbaEngine` and supplies a runner; one that translates the plan reuses
`decikit._engine.segments` and supplies an emitter and a boundary.
