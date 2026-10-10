# The polars engine

`pipeline.run(lazy_frame)` runs a pipeline as a polars `LazyFrame` plan, on the polars engine whatever the
default engine is, and `pipeline.compile("polars")` binds it once. Nothing in your pipeline changes:
every step still runs on the fused engine's numba kernels, on each batch of rows polars hands it, and the
answers are the fused engine's. What changes is who drives the batch: polars reads the input, splits it into
batches, runs them on its threads and writes the result, so a batch larger than memory can be scanned from
parquet, scored and sunk back to parquet. This page shows how to run it, where the stream breaks, what it
refuses and why, and how to test it.

## Run a batch as a plan

```python
import polars as pl
from decikit import flow, frame_step, param

def ratio(income: float, debt: float) -> float:
    return debt / income

def approved(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit

@frame_step(reads=["income"], writes=["income_rank"])
def rank_income(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("income").rank().alias("income_rank"))

def tier(income_rank: float, approved: bool) -> int:
    return 2 if approved and income_rank > 2 else 1

pipeline = flow(ratio, approved, rank_income, tier, name="credit")

plan = pipeline.run(pl.scan_parquet("applications.parquet"))   # a LazyFrame; nothing has run yet
plan.sink_parquet("decisions.parquet")                       # streamed batch by batch
plan.collect()                                               # or everything in memory
plan.collect(engine="streaming")                             # or streamed into memory
```

`pipeline.run(lf, engine="fused")` names a numba engine instead: it collects the frame and returns a
`DataFrame`. Bind once with `exe = pipeline.compile("polars")` to build many plans; its `run` takes three
kinds of input:

| `run(data)` | Returns | How |
|---|---|---|
| a `pl.LazyFrame` | a `pl.LazyFrame` | the plan; collect, stream or sink it as you like, or keep composing it |
| a `pl.DataFrame` | a `pl.DataFrame` | `run(df.lazy()).collect()` |
| a dict | a dict | one record on the fused engine directly, the serving path |

The output equals `pipeline.compile("fused").run(df)`: the same columns in the same order, the same dtypes,
the same values.

`PolarsEngine` is an engine document like the others. Its settings for the kernels live on `segments`, the
fused engine that runs every batch:

```python
from decikit import FusedEngine, PolarsEngine

PolarsEngine(segments=FusedEngine(strict_compile=True)).bind(pipeline)       # refuse Python fallbacks
PolarsEngine(params_validation="lazy", segments=FusedEngine(params_validation="lazy"))
```

`params_validation` is one setting: the segments validate the document, so it must be the same on both, and
an engine where they differ is refused when it is built.

## Where the stream breaks: segments

A frame step needs the whole frame: a rank, a join, a group-by computed on one batch would be silently
wrong. So the engine splits the pipeline at its frame steps. Each run of row-local steps between them is a
**segment** that polars may stream; each frame step runs once, on every row, and the frame materialises there.

```python
exe.segments
```

```text
((('credit/ratio', 'credit/approved'), True), (('credit/rank_income',), False), (('credit/tier',), True))
```

Each entry is the step paths of one segment and whether it streams. The split happens between the pipeline's
top-level steps: a branch, loop or named flow that holds a frame step anywhere inside runs whole, on the whole
frame. A pipeline without frame steps is one streaming segment.

A batch larger than memory works when the frame steps' outputs fit in memory: everything before a frame step
streams into it, and everything after streams out of it. With `collect()` (not streaming) polars hands each
segment the whole frame once.

`exe.fallbacks()` lists every step that runs in Python instead of a kernel, keyed by path, as on the fused
engine; the polars engine adds none of its own.

## Params

Pass the params document to `run`, as on every engine:

```python
exe.run(pl.scan_parquet("applications.parquet"), params={"credit": {"approved": {"limit": 0.3}}})
```

It is validated against the whole pipeline when `run` builds the plan, so an invalid document or an entry for
a step that does not exist fails there, with the fused engine's message, before any row is read.

## What it refuses, and why

The engine refuses what it cannot answer exactly as the fused engine would, with an `EngineError` saying what
to do instead.

| Refused | Why | Do instead |
|---|---|---|
| `run(..., trace=sink)` | a trace is written by the numba driver, which a polars plan does not have | trace on the fused engine: same pipeline, same answers |
| `session()` | a session pauses between nodes; polars runs the plan on its own | debug with `pipeline.session(df)` |
| `.emit("x@path")`, a qualified emit | the output holds plain column names | relabel the step so that version has its own name |
| an output with no polars dtype until it holds values: a plain `dict`, a bare `list`, `Any` | polars needs every column's type before the first row arrives | annotate it: a TypedDict, `list[float]`, `str` |
| an `each` whose gathered list leaves a segment or is an output | the list's item fields are known only once it holds items | run the pipeline on the numba engine |
| a frame step that fails on an empty frame | the engine runs each frame step once on zero rows to learn the columns it returns | make it accept an empty frame |
| a frame step that leaves a column empty on zero rows | its dtype is unknown until values arrive | declare it: `@frame_step(writes={"band": pl.String})` |

An untyped value only matters where polars sees it: as an output, or as a value one segment hands to the
next. A `dict` that one step writes and the next step reads, inside one segment, is fine.

## Errors while it runs

A step that raises inside a batch raises from `collect()` or the sink with its own exception type and the
same `in step <path>` note the fused engine adds. polars does not wrap it. A step that runs in Python (a
fallback) also notes the row, counted from the start of its batch when the plan streams.

## Testing it

`decikit.lab.testing.assert_equivalent` runs the polars engine with the others by default: its
`run(frame.lazy()).collect()` must equal the first engine's `run(frame)`, and `run(record)` the matching row.
An engine that refuses the pipeline is skipped with a warning naming the engine and the reason; one that
raises where the first engine did not is a divergence.

```python
from decikit.lab.testing import assert_equivalent

assert_equivalent(pipeline, df)                                  # interpreted, stepped, fused, polars
assert_equivalent(pipeline, df, engines=("fused", "polars"))
```

## See also

- [Compiled modes and fallback](compile-and-fallback.md) for what compiles and the fused engine's settings.
- [Branches, loops and frame steps](branches-loops-frames.md) for frame steps.
- [Polars engine architecture](../architecture/engine/polars.md) for how the plan is built.
