# Semantics across engines

The contract every engine must meet, how the engine-owned parts (nulls, params, outputs) map onto polars and
Spark, where author-written polars bodies are known to diverge, and how `assert_equivalent` checks it.

## The contract

For the same pipeline, params document and input rows, every engine returns the same output names, dtypes
and values, nulls included; a `REQUIRED` input that is null raises `MissingInputError` in every engine; and an
exception a body raises propagates in every engine. `run(record)` equals the matching row of `run(frame)`.

Where an engine cannot meet a clause, it refuses the pipeline or the feature with an error rather than
answering differently: a `frame_step` on Spark, `trace=` on a translator engine, a `Decimal` input to a polars
body.

The v1 polars host meets the contract by construction: every step runs its Python body inside a numba
sub-executable, and the only polars-side logic is splitting rows into batches, which no row-local step can
observe. Everything below about native segments applies to v2 only.

## What the engine owns, and how it maps

| Concern | Numba engine, and every boundary | Native polars segment (v2) | Spark native (later) |
|---|---|---|---|
| `REQUIRED` input null | `MissingInputError` naming input, step and null count, at the boundary read | a streamable check UDF before the segment raises `MissingInputError` with step and count; no row index | same, per partition |
| `MISSING_AS` fill | the fill value per row (a deep copy per row for list/dict fills) | `pl.col(x).fill_null(lit)` wrapped around the input Expr | `F.coalesce(col, lit)` |
| `OPTIONAL` | `None` per row | nulls present in the Expr; the body must handle them | same |
| params | validated once per document; a bundle of Python values passed at run time | validated through the same `NodeParams`; baked as `pl.lit` into the plan | baked as `F.lit` |
| `param_table` | rows as a typed record array | not supported in a polars body | not supported |
| a `None` result | a null in the output | a null | a null |
| several outputs | a tuple per row | a tuple of Expr | a tuple of Column |
| `Literal` outputs | coded table; stored as an index | emitted as strings; cast on output | same |
| `Struct[Item]` / `Columnar[Item]` inputs | per-row record or per-field arrays | the column's Expr | the column |
| `Text` inputs | `(address, length)` spans in a kernel | `pl.col(x)` | `F.col(x)` |
| `Decimal`, `Currency` | int64 mantissa, half-even operators | refused at decoration | refused |
| output dtype | from the annotation; lossy casts refused (`cast_values`, `same_kind`) | schema resolved after each step and checked against the annotation at plan build | schema checked at plan build |
| a step inside a `loop` | runs as today | never native: the loop is a boundary | same |

## Where polars bodies are known to diverge

These are properties of polars and Spark, not bugs an engine can fix. A body's author owns them; the tests
below detect them.

| Operation | Python / numba | polars | Spark |
|---|---|---|---|
| `x / 0.0` | raises `ZeroDivisionError` (the interpreted driver passes Python scalars to make sure of it) | `inf`, `-inf` or `NaN`; integer division gives null | null |
| `x // y`, `x % y` with negatives | floor semantics | floor semantics; check | truncation semantics |
| int64 overflow | wraps silently in a kernel; Python ints are unbounded | wraps or raises depending on the operation; check | overflow to null or error depending on ANSI mode |
| `round(x, n)` | CPython half-even, reproduced in kernels | `Expr.round` has a `mode` argument; the default must be checked against CPython | `round` is half-up; `bround` is half-even |
| `x ** n` | CPython; a literal `x ** 2` may differ from libm by one ulp (known limitation) | `pow` | `pow` |
| string methods | Python `str` semantics (`casefold`, `strip` on all Unicode whitespace) | `.str.*` with their own Unicode and whitespace rules | similar caveats |
| date arithmetic | `datetime.date` | `.dt.*`; month arithmetic differs from a Python `relativedelta` | `F.months_between` rounds |
| `NaN` | a float value; `NaN != NaN` | a value distinct from null | a value distinct from null |

Rule for authors: write the Python body first, make the polars body agree on the test corpus, and guard the
divergence cases in both (a `where` for the zero divisor, an explicit rounding mode) rather than relying on
either engine's default.

## How `assert_equivalent` extends

`assert_equivalent(step, frame, params=None, engines=None, script=None)`:

- `engines` defaults to the built-in numba engines plus `polars`; a tag or an `Engine` instance each.
- For each numba engine: `run(frame)` equals the first engine's; `run(record)` per row equals that row;
  a session resumed to the end equals `run`; the scripted session equals across engines. These are today's
  checks with `score` renamed.
- For `polars`: `run(frame.lazy()).collect()` equals the first engine's `run(frame)`, and `run(record)` (which
  delegates to its `segments`) equals the row. No session checks. In v2 this is the check that compares a
  polars body with its Python body.
- For an engine that refuses the pipeline (a `frame_step` on Spark), the refusal is reported as a skip, not a
  divergence; raising only in a later engine is a divergence, as today.

`MODES` in `decider/lab/testing/_equivalence.py` goes; the engine list comes from the registry's built-ins
through a public facade, so the lab does not import `_engine`.

## What the tests must cover before the contract is claimed

- v1: every `examples/*` flow equal between fused and the polars host, in memory and streamed from parquet
  with a batch size smaller than the frame, with and without frame steps.
- `run` on dict, `DataFrame` and `LazyFrame` for each engine, including the refusals.
- v2 only: a step with a polars body on a frame and a record, inside a branch arm (native and non-native
  branches), and inside a loop body (asserting the Python body ran); each null policy through a native
  segment, with and without nulls present; each divergence class above on one deliberate case, asserting that
  `assert_equivalent` reports it; a pipeline with alternating native and boundary segments.
