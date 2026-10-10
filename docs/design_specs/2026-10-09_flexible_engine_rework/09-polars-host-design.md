# Polars host: detailed design (phase P4)

Everything needed to implement `PolarsEngine` v1 without having been in the design discussion. Facts marked
**verified** were checked by a spike on 2026-10-09 against polars 1.41.2 at commit `54ce33f3`; the spike
code is at the end so the facts can be re-checked when polars or decider moves.

## What v1 does

`PolarsEngine().bind(pipeline).run(lf)` returns a `LazyFrame` whose plan runs each row-local part of the
pipeline through the existing fused `Executable` on every batch polars hands it, runs each frame step on the
whole frame, and ends with a `select` so the columns equal `FusedEngine().bind(pipeline).run(df)`. No step
runs anywhere but the numba engine; polars only decides how rows are batched and threaded.

## Verified facts

1. `lf.map_batches(exe.run, schema=..., streamable=True)` hosts a fused `Executable`. With
   `pl.Config.set_streaming_chunk_size(7)` and `collect(engine="streaming")`, a 100-row frame called each
   row-local segment 25 times and the `streamable=False` frame step once. Plain `collect()` called each once.
   Answers equal the fused run after the final select.
2. `SequenceNode(ir.origin, children[i:j], emits=(...))`, built from an existing IR's children, resolves and
   binds (`Engine().bind(node, "fused")`). Its `plan.inputs` are the names it reads from outside.
3. Running a bound `Executable` on a zero-row frame of the input schema (`df.clear()`) returns a frame with
   the exact output schema, so the `schema=` that `map_batches` requires can be derived without a dtype
   table.
4. A sub-executable rejects a params document entry for a step outside it:
   `ParamsError: params document: no step with params at 'p'. This pipeline has no params...`
   (`check_namespaces` in `decider/_engine/params/validate.py:196`). The document must be pruned per
   segment.
5. A name written on both sides of a frame step: when the later segment also reads it, the sub-executable
   accepts a batch that already holds the column (a self-read is an input, and `_produced` excludes inputs)
   and the answer equals fused. When the later segment only writes it, `_check_shadowing` raises
   `EngineError: '<name>' is produced by this pipeline and is also a column of the input frame`. The host
   drops that column before the segment.
6. Earlier segments' intermediate outputs stay in the frame (`['income', 'debt', 'affordable', 'ratio',
   'rank', 'flag']` against the fused `['income', 'debt', 'flag']`), so a final `select` is required.
7. `LazyFrame.map_batches(function, *, schema=None, validate_output_schema=True, streamable=False)`;
   `schema=None` means "unchanged", which is never true for a segment, so the schema is always passed.
   `validate_output_schema` stays `True`: a wrong derived schema must fail loudly.

## Data model

```python
@dataclass(frozen=True)
class Segment:
    nodes: tuple[IRNode, ...]        # consecutive top-level children of the root sequence
    streamable: bool                 # False when any node contains a frame call
    executable: Executable           # self.segments.bind(SequenceNode(root.origin, nodes, emits=writes))
    reads: frozenset[str]            # {i.name for i in executable.plan.inputs}
    writes: frozenset[str]           # every output name of every CallNode under nodes (None outputs: nothing)
    params_paths: frozenset[str]     # {n.path for n in executable.nodes.values()}, for the params prune
    shared_keys: frozenset[str]      # shared params its nodes use, for the params prune
```

`PolarsExecutable` holds `segments: tuple[Segment, ...]`, `full: Executable` (the whole pipeline bound with
`self.segments`, used for the shadowing check, the record path and the final column list) and the `Engine`
document. Nothing else; `run` writes nothing to `self`.

## Build (once, in `PolarsExecutable.__init__`)

1. `ir = to_ir(step_or_ir)`; `full = engine.segments.bind(ir)`.
2. Top-level children: `kids = ir.children()` if `ir` is a `SequenceNode`, else `(ir,)`.
3. Group in order. `contains_frame(node)` is true when any `CallNode` under `iter_nodes(node)` has
   `kind == "frame"`. A child with a frame call is a segment on its own with `streamable=False` (a bare
   frame step, or a branch or loop that holds one: the whole node sees the whole frame, which is the only
   correct choice). Consecutive children without one form a streamable segment.
4. Refuse up front, with `EngineError`, what v1 does not handle: a qualified output in `full.plan.outputs`
   (`"name@path"`), because the final select works on plain names.
5. For each group build `SequenceNode(ir.origin, nodes, emits=tuple(sorted(writes)))` and bind it. Emitting
   every written name is deliberate: no per-segment pruning, the final select trims. A name nothing later
   reads costs memory for the length of the plan and nothing else.

## Run

```python
def run(self, data, params=None, trace=None):
    if trace is not None:
        raise EngineError("the polars engine records no trace; run the pipeline on the numba engine for a TraceSink")
    if isinstance(data, Mapping):
        return self.full.run(data, params)
    if isinstance(data, pl.DataFrame):
        return self.run(data.lazy(), params).collect()
    if not isinstance(data, pl.LazyFrame):
        raise TypeError(f"run() takes a dict record, a polars DataFrame or LazyFrame, not {type(data).__name__}")
    lf = data
    schema = lf.collect_schema()
    self.full._check_shadowing(schema.names())
    present = set(schema.names())
    for seg in self.segments:
        doc = _params_for(params, seg)
        out_schema = _output_schema(seg, schema, doc)
        for name in seg.writes & present - seg.reads:          # fact 5: a blind overwrite must not shadow
            lf = lf.drop(name)
        lf = lf.map_batches(partial(seg.executable.run, params=doc), schema=out_schema,
                            streamable=seg.streamable)
        schema = out_schema
        present = set(schema)
    return lf.select(_final_columns(self.full, data_schema=lf_input_schema, params))
```

- `_output_schema(seg, schema_in, doc)`: run `seg.executable.run(pl.DataFrame(schema=schema_in), doc)` and
  take `.schema` (fact 3). If a frame step inside the segment cannot run on an empty frame (user code that
  indexes row 0, say), the exception is re-raised as `EngineError` telling the author to declare
  `writes={name: dtype}` on the frame step; when `FrameStep.write_annotations` holds dtypes for every column
  the step writes, the host uses them instead of running the step, and runs only the scalar parts. For v1 it
  is acceptable to implement the zero-row run first and the declared-dtype path when the first real pipeline
  needs it; the error message must exist from day one.
- `_final_columns(full, schema_in, params)`: the column list of `full.run(pl.DataFrame(schema=schema_in),
  params)`. Same zero-row trick; one extra zero-row run per `run` call.
- `_params_for(doc, seg)`: a copy of the params document holding only what this segment's nodes accept, or
  `None` when nothing remains (fact 4). Keep `doc["shared"]` entries whose key is in `seg.shared_keys`. For
  the rest, walk the nested mapping the way `check_namespaces._walk` does: a key at prefix `p` is kept if
  `p + key` is in `seg.params_paths` (keep the whole subtree), descended into if some path in
  `seg.params_paths` starts with `p + key + "/"`, dropped otherwise. Empty entries (`{}`) may be dropped. A
  fresh dict per call defeats the sub-executable's "hash a reused document once" cache; at 10k+ rows that
  is noise, so v1 prunes per call and leaves a `# ponytail:` comment naming the cache as the upgrade.
- Zero-row runs validate params too, so an invalid document fails before any batch runs, as it does on the
  numba engine with eager validation.

## Streaming semantics

- `out.collect()` runs every segment once on the whole frame. `out.collect(engine="streaming")`,
  `out.sink_parquet(...)` and the other sinks run streamable segments once per batch polars chooses and the
  frame-step segments once on everything. Row-local segments give the same answer either way; that is what
  makes `streamable=True` honest.
- The host never assumes a batch size. Tests force small batches with `pl.Config.set_streaming_chunk_size`
  (fact 1) and must reset it afterwards (`pl.Config.restore_defaults()` or a fixture).
- A frame step is a pipeline breaker: the whole frame materialises there. Larger-than-memory input works
  only for pipelines whose frame steps' output fits in memory.

## Errors

- `TypeError` for an input that is none of dict, `DataFrame`, `LazyFrame`, naming the accepted types.
- `EngineError` for `trace=`, for `session()`, for a qualified emit, and for a frame step that cannot give
  its schema; each says what to do instead.
- An exception raised by a step inside a segment propagates from `collect()`. Check what type reaches the
  caller: polars may re-raise the Python exception as is or wrap it in `ComputeError`. `assert_equivalent`
  treats a different exception type as a divergence, so if polars wraps, unwrap in the host (catch
  `ComputeError`, re-raise `__cause__` when it is set) and record the choice in the decision note. The step
  note the numba driver adds (`in step p/flag`) must survive.

## Thread safety and conventions

- `PolarsExecutable.run` writes nothing to `self`; segments and `full` are built in `__init__`. Add
  `PolarsExecutable` to the walk in `test_runners_and_executable_keep_no_per_call_state`.
- Sub-executables are ordinary `Executable`s and share their thread-safety contract. Two threads can call
  `run` on one `PolarsExecutable` at once.
- `decider/_engine/polars/engine.py` and `emit.py` are private; no `__all__`. Each under 500 lines.

## `fallbacks()` and inspection

- `fallbacks()` returns the union of every segment's `executable.fallbacks()` (the numba fallbacks inside
  boundaries, keyed by step path as today). Nothing is prefixed; paths are already unique.
- `segments` is public on the executable: `[(tuple(n.origin.path for n in s.nodes), s.streamable) for s in
  self.segments]`, so a user can see where the frame breaks.

## Tests (`tests/polars_engine/`)

Each name states the behaviour; the fixture pipeline is the three-segment one from the spike (two scalar
steps, a `rank` frame step, a scalar step reading `rank`).

- `test_pipeline_without_frame_steps_is_one_streamable_segment`
- `test_frame_step_runs_once_on_the_whole_frame_while_segments_run_per_batch` (chunk size 7, counters)
- `test_streaming_collect_and_in_memory_collect_give_the_same_frame`
- `test_sink_parquet_round_trip_equals_fused`
- `test_intermediate_columns_are_dropped_as_fused_drops_them`
- `test_params_document_is_pruned_per_segment` (a param in segment 3, a document covering it; and a document
  with an unknown step, which must still fail as on numba)
- `test_shared_param_used_by_one_segment_only`
- `test_name_read_and_rewritten_after_a_frame_step` and `test_name_overwritten_after_a_frame_step_is_dropped_first`
- `test_frame_step_inside_a_branch_makes_the_whole_branch_a_whole_frame_segment`
- `test_frame_step_that_cannot_run_on_zero_rows_gets_a_clear_error` (and, when built, the declared-dtype path)
- `test_dict_record_runs_on_the_segments_engine`
- `test_dataframe_input_returns_a_collected_dataframe`
- `test_trace_and_session_are_refused_with_a_pointer_to_the_numba_engine`
- `test_qualified_emit_is_refused`
- `test_step_exception_type_and_note_survive_collect`
- `test_every_example_flow_equals_fused` (parametrised over `examples/*`; a refusal is a skip with the reason)
- `test_two_threads_share_one_polars_executable`

## Out of scope for v1

Native segments and `impl` (P6), the `REQUIRED` null UDF (only native segments need it), lazy-aware frame
steps, per-segment column pruning, caching pruned params documents.

## Spike

Run with `uv run python <file>`; it takes a few seconds for the numba compile.

```python
import polars as pl
from decider import Engine, flow, frame_step, param
from decider._engine.ir.context import to_ir
from decider._engine.ir.nodes import SequenceNode

def ratio(income: float, debt: float) -> float:
    return debt / income

def affordable(ratio: float, limit: float = param(0.4)) -> bool:
    return ratio <= limit

@frame_step(reads=["income"], writes=["rank"])
def rank_income(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("income").rank().alias("rank"))

def flag(rank: float, affordable: bool) -> bool:
    return affordable and rank > 2

pipe = flow(ratio, affordable, rank_income, flag, name="p")
df = pl.DataFrame({"income": [1000.0 * (i + 1) for i in range(100)], "debt": [200.0] * 100})
ref = Engine().bind(pipe, "fused").run(df)                       # ['income', 'debt', 'flag']

ir = to_ir(pipe)
kids = ir.children()                                             # four CallNodes
seg1 = Engine().bind(SequenceNode(ir.origin, kids[:2], emits=("ratio", "affordable")), "fused")
seg2 = Engine().bind(SequenceNode(ir.origin, kids[3:], emits=()), "fused")

calls = {"s1": 0, "f": 0, "s2": 0}
def s1(b): calls["s1"] += 1; return seg1.run(b)
def fr(b): calls["f"] += 1; return rank_income(b)
def s2(b): calls["s2"] += 1; return seg2.run(b)
sch1 = dict(seg1.run(df.clear()).schema)                         # fact 3
schf = {**sch1, "rank": pl.Float64}
sch2 = dict(seg2.run(pl.DataFrame(schema=schf)).schema)
lf = (df.lazy()
      .map_batches(s1, schema=sch1, streamable=True)
      .map_batches(fr, schema=schf, streamable=False)
      .map_batches(s2, schema=sch2, streamable=True))

pl.Config.set_streaming_chunk_size(7)
out = lf.collect(engine="streaming")
assert out.select(ref.columns).equals(ref)
print(calls)                                                     # {'s1': 25, 'f': 1, 's2': 25}
```

The params and overwrite checks (facts 4 and 5) are two more short runs: give `seg1` a document with an
entry for a step in `seg2` and expect `ParamsError`; make a step after a frame step write a name a step
before it wrote, once reading it first (accepted) and once not (`EngineError` on shadowing).
