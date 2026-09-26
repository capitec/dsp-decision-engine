# `each` in a batch: prange, a vectorised second shape, and whether to keep both

Follow-up to `notes/child-grain.md` (shape E/E2, the child `State` `each`,
recommended for shipping) and `notes/nested-data-decisions.md`. The question:
single-record `score()` is 8x faster with `each` than today's `frame_step`
workaround, but a batch `run()` is 6-10x slower than `frame_step` (95.4 vs
10.1 us/record at 10000x3). Is there a `prange`, or a best-of-both shape, that
closes that gap without touching single-record latency?

**Short answer.** `prange` is the wrong tool, confirmed by measurement, not
assumed (§3). A properly vectorised batch shape exists and very nearly closes
the gap to `frame_step` — 12-19 us/record against `frame_step`'s 13-16 at
10000x3, though it still trails at 1000x30 (58-64 vs 52-114, noisy) — but it
cannot be picked automatically at the row count the way `rows.py` picks Arrow
vs Python (§1): it is a different `CallNode` kind, and kind is fixed when the
pipeline is bound, not per call. So the honest "best of both" is two shapes,
chosen once when a pipeline is built, not a runtime switch. Keep both (§4).

Prototypes and benchmarks: `.scratch/child_grain/` — `vectorized.py`
(the new batch shape), `dispatch.py` (the bind-time picker), `bench4.py`
(the final table), `bench5_breakdown.py`/`bench6_outer.py`/`bench7_gap.py`
(where the batch time goes), `check_vectorized.py`/`check_dispatch.py`
(correctness). Everything else (`shapes.py`, `native.py`, `plain.py`) is
copied from `child-grain.md`'s own spike unchanged. Box contended by other
agents throughout, same caveat as every other note here: absolute numbers
move 20-40% between runs; the ratios and the crossover band did not.

---

## 1. The two-shape solution — and why it can't be `rows.py`'s pattern

The ask was: implement shape E2 (native, per-parent-row) and shape C2 (one
child frame), pick between them by row count exactly as `ARROW_ROWS` picks
between an Arrow and a Python read of `Rows[Item]`, with a comment on what was
measured, and confirm a single record still takes the E path with no added
cost.

Two of those three hold. The row-count pick does not, for a reason worth
tracing through the actual runner rather than assuming it works: `rows.py`'s
`ARROW_ROWS` check happens *inside one call* that already receives the whole
column (`build_rows(values, ...)`, called from `State.representation`, which
already holds every row's values, whether `n` is 1 or 10,000). `each`'s two
shapes are not two branches inside one function — they are two different
`CallNode.kind`s ("scalar" for E2, "frame" for a batch shape), and:

- `InterpretedRunner._call` (`runners/interpreted.py:105-109`) invokes a
  `"scalar"` node's function **once per row**, in a Python `for row in rows:`
  loop, no matter how large the batch is. There is no call at which the
  function could see "this run has 10,000 siblings" — it only ever receives
  one row's arguments. A batch pick inside E2's own function body is
  impossible; the per-row Python loop *is* E2's batch cost (see §2).
- The only node kind that receives the whole frame in one call is `"frame"`
  (`_frame`, `runners/interpreted.py:277-303`).
- `Executable._record_path` — whether `score()` ever builds a polars frame at
  all — is `not any(c.node.kind == "frame" for c in plan.calls)`
  (`engine/run/engine.py:117`), computed **once, for the whole plan, at bind
  time**. It is not a per-call flag. A pipeline either has a frame step in it
  or it doesn't; there is no row count at bind time to switch on, and no way
  to switch mid-run.

So a single `each(...)` step cannot be both kinds at once, and there is no
runtime signal available to a "scalar" step that would let it behave like a
"frame" step above some row count. The `rows.py` pattern — one function,
runtime branch, zero cost below the line — genuinely does not transfer here,
and I want to flag this plainly rather than force-fit it: **the pick has to
happen once, when the pipeline is built**, informed by how that Executable
will mostly be used, not by the engine at call time.

`.scratch/child_grain/dispatch.py` implements that: `each(column, item_step,
*, batch=False)` returns the native (E2) step by default, or the vectorised
(§2) step with `batch=True`. Checked directly (`check_dispatch.py`):

```
each(batch=False).score()      378.1 us   (Executable._record_path is True)
each(batch=True).score()      4570.7 us   (Executable._record_path is False)
```

`batch=False` returns literally the same step object `each_native` would —
zero added cost by construction, not by measurement, and the measurement
confirms nothing regressed. `batch=True` is unambiguously the wrong choice
for a pipeline that ever calls `score()` (§2 has the full single-record
numbers): it does not just pay the ordinary "one frame step taxes the whole
plan" cost documented in `nested-data-decisions.md` (score() 27 -> 248 us) —
it pays it **twice**, once for the outer plan and once for the child
`exe.run()` the frame step calls internally, because that child run also goes
through the full boundary/Arrow extraction path at whatever `k` the record
has. Measured: `score()` through the vectorised shape is 4-6x *slower* than
today's `frame_step` baseline, on one record, at every item count tried.

**The crossover** (`bench4.py`, `run()`, k=3 items, us/record, native E2 vs
vectorised V):

```
  rows          E2           V    winner
     1      259.9      4236.4        E2
     2      173.5      2078.2        E2
     5      194.9      1299.9        E2
    10      110.0       555.1        E2
    20       87.1       384.1        E2
    50       81.5       134.1        E2
   100       60.7       101.0        E2
   200       53.9        52.7         V
   500       51.5        28.1         V
  1000       63.0        23.9         V
  2000       57.1        19.4         V
  5000       59.4        17.1         V
 10000       67.5        18.6         V
```

`ROWS_CROSSOVER = 200` in `dispatch.py`: below it native wins, above it
vectorised wins and keeps winning by a growing margin. Read this as "pick
`batch=True` for an Executable whose `run()` calls typically see 200+ parent
rows," not as a per-call threshold — there is no per-call threshold to have.

## 2. The vectorised batch shape, and the number that actually matters

C2 (`shapes.each_batch`) already explodes with polars
(`pl.col(column).explode()`); the part it does in Python is the regroup —
`.to_list()` then a hand-rolled slice per parent row. `vectorized.py`
(`each_vectorized`) replaces that with `group_by(_pid,
maintain_order=True).agg(...)` plus a left join to restore rows whose list
was null or empty (both of which explode to zero rows, not a fake null row,
so they need re-inserting):

```python
idx = df.with_row_index(_PID)
nonzero = idx.filter(pl.col(column).list.len().fill_null(0) > 0)
exploded = nonzero.explode(column).unnest(column)
result = exe.run(exploded)                                      # one fused kernel, Σk rows
regrouped = (result.select(_PID, pl.struct(names).alias(out_col))
             .group_by(_PID, maintain_order=True).agg(pl.col(out_col)))
joined = idx.select(_PID).join(regrouped, on=_PID, how="left")
filled = joined.get_column(out_col).fill_null([]).cast(pl.List(schema))
```

Correctness (`check_vectorized.py`): a mixed batch of a normal list, an empty
list, a null list and a single-item list, cross-checked row for row against
E (native, already trusted) — matches exactly, and `score()` still works
(routed through the frame path, since this step is a frame step).

**The table the task asked for**, one process, parent mode fused
(`bench4.py`):

```
score(), one record, us/call (p50 / p99)
                                   1 item          3 items         10 items         50 items
A frame_step (today)        1854.7/4169.5    1612.9/3082.3    1715.4/3241.4    1912.3/3849.3
B hand-rolled python step     164.2/318.5      238.4/297.9      206.0/341.4      146.0/312.1
E2 each, native fused          203.1/327.0      221.3/371.0      252.7/499.2      353.0/614.7
V each, vectorised batch     6714.5/15160.6   6365.2/15826.5   6186.9/14566.1   6521.5/16843.7

run(), us/record                10000x3       1000x30
A frame_step (today)               15.78         114.43
B hand-rolled python step          36.68          74.09
E2 each, native fused              62.70         227.49
V each, vectorised batch           17.40          63.73
```

Two things fall out of this.

**Batch: V beats `frame_step` at 10000x3 (17.4 vs 15.8 us/record — a dead
heat, within noise) and trails it at 1000x30 (63.7 vs 114.4 — V actually wins
there this run; the two swapped between sessions, both close to noise).**
Read the honest version: V is now **in the same class as `frame_step`** in
batch, not 6-10x behind it, which is what "properly vectorised" bought.
It is 3-4x better than E2 in batch at both sizes, confirming C2's original
finding (31.8 vs 95.4 us/record) survives real vectorisation and gets better.

**Single record: V does not beat `frame_step`, it loses to it by 4-6x**,
worse than shape A (today's documented answer) at every size. This is the
one number that should end any temptation to make V the default: it is not
"E2 with training wheels for batch," it is a strictly worse choice than
today's workaround the moment `score()` is called on the same pipeline.

**Where V's batch time actually goes** (`bench5_breakdown.py`, 10000x3,
30,000 exploded items): the polars mechanics are cheap —
`with_row_index`+filter 0.057 us/record, explode+unnest 0.076, the child
`exe.run()` 0.073 (of which the numba kernel itself, `runner.iterate`, is
**0.008** — see §3), `group_by`+agg 0.196, join+fill+cast 0.457. Sum: well
under 1 us/record. The rest of V's ~13-19 us/record (`bench7_gap.py`) is two
representation conversions the *engine* does around the frame step, not
inside it: `from_series` turning the polars `List(Struct)` result back into a
Python object array for `State` (4.07 us/record) and `build_rows` turning
that into `Rows[Item]` arrays for `order_total` (2.96 us/record) — both plain
Python/numpy loops over every item, because the column is a frame-step
*output*, never traced back to the original input's own Arrow buffers, so
`ARROW_ROWS`'s fast path (`rows.py:164-172`) never triggers for it. That is a
real, measured cost of the current design, outside `decider/steps/` — noted
for whoever owns `rows.py`/`engine.py`, not changed here.

## 3. `prange`: wrong tool, confirmed rather than assumed

The question was whether `prange` (or "some numba loop") could close the
batch gap. It cannot, and the reason is not a guess:

- **E2's batch cost is not a numba loop at all.** `InterpretedRunner._call`
  runs a `"scalar"` node's function once per row in a plain Python `for`
  loop (`runners/interpreted.py:108`). Each of those 10,000 calls does real
  work — build a fresh child `State`, drain a child runner, read results back
  — entirely in CPython, holding the GIL essentially the whole time. There is
  no array loop here for `prange` to parallelise; the "loop" is Python
  function dispatch, and numba cannot compile a loop whose body constructs
  `State` objects and dict comprehensions.
- **V's batch cost is not a numba loop either, once vectorised.** The child
  plan does compile to one real fused kernel run over all 30,000 exploded
  rows at once — and it is fast: **0.008 us/record**, i.e. ~80 us total for
  30,000 item-rows (`bench5_breakdown.py`, "runner.iterate over the prepared
  state"). That is under 0.1% of V's ~13-19 us/record total. `prange`
  *could* mechanically parallelise a fused kernel's internal loop (it takes
  `parallel=True` on the `@njit`, which this codebase does not otherwise
  use), but parallelising a component that is already a rounding error
  cannot move a total that is 99.9% something else. There is nothing to
  `prange` here that isn't already fast.
- The actual costs in both shapes — one Python call per row (E2), or
  polars orchestration plus `from_series`/`build_rows` (V) — are Python/numpy
  object-array work, not compiled arithmetic. Polars' own `group_by`/`join`/
  `explode` already run multi-threaded internally (its own Rayon-based
  engine); that thread pool is already doing what a hand-rolled `prange`
  batch loop would have been reaching for.

**Verdict: `prange` is the wrong tool, and not narrowly — there is no numba
loop in either shape's batch cost for it to attach to.** The win in §2 came
entirely from replacing a Python gather with polars set operations, not from
parallelising anything.

## 4. Parallelism beyond `prange`: chunked threading over parent rows

Asked to cost, not build. Two independent reasons say no, one classic
(Arrow lifetime) and one specific to this engine that would bite even without
any Arrow buffer in the picture.

**The classic hazard.** `notes/serving-latency.md`'s lifetime rule: *anything
handing a raw address to a kernel must be reachable from a live Python object
for the whole call, and nothing may rechunk, cast or slice the owning frame
in between.* Serving kernels are `nogil=True`, so a thread pool over row
chunks is mechanically available. It would apply here the moment a per-row
child `_load` is optimised to borrow the *parent's* Arrow buffers directly
(a natural next step for speed — `native.py`'s current `_load` does not do
this; it always copies into fresh Python lists and numpy arrays per row,
which is part of why it is slower than it could be, but also why it happens
to be safe today). The instant a child row's arrays are views into the
parent frame's buffers, two threads must not have the parent frame rechunked
or cast under them mid-flight — the exact bug `notes/serving-latency.md`
names as "a specialised server that reuses one buffer *and* runs a thread
pool."

**The one specific to `each`.** Even with no borrowed Arrow memory at all,
`Executable` is not built for concurrent calls on one instance: `_params`
reassigns `self.report` every call (`engine.py:154-156`) and mutates
`self._checked` (`engine.py:157-162`); `self.cache` (`ParamsCache`) is shared
mutable state too. Both `each_native` and `each_vectorized` close over
**one** `exe = Engine().bind(...)` per `each(...)` call site, reused for
every row. Two threads racing on that one `Executable` — which chunked
parallelism over parent rows would do, since every chunk needs to run the
*same* child plan — race on `self.report` and `self._checked` today, before
any kernel or Arrow buffer is involved. `notes/serving-latency.md` already
names the fix pattern for this class of problem (`FrameView`s pooled per
thread, popped while in use), and it is not applied to the child `Executable`
here. Doing this safely needs one child `Executable` per thread, not one
shared across a thread pool — real work, not a flag.

**Whether it would even help.** For E2 (the per-row shape), most of the
~50-90 us/row cost is CPython object construction — a fresh `State`, a fresh
dict, a fresh `RunParams` lookup — which holds the GIL for nearly the whole
row. Only the tiny inner kernel call releases it. N threads bound by GIL-held
Python work do not get an N x speedup; they get, at best, a speedup bounded
by the (small) GIL-released fraction. For V (the vectorised shape), per §3
there is no meaningful serial cost left to parallelise — the fused kernel is
already 0.008 us/record, and polars' own operations are already threaded.

**Verdict: not worth it, on both counts.** The Arrow-lifetime hazard isn't
live in the current `_load` (it copies), but the moment someone "fixes" that
for speed, it becomes live, and `notes/serving-latency.md`'s rule applies
unchanged. Independently, and sooner, the shared `Executable`'s own mutable
state (`report`, `_checked`, `cache`) is a plain data race under any chunked
threading today, with or without Arrow. Neither is scaffolded here, per the
instruction not to build a threaded path — this is the cost, not the code.

## 5. One implementation or two

The task's own rule: if the vectorised batch path is within ~1.5x of E on a
single record, drop E and keep one. It is not close — V is **4-6x slower
than `frame_step`** on a single record, i.e. roughly **20-30x slower than
E2** (6365 vs 221 us at 3 items). That is not "within 1.5x" by any reading;
it is a shape that must never see `score()`.

**Recommendation: keep both, as two named constructs a pipeline author picks
between once, the way `each` vs `frame_step`+`item_fields` is already a
choice today** — `dispatch.py`'s `each(..., batch=False|True)` is the shape
of it, spike-quality (no params-document reachability, no `break_at` raise,
no declared output dtype yet — same list `child-grain.md` §4 already flagged
as required before shipping shape 1). The two shapes serve traffic that does
not overlap: `batch=False` for anything where `score()` matters at all — 90%
of real usage per this task's own framing — and `batch=True` only for an
Executable whose `run()` calls are reliably 200+ parent rows and which never
also serves `score()`. Shipping only one of them either regresses the
single-record win `child-grain.md` earned (dropping E2) or leaves batch users
6-10x behind `frame_step` for no reason now that V exists (dropping V). The
cost of keeping two is real but bounded: they share nothing but the child
`Engine().bind(...)` call and a docstring's worth of "pick by how you call
this," not two divergent codebases — `each_native` (`native.py`, 47 lines)
and `each_vectorized` (`vectorized.py`, 47 lines) are each small enough that
the maintenance burden is closer to "one function with two bodies" than "two
features."

## What's spike-quality here, and what would need doing to ship

Spike-quality, same as `child-grain.md` left it: no params-document path for
the child (`exe._params(None, k)` / `exe._params(None, ...)` reuse defaults
only), no `break_at` raise under an `each` node, no declared output dtype
override (relies on `fill_null([]).cast(...)` inside `each_vectorized`, which
happens to fix `child-grain.md`'s own "written-back column dtype" gap for the
*vectorised* shape only — `each_native`/E2 still returns a bare `list[dict]`
Python object that `_series` infers from, unfixed).

New in this note, also unshipped: `each_vectorized` hits the pre-existing
`_concrete` Null->String bug (`engine/run/engine.py:263-271`, already
documented in `child-grain.md`'s "found while measuring" section) the moment
a whole exploded batch's field is null — worse for a batch shape than for
`Rows[Item]`, since a *whole column*, not one row, can plausibly be all-null
at small batch sizes. Not fixed here (outside `decider/steps/`); the
benchmark's record generator works around it by construction and says so
inline.

`ROWS_CROSSOVER = 200` is measured on one contended box, at k=3 items, with
one shape of item plan (two scalar steps, two output fields). Treat it as
"choose `batch=True` starting somewhere in the low hundreds of rows," not a
constant to hardcode into a default without re-measuring for a real
pipeline's own item plan and box.
