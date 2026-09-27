# Can a `Rows[Item]` step join the shared array kernel?

**Question.** A `Rows[Item]` step runs compiled but as a `Fallback`: one dispatcher
call per row, because each parent row owns a different number of children, so the
value can't be an ordinary kernel argument. A `decider.loop` whose body reads
`items: Rows[Item]` therefore does not pack. `notes/tmcfeval/DECIDER_GAPS.md` §3
claims that is the difference between 119 ms and 1.4 ms for a bundle search, and
calls it the thing that decides whether native decider is fast. Settle it.

**Answer.** It works, it needs no new numba machinery, and it is the largest
speedup measured in this codebase: the doc's bundle search goes from **49.5 ms to
0.21 ms** per `score()`, and at the doc's larger size it lands within **1.2x of a
hand-written `@njit` search**. The claim holds and understates the ratio. Ship it,
after two named fixes.

Everything here is behind `DECIDER_RAGGED_IN_KERNEL=1` and the flag is off by
default, so the change is inert until someone turns it on. What is shippable and
what is spike-quality is listed at the end.

## What numba accepts (numba 0.67.0)

The idea: pass the flat per-field arrays as ordinary kernel arguments and build the
step's value *inside* the kernel, per row, by slicing them. Three forms were
probed. All three work, first try, and none needed a new numba type:

1. **Plain nopython.** `Item(price[lo:hi], qty[lo:hi])` inside an `@njit` function
   compiles and runs. numba supports calling a namedtuple class in nopython code,
   and a step-1 slice of a C-contiguous 1-d array is itself `array(T, 1d, C)`.
2. **An `@intrinsic` building it at LLVM level.** `context.compile_internal` for
   the slice per field, then `context.make_tuple` on the `NamedTuple` type. This
   is the form that matters: `kernel.py` lowers its whole row body to IR and has
   no Python source to put a slice in.
3. **The value satisfies the step's existing signature.** The critical check.
   `typeof(rows_probe(schema))` — the type `_probe_signature` already uses to
   pre-compile a `Rows[...]` step — is exactly what the intrinsic produces.
   Feeding the intrinsic's value to a dispatcher already compiled for that
   signature added **no new signature**: `step.signatures` was unchanged after.

So the whole `span.py` apparatus — a custom `types.Type`, `register_model`, an
`@intrinsic` of our own — is **not needed**. A `Rows[Item]` value is already a
first-class numba type (`NamedTuple` of `Array`) and the step is already compiled
for it; the only missing piece was a way to *construct* one inside a kernel, and
`context.compile_internal(builder, lambda arr, lo, hi: arr[lo:hi], ...)` is it,
with numba's own array-slicing lowering doing the work. No code generation, no
rendered source, no `exec`.

### What numba refused, once

One thing bit, and only on a **batch past `ARROW_ROWS`**, never on `score()`:
field arrays read straight out of the Arrow buffers are **read-only**, and
`readonly array(float64, 1d, C)` is a numba type distinct from
`array(float64, 1d, C)`. Typing the slice's return from `typeof(rows_probe(schema))`,
which builds writable `np.empty` arrays, gives:

```
numba.core.errors.TypingError: Failed in nopython mode pipeline (step: native lowering)
No conversion from readonly array(float64, 1d, C) to array(float64, 1d, C)
for '$10return_value.6', defined at None

File "decider/engine/compile/kernel.py", line 17:
def _slice(arr, lo, hi):
    return arr[lo:hi]
```

The fix is one line of principle: **type the value from the arrays, not from a
probe.** `rag()` reads each field's type out of `flats.types[...]` and builds
`types.NamedTuple(fields, rows_class(schema))` from those, so a read-only column
gives a read-only namedtuple and the step compiles a second signature for it. The
per-row dispatcher has always done exactly this, silently. Worth knowing for
anything else that ever hands an Arrow-backed array to a kernel argument.

**Otherwise numba refuses nothing in this design.** The one thing that does not
work is a *different* design — `Rows[Item] | None`. One kernel has one signature,
and a namedtuple of array views has no in-band `None`; `types.Optional` of it would
need the step body to survive an `is None` guard on an Optional-of-tuple-of-arrays.
So an OPTIONAL rows input keeps the per-row dispatcher (3 lines in `compile_call`),
and `test_an_optional_rows_input_gives_the_step_none_for_a_null_list` still passes.

## The design

`Layout.source` gives a `Rows[...]` input the source `("rag", schema, base)`.
`Kernel` gains `ragged: tuple[Version, ...]`, and the kernel a fifth argument
`flats`: a flat tuple of `(lo, hi, *fields)` per ragged version, so `flats[base]`
and `flats[base + 1]` are the row's `[lo, hi)` and `flats[base + 2 + k]` is field
`k` of the whole column. `load(("rag", ...))` reads `lo[i]`, `hi[i]` through the
same `element()` helper every other column uses, slices each field array, and makes
the namedtuple. `build_ragged` in `rows.py` is `build_rows` stopped one step early:
same Arrow read, same null rules, minus the per-row namedtuple assembly.

`lo`/`hi` are **per-row arrays, not an `offsets` array of `n + 1` entries.** That
is the design choice worth recording: a branch or loop row subset is then
`Ragged(lo[rows], hi[rows], fields)` — two fancy indexes, child arrays shared
unchanged, no gather and no copy, and the item index a null error reports stays in
the full column's coordinates. With `offsets[r]:offsets[r + 1]` a row subset has to
rebase the offsets and either gather the children or carry a separate index map.

## The prize, measured

`benchmarks/bundle_search.py` builds the doc's shape for real: a `decider.loop` over
bundle masks carrying `mask` and `best`, body `bundle_total(mask, items: Rows[Item])`
summing `price * weight` over the items the mask selects, then `keep_best`, then
`advance`. `score()` runs the whole search for one order. "today" is HEAD's path
(the value is a `Fallback`, one dispatcher call per row); "in kernel" is this change;
"hand-written" is the same search as one `@njit` over two numpy arrays. `mode="fused"`
throughout, one process per column, back to back. **This box is loaded and shared**,
so read the ratios; `score()` p50 for the 9-item case was 0.082 ms on a quieter run
and 0.206 ms here, i.e. +/- 2.5x of run-to-run noise on the small numbers.

**`score()`, one order, p50 / p99:**

| items | bundles | today | in kernel | hand-written `@njit` | packs? |
|---:|---:|---:|---:|---:|:--|
| 9 | 511 | 49.5 / 64.8 ms | **0.21 / 0.25 ms** | 0.009 / 0.027 ms | no -> **yes** |
| 16 | 65,535 | 7,174 / 7,704 ms | **7.53 / 7.73 ms** | 6.29 / 6.48 ms | no -> **yes** |

**`run()`, p50:**

| items | bundles | today | in kernel | hand-written |
|---:|---:|---:|---:|---:|
| 9 | 511 | 469.6 ms / 100 orders | **4.73 ms / 100 orders** | 0.73 ms / 100 orders |
| 16 | 65,535 | 13,262 ms / **1** order | **30.2 ms / 4 orders** | 24.4 ms / 4 orders |

(The 16-item "today" column is one order, because four of them is three and a half
minutes per repetition. Per order that is 13,262 ms against 7.6 ms.)

And the fallback report, which is the whole question in two lines:

```
today:      packed loops: []               fallbacks: {'order/search/body/bundle_total':
                                             "reads 'items' as Rows[...], which runs one call
                                              per row, outside the shared kernel"}
in kernel:  packed loops: ['order/search']  fallbacks: {}
```

**Where the residual goes.** The same pipeline at 1 item / 1 bundle — the loop
machinery with no work in it — costs 0.179 ms per `score()` today and **0.069 ms**
in kernel (hand-written: 0.001 ms). So of the in-kernel 9-item 0.21 ms, ~0.07 ms is
the fixed cost of a `score()` call and the search itself is tens of microseconds
against the hand-written 0.009 ms. At 16 items the fixed cost is 1% of the total and
decider is within 1.2x of hand-written numba. **The loop is no longer the cost; the
call around it is.**

### Does the doc's 119 ms -> 1.4 ms claim hold?

**It holds, and the ratio is larger than claimed.** The doc claims 85x. Measured on
the real construct: **240x** at 9 items/511 bundles (49.5 -> 0.21 ms, or 600x on the
quieter run) and **953x** at 16 items/65,535 bundles (7,174 -> 7.53 ms).

Two honest qualifications, in opposite directions:

- **The absolute numbers are not comparable.** The doc's hand-written ceiling is
  2.4 ms for 511 bundles — 4.7 us *per bundle*; ours is 9 us for the whole search,
  18 ns per bundle. The doc's body is the real project's (a nested rate-card loop
  and a branch inside the bundle loop) and does roughly 250x more work per bundle
  than this sum. Read the doc's 119/1.4/2.4 as that body's numbers and this
  table's ratios as the construct's.
- **The doc's "today" was pessimistic about the baseline.** It measured a *Python*
  step at ~0.2 ms per iteration. Today's real path is a compiled dispatcher, which
  is 0.097 ms per iteration at 9 items — about half. So the gap being closed is
  half as wide as the doc implies, and the ratio is still 240x, because the closed
  gap is essentially the whole thing.

The claim that matters — *"with list inputs in kernels, the native pipeline is the
fast pipeline"* — is confirmed with room to spare. The doc's stand-in projected
19 ms against a 61 ms hand-written search at 16 items, i.e. faster than hand-written;
the real thing lands at 1.2x of it, which is the outcome that could have gone wrong
and didn't.

## Single-record latency: no trade, in either shape

The 90% case is `score(dict)`, and it gets faster in every shape measured, because a
`Rows[...]` step stops being a separate per-row dispatcher call and joins the kernel
its neighbours are already in. Two `Rows[Item]`
steps over a `list[dict]` column, no loop at all, 20k rows, `mode="fused"`.

| items | `score()` today | `score()` in kernel | `run(20k)` today | `run(20k)` in kernel |
|---:|---:|---:|---:|---:|
| 3 | 97.3 us | **42.4 us** (2.3x) | 252.1 ms | **68.5 ms** (3.7x) |
| 30 | 93.0 us | **67.4 us** (1.4x) | 580.0 ms | **367.8 ms** (1.6x) |

The win is *larger* at 3 items than at 30, which is the shape `notes/nested-rows.md`
predicted: what goes away is the fixed ~23 us per call (one namedtuple of views per
row in Python, plus entering a dispatcher), and that dominates when the per-item work
is small. Two `Rows` steps also now fuse into one kernel instead of being two
`Fallback` units, which is part of the batch number.

`mode="stepped"` improves too (0.453 -> 0.160 ms at 1 item/1 bundle; 57.0 -> 41.3 ms
at 9 items/511). An earlier run showed stepped `score()` 15% *worse* at 9 items and a
later one 28% better, so the per-iteration cost of launching a one-row kernel instead
of calling a dispatcher is inside this box's noise. **No reproducible regression on
any path.**

## Cost of making it the default

```
 decider/engine/compile/kernel.py      | 42 +++++++++++++++++++++++++--------
 decider/engine/compile/njit.py        | 12 ++++++++--
 decider/engine/compile/packed.py      |  2 +-
 decider/engine/compile/rows.py        | 44 +++++++++++++++++++++++++++++++++++
 decider/engine/compile/units.py       | 27 +++++++++++++++++----
 decider/engine/run/runners/stepped.py | 14 +++++++----
 6 files changed, 120 insertions(+), 21 deletions(-)
```

`uv run pytest -q -p no:randomly`, which is the honest measure of how much this
disturbs: **flag off, 1925 passed** (the committed state). **Flag on, 4 failed /
1921 passed**, and all four failures are the same two tests in two modes, both of
which exist to assert the limitation being removed:

- `test_rows_compiles_but_cannot_join_the_shared_array_kernel[stepped|fused]` —
  asserts the fallback reason is present.
- `test_only_the_shapes_a_kernel_holds_stay_in_it[stepped|fused-rows_of_items]` —
  `rows_of_items` is listed in `SHAPES` with `kernel=False`.

Nothing about *values* failed: `test_every_shape_gives_the_same_answer_in_every_mode`
and `test_every_shape_scores_one_record` pass for `rows_of_items` with the flag on,
as do all of `tests/run/test_rows.py`'s null, `missing_as`, Arrow-slice and
lifetime tests. So the cost in tests is: rewrite two assertions, flip one table
entry, and add the coverage that does not exist yet (a ragged read inside a packed
loop). The suite also runs **563 s -> 250 s** with the flag on, which is the same
effect the benchmarks measure showing up as wall clock.

Per item the question asked about:

- **A new source kind in `load()`** — 13 lines in `kernel.py`, plus threading a
  `flats` argument through `fused_kernel`, its `run`, and the `body` intrinsic's
  signature: 7 one-line changes. No new numba type, no `@register_model`, no
  `@overload`. Every existing kernel gains one empty tuple argument.
- **A version-to-many-columns change in the layout** — **avoided**, and this is why
  the diff is small. `cols[j]` is indexed by the kernel's row counter and a row
  subset is applied per version by the runner, so one version cannot yield both
  per-row and whole-column arrays through `cols`. The fifth `flats` argument
  sidesteps it: `Layout` keeps a `ragged` list and a `v.id -> ("rag", schema, base)`
  map, `Kernel.run` builds `flats` in one comprehension, and `self.reads`,
  `self.optional` and every existing source kind are untouched.
- **`valids`** — a REQUIRED `Rows[...]` with a null row still raises
  `MissingInputError` in the runner before the kernel launches: unchanged. An
  OPTIONAL one cannot join a kernel at all (above) and keeps today's path.
- **A `missing_as` fill** — free. `missing_as([])` and a column absent from the
  frame both give `lo[i] == hi[i]`, a zero-length view, which is exactly "a row with
  no items". `_fills` needed one line to learn the representation's new name.
- **A branch or loop row subset** — free, because of the `lo`/`hi` choice:
  `Ragged.__getitem__(rows)` is two array indexes and shares the children.
- **The single-record path** — `State(plan, _NO_FRAME, 1)` has no Arrow source, so
  `build_ragged` takes the Python read as `build_rows` does (`_from_arrow` returns
  `None` below `ARROW_ROWS` regardless). For n = 1 it is `lo=[0]`, `hi=[total]`, and
  it *skips* the per-row namedtuple assembly `notes/nested-rows.md` measured at ~17
  of the ~23 us fixed cost. It gets faster, not slower. Measured above.

### What you would have to give up

1. **`Rows[Item] | None` stays a per-row dispatcher.** Not a regression — it is
   today's behaviour — but the fast path is not universal and the `FallbackWarning`
   for it has to keep existing.
2. **`Kernel.run`'s `FALLBACK_ERRORS` retry is broken for a ragged read, and must be
   fixed before this ships.** When a kernel numba built but cannot launch falls back
   to running its calls one by one (`self._python`), the `Fallback` gets the `Ragged`
   out of `values` and `_per_row` asks it for `.dtype`: `AttributeError` instead of a
   fallback. Either give `Ragged` an int `__getitem__` returning that row's
   namedtuple plus a `dtype` shim (~6 lines), or have the retry go back through the
   runner for a fresh representation (cleaner; the runner already knows how).
   **Not fixed in this spike.**
3. **Two different `Item` types over one column in one kernel.** There is one `flats`
   block per column, so the second reader would silently get the first's fields.
   Guarded with a `TypeError` at kernel-build time (3 lines); doing it properly means
   one block per (version, schema) pair, which the runner has to supply. No test in
   the suite does this today.
4. **A `Rows[...]` input on a row node** (a tree/table/scorecard feature) is excluded
   by `node.kind == "scalar"` in `compile_call`. `Layout.spec` would wrap the rag
   source in the node's `("row", ...)` feature tuple, which nothing reads today;
   leaving it out costs nothing and avoids an untested shape.
5. **`_boxed(..., kernel)`** — the representation now depends on whether the unit is
   a `Kernel` or a `Fallback`, which is threaded through `_external` as a boolean.
   It works, but the choice belongs with the unit; this is the ugliest line of the
   change.

## Verdict: ship it

~120 lines across six files, no new numba machinery, and it removes the largest
performance cliff in the framework. Fix items 2 and 3 above, drop the flag, and make
it the behaviour. `DECIDER_GAPS.md` §3 calls the parent-grain half of this gap "what
decides whether native decider is fast". On this evidence it does, and the answer is
yes: the construct the project's real workload is built out of — a `decider.loop`
over bundles reading the order's items — lands within 1.2x of hand-written numba.

**Shippable as written:** `Ragged` and `build_ragged` in `rows.py`, the `("rag", ...)`
source and the `flats` argument in `kernel.py`, the `Layout`/`Kernel` plumbing in
`units.py`, the `packed.py` one-liner.
**Spike-quality:** the `RAGGED_IN_KERNEL` env flag (it should just become the
behaviour), the `_boxed(..., kernel)` boolean, and the missing fallback-retry fix.
**Not written:** tests. The existing `tests/run/test_rows.py`,
`test_mixed_representations.py` and `test_value_shapes.py` agree on every value with
the flag on, but they were written for the `Fallback` path; none exercises a ragged
read inside a packed loop, which is the whole point, and two of them assert the
limitation (see above).

## Not done, and why

- **The `each("items", item_flow)` child-grain half of §3.** This costs the
  *parent*-grain half, the one the benchmark table is about. Per-item rules as
  ordinary steps are a separate, larger change and this says nothing about them.
- **A `decider.List[Item]` / `for i in items:` protocol** (the user's follow-up in
  the doc). `Rows[Item]` is arrays-per-field; iterating items as records needs a
  namedtuple-per-item value — numba can build one, but it is a different
  representation question.
- **`str` fields.** Unchanged: still refused, still waiting on the semantic-`str`
  work. `notes/nested-rows.md` records why a span would be the natural fit.
