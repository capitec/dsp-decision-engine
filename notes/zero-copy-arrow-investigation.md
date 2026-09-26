# Zero-copy Arrow gather in kernels (spike)

**Question.** Can kernels read every input column straight out of polars'
Arrow buffers (values plus validity bitmap, read per row inside the kernel)
instead of copying it into numpy first? What would that do to the design and
to speed? Also: can `score(dict)` go through polars/Arrow too, so there is
only one input path?

**Method.** The prototype is behind `DECIDER_ZERO_COPY=1` and off by default.
With the flag off, the boundary works as it does today. With it on,
`sm_columns` does not copy float64/int64/int32 columns that have nulls, or any
bool column. It hands back the Arrow values address, the validity bitmap
address and the bit offset. Kernels read each validity as a
`(uint8 bitmap, bit offset)` pair, and a bool column as the same kind of pair,
using a few LLVM instructions in the fused-kernel intrinsic. MISSING_AS fills
happen in the kernel (`("fill", col, mask, value)`). A REQUIRED null is still
an error; the count comes from Arrow's null count, with no scan.

Engine arrays reach kernels in the same shape: a mask is `np.packbits`ed, a
column with no nulls gets one shared all-ones bitmap, and `score()` packs its
bools. So each kernel compiles one signature, whatever the source. `State`
keeps an in-place column packed and unpacks it the first time Python reads it
(interpreted mode, fallbacks, frame steps, branch/loop row subsets,
sessions). Numeric columns under 4096 rows are still copied
(`DECIDER_BORROW_ROWS`).

Pipelines measured:
- flagship: 5 clean floats, 3 steps.
- wide: 40 clean floats, 40 floats with 10% nulls (half MISSING_AS, half
  OPTIONAL) and 4 bools, in 20 fused steps.
- tree: 127 nodes, 16 float and 2 int inputs, all clean.

The benchmark scripts are `benchmarks/zero_copy_spike.py` (fused `run()`
split into extract / driver / kernel / output) and
`benchmarks/score_through_arrow.py`. The baseline is the parent commit
`69c5040`. The dev box was shared with swap full: batch numbers move ±30%;
single-record numbers are medians of three alternating base/prototype runs.

**Results** (fused mode; µs for single-record, ms for batches; batch cells
are base → prototype):

| case | score p50/p99 | 1-row `run` p50 | 100k rows | 1M rows | 1M split, base (extract / driver / kernel / output) | 1M split, prototype |
|---|---|---|---|---|---|---|
| flagship, base | 29.8 / 37 | 128 | 0.73 | 9.5 | 0.7 / 0.5 / 8.2 / 0.07 | |
| flagship, prototype | 30.4 / 36 | 128 | 0.72 | 9.5 | | 0.8 / 0.5 / 8.1 / 0.06 |
| wide, base | 290 / 367 | 844 | 83–108 | 2354–2726 | 1330–1760 / 495–568 / 400–530 / 0.25 | |
| wide, prototype | 291 / 339 | 907 | **24.5** | **364–374** | | 1.5 / 11 / 350–361 / 0.2 |
| tree, base | 60.5 / 70 | 229 | 21.1 | 207–219 | 0.5 / 9.5 / 155–163 / 43–45 | |
| tree, prototype | 65 / 73 | 234 | 21.0 | 207–209 | | 0.6 / 9.4 / 155 / 43 |

Checks:
- **Correctness.** Outputs are bit-for-bit identical to the baseline in 48
  cases (3 pipelines × fused/stepped/interpreted × 200k-row, 50k-row sliced,
  9k-row oddly sliced, 100-row and 1-row frames, plus 285 `score` dicts), in
  three settings: flag off, flag on, and flag on with every frame size read
  in place. With the flag off the suite passes (1400 passed). With it on,
  all pipeline tests pass. The 17 failures are all boundary unit tests that
  inspect `ExtractedColumn.values` and expect a filled numpy copy.
- **Compile time (cold cache).** flagship 0.58 s → 0.58 s, tree 7.5 s →
  7.6 s. Wide was 3.2 s for `run` plus 3.1 s for `score` (two signatures);
  it is now 4.7 s in total (one signature: bit reads and fill selects add IR).
- **Output assembly.** `pl.Series(float64 ndarray)` already shares memory, so
  output costs 0.03–0.25 ms for numeric outputs. Two costs are unrelated to
  inputs:
  - A nullable output uses `scatter(None)`, 3.5 ms per 1M-row column;
    `Series.set(mask, None)` takes 0.3 ms.
  - Tree `Literal` outputs are decoded through an object array, which is
    43 ms of the tree's 207 ms at 1M rows;
    `pl.Series(choices, dtype=pl.Enum(choices)).gather(codes)` takes 3–4 ms.

  Handing polars a validity bitmap with no copy at all would mean exporting a
  C `ArrowArray` with a release callback that holds a Python reference. For a
  gain of 0.3 ms per column it isn't worth it.
- **`score()` through polars/Arrow**, p50 µs, with `score()` today →
  through the frame:
  - flagship: 30.9 → 126. Of that, `pl.DataFrame([record])` is 15,
    `State.from_frame` 55–60 (of which the export is 1.7) and kernels plus
    the dict about 35.
  - tree: 60 → 231 (frame 35, `from_frame` 123–130).
  - wide: 295 → 995 (frame 118–140, `from_frame` 560–580).
  - `pl.DataFrame({k: [v]})` is 1.5–2× slower again.
  - Most of `extract_frame`'s one-row cost is Python: hashing the tuple of
    `Input`s for the plan cache, the per-column numpy views and
    NamedTuples. If all of that were cut, the floor is still frame +
    export + one C call ≈ 20 µs on flagship, which grows with width (building
    the frame alone is 140 µs at 84 inputs).

**What was found on the way.** Today's nullable column path is the only
expensive part of the boundary. Each 1M-row nullable column costs 30–45 ms
on this box: a branchy fill loop (~4 ms), 9 bytes a row of fresh memory
page-faulted under memory pressure, then the driver's MISSING_AS
`np.where(...).astype` (~25 ms). Clean numeric columns are already zero-copy,
strings are already spans, and the kernel itself is unchanged. That is why
wide goes 6–7× faster at 1M rows and 4× at 100k, while flagship and tree
don't move.

**Design impact.**
- **Kernel ABI.** A validity is always a `(bitmap, offset)` pair; a bool
  column is a pair or an array. One specialisation serves Arrow input,
  `score()` dicts and engine intermediates, with or without nulls; before,
  wide compiled twice. A `missing_as` fill is baked into the kernel key: it is
  IR, not a param, so changing it recompiles.
- **Multi-chunk and sliced frames.** Unchanged: polars rechunks on export,
  and bit offsets cover slices (tested at offsets 3 and 13).
- **Lifetime.** Bitmaps and values are views owned by the moved `ArrowArray`
  (`_Export`), as clean columns already are. The export lives as long as the
  `State` does, which for a session is the session.
- **Sessions.** `set`, `rewind` and `replace` go through
  `State.read`/`write`/`restore`, which unpack on demand; `edit.py` now
  reads through `read()`.
- **Where zero-copy stops.** A branch arm or loop body (a row subset) unpacks
  (a copy). A packed branch/loop whose guarded input has nulls takes the
  unpacked path, which was already the rule.
- **Modes.** Interpreted mode, Python fallbacks and frame steps see unpacked
  arrays, so the three modes stay equivalent (verified bit for bit).
- **Row nodes.** Trees and tables read inputs through the same `Layout`
  sources, so they need no change.
- **`score(dict)`.** It never touches Arrow; it now passes the shared
  all-ones bitmap instead of an `np.ones` per OPTIONAL column. p50 moved by
  0 to +4 µs (tree), within noise.
- **Size.** +170/−22 lines in `decider/`. Adopting it properly would delete
  the byte-mask and fill loops in `sm_columns` (~40 lines of C) and the
  driver fill, and would rewrite ~17 boundary unit tests.
- **Risks.**
  - The value under a null slot is whatever polars left there, not NaN, and
    `State.values` can now hold a `(bitmap, offset)` tuple for a bool. Only
    `State` internals see raw values (public reads unpack), but any future
    code that assumes a filled ndarray will be wrong.
  - Wide's one-row `run` is +60 µs (7%) from per-input packing on 84
    inputs.

**Recommendation: adopt for specific column kinds, not as a unified single
path.**
1. Nullable numeric columns and bool columns in batches of 4096 rows or more
   should be read in place, with validity passed to kernels as bitmaps and
   `missing_as` fills applied in the kernel. That is the only case with a
   real gain (6–7× on null-heavy wide frames), and it removes the one
   pathological cost in the boundary. Clean numeric columns and string spans
   already work this way, so this finishes "every kind is zero-copy" for
   batches.
2. Adopt the kernel-side half (bitmap validity, in-kernel fills) regardless:
   it gives one kernel specialisation per unit and no `np.ones` per call.
3. Don't route `score(dict)` through polars. It costs +95 µs p50 on flagship
   (4×), +170 µs on the tree and +700 µs on wide, and even an ideal Arrow
   builder would stay ~20 µs+ above today's path and grow with width.
   Unify at the kernel ABI instead: `score()` already builds the same
   `(values, bitmap)` columns from the dict that the shim builds from Arrow,
   so the kernels, `Layout` and null handling are already one
   implementation. Only the 30-line dict loader is separate.
4. Independent batch wins worth taking first, each a few lines:
   `Enum.gather` for `Literal` outputs (−40 ms per 1M tree rows) and
   `Series.set(mask, None)` for nullable outputs.

**How to switch it on:** `DECIDER_ZERO_COPY=1` (read at import time);
`DECIDER_BORROW_ROWS=<n>` sets the in-place threshold for numeric columns.
