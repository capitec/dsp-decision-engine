# Landing `Rows[...]` in the shared array kernel

Finishes `notes/nested-rows-in-kernel.md`: the three items it left owed, plus the
one-line `_probe_signature` fix from `notes/nested-item-fields.md`, made the
default behaviour, and re-measured. `DECIDER_RAGGED_IN_KERNEL` no longer exists;
a `Rows[Item]` step (scalar, all its `Rows[...]` inputs REQUIRED or `missing_as`)
now always joins the shared kernel instead of running one dispatcher call per row.

I chose to add this note rather than extend the spike note: that one is a
record of the exploration (three probed forms, the readonly-array trap, the
measured ratios); this one is the shipping diff. Keeping them separate keeps
the spike's "here's what I tried and why" readable on its own.

## What changed

1. **The fallback retry, fixed.** `Kernel.run` catches a runtime `FALLBACK_ERRORS`
   from the fused kernel and retries by calling each of its steps once per row
   in Python (`Fallback.run`). That retry asked `_per_row` for `.dtype` on
   whatever `values[v.id]` held; for a ragged `Rows[...]` input that's a
   `Ragged` (flat field arrays plus per-row `lo`/`hi`), which has no `.dtype`
   and blew up with `AttributeError` instead of falling back.

   Fix: `Ragged` now carries the schema's namedtuple class (`nt`) alongside
   its arrays, and a `Ragged.row(i)` method returns row `i` in exactly the
   shape `build_rows` would have given it — the shape the per-row dispatcher
   already expects. `_per_row` special-cases `Ragged` to build that list.
   I considered the note's other option ("have the retry go back through the
   runner for a fresh representation") and didn't take it: it would thread a
   rebuild callback from `stepped.py` into `Kernel`, which doesn't otherwise
   know about runners or column-building. Giving `Ragged` the one field it was
   missing keeps the fix inside `Kernel.run`'s existing shape.
   Test: `tests/run/test_rows.py::test_a_kernel_that_fails_at_run_time_falls_back_to_the_per_row_dispatcher`
   monkeypatches a real, already-joined kernel's `.fn` to raise a `TypingError`
   and checks the run still gives the right answer and that `unit._python`
   got populated (proof the retry, not the original path, ran).

2. **The flag dropped.** `njit.py`'s `compile_call` no longer gates on
   `RAGGED_IN_KERNEL`; a `Rows[...]` input joins the kernel whenever the node
   is `scalar` and none of its `Rows[...]` inputs is OPTIONAL — full stop.
   `stepped.py`'s `_boxed` lost the same guard. `benchmarks/bundle_search.py`'s
   docstring no longer mentions the env var.

3. **The four tests, repointed at the new behaviour.**
   - `tests/run/test_rows.py::test_rows_compiles_but_cannot_join_the_shared_array_kernel`
     → `test_rows_joins_the_shared_array_kernel`: asserts `exe.fallbacks() == {}`
     and that the unit's `.ragged` is non-empty (flat arrays reached the
     kernel), for both `stepped` and `fused`.
   - `tests/run/test_value_shapes.py`'s `SHAPES` table: `rows_of_items`'s
     "does a kernel hold it" flag flipped `False` → `True`, same table, no
     special-casing.
   - Added `test_a_loop_reading_rows_packs_into_one_kernel` (`test_rows.py`):
     a real `decider.loop` whose body reads `Rows[Item]`, asserting
     `"p/search" in exe.runner.packed`. Neither test above used a loop, and
     the note itself flagged this as the one thing nothing in the suite
     exercised ("a ragged read inside a packed loop, which is the whole
     point"). `packed.py` needed **no changes** — `Layout.source`'s ragged
     handling was already generic across a plain kernel and a whole
     branch/loop kernel, so once `compile_call` stopped refusing the join,
     loops started packing for free. Verified by hand first, then pinned.

4. **The documented limitations, kept honest and tested.**
   - `Rows[Item] | None` still can't join (one kernel signature, no in-band
     `None` for a namedtuple of array views): unchanged in `njit.py`, now with
     a comment explaining why at the point that excludes it, and
     `test_an_optional_rows_input_gives_the_step_none_for_a_null_list` now
     also asserts the fallback reason is still reported for it.
   - Two different `Item` types over one column: `Layout.source`'s 3-line
     `TypeError` guard is unchanged. Worth recording what I found tracing it:
     **it's unreachable through `flow`/`branch`/`loop`.** `resolve()`'s wiring
     pass already refuses two readers of one *input column* declaring
     different types (`WiringError`, at plan-build time, before
     `compile_plan` exists), and a `Rows[...]` value can only ever be a leaf
     input — nothing can produce one as a step output — so any two `Rows[...]`
     readers of the same column name anywhere in a pipeline hit that check
     first, with a different error. The guard in `Layout.source` only matters
     for a caller that drives `Layout` directly (as the low-level
     `tests/compile/` suite already does elsewhere), so that's where I tested
     it: `tests/compile/test_compile_units.py::test_two_item_types_over_one_column_in_one_kernel_is_refused`
     builds one `Version` and calls `Layout.source` on it twice with two
     schemas. My first attempt wrote this as an `Engine().bind(...).run(...)`
     test with two steps in one `flow`; it failed with `WiringError` from
     `resolve()`, not the `TypeError` I expected — which is how I found the
     above.
   - A `Rows[...]` input on a row node (tree/table/scorecard): still excluded
     by `node.kind == "scalar"` in the join condition; comment updated to say
     so. No test added — nothing in the suite gives a row node a `Rows[...]`
     input today, and building one is a separate feature.

5. **The `_probe_signature` span fix (`notes/nested-item-fields.md`).** A step
   comparing a `Rows[Item]` `str` field against a `str = param(...)` fell back
   to Python. The note diagnosed one line: `_probe_signature`'s `SPAN in ins`
   test only sees a top-level `bytes` input typed `SPAN`; a `Rows[...]`
   input's numba type is a `NamedTuple`, so a `str` field's span lives at
   `ins[j].types[k].dtype`, invisible to `in`. Fixed with a small `_has_span`
   helper that also looks inside a `NamedTuple`/`NamedUniTuple` input.

   That line was necessary but **not sufficient** — I found this by
   reproducing the failure before touching anything, then again after the
   probe fix, and it still failed, just later and with the previous spike's
   *other* owed bug (item 1 above, before I'd fixed it): a `str` param's
   runtime value is a plain Python `str` unless `stepped.py`'s `_bundle`
   converts it to a UTF-8 `(address, length)` span first, and that conversion
   was gated on `_reads_bytes(call)` — true only for a *top-level* `bytes`
   input, never for a `Rows[...]` item's `str` field. So the probe now
   compiles the right signature, but the kernel launch got handed a numba
   `unicode_type` where it wanted a span, which is itself a `FALLBACK_ERRORS`
   at first-call lazy compilation of the fused kernel body — the exact retry
   path item 1 fixes. `_reads_bytes` is renamed `_reads_span` and now also
   matches a `Rows[...]` input whose schema has a `str` field. With both
   fixes in, `eq_param` (`items.left[j] == want`) compiles and runs in the
   kernel with no warning.
   Test: `tests/run/test_nested_str_fields.py::test_a_str_field_compared_against_a_param_runs_in_the_kernel`.

## Suite

`uv run pytest -q -p no:randomly`: **1959 passed, 1 xfailed**, 0 failed — the
1955 the spike measured with the flag off, plus the new coverage above, minus
nothing (the four tests were repointed, not deleted).

## Re-measured

Same box, same methodology as the spike note (one process, back to back,
`mode="fused"` unless noted); read the ratios, not the absolute numbers — this
box runs other people's test suites in the background.

### `benchmarks/bundle_search.py`

**`score()`, one order, p50 / p99** (spike's "today" column is its own
measurement of the flag-off path, reproduced here for reference since the
flag no longer exists to re-run it):

| items | bundles | today (spike, flag off) | landed (this branch) | hand-written `@njit` |
|---:|---:|---:|---:|---:|
| 9 | 511 | 49.5 / 64.8 ms | **0.327 / 0.451 ms** | 0.010 / 0.012 ms |
| 16 | 65,535 | 7,174 / 7,704 ms | **10.40 / 13.21 ms** | 9.74 / 11.13 ms |

`run()`, p50:

| items | bundles | today (spike) | landed | hand-written |
|---:|---:|---:|---:|---:|
| 9 | 511 | 469.6 ms / 100 orders | **6.17 ms / 100 orders** | 0.83 ms / 100 orders |
| 16 | 65,535 | 13,262 ms / 1 order | **49.6 ms / 4 orders** | 38.8 ms / 4 orders |

`[fused] packed loops: ['order/search']`, `[fused] fallbacks: {}` at both
sizes — same two-line proof the spike led with. At 16 items the landed
`score()` is **1.07x** of hand-written (was 1.2x in the spike's own run,
which was itself already inside this box's noise band); the headline claim
("with list inputs in kernels, the native pipeline is the fast pipeline")
holds with a little more room than the spike measured, not less.

`stepped` mode, included for completeness, is unaffected by any of this:
stepped never packs a loop (its whole design pauses at every node), so a
`decider.loop` over `Rows[Item]` in `stepped` still costs one Python-level
iteration per mask — 159 ms / score() at 9 items, 9.0 s at 16 — exactly the
"today" shape, because nothing about stepped mode's loop execution changed.
The fix is a `fused`-mode (and packed-branch/loop) story; `stepped` runs each
`Rows[Item]` call as its own one-call kernel now, which is faster in
isolation, but the loop around it still isn't packed in that mode by design.

### `benchmarks/nested_item_fields.py`

Unaffected in shape — this branch doesn't touch `str` field handling itself,
only the one param-typing line above — and the numbers land within this box's
usual run-to-run noise of the note's own (`int + str, == literal` 73.6 us here
against 62.0 us there; both boxes were "loaded and shared" at the time).
Full output: nothing regressed, nothing warns that didn't before.

## Item 7: does the record-array argument still hold?

**Short answer: no, not as a performance argument. What's left is ergonomics.**

`notes/nested-item-fields.md`'s case for a record array per row rests on one
number: constructing a namedtuple-of-arrays argument for a *per-row dispatcher
call* costs ~0.2 us per field (1.20 us at k=2, 2.35 us at k=8), because numba
unboxes one tuple member at a time across the Python↔numba call boundary. The
in-kernel design removes exactly that boundary: for an `n`-row kernel launch,
`Item` is built *inside* the compiled loop, once per row, by slicing `flats`'
arrays — no call, no boxing, per row.

Measured directly (`Kernel.run` timed on a pre-built `Ragged`, so
`build_ragged`'s own column-building cost — which scales with k regardless of
design, and dominates the two field counts almost identically — is held out;
`.scratch/rows_kernel_only_bench.py`, not committed, numbers below):

| items in the one row | k=2, p50 | k=8, p50 |
|---:|---:|---:|
| 3 | 4.36 us | 5.62 us |
| 30 | 4.40 us | 5.62 us |
| 300 | 4.66 us | 6.19 us |
| 3000 | 4.84 us | 5.96 us |

Two things fall out of this table:

- **Flat against item count, for a fixed k.** Building the per-row `Item`
  inside the kernel is O(1) in how many items the row holds, at both k=2 and
  k=8 — confirming the per-*row* unboxing cost the note measured is gone, not
  just reduced.
- **k still costs something — ~0.2 us/field, the same magnitude as before —
  but it moved.** It's not the per-row construction; it's `flats` itself
  being a wider argument tuple (`2 + k` arrays instead of 4), which numba's
  dispatcher still has to individually type-check and marshal once per
  **kernel launch**. That's the same mechanism the note found, relocated from
  "once per row" to "once per `score()`/`run()` call, whatever it does inside".

That relocation is the whole story. For `score()` (one row), the k-dependent
cost is the entire story too — a real but small 1.2–1.5 us against a
total `score()` cost of tens of microseconds (my end-to-end run,
`.scratch/rows_field_count_bench.py`: k=2 "find" at 3 items/row was 59.0 us
p50, k=8 was 96.6 us — most of that gap is `build_ragged` doing 8 Python-level
field extractions instead of 2 from the record's own dict, not the kernel
launch; the isolated table above is the part attributable to the kernel
itself). For `run()` or a `loop()` over N rows or iterations — the shape this
whole feature exists for — that same fixed launch cost is paid **once**,
however large N is, so it disappears into the noise; `bundle_search.py`'s
9-item, 511-mask search above pays it once for the whole search, not once per
mask.

So: a record array would still shave that last ~1–1.5 us off a `score()` call
against a schema with several unused fields, by making `flats` one array
instead of k — a real number, but under 2% of `score()`'s own floor, and
irrelevant to `run()`/`loop()` throughput, which is where this feature's
numbers matter (531x, 953x). The remaining case for it is what the note's own
numbers already said louder than the performance one: `for i in items:` and
`i.el_1` are what the user asked for, and the positional form
(`items.el_1[j]`) is what `Rows[Item]` makes them write instead. I'd only
revisit the performance angle if a workload turns up needing very high
`score()` call rates against `Item` schemas with dozens of unused fields —
nothing in the suite or the benchmarks looks like that today.
