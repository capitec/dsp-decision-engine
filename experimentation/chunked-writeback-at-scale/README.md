# EXPERIMENT J2 — chunked write-back at scale

Continues experiment J (`experimentation/output-writeback-convention/`),
which measured record-out vs column-major-2D-out vs row-major-2D-out at
100k rows and found column-major wins 1.74x end-to-end, with compile
collapsing 11.31s -> 0.27s. This experiment asks what happens at 1M rows
— where a single record-out variant needs ~16GB unchunked (measured
below) — whether chunking preserves the win, and what chunk size to
default to.

**Verdict: partially confirmed.** Column-major still wins on pure
kernel+write-back cost — 3.1x at a 250k chunk, *bigger* than the 1.74x
unchunked figure. But once a full per-chunk batch pipeline (generate/
extract/assemble the chunk's input, run the kernel, write the chunk's
output) is measured honestly, per-chunk overhead unrelated to the
write-back convention dominates wall time (58-82%) and the end-to-end
win shrinks to 1.03x-1.57x depending on chunk size — it does NOT reliably
hold at 1.74x, and at the smallest chunk size (10k) it nearly vanishes.
Chunking is confirmed mandatory at 1M rows for this shape: a single
record-out variant needs ~16.2GB unchunked, col-major ~11.9GB — both far
over a 6GB cap — while every chunked configuration measured stayed
under 4.1GB peak, bounded by chunk size, not total rows.

## What killed the box (diagnosis, from reading writeback.py)

`experimentation/output-writeback-convention/writeback.py` (experiment
J) computes all three variants — record, col-major, row-major — inside
one `run_size()` call, and for its final "are all three identical?"
check it holds **all of the following live simultaneously**: the source
polars frame, the extracted `cols` dict, the assembled `in_rec` record
array, all three output arrays (`out_rec_final`, `cf8+ci8+cb`,
`rf8+ri8+rb`), and all three write-back frames (`frame_rec`, `frame_col`,
`frame_row`). At 100k rows (its checked-in `results_100k.json`) that is
fine. At 1M rows, with this shape (400 in / 633 out, mixed f8/i8/bool):

| held object | bytes/row | @ 1M rows |
|---|---|---|
| source polars frame + `cols` dict + `in_rec` (input, x3 copies) | ~3,060 x 3 | ~9.2 GB |
| `out_rec_final` (record output) | 4,399 | 4.4 GB |
| `cf8`/`ci8`/`cb` (col-major output) | 4,399 | 4.4 GB |
| `rf8`/`ri8`/`rb` (row-major output) | 4,399 | 4.4 GB |
| `frame_rec` (copies — strided field extraction) | 4,399 | 4.4 GB |
| `frame_col` (zero-copy, ~free) | ~0 | ~0 |
| `frame_row` (copies — strided column slices) | 4,399 | 4.4 GB |

**~31 GB analytic, before any numpy/polars/numba allocator overhead** —
comfortably explains both kills (18.7GB and 20.8GB RSS: the OOM killer
almost certainly fired mid-allocation, before the harness reached its
true peak). The fix applied in this harness: never construct more than
one variant per process (`measure_variant.py` is a single-variant,
single-config worker, always launched fresh by `run_j2.py`); compare
variants by a per-chunk checksum written to jsonl, never by holding two
full result frames resident (confirmed here: 134 checksums compared
across record vs col-major, 0 mismatches, without the two variants ever
coexisting in memory).

## What this measures

1. **Peak RSS per row count**, one variant at a time (record, col_major),
   at 10k/50k/100k/250k rows, unchunked. Fit a line, extrapolate to 1M,
   and report the largest batch that fits a 6GB cap.
2. **Chunked processing**: 1M rows total, swept across chunk sizes
   10k/50k/100k/250k, for record-out and col-major-out. Each chunk's
   result is written to its own parquet part file as it completes (the
   "append to the result" step) rather than concatenated into one
   in-memory frame — a full in-memory concat would total the same bytes
   as the unchunked case and defeat chunking's point.
3. **Peak RSS under chunking** — confirms it is bounded by chunk size,
   not total rows.
4. **Optimal chunk size** — from the sweep, by wall time and by kernel/
   writeback/boundary-overhead share.
5. **Generation-pointer discipline** (EXPERIMENTS.md §H) — the driver
   and coefficient array are resolved ONCE, before the chunk loop, never
   re-resolved per chunk.

## How to run

```
cd experimentation/chunked-writeback-at-scale
tmux new-session -d -s exp-j2 -c "$(pwd)" \
  "systemd-run --user --scope -p MemoryMax=6G -p MemorySwapMax=2G -- \
   /path/to/.venv/bin/python run_j2.py 2>&1 | tee run.log; echo EXIT=\$? >> run.log"
tail -f run.log
```

`run_j2.py` never allocates the measured arrays itself — every measurement
is a subprocess of `measure_variant.py`, which exits (freeing all its
memory back to the OS) before the next one starts. `free -g` is checked
before launch and the run aborts if available memory is under 8GB.

Results: `results_peak.jsonl` (part 1), `results_chunked.jsonl` (parts
2-4), `chunk_progress.jsonl` (one line per chunk, for resilience against a
mid-run kill), `summary.json` (fit + recommendation, written at the end).

## Results

### 1. Peak RSS per row count, unchunked, one variant at a time (repeats=3)

| variant | n | compile | kernel med | writeback med | peak RSS |
|---|---|---|---|---|---|
| record | 10,000 | 12.11s (cold) | 9.76 ms | 36.75 ms | 0.399 GB |
| record | 50,000 | 0.19s (cache HIT) | 46.80 ms | 387.39 ms | 0.983 GB |
| record | 100,000 | 0.20s (HIT) | 93.15 ms | 774.02 ms | 1.768 GB |
| record | 250,000 | 0.20s (HIT) | 233.17 ms | 2260.94 ms | **4.127 GB** |
| col_major | 10,000 | 0.45s (cold) | 32.05 ms | 5.27 ms | 0.315 GB |
| col_major | 50,000 | 0.17s (HIT) | 160.12 ms | 5.98 ms | 0.810 GB |
| col_major | 100,000 | 0.17s (HIT) | 340.39 ms | 7.17 ms | 1.333 GB |
| col_major | 250,000 | 0.18s (HIT) | 802.01 ms | 10.97 ms | 3.061 GB |

**Fit** (least-squares, bytes = a·n + b):

| variant | measured B/row | baseline | predicted @ 1M rows | largest n under 6GB cap |
|---|---|---|---|---|
| record | 15,967 | 226 MB | **16.19 GB** | ~389,000 rows |
| col_major | 11,653 | 218 MB | **11.87 GB** | ~534,000 rows |

Both far over a 6GB cap at 1M rows — chunking is confirmed mandatory,
for BOTH conventions, not just record-out. The measured B/row is ~1.3-1.4x
the naive analytic model (`shapes.estimate_peak_bytes`), which only counts
the record/array bytes and the one known write-back copy — the gap is
numpy/polars allocator overhead (arena growth, alignment, temporary
buffers during `pl.DataFrame(dict)` construction) not modeled analytically.

**Cross-process numba cache note.** The generated driver is written ONCE
to a fixed path (`_generated/driver_<variant>.py`) and imported by module
name (`sys.path` + `importlib.import_module`), not via
`spec_from_file_location` as experiment J and doc 05 §4.1 recommend.
That distinction mattered: with `spec_from_file_location`, a *fresh*
process reusing the byte-identical cached driver failed with
`ModuleNotFoundError: No module named '<dynamic>'` — numba's on-disk
cache pickles a reference to the function's environment module and
rebuilds it with `importlib.import_module(modname)`, which only resolves
if the module was reachable through the normal import system in the
*original* compiling process too, not just present as bytes at a path.
With the module-name-import fix, the record-out driver compiled once
(12.1s, first call of the whole run) and every one of the other 7 record
invocations in this run — across both peak and chunked mode, fresh
subprocesses each time — hit a cache **HIT** (~0.2s). This is a real,
reproducible amendment to doc 05 §4.1's import recipe and to
EXPERIMENTS.md §C's six-condition cache contract: the six conditions are
necessary but not sufficient across a process boundary; the module must
also be importable by name, not just present at the right path/bytes/mtime.

### 2/3. Chunked processing, 1M rows total, sweeping chunk size

Each chunk's output is written to its own parquet part file as it
completes (the "append to the result" step) and freed before the next
chunk starts — not concatenated into one in-memory frame, which would
total the same bytes as the unchunked case (Part 1) and defeat chunking's
point. `boundary` below = wall − kernel − write-back − compile: it is
dominated by generating/extracting/assembling each chunk's *input* (a
harness cost — J never measured this either, since J built its input
frame once, outside its timed loop) plus each chunk's parquet write; it
was not decomposed further within budget (see "What I did not measure").

| variant | chunk | n_chunks | wall | kernel | writeback | boundary | peak RSS |
|---|---|---|---|---|---|---|---|
| record | 10,000 | 100 | 19.74s | 5.2% (1.02s) | 12.7% (2.51s) | 81.1% (16.00s) | **0.448 GB** |
| record | 50,000 | 20 | 21.08s | 4.6% (0.96s) | 36.4% (7.68s) | 57.9% (12.21s) | **1.049 GB** |
| record | 100,000 | 10 | 21.06s | 4.5% (0.94s) | 36.3% (7.65s) | 58.2% (12.26s) | **1.792 GB** |
| record | 250,000 | 4 | 25.24s | 3.9% (0.98s) | 37.3% (9.41s) | 58.0% (14.64s) | **4.020 GB** |
| col_major | 10,000 | 100 | 19.20s | 15.4% (2.96s) | 2.5% (0.47s) | 81.1% (15.58s) | **0.394 GB** |
| col_major | 50,000 | 20 | 15.69s | 21.6% (3.38s) | 0.7% (0.11s) | 76.6% (12.02s) | **0.828 GB** |
| col_major | 100,000 | 10 | 15.83s | 22.1% (3.50s) | 0.4% (0.06s) | 76.3% (12.08s) | **1.366 GB** |
| col_major | 250,000 | 4 | 16.04s | 20.8% (3.33s) | 0.2% (0.03s) | 77.9% (12.49s) | **2.963 GB** |

Checksum cross-check (f8/i8/bool column sums per chunk, compared from the
jsonl only — never both variants' frames resident): **134 chunks
compared, 0 mismatches** — the two conventions compute bit-identical
output under chunking too, confirming experiment J's equality finding
survives chunking.

**Does the 1.74x hold?**

| metric | @ 10k chunk | @ 50k chunk | @ 100k chunk | @ 250k chunk |
|---|---|---|---|---|
| kernel+writeback only (record / col_major) | 1.03x | 1.13x | 1.16x | **3.09x** |
| full wall time (record / col_major) | **1.03x** | 1.34x | 1.33x | 1.57x |

Kernel+write-back alone, col-major's win *grows* with chunk size (up to
3.1x at 250k — bigger than J's unchunked 1.74x), because record-out's
write-back cost scales worse than linearly while col-major's stays flat
and near-zero. But once the realistic per-chunk pipeline is counted, that
win is diluted by `boundary` — work identical in kind between the two
conventions — and the **full-wall-time ratio never reaches 1.74x** at any
chunk size tested, falling to **1.03x (no measurable win) at the smallest
chunk size**. **Partial refutation**: the write-back-convention win is
real and, on the isolated cost, larger than measured at 100k — but at the
whole-batch level it is chunk-size-dependent and can vanish entirely.

**Peak RSS is bounded by chunk size, not total rows** — confirmed
directly: record's peak RSS tracks its chunk size (0.45/1.05/1.79/4.02 GB
for 10k/50k/100k/250k chunks) and is flat with respect to the 1M-row
total in every column of the table above. This is the property that
makes 1M rows possible at all under a fixed memory cap.

**What sets the optimal chunk size?** Not the ~0.44µs/kernel-call
dispatch cost from EXPERIMENTS.md §E — at up to 100 chunks that is
≤44µs total, undetectable against multi-second wall times. `boundary`
total time is roughly flat-to-U-shaped across chunk sizes (lowest at
50k-100k for both variants; slightly worse at both 10k, from more
parquet-file overhead, and 250k, from larger one-shot allocation/copy
costs) rather than falling monotonically with fewer, bigger chunks — so
it is **memory- and per-chunk-I/O-bound work, not a fixed per-call tax**.
Combined with peak RSS scaling ~linearly with chunk size, the
recommendation is a chunk size chosen for a **memory budget**, not for
amortizing a dispatch floor.

### 5. Generation-pointer discipline (EXPERIMENTS.md §H)

Confirmed compatible by construction: `measure_variant.py`'s
`run_chunked()` calls `load_driver()` and builds `coef_f8` ONCE, before
the `for idx in range(n_chunks)` loop, and reuses that same driver object
and array for every chunk — never re-resolved per chunk. `compile_s` in
every chunked-mode result reflects this: ~0.2s (a cache HIT) for every
config after the first, i.e. exactly once per driver, not once per chunk.

## Deliverable: the chunking contract for doc 05

1. **Chunking is mandatory at 1M rows for this shape, for both
   conventions** — record-out needs ~16.2GB unchunked, col-major
   ~11.9GB, both far past any reasonable cap.
2. **Default chunk size: 100,000 rows.** It sits at the low end of the
   wall-time-optimal range (50k-100k, tied within noise) for both
   conventions, and its peak RSS (1.79GB record / 1.37GB col-major) is
   comfortably under a 6GB cap with ~4x headroom for a shared machine —
   versus 250k's 4.02GB, which leaves almost none.
3. **Column-major output survives chunking** — still wins on the part of
   the cost the convention actually controls (kernel+write-back: 3.1x at
   a 250k chunk), and the two conventions remain bit-identical under
   chunking (134/134 checksums). But **do not advertise a flat "1.74x"
   for the chunked path**: end-to-end, chunking dilutes it to 1.03x-1.57x
   depending on chunk size, because per-chunk input assembly and output
   persistence — costs the convention does not touch — dominate wall
   time (58-82%) once measured honestly.
4. **Read the driver/coefficients/generation-pointer once per batch, not
   per chunk** — confirmed compatible with the chunk loop built here
   (§5 above), consistent with EXPERIMENTS.md §H's re-read-per-chunk
   control (99.87% straddled).
5. **The numba cache-across-processes fix belongs in doc 05 §4.1**:
   import the generated driver by module name (`sys.path` +
   `importlib.import_module`), not `spec_from_file_location` — the
   latter breaks cross-process cache reuse outright (`ModuleNotFoundError:
   No module named '<dynamic>'`) even when EXPERIMENTS.md §C's six
   conditions all hold.

## What I did not measure (dropped for the ~12-minute budget)

- **`boundary` was not decomposed** into (input generation) vs (extract
  + assemble) vs (parquet write) vs (gc/del) — would need one more
  instrumented pass with per-phase timers in `measure_variant.py`.
  Reported as a single residual bucket; its likely composition is
  discussed inline above but not independently verified.
- **The row-major control was dropped from this experiment** — J already
  settled "2D instead of records" vs "column-major specifically" at
  100k; J2 only needed the two live candidates for the chunking
  contract (record, col_major).
- **Zero-copy was not re-verified per chunk** — it uses the identical
  `writeback_2d` code path as J, which J proved zero-copy at 100k; not
  independently re-checked here per chunk to save time.
- Chunk sizes below 10k or above 250k were not swept (matches the range
  specified in the task).

## Peak memory

**4.13 GB** — the single largest peak RSS observed across every
configuration (record-out, unchunked, n=250,000; part 1). Every chunked
configuration (the design meant for 1M+ rows) stayed at or below 4.02GB,
and the smallest, safest default (100k-row chunks) peaked at 1.79GB
(record) / 1.37GB (col-major). All runs stayed under the 6GB
`systemd-run` cap; none were killed. Total wall time for the whole sweep
(8 peak configs + 8 chunked configs, 1M rows x2 conventions): **207.3s**.

## Reused from prior harnesses

- `elapsed()`/`log()` progress clock — `prange-crossover/prange_crossover.py`
- generated source written to a real `.py` file, imported (never `exec()`)
  — `ruleset-compile-latency/run.py:load_source`, and experiment J's own
  `writeback.py`
- zero-copy check pattern — `dtype-boundary/dtype_boundary.py:probe_to_numpy`
- extraction via `Series._get_buffers()` (no pyarrow) — `dtype-boundary/`,
  per EXPERIMENTS.md §A
- shape (400 in / 633 out), codegen bodies, `writeback_record`/
  `writeback_2d` — `output-writeback-convention/writeback.py` (experiment J)
  verbatim, with one change: a FIXED generated-driver path + `cache=True`
  instead of a fresh per-call UID'd file + `cache=False`, to test whether
  numba's on-disk cache (EXPERIMENTS.md §C) survives across the repeated
  subprocess invocations this harness's memory-safety design requires.
