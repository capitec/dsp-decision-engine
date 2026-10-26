# Does the polars host lose to native polars expressions?

Date: 2026-10-10 · Script: `benchmarks/polars_engine.py`

## Purpose
Find out whether translating steps into native polars expressions would beat the host (the fused pipeline inside `map_batches`), and by how much, on pipelines people actually run. If it would not, polars bodies and native segments are not built.

## Method
Three pipelines: `flagship` (the three arithmetic steps of `benchmarks/modes.py`, 5 float columns); `fallback` (ten plain Python steps over 3 `date` and 3 `str` columns that the numba engine runs as fallbacks: tenure, age, weekend and month-end flags, name normalisation, code prefix/suffix tests, a region lookup, a parsed branch number, a segment string); `tree` (the 127-node document of `benchmarks/tree_walk.py`).

Configurations: `fused-mem` (eager `run(df)`), `fused-io` (read parquet, run, write parquet); `host-mem` (`run(df.lazy()).collect()`), `host-memstream` (the same with `collect(engine="streaming")`), `host-stream` (`run(scan_parquet(...)).sink_parquet(...)`); `native-mem` and `native-stream` (hand-written polars for the same logic, the ceiling a translator could reach; the tree as one `pl.when` chain); for `fallback`, `frames-*` (ten `frame_step`s with polars bodies) and `frame1-*` (one); `copy-stream` (a plain scan into a sink). Compare `*-stream` with `fused-io` and `*-mem` with each other.

Each configuration runs in its own child process under a 7 GB cap, after a warm-up call on 1000 rows (the build is reported apart); configurations alternate, 3 repeats, median and min-max; a run under 3 s repeats inside the child and reports its median. Peak RSS is `ru_maxrss`. Correctness first: at 10k rows all 19 configurations equal eager fused exactly, dtypes included, in memory and streamed with a 1000-row chunk. Box: polars 1.41.2, 28 threads, 31 GB shared with other sessions (one-minute load average 6 to 106 during the runs, so the spread is reported).

## Result
Median wall seconds (min-max); RSS in MB.

**1M rows**

| Pipeline | fused-mem | host-mem | native-mem | fused-io | host-stream | native-stream |
|---|---|---|---|---|---|---|
| flagship | 0.0173 (0.0119-0.0204) | 0.0243 (0.0127-0.0279) | 0.0145 (0.0093-0.0149) | 0.372 (0.224-0.418) | 0.255 (0.216-0.406) | 0.335 (0.224-0.458) |
| tree | 0.189 (0.184-0.190) | 0.197 (0.186-0.200) | 0.197 (0.177-0.206) | 0.852 (0.827-0.904) | 0.791 (0.700-0.804) | 0.710 (0.691-0.788) |
| fallback | 25.3 (25.0-26.6) | 26.2 (24.2-28.9) | 0.707 (0.631-0.767) | 27.3 (25.2-27.3) | 33.0 (31.3-33.7) | 0.375 (0.369-0.494) |

**Fallback rewritten as frame steps, 1M rows**

| | host | `frames` (10 frame steps) | `frame1` (1 frame step) | native |
|---|---|---|---|---|
| in memory | 26.2 (24.2-28.9) | 4.88 (4.85-5.47) | 2.47 (2.47-3.05) | 0.707 (0.631-0.767) |
| streamed | 33.0 (31.3-33.7) | 6.11 (5.28-6.12) | 3.72 (3.69-5.57) | 0.375 (0.369-0.494) |

**Flagship, 50M rows** (frame 2.0 GB in memory)

| | fused-mem | host-mem | host-memstream | native-mem | fused-io | host-stream | native-stream |
|---|---|---|---|---|---|---|---|
| wall s | 4.56 (3.73-4.69) | 3.70 (3.30-3.86) | 0.930 (0.726-0.960) | 0.200-0.218 | 10.3 (9.03-10.95) | 4.60 (4.27-5.70) | 4.15 (3.93-5.27) |
| RSS | 4811 | 4811 | 3174 | 3495-3769 | 4149 | 4010 | 4085 |

(`host-memstream` and its `native-mem` partner, 0.218 s, come from a second session.)

**Flagship, 50M rows streamed: peak RSS by input layout** (median MB; 20 parts of 2.5M rows, or one merged file)

| Input | threads | copy-stream | native-stream | host-stream |
|---|---|---|---|---|
| 20 parts, row group 1M | 28 | 2823 | 2752 | 2757 |
| 20 parts, row group 1M | 4 | 869 | 1009 | 1015 |
| 20 parts, row group 100k | 28 | 1037 | 2420 | 2457 |
| 20 parts, row group 100k | 4 | 368 | 628 | 684 |
| one file, row group 1M | 28 | 3522 | 4069 | 4032 |
| one file, row group 1M | 4 | 2018 | 2187 | 2293 |

**Tree, 50M rows streamed** (7.2 GB of columns): the host finished in every setting but 28 threads with 1M row groups (16.2 to 21.3 s; 1.1 GB at 4 threads and 100k row groups); the hand-written `when` chain ran out of the 7 GB cap in all four settings.

- In memory the host is the eager fused run batch by batch: tree and fallback within 4% of `fused-mem` at 1M; the flagship at 1M has a median 40% above fused with overlapping ranges, and at 50M is 19% faster with overlapping ranges.
- Streamed, the host is within noise of hand-written polars on the flagship (1M and 50M, first session) and the tree (1M); in a second session over six input layouts it was 10 to 26% slower on the flagship (non-overlapping ranges in three of six), with peak RSS within 2 to 9% of hand-written polars. Both sit near a plain scan-to-sink copy: the streamed flagship is I/O bound, and the peak follows the input layout and thread count, not the host.
- In memory, hand-written polars beats the host on the flagship: 18x at 50M with plain `collect()` (one single-threaded numba pass against 28 polars threads); `collect(engine="streaming")` narrows that to 4.3x (0.93 against 0.218 s) and makes the host 3.4 to 3.6x faster than the hand-written chain on the tree at 1M, but is 2.8x slower than plain `collect()` on the flagship at 1M and 17% slower on the fallback pipeline.
- The fallback pipeline is where native polars wins: 37x in memory at 1M. Rewriting the steps as frame steps closes 84% (ten) to 93% (one) of the host-to-native gap in memory and 82% to 90% streamed, leaving 3.5x to 16x; at 5M rows streamed (one run each) the host ran at 35k rows/s, `frames` 197k, `frame1` 275k, native 3.6M. A frame step is a pipeline breaker: its peak grew 3.0 to 4.0x for 5x the rows.
- At 10k rows the per-call cost of `map_batches` and the zero-row schema runs dominates (flagship host-mem 1.7 ms against 0.3 ms fused); use the numba engines for small frames.

## Decision
P6 is not built (no-go, 2026-10-10): `impl("polars")`, native segments and experiments E3 to E5 leave the spec. The rule asked for the host within noise of hand-written polars on the flagship and the tree, and the frame-step rewrite closing most of the fallback gap. Streamed, the host is at or near native on both (within noise, or 10 to 26% on an I/O-bound flagship), and on the 50M tree only the host fits in 7 GB; the rewrite closes 82 to 93% of the gap. The one miss, the in-memory flagship at 50M (4.3x with the streaming engine), is pure arithmetic that already runs at 54M rows/s, not worth a second implementation of every step and an equivalence burden on authors. `run(DataFrame)` keeps plain `collect()`; the guide shows `collect(engine="streaming")` for large frames. P4's memory exit holds only with polars configured (0.68 GB against the 2.0 GB frame at 4 threads and 100k row groups); at the defaults the peak is polars' own scan and sink buffering, the same for hand-written polars. See [engine](../decisions/engine.md), [polars engine](../../architecture/engine/polars.md).
