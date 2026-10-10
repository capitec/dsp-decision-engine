# Does the polars host lose to native polars expressions?

Date: 2026-10-10 · Script: `benchmarks/polars_engine.py`

Draft: the numbers are measured, the decision paragraph is factual and the manager writes the final one.

## Purpose
Find out whether translating steps into native polars expressions would beat the host (the fused pipeline inside `map_batches`), and by how much, on pipelines people actually run. If it would not, polars bodies and native segments are not built.

## Method
Script `benchmarks/polars_engine.py`; every configuration runs in its own child process (a fresh numba start, a warm-up call on 1000 rows excluded from the timing, the build reported apart), in its own 7 GB memory scope. Configurations alternate within one session and repeat 3 times; a run under 3 s repeats inside the child (up to 20 times) and reports the median of those. Wall seconds, rows/s and the child's peak RSS (`ru_maxrss`) are reported.

Pipelines:
- `flagship`: the three arithmetic steps of `benchmarks/modes.py`; 5 float columns.
- `fallback`: ten steps over `date` and `str` columns, each a plain Python body that the numba engine runs as a fallback (`fallbacks()` lists all ten): tenure in months from two dates, age from a birth date, opened-on-weekend, is-month-end, name normalisation (`" ".join(name.split()).upper()`), product prefix and closed-suffix tests on an account code, a region lookup by branch prefix, a branch number parsed from the code, and a segment string from region and tenure. 3 `date` and 3 `str` columns.
- `tree`: the 127-node document of `benchmarks/tree_walk.py` (63 internal nodes, 64 leaves; 16 float and 2 int features; a String, a Float64 and an Int64 output).

Configurations (`<engine>-<mode>`):
- `fused-mem`: eager `run(df)` on a frame already in memory. `fused-io`: `read_parquet`, `run`, `write_parquet` (the like-for-like of the streamed runs).
- `host-mem`: `run(df.lazy()).collect()`. `host-stream`: `run(scan_parquet(...)).sink_parquet(...)`.
- `native-mem`, `native-stream`: hand-written polars for the same logic, the ceiling a native translator could reach. The tree is one `pl.when` chain that finds the leaf, then one `replace_strict` lookup per output column.
- `frames-mem`, `frames-stream` (`fallback` only): the host with the ten steps rewritten as ten `frame_step`s whose bodies are one `with_columns` expression each. `frame1-*`: one `frame_step` holding all ten expressions.
- `copy-stream` (`flagship` only): a plain scan into a sink, to separate what the sink costs.

Streamed runs include reading and writing parquet, in-memory runs do not, so compare `*-stream` with `fused-io` and `*-mem` with each other.

Correctness first: on 10k rows `check` compares every configuration's output with eager fused, in memory and streamed with a 1000-row streaming chunk, column by column with exact equality and dtype. All 16 comparisons were equal (no float tolerance was needed: the polars arithmetic and the Python arithmetic agree bit for bit on these data).

Box: 28 polars threads (`pl.thread_pool_size()`), polars 1.41.2, Python 3.14.5, 31 GB shared with other sessions. One-minute load average at the start and end of the four timed invocations: 106 and 15 (10k and 1M, first pass), 7.3 and 9.0 and 8.8 and 12.7 (the re-run of flagship and tree at 10k and 1M, then of fallback at 1M), 8.9 (50M flagship, no end record). Other sessions' work is in the spread, which is why the min-max is given.

## Result

Median wall seconds (min-max over 3 child runs); peak RSS in MB.

**1M rows**

| Pipeline | fused-mem | host-mem | native-mem | fused-io | host-stream | native-stream |
|---|---|---|---|---|---|---|
| flagship | 0.0173 (0.0119-0.0204) | 0.0243 (0.0127-0.0279) | 0.0145 (0.0093-0.0149) | 0.372 (0.224-0.418) | 0.255 (0.216-0.406) | 0.335 (0.224-0.458) |
| tree | 0.189 (0.184-0.190) | 0.197 (0.186-0.200) | 0.197 (0.177-0.206) | 0.852 (0.827-0.904) | 0.791 (0.700-0.804) | 0.710 (0.691-0.788) |
| fallback | 25.3 (25.0-26.6) | 26.2 (24.2-28.9) | 0.707 (0.631-0.767) | 27.3 (25.2-27.3) | 33.0 (31.3-33.7) | 0.375 (0.369-0.494) |

Fallback with the steps as frame steps, 1M rows:

| | host (Python fallbacks) | `frames` (10 frame steps) | `frame1` (1 frame step) | native polars |
|---|---|---|---|---|
| in memory | 26.2 (24.2-28.9) | 4.88 (4.85-5.47) | 2.47 (2.47-3.05) | 0.707 (0.631-0.767) |
| streamed | 33.0 (31.3-33.7) | 6.11 (5.28-6.12) | 3.72 (3.69-5.57) | 0.375 (0.369-0.494) |

Share of the host-to-native gap that the rewrite closes: in memory 84% (ten frame steps) and 93% (one); streamed 82% and 89%. What is left is 6.9x and 3.5x native in memory, 16x and 9.9x streamed.

**50M rows, flagship** (3 child runs; the frame is 2.0 GB in memory: 5 float64 columns)

| Config | wall s | rows/s | peak RSS MB |
|---|---|---|---|
| fused-mem | 4.56 (3.73-4.69) | 11.0M | 4811 (4811-4812) |
| host-mem | 3.70 (3.30-3.86) | 13.5M | 4811 (4810-4812) |
| native-mem | 0.200 (0.178-0.213) | 250M | 3769 (3763-3905) |
| fused-io | 10.3 (9.03-10.95) | 4.9M | 4149 (4131-4152) |
| host-stream | 4.60 (4.27-5.70) | 10.9M | 4010 (3969-4086) |
| native-stream | 4.15 (3.93-5.27) | 12.1M | 4085 (4023-4095) |

10k rows (median of 20 calls in the child; cold build apart): flagship fused-mem 0.3 ms, host-mem 1.7 ms, native-mem 0.8 ms; host-stream 29 ms, native-stream 6.5 ms. Fallback: fused 253 ms, host 253 ms, native 10.8 ms, frames 78 ms, frame1 36 ms. Tree: fused 2.5 ms, host 5.5 ms, native 9.7 ms. At this size the per-call cost of `map_batches` and the zero-row schema runs dominates.

Cold build (executable build plus the warm-up call, seconds): flagship 0.5 to 1.0 host and fused, 0.03 hand-written; tree 1.2 to 1.5 both (the walker compiles), 0.06 hand-written; fallback 0.07 to 0.15 (nothing compiles).

What the numbers show:
- The host is the eager fused run, batch by batch: in memory it is within noise of `fused-mem` on every pipeline (flagship 50M: 3.70 s against 4.56 s, the host faster by 19%; tree and fallback 1M within 4%). Streamed, it is faster than `fused-io`'s read-run-write on the flagship (50M: 4.60 s against 10.3 s; 1M: 0.255 against 0.372 s) and the tree (0.79 against 0.85 s at 1M) and 21% slower on the fallback pipeline (33.0 against 27.3 s at 1M), with about the same peak RSS.
- Streamed, the host is within noise of hand-written polars on the flagship (50M: 4.60 (4.27-5.70) s against 4.15 (3.93-5.27) s; 1M: 0.255 against 0.335 s) and on the tree (1M: 0.79 (0.70-0.80) against 0.71 (0.69-0.79) s). The streamed flagship is bound by reading and writing parquet.
- In memory the host is not within noise of hand-written polars on the flagship: 50M rows 3.70 s against 0.200 s (18x), because the host runs one single-threaded numba pass over the whole frame while polars spreads the arithmetic over 28 threads. At 1M rows the gap is 1.7x with overlapping ranges (host 12.7-27.9 ms, native 9.3-14.9 ms). On the tree the two are equal in memory at 1M (0.197 s each).
- The fallback-heavy pipeline is where native polars wins by two orders of magnitude (26 s against 0.7 s per 1M rows, 37x). Rewriting the steps as frame steps closes most of that: 5.4x (ten steps) and 10.6x (one step) faster than the host in memory.
- A frame step costs more than its polars body. Profiling `frame1` at 1M rows (cProfile, one run): 2.6 s in all, of which 1.6 s is `State.from_series` turning the frame's `str` and `date` columns into the engine's run state (42 calls in three `Engine.run`s, two of them on zero rows), against 0.7 s for the polars expressions themselves. That is why the one-step rewrite is 3.5x and not 1x native.
- A frame step is a pipeline breaker: the segments before and after it stream, but the step runs once on the whole frame, so its input and output columns are in memory together. `frames-stream` and `frame1-stream` at 1M rows peak at 628 and 908 MB against 784 MB for `host-stream`; at 1M rows this does not show, and at 50M it was not run.
- Memory, 50M rows, flagship, first session (one merged 1.5 GB parquet file): streamed peak RSS 3969 to 4086 MB for the host and 4023 to 4095 MB for hand-written polars, twice the 2.0 GB frame. The second round below shows this is the input layout and polars' scan/sink, not the host.

### Round 2: the in-memory frame through the streaming engine (`host-memstream`)

`exe.run(df.lazy()).collect(engine="streaming")` with the default chunk size. The host hands the segment 28 batches at 1M rows and 504 at 50M (about 36k and 99k rows each). 3 child runs alternating with `host-mem` and `native-mem`; median seconds (min-max), peak RSS MB.

| Pipeline, rows | host-mem | host-memstream | native-mem |
|---|---|---|---|
| flagship 1M | 0.0106 (0.0102-0.0115) | 0.0293 (0.0287-0.0296) | 0.0108 (0.0068-0.0146) |
| flagship 50M | 3.43 (2.43-4.08), RSS 4814 | 0.930 (0.726-0.960), RSS 3174 | 0.218 (0.216-0.222), RSS 3495 |
| tree 1M | 0.190 (0.190-0.194) | 0.0533 (0.0500-0.0611) | 0.184 (0.173-0.198) |
| fallback 1M | 24.4 (23.2-24.4) | 28.5 (28.0-30.0) | |

- At 50M rows the streaming engine makes the host 3.7x faster than its own in-memory `collect()`, and it is still 4.3x slower than hand-written polars in memory.
- On the tree at 1M it is 3.6x faster than both the host's in-memory collect and the hand-written chain.
- At 1M on the flagship it is 2.8x slower than in-memory `collect()` (29 ms against 11 ms): the fixed cost of the streaming engine and 28 batches outweighs 10 ms of work.
- The fallback pipeline stays single-threaded (the Python bodies hold the GIL): 17% slower than in-memory `collect()` (28.5 against 24.4 s), RSS 756 against 881 MB.

### Round 2: what the 4 GB streamed peak is (flagship, 50M rows, 3 child runs each)

The frame is 2.0 GB in memory. Median peak RSS in MB (median wall seconds). `copy-stream` is a scan straight into a sink with no step. "20 parts" is 20 files of 2.5M rows (about 1.4 GB in all, 75 MB each); "one file" is the same data merged into one 1.4 GB file. Row groups of 100,000 rows made the files no larger (75.14 MB against 75.12 MB per part). The sink's own row-group size was left at its default.

| Input layout | threads | copy-stream | native-stream | host-stream |
|---|---|---|---|---|
| 20 parts, row group 1M | 28 | 2823 (2.51 s) | 2752 (3.27 s) | 2757 (3.80 s) |
| 20 parts, row group 1M | 4 | 869 (3.45 s) | 1009 (3.73 s) | 1015 (4.19 s) |
| 20 parts, row group 100k | 28 | 1037 (2.57 s) | 2420 (3.14 s) | 2457 (3.88 s) |
| 20 parts, row group 100k | 4 | 368 (3.37 s) | 628 (4.04 s) | 684 (4.47 s) |
| one file, row group 1M | 28 | 3522 (2.82 s) | 4069 (3.33 s) | 4032 (4.20 s) |
| one file, row group 1M | 4 | 2018 (3.57 s) | 2187 (4.51 s) | 2293 (4.97 s) |

- The host's peak RSS is within 2% of hand-written polars with 28 threads and within 9% with 4 (684 against 628 MB at the smallest). In this session its wall time was 10 to 26% above hand-written polars, with non-overlapping ranges in four of the six rows (the first session's 4.60 (4.27-5.70) against 4.15 (3.93-5.27) s overlapped).
- The peak depends on the input layout and on threads, not on the host: one merged file reads at 3.5 GB even for the plain copy, twenty parts at 2.8 GB; four threads cut it to 0.9 and 2.0 GB; smaller row groups with 4 threads cut it to 0.37 GB.
- Hand-written polars and the host peak 1.4 GB above `copy-stream` with 28 threads and small row groups (2.4 against 1.0 GB), and 0.26 to 0.32 GB above it with 4 threads: the two new columns' buffers. A run is under half the frame (0.63 and 0.68 GB against 2.0 GB) only with 4 threads and 100k row groups.

### Round 2: the tree at 50M rows streamed (7.2 GB of columns in 20 parts, 5.7 GB on disk)

| Row group | threads | host-stream | native-stream |
|---|---|---|---|
| 1M | 28 | out of memory in 6 to 8 s, 3 of 3 | out of memory in 6 to 8 s, 3 of 3 |
| 1M | 4 | 20.7 s (20.2-21.6), RSS 2391 MB | out of memory (10 to 28 s), 3 of 3 |
| 100k | 28 | 16.2 s (14.6-18.1), RSS 6780 MB | out of memory (8 to 360 s), 3 of 3 |
| 100k | 4 | 21.3 s (21.2-21.6), RSS 1107 MB | out of memory (10 to 25 s), 3 of 3 |

The host streams the 127-node tree at 50M rows (2.3 to 3.1M rows/s) in 1.1 GB with 4 threads and 100k row groups; the hand-written `when` chain runs out of the 7 GB cap in all four settings. I did not look into why polars' own streaming of the chain buffers so much; at 5M rows, 28 threads, 1M row groups, the chain peaks at 1786 MB and the host at 1549 MB (2.1 and 3.1 s).

### Round 2: the fallback pipeline at 5M rows streamed (1 run each)

| | host-stream | frames-stream | frame1-stream | native-stream |
|---|---|---|---|---|
| wall s | 144 | 25.4 | 18.2 | 1.38 |
| rows/s | 34.7k | 197k | 275k | 3.6M |
| peak RSS MB (1M rows) | 1994 (784) | 1846 (628) | 3601 (908) | 1585 (756) |

The host's rate is steady from 1M to 5M (30.3k then 34.7k rows/s), so 50M rows would take about 24 minutes; that was extrapolated, not run. The frame-step variants hold more of the frame as rows grow (a frame step is a pipeline breaker): 3.0x and 4.0x the 1M peak for 5x the rows, so at 50M they would pass the 7 GB cap if that growth continued; also extrapolated, not run. The eager variants were not run at 5M or 50M.

Not measured: the fallback pipeline at 50M, and the eager tree and eager fallback at 50M (the tree's 7.2 GB of columns exceeds the 7 GB cap by size alone; the fallback's 880 MB per million rows puts it over the cap at about 8M rows).

## Decides
The rule from the spec: P6 (polars bodies and native segments) is not built if the host is within noise of hand-written polars on the flagship and the tree, and the `frame_step` rewrite closes most of the gap on the fallback-heavy pipeline.

- Second clause: met on the 1M numbers. The rewrite closes 84% (ten frame steps) to 93% (one frame step) of the gap in memory and 82 to 89% streamed.
- First clause, streamed: on the flagship the host's wall time is within noise of hand-written polars in the first session (50M and 1M) and 10 to 26% above it in the second (50M, six layouts); on the tree at 1M it is within noise, and at 50M only the host finishes inside the 7 GB cap. First clause, in memory: met on the tree (1M) and not on the flagship, where hand-written polars was 18x faster at 50M rows (and 1.7x at 1M with overlapping ranges). Collecting the in-memory frame with the streaming engine narrows the 50M flagship gap to 4.3x (0.93 against 0.218 s) and makes the host 3.6x faster than hand-written polars on the tree at 1M.
- The fallback-heavy gap is the Python bodies, which only a rewrite by the author (frame steps, which the engine already supports) or a translator of Python to expressions would remove; the rewrite is 5x to 10x faster than the host at 1M and 5.7x to 7.9x at 5M, with a memory cost that grows with the rows.
- P4's exit asked for peak RSS well under the frame size at 50M rows, streamed, on a pipeline without frame steps. With the host: 4.0 GB with one merged input file and default threads, 2.8 GB with 20 parts, 2.5 GB with 20 parts and 100k row groups, 0.68 GB with 4 threads and 100k row groups (frame 2.0 GB). In each setting hand-written polars peaks within 2 to 9% of the host's figure and a plain scan-to-sink copy at 0.37 to 3.5 GB.
