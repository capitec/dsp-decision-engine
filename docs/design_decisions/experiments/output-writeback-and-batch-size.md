# How should a kernel's outputs reach polars, and how many rows can one batch hold?

Date: 2026-09-19 to 2026-09-23 · Script: spikes, in decider-experimentation/ `output-writeback-convention`, `chunked-writeback-at-scale`

## Purpose
Settle whether outputs are written as a record array, a row-major 2D array or one column array per name, and how large a batch a wide pipeline can take before memory, not time, decides.

## Method
A generated kernel with 400 inputs (300 float64, 80 int64, 20 bool) and 633 outputs (380, 158, 95), identical arithmetic in three conventions: record in and out, record in with dtype-grouped column-major 2D out, and the same row-major as a control. Timed at 100k and 1M rows (median of 5-7), write-back being the copy into polars frames. The 1M-row run was repeated chunked, one variant per subprocess under a 6 GB cap, with a checksum per chunk to compare conventions (134 chunks, 0 mismatches). The first attempt held all variants at once, about 31 GB, and was killed twice.

## Result
At 100k rows, milliseconds:

| convention | compile | kernel | write-back | total |
|---|---|---|---|---|
| record out | 11.3 s | 94 | 776 (64%) | 1,212 |
| row-major 2D out | 0.25 s | 80 | 506 | 928 |
| column-major 2D out | 0.27 s | 350 | 5.7 (zero-copy) | 697 |

- Column-major is 1.74x end to end. The kernel is 3.7x slower, writing 633 strided columns, but turning the result into polars frames costs nothing. Compile falls from 11.3 s because no record type is built.
- Chunked at 1M rows, kernel plus write-back is 3.1x faster at 250k-row chunks, but the whole pipeline only 1.03x (10k chunks) to 1.57x (250k): building each chunk's input and writing parquet are 58-82% of wall time, whatever the convention.
- Unchunked peak RSS grows 16.0 KB/row for record out (16.2 GB at 1M rows) and 11.7 KB/row column-major (11.9 GB); the largest batch under 6 GB is about 390k and 534k rows. Chunked, peak follows chunk size, not total rows: 100k rows peaked at 1.79 GB (record) and 1.37 GB (column-major), 250k at 4.0 and 3.0.
- Wall time is flat to U-shaped from 50k to 100k rows per chunk; the dispatch floor (about 0.44 µs a call) is irrelevant, so chunk size is a memory budget.
- A generated driver must be imported by module name for another process to reuse its numba cache; loaded by file path, a fresh process raised `ModuleNotFoundError: No module named '<dynamic>'`.

## Decision
`State` holds one column array per produced name, with no record array and no row-major output; each becomes a polars series. `Executable.run` does not chunk: the experiment runner slices its input at `batch_size=100_000` (the spike's default), and any other caller with a wide pipeline and millions of rows should do the same. Generated drivers no longer exist, so the module-name trap is moot. See [engine](../decisions/engine.md), [experiment runner scale](experiment-runner-scale.md).
