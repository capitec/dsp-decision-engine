# Can kernels read every Arrow column in place, validity bitmaps included?

Date: 2026-09-26 · Script: spike, not kept (it ran behind `DECIDER_ZERO_COPY=1`, since removed)

## Purpose
Find out whether nullable and bool columns could be read straight from polars' Arrow buffers, and whether `score(dict)` could share one Arrow input path.

## Method
Kernels took each validity as a `(bitmap, bit offset)` pair, with `missing_as` fills applied in the kernel. Three pipelines (flagship: 5 clean floats; wide: 40 clean floats, 40 floats with 10% nulls and 4 bools; a 127-node tree) in fused, stepped and interpreted modes; outputs checked bit for bit on 48 cases. Shared box, batch numbers +-30%.

## Result
| Case | Before | In place |
|---|---|---|
| wide, 100k rows | 83-108 ms | 24.5 ms |
| wide, 1M rows | 2354-2726 ms | 364-374 ms |
| flagship / tree, 1M rows | 9.5 / 207-219 ms | unchanged |
| `score()` through a frame (flagship / tree / wide) | 30.9 / 60 / 295 us | 126 / 231 / 995 us |

- The only expensive part of the boundary is a nullable column: 30-45 ms per 1M-row column (fill loop, page faults, the driver's `np.where`). Clean numerics and string spans were already zero-copy.
- One kernel signature then serves Arrow input, `score()` dicts and engine intermediates (wide compiled twice before). Cost: +170/-22 lines, 17 boundary tests rewritten, a one-row `run` on 84 inputs +60 us.
- Output side: `Series.set(mask, None)` is 0.3 ms against 3.5 ms for `scatter(None)`; `Enum.gather` decodes tree `Literal` outputs in 3-4 ms against 43 ms.

## Decision
Not adopted; `score(dict)` stays off Arrow. Reading nullable columns in place for batches of 4096+ rows is the recommended next step. See [serving](../decisions/serving.md).
