# How fast can an open debug session pick up an edited step?

Date: 2026-09-23 · Script: spike, not kept (behaviour is pinned by `tests/debug/test_session_reload.py`)

## Purpose
Can `Session` follow an edited step by itself, after a notebook cell re-run or a file save, and is one rebuild primitive enough?

## Method
A prototype `Session.reload(pipeline)` (diff checkpoint contents, rebuild, replay) fed by an IPython `post_run_cell` hook or a module poller. Time from the edit to the updated `output()` on a 28-core Linux box, numba 0.67, CPython 3.14; compare per-path `replace` calls with one `reload`.

## Result
| edit | interpreted | stepped | fused |
|---|---|---|---|
| file, 3-step flow, constant in step 2 | 5 to 6 ms, 0 compiles | 186 ms, 2 | 142 to 151 ms, 1 |
| file, back to an earlier version | | 7 ms, 0 | 7 ms, 0 |
| larger example, edit a param default | 19 to 28 ms | 26 to 39 ms | 29 to 34 ms |
| cell: `reload` vs per-path `replace` | 1.5 vs 0.9 ms | 182 vs 189 ms | 148 vs 156 ms |

- Compiled modes pay for the edited step's dispatcher plus its fused kernel; unchanged steps and earlier versions hit the content-keyed cache; a param-default edit never compiles. Computing the checkpoint keys costs 1.6 ms.
- A step reading a global DataFrame is keyed by identity, so every re-import re-runs from it (21 to 39 ms on the example): noisy, never stale.
- Serving would pay 142 to 186 ms per edited step on the request path against a 60 us p99 budget.

## Decision
One first-difference rebuild (cost equals per-path edits, and it can insert and delete); no hot reload in serving. See [debugger and editors](../decisions/debugger-and-editors.md).
