# Which layout keeps a 1,000 to 10,000 call flow interactive?

Date: 2026-09-30 (dagre baseline), 2026-10-05 (custom layout and ELK) ·
Script: `ui/lib/bench/layout.ts` (`cd ui/lib && pnpm bench`; times the custom layout only. The dagre and ELK runs and the payload-size generator were spikes, not kept)

## Purpose
The graph used dagre. Find out how big a flow it can draw, whether to swap the layout, and what budgets to set.

## Method
Generated nested flows (groups of 3, every 5th a branch, every 7th a loop, each call reads the previous two outputs), plus the 94-call retail-credit example (about 110 KB `describe`). Median of 7 runs after a warm-up, Apple M2 Pro, Node 23. dagre and ELK 0.10.2 (`layered`, compound nodes) timed on equivalent chains.

## Result
| calls | dagre | ELK | custom layout | fold + layout |
|---|---|---|---|---|
| 100 | 72 ms (330 ms on the first, slower run) | 50 ms | 0.2 ms | 0.1 ms |
| 1,000 | 1.2 s (5.4 s) | 0.46 s | 0.8 ms | 0.1 ms |
| 2,000 | stack overflow | 0.84 s | 1.2 ms | 0.1 ms |
| 10,000 | n/a | 4.65 s | 8.8 ms | 0.4 ms |

- A re-run on a loaded Linux box (`heavy.sh`, Node 22) gave 2.4 ms at 1,000 and 20.8 ms at 10,000 for the custom layout: same order, so the conclusion holds.
- `dataEdges` is the one superlinear pass (53 ms at 10,000 calls on the M2, 133 ms on the re-run). It scans back for each input's last writer.
- Culling costs about 1 ms at 10,000 calls. Real payloads are 1 KB a node (5 to 10 MB at 5,000 to 10,000 calls), so `postMessage` and JSON parse, then DOM size, are the next ceilings. Interaction latency and webview memory were not measured.
- Layout quality (ELK routes edges for any graph; the custom layout places nodes from the flow's tree) was not compared.

## Decision
Replace dagre with a dependency-free tree-walking layout, fold groups by default (flows over 80 steps), cull to the viewport, and keep per-column edges selection-only. Budgets: at most 500 unfolded and 5,000 total calls, `describe` at most 2 MB. See [debugger and editors](../decisions/debugger-and-editors.md).
