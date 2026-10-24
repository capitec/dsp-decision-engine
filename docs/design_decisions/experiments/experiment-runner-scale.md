# How fast can experiments run locally, and what do revisions cost?

Date: 2026-10-02 · Script: `benchmarks/experiment_throughput.py` (the earlier fork/fused comparison was a spike, not kept)

## Purpose
Choose how a scenario executes (session fork, fused, stepped, subprocess) so a sweep stays interactive at millions of rows.

## Method
Fused, stepped and interpreted `run()` on a 3-step flow; five scenarios via the old session `fork`; then a whole experiment (baseline, determinism rerun, one param scenario) with a global metric (fused) and node metrics (stepped). Revision cost measured on a two-commit project.

## Result
| Measurement | Result |
|---|---|
| 1M rows: fused / stepped / interpreted `run()` | 2.3 ms / 4.6 ms / 4.6 s |
| 5 scenarios x 1M rows: session fork / fused param sweep | 31 s / 42 ms (~750x) |
| 52-step nested-column pipeline, interpreted | ~67 records/s |
| Stepped with a per-node callback reading columns | ~300M rows/s vs 322-630M fused (~25%) |
| Whole experiment, 100k / 1M / 10M rows, global metric | 26 / 87 / 605 ms (this run; 80 / 104 / 382 ms in the original) |
| Same, node metrics | 45 / 231 / 2,080 ms (96 / 168 / 897 ms originally) |
| Revision scenario | ~0.5 s checkout + ~0.8 s subprocess start; cold `uv sync` per revision 23 s |

The original and current numbers differ by machine load; the ordering and the ~2-4x node-metric cost hold.

## Decision
Route by capability (fused, else stepped, never session replay), run other revisions whole in a subprocess under today's engine, and use lockstep batches: [lab-and-mcp](../decisions/lab-and-mcp.md).
