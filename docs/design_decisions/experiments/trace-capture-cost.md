# What does decision-trace capture cost, and where?

Date: 2026-09-30 · Script: `benchmarks/trace_overhead.py` (the first spike's scripts were not kept)

## Purpose
Can a compiled `nogil` kernel record ordered decision evidence cheaply, and what bounds the cost of tracing a batch?

## Method
Synthetic 10-step scalar chains and a branch+loop through the real engine in all three modes (spike), then the shipped `TraceSink` on a 3-step fused flow (benchmark, 1M rows). Concurrency: 4 and 8 threads over one capture kernel. Machine was loaded in the follow-up, so read the ratios, not the absolutes.

## Result
| Measurement | Result |
|---|---|
| In-kernel write | ~1.2G events/s (~0.8 ns/event), 8 bytes/event |
| `score()`, 10 steps: none / one shared int64 / int64 per step | 25.9 / 26.6 / 60.3 us |
| Batch rows/s: none / one shared int64 | 888M / 440M |
| Decode: per-event Python / vectorised unpack / kind histogram | 1.2M / 53M / 162M events/s |
| Shipped sink, 1M rows (this run) | 4.8 ms untraced, 10.2 s traced (~98k rows/s) |
| `score()` traced vs not (this run) | 206 vs 230 us (noise) |
| Threads, 4x the work: `nogil=False` / `nogil=True` | 5.7x / 2.0x one thread |
| 8 concurrent writers, own buffers | 12.8M events, all conserved, order intact |
| Bounded adapter, queue of 16 | 1.6M events dropped, producer never blocked |
| 1M rows x 10 steps | 80 MB at full rate, 1 MB at 1% sampling |

The kernel write is free; the cost is turning int64s into `TraceEvent` objects. Every evidence type fits one int64 (index, bit-cast float, constant-table string).

## Decision
One shared buffer per invocation, vectorised or lazy decode, opt-in bounded batch tracing: [lab-and-mcp](../decisions/lab-and-mcp.md).
