# Fusion is explicit, never implicit

**Decision:** Kernel grouping follows a fixed, predictable rule. The compiler
never merges or splits kernels on a cost heuristic; any other grouping is
authored. A fused variant must give the same answers as the unfused one.

**Why:** Fusion is non-monotone, and whether it helps depends on how much work
each step's body does, not on how many modules there are.
- **E7 (branchy steps):** fused vs one kernel per module was 1.33–1.61× faster
  for one branch group, then 0.75× at 4 modules, 0.47× at 8, 0.17× at 16
  (58.7 ms vs 9.8 ms) and 0.11× at 32.
- **E5 (1M rows):** 0.90 at 3 modules, 0.76 at 5, 0.51 at 10 and 0.33 at 20.
- **EXPERIMENTS §E**, at a fixed 1M rows, ratio = split/fused (>1 means fusion
  wins):

  | modules | cheap straight-line body | heavy body |
  |---|---|---|
  | 3 | 3.26 | 0.76 |
  | 20 | 13.00 | 0.22 |
  | 40 | 11.20 | 0.18 |

  That is a 62× spread from body cost alone, and the sign of the decision flips.
- **Mechanism:** register pressure. Split kernels unroll about 5× using 7 of 16
  ymm registers with no spills. Fused at M=20, the kernel pins all 16 registers,
  unrolls about 2× and spills 47 times per iteration, so it becomes
  latency-bound.
- **Short-circuit skip:** when a heavy arm is skipped, fusion wins by 1071×. No
  size constant covers a range from 0.11× to 1071×.
- **At N=1** fusion wins 9.4–48× over per-kernel dispatch (0.44 µs/kernel).
  That is microseconds: 30 unfused kernels cost about 13 µs, 0.07% of a 20 ms
  budget.

**What we tried:**
- A cap of about 6–9 steps per fused group. It named the range the data shows is
  already worse than not fusing (0.72–0.90 at 100k–1M rows), so it was withdrawn.
- A "break-even at ~10k rows" rule. There is no common break-even: cheap
  straight-line bodies never cross, and heavy branchy ones cross below 1000
  rows.
- Using lost auto-vectorisation (read from `.inspect_llvm()`) as the observable.
  It is refuted: the packed-FP count *rises* with M, and the worst case (0.18×)
  is fully vectorised. There is no static signal that says whether fusing would
  help; only measuring the group does.
- One grouping serves both batch and `score()`, so there is one codegen path and
  one cache surface.

**Source:** `decider2/docs/EXPERIMENTS.md` (§E);
`decider2/docs/01-motivation-and-evidence.md` (§4b, §4c);
`decider2/docs/05-boundary-and-compilation.md` (§7);
`experimentation/fusion-body-cost/`
