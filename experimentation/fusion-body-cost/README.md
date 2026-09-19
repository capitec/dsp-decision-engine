# E-fusion-body-cost — fusion vs one-kernel-per-module, with body cost as an axis

## What this measures

Doc 01 §4c (E5) published a fusion table indexed by **module count** and **row
count** only. Across that whole table `split ns/step` is flat at 0.20, which is
the tell that **per-step body cost was held roughly constant**. This harness
re-runs the same comparison with body cost made explicit, and adds branchiness.

Four axes:

| axis | values |
|---|---|
| module count | 1, 2, 3, 4, 5, 8, 10, 20, 40 |
| row count | 1, 1 000, 10 000, 100 000, 1 000 000 |
| **per-step body cost** | `trivial` (one add) / `medium` (~10 flops) / `heavy` (8-iteration inner loop) |
| **branchiness** | `straight` (no branches) / `branchy` (3 branch groups per module) |

Reported as **`ratio = t_split / t_fused`**, so **`ratio > 1` means fusion wins**
— the same orientation as doc 01 §4c.

It also checks doc 01 §4b's *mechanism* claim ("vector IR values drop to 0 at
M>=4") directly, by reading `.inspect_llvm()` and `.inspect_asm()` off every
compiled kernel and counting:

- LLVM IR values with a vector type, and the vector-typed FP ops among them;
- x86 FP arithmetic split three ways — scalar (`*sd`), packed in `xmm`, packed
  in `ymm`/`zmm` — and the SIMD fraction of all FP arithmetic.

And it measures the **boundary-cost floor**: numba dispatch overhead plus one
`load + store` per row, i.e. the price of adding one more split kernel before
any body runs. Doc 01 §4c calls that "near-free"; this puts a number on it.

## Model

A **module here is one step**: one chunk of body source updating a running
scalar `v` from two input columns `x`, `y`.

- **fused** — one `njit` kernel, one row loop, all M module bodies inlined in
  sequence.
- **split** — M `njit` kernels, each with its own row loop, called in sequence by
  a Python driver, ping-ponging between two intermediate `float64` buffers. This
  is the `apply()` default of doc 02 §1.2, "never fuse across module
  boundaries".

Both sides emit the **same source text** per module with the **same per-module
literal constants** (the constants vary with module index so the fused body
cannot be collapsed by CSE), so LLVM sees identical arithmetic. Every
(body, branch, M, rows) cell asserts fused output == split output; all cells in
the recorded run were **bitwise exact**.

Split kernel `k` is compiled once and reused for every M that contains module
`k` — this only saves compile time, the emitted code is the same.

`fastmath=False`, `cache=False`, `parallel=False` on both sides. The bodies are
elementwise in `i`, so auto-vectorisation does not need `fastmath`.

## Run

```
/path/to/.venv/bin/python fusion_body_cost.py           # full sweep (~4 min)
/path/to/.venv/bin/python fusion_body_cost.py --quick   # ~25 s smoke sweep
/path/to/.venv/bin/python fusion_body_cost.py \
    --modules 1,4,20 --rows 1000,1000000 \
    --bodies trivial,heavy --branch straight,branchy
/path/to/.venv/bin/python analyse.py results.json       # render the tables
```

Results stream to stdout and are written incrementally to `results.json`, so a
killed run still leaves usable data. `sweep.log` is the recorded run.

## Machine of record

Intel Core i7-14700HX, 28 logical cores, `cpu max MHz` reported as 2100.
L1d 768 KiB, L2 28 MiB, L3 33 MiB. Python 3.14.5, numba 0.67.0, llvmlite 0.49.0,
numpy 2.4.6. Single-threaded throughout.

This matters for comparison with doc 01, whose numbers were taken on an
**M2 Pro** — see the boundary-floor table, which is where the two machines
disagree most.

---

## Recorded run — headline numbers

Full sweep wall time **103 s** (270 cells). **270/270 cells bitwise-exact**
fused == split. Raw output in `sweep.log` / `results.json`; rendered tables in
`results_tables.txt`; hot-loop disassembly summary in `hotloop.txt`
(`inspect_hotloop.py`).

### The sign of the fusion decision flips with body cost (ratio at 1 M rows)

| modules | trivial/straight | trivial/branchy | medium/straight | medium/branchy | heavy/straight | heavy/branchy |
|---|---|---|---|---|---|---|
| 1 | 0.99 | 0.98 | 0.99 | 0.89 | 0.98 | 0.94 |
| 3 | 3.26 | 1.30 | 3.06 | 0.93 | 0.76 | 0.56 |
| 5 | 5.18 | 1.15 | 2.20 | 0.76 | 0.46 | 0.49 |
| 10 | **11.18** | 0.80 | 1.93 | 0.63 | 0.42 | 0.37 |
| 20 | **13.00** | 0.62 | 1.42 | 0.56 | **0.22** | 0.34 |
| 40 | 11.20 | 0.55 | 1.02 | 0.43 | **0.18** | 0.31 |

Same module count, same row count, same machine: **13.00× to 0.18×, a 62× spread
from body cost alone.** No constant in (modules, rows) can express that.

### Break-even is not a row count

Doc 01 §4c: *"Break-even is ~10k rows. Below it fusion wins (<=1.4x)."* Neither
half holds here. `trivial/straight` fusion wins at **every** row count from
1 000 to 1 000 000 (2.1×–13×); `heavy/branchy` fusion **loses** at every row
count from 1 000 up (0.3–0.8×). Below 1 000 rows fusion always wins, but by far
more than 1.4× — up to **48×** at n=1, M=40, because per-call dispatch dominates.

### Split does not hold a flat 0.20 ns/step here

`split ns/step @1M`: trivial 0.82–0.96, medium 1.56–2.38, heavy 0.85–3.30. The
measured **boundary floor** (numba dispatch + one load+store per row, empty
body) is **0.32–0.76 ns/row**, i.e. doc 01's 0.20 ns/step is *below the floor of
one kernel call on this machine*. On this box `split ns/step` is flat only
*within* a body cost, and its floor is set by per-core load/store bandwidth
(21–49 GB/s measured single-threaded).

### Vectorisation loss is NOT the mechanism, and NOT a usable signal

Doc 01 §4b: *"past ~3-4 branch groups the fused body loses LLVM
auto-vectorisation (vector IR values drop to 0 at M>=4)"*.

Across all 54 fused kernels, packed-SIMD FP ops **never reached 0** — they grow
linearly with M. In the vectorised main loop of the *worst* case
(`heavy/branchy`, M=20, ratio 0.34): **652 packed FP ops, 16 ymm registers, 0
scalar**. It is fully vectorised and still 3× slower per step than split.

What actually grows is **register spilling** — `%rsp` references inside the
vectorised main loop:

| kernel | ymm regs used | spill refs in main loop | unroll | ratio @1M |
|---|---|---|---|---|
| heavy/straight, split (M=1) | 7 | **0** | ~5× | — |
| heavy/straight, fused M=4 | 14 | **0** | ~4× | 0.88 |
| heavy/straight, fused M=20 | **16** | **47** | ~2× | 0.22 |
| medium/straight, fused M=20 | **16** | **82** | ~1× | 1.42 |
| heavy/branchy, fused M=20 | **16** | **130** | ~1× | 0.34 |

Every fused kernel at M>=4 pins all 16 architectural ymm registers. The cost is
not losing SIMD, it is losing **row-loop unrolling**: split keeps the body small
enough to unroll ~5× and hide FMA latency, the fused body cannot. So spilling is
a better signal than vector-value counts — but it is still not sufficient
(medium/straight has *more* spills than heavy/straight and is 6× better),
because it does not see dependency-chain length.

**Consequence for doc 02 §1.2's `explain_kernels()`:** a vector-IR-value count
would have reported "vectorised, fine" for every regression measured here. Do
not ship it as the fusion signal.
