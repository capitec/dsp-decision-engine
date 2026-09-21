# A cfunc-pointer generic interpreter for Branch/Loop — the mechanism is free, the shape is not

**Question:** can decider2's control flow (`Branch`/`Loop`) be driven by ONE
generic kernel over data, calling steps through `@numba.cfunc` raw function
pointers, instead of generating Python source text per pipeline shape? §S
measured the "obvious" version of this idea (dispatch through
`numba.typed.List[FunctionType]`) and it lost badly — 28-41x codegen. §V
measured a *different* mechanism (a raw C-ABI pointer, address known at
*compile* time) at 2.4-3.84 ns/call. Nobody had measured a generic
interpreter dispatching through `@cfunc` addresses carried as *runtime
data* — the actual mechanism a "call step k, where k is data" kernel needs.

**Answer, in one paragraph:** the call *mechanism* is not the problem — a
raw address read out of a numpy array at a runtime index costs the same as
numba's own blessed `ctypes`-global call (confirmed identically across
eight (signature × callee-origin × dispatch-mechanism) combinations,
~7.2 ns/call for a float64 cfunc, matching AOT and JIT-compiled callees
alike), and calling through a genuinely *varying* index is not slower than
a fixed one. That is a real, previously-unmeasured result and it closes
the gap between §S and §V cleanly: raw pointers are not "near §S" at all.
But a generic interpreter that dispatches through the pointer table **once
per control-flow node** pays that ~7 ns tax **many times per row** — a
nested `Loop(Branch(...))` visits ~15-25 nodes per row — and that
frequency, not the per-call cost, is what makes it **8.8x slower than
decider2's real codegen** on decider2's own nested `Loop(Branch(steps1,
Loop(steps2)))` example, at bit-identical answers, 100k rows, stable
across 5 independent reruns (8.82-8.90x). That is worse than §Q's plain
numba array interpreter (2.0-2.7x slower than codegen, at full-binary
depth ≤7) and meaningfully better than §S's function-value dispatch
(28-41x) — a real, new data point, but not a win. On the two questions §V
left open, this mechanism wins outright: it stays fused into one
`CompiledSegment` (confirmed against decider2's real `build_driver`), and —
unlike ctypes/`.so` symbols — **it survives numba's on-disk cache across a
process restart** (confirmed against a persistent, non-temp `__pycache__`,
with `NUMBA_DEBUG_CACHE=1` showing genuine "index loaded"/"data loaded"
hits, not recompiles) because the address is an ordinary function argument,
never a compile-time global. **Recommendation at the bottom.**

Everything below is under `experimentation/cfunc-pointer-interpreter/`.
`decider2/src/` was read and its real `Branch`/`Loop`/`build_driver` were
imported and run, never modified or reimplemented. Re-run everything with
`./run_all.sh` (also regenerates `results.jsonl`, `run_all.log`). Machine:
28-core `i7-14700HX`, Python 3.14.5 / numba 0.67.0 / llvmlite 0.49.0 /
numpy 2.4.6 — same environment EXPERIMENTS.md's other entries used. Every
process ran under `ulimit -v 2000000` (2 GB), `free -h` showed 8-18 GB
available throughout (checked in `run_all.log`).

---

## 1. Call overhead — the decisive number, first

`bench_call_overhead.py` — a trivial `float64(float64)` `@cfunc`, called
5M times/rep, 5 reps, best-of-5 shown, full range in `results.jsonl`:

| mechanism | ns/call | range |
|---|---|---|
| `njit -> njit` (inlinable; real loop overhead, not algebraic elimination) | **0.96** | 0.96-0.99 |
| `njit -> cfunc`, address baked in as a compile-time constant | 7.24 | 7.24-7.27 |
| `njit -> cfunc`, 1-entry pointer table, **fixed** runtime index | 7.28 | 7.28-7.56 |
| `njit -> cfunc`, 3-entry pointer table, **varying** runtime index (round-robin) | 4.43-4.79 | (faster than fixed — see note) |

Reference, EXPERIMENTS.md, not re-derived: §V `njit -> njit` 0.00 ns
(algebraic loop elimination for an integer accumulator — not a real "call
costs nothing" result; the float64 loop above can't be eliminated that way
and gives the real ~0.96 ns floor); §V `njit -> ctypes-global extern "C"`
2.4-3.84 ns (a real Rust cdylib, `int32(int32,int32)`); §S
`numba.typed.List[FunctionType]` 28-41x codegen, no bare ns/call figure
(measured embedded in a tree walk).

**The varying-index case being *faster* than the fixed-index case is
real and reproducible** (confirmed across the full `run_all.sh` run and
several exploratory reruns), not noise — the likely cause is that LLVM can
unroll the 3-way round-robin loop and overlap three independent,
data-hazard-free indirect calls, where the fixed-index loop has one
tight serial dependency chain through the accumulator. Flagged as an
observation with a plausible mechanism, not fully diagnosed (would need
disassembly to confirm) — but it means "dispatch varies every call" is not
a tax on top of "dispatch is fixed," which is the realistic shape for an
interpreter walking heterogeneous nodes.

### Why 7.2-7.3 ns and not §V's 2.4-3.84 ns — signature and origin, not mechanism

This gap looked, at first, like it might mean the raw-pointer mechanism
itself was expensive. `bench_signature_matrix.py` isolates it: the SAME
trivial function, compiled two ways (gcc `-O3` AOT `.so` vs `@numba.cfunc`
JIT), called two ways (numba's own `ctypes`-global mechanism vs the
raw-address `@intrinsic` this experiment introduces), at two signatures
(`int32(int32,int32)` — §V's own shape — and `float64(float64)` — decider2's
actual boundary type, doc 05 §1.5). Two full interleaved rounds, reproduced
again in `run_all.sh`:

| signature | callee origin | ctypes-global | raw-address (this experiment) |
|---|---|---|---|
| int32 | AOT (gcc -O3 `.so`) | **1.92 ns** | **1.92 ns** |
| int32 | JIT (`@cfunc`) | 8.6-9.3 ns | 8.6-9.3 ns |
| float64 | AOT (gcc -O3 `.so`) | 7.24 ns | 7.25 ns |
| float64 | JIT (`@cfunc`) | 7.18-7.28 ns | 7.15-7.27 ns |

**In every one of the 8 cells, the dispatch mechanism (ctypes-global vs a
raw address via `@intrinsic`/`inttoptr`) is identical within noise.** What
actually moves the number is (a) argument width/type and (b) — for narrow
integer args specifically — whether the callee was numba/LLVM-JIT'd or
AOT-compiled (int32 JIT costs 4.5x its AOT twin; float64 shows no such
gap). §V's 2.4-3.84 ns is the `(int32, AOT)` cell's neighbourhood, not a
universal "C-ABI call" constant. decider2 steps are float64/int64/bool at
the kernel boundary (doc 05 §1.5), so **~7.2-7.3 ns/call is the number that
actually applies**, and it is identical whether the pointer arrives as a
numba global or as data in a runtime-indexed array — reproduced with a
gcc-compiled `.so` as an independent third data point
(`c_stand_ins/trivial_i32.c`, `trivial_f64.c`), not just the two mechanisms
under test.

**Verdict on item 1: does not land near §S** (no bare ns/call there, but
28-41x codegen on a tree walk is a different order of magnitude from a
~7.5x call-overhead multiple over an inlined call) — **and does not land
at §V's cited 2.4-3.84 ns either**, because that number was never a
mechanism constant to begin with. The real, apples-to-apples finding is
narrower and cleaner: **the pointer-table indirection itself costs
nothing extra over any other way of calling the same compiled code from
njit.** That is the major result the brief asked to flag plainly if it
showed up, and it did.

---

## 2. A generic control-flow interpreter — one kernel, flat arrays, no codegen

`interpreter.py`: one `@njit(cache=True)` kernel, `run_program`, walking 9
opcodes (`STEP1/STEP3`, `COND1/2/3`, `SET_ZERO`, `INCR`, `JUMP`, `LEAF`)
over flat int32 arrays (`op`, `step_idx`, `arg0-2`, `dest`, `next_`, `alt`)
plus a per-kernel `param_template` and a per-row `row_regs` array. A loop
head is a `COND` node whose false edge exits; init/increment are their own
tiny opcodes so a loop's counter resets on entry and survives the back-edge
— the flat-array analogue of a generated `while`'s own local variable.
Steps are called through `ptr_table[step_idx[pc]]` via the `@intrinsic`
raw-address mechanism from §1 — nothing about the callee is known to LLVM
until runtime.

`test_shapes.py` verifies three shapes against an independent
non-generic-kernel reference (`np.where`/a plain Python loop), all exact
(`np.array_equal`):

- **(a) straight-line**: `(x+1)*2-3` as three chained `STEP1`s, 10k rows.
- **(b) Branch**: `x+10 if x>50 else x-5`, 10k rows.
- **(c) Loop with a real, data-dependent early exit**: extend `term_cap` by
  1.0/iteration while `principal/term_cap > ceiling`, bounded at 1000 —
  same shape as decider2's own `ExtendTermToFit`. 5k rows: 2,056 rows exit
  on iteration 0 (already under ceiling), 0 rows hit the bound — a genuine,
  per-row-varying early exit, not a fixed trip count.

**(d) the nested shape, `Loop(Branch(steps1, Loop(steps2)))`**, is
`nested_program.py` — see §3, which checks it against decider2's own real
compiler rather than a hand reference (the stronger check).

---

## 3. Identical work against decider2's real codegen

`compare_vs_codegen.py` imports `decider2.examples.playground_control_flow.
SearchForViableTerm` UNCHANGED — a real, already-tested decider2 pipeline:
the outer `Loop` (carries `term_cap`, `should_continue=term_still_short`)
wraps `AdjustmentStrategy` (a `Branch` on `is_high_income`) whose True arm
is `FastTrackBump` (an inner `Loop`) and False arm is `slow_bump` — i.e.
`Loop(Branch(steps1, Loop(steps2)))` in full, with two independent
data-dependent early exits (outer loop stops when `term_cap` reaches
`floor`; which arm the Branch takes changes how fast it gets there per
row). `steps_nested.py` reimplements the SAME five step functions as
`@cfunc`s with identical arithmetic (verified byte-for-byte against
`build_driver`'s own known output on the doc's worked rows:
`[48, 42, 48, 42, 48]`, before scaling to 100k).

100k rows, 5 independent full process reruns (`np.random.default_rng`
seeded — §Q's lesson: never `hash()` a string, CPython randomises string
hashing per process):

| | codegen (real `Branch`/`Loop`) | cfunc-pointer interpreter |
|---|---|---|
| ns/row, best-of-5-within-run | 25.05-25.76 | 222.43-224.10 |
| ns/row, full range across all reps/runs | **25.05-27.40** | **222.43-256.43** |
| ratio (interp/codegen), per run | — | **8.82-8.90x** |
| answers | — | **`np.array_equal` exact, every run, all 100,000 rows, both before and after a retune** |

Both far more stable than §Q's d9/d10 codegen instability (80% spread on a
2,000+-line generated function) — unsurprising, since this pipeline emits
only five short generated functions, nowhere near that line count.

**Reading this against §Q/§S**: 8.8x is a real, new number — not §S's
28-41x (function-value dispatch), and worse than §Q's 2.0-2.7x (a plain
numba array walk, no `@cfunc`/`@intrinsic` machinery at all). The
mechanism win from §1 is real but it does not close this gap, because
**the gap is about call FREQUENCY, not call COST**: §V's near-parity result
(a C-ABI call 26-35% *faster* than the numba walker) came from crossing the
boundary **once per row** — the whole tree walk happens inside one foreign
call. This interpreter crosses the boundary **once per control-flow node**
— the nested pipeline above visits roughly 15-25 nodes per row (an outer
loop iteration is itself ≥3 nodes, an inner `FastTrackBump` iteration
taken 0-3 times adds 2 more each) — so the ~7.2 ns tax from §1 is paid
that many times, not once. A per-call cost of effectively zero extra
overhead (§1) still adds up when the call count itself is what scales with
program complexity. This is the sharper, mechanism-isolated version of
what §S already found by a different (slower) route: a fully generic,
per-node-dispatch interpreter is structurally disadvantaged against fused,
inlined machine code, independent of how cheap any one dispatch is made.

---

## 4. Build cost

`build_cost.py`, against the same `SearchForViableTerm` pipeline:

| | codegen | interpreter |
|---|---|---|
| cold (fresh build dir, whole 5-function pipeline) | 146-525 ms | — |
| **rebuilding the SAME shape** (warm on-disk cache, fresh `Driver`) | **2.8-5.7 ms** | **0.06 ms** (pure Python/numpy, no compiler invoked) |
| building a **genuinely different shape**, reusing existing steps | n/a (codegen always emits new source per shape) | **0.04-0.06 ms** |
| adding **one brand-new custom step function** | would require a new `Branch`/`Loop` compile | 25-28 ms (one `@cfunc` compile — a single small function, not a whole driver) |
| `run_program.signatures` count, across every shape/rerun above | n/a | **stayed at 1** |

The interpreter's own generic kernel compiles exactly **once, ever** — the
first call in `cache_check.py`'s cold run. Every shape after that (item 2's
three standalone shapes, item 3's nested pipeline, and the brand-new
straight-line-plus-new-cfunc case above) reuses it with zero additional
`run_program` specialisations, confirmed by `.signatures` staying at 1
throughout. Even introducing a genuinely new step function costs ~25-28 ms
for that one function — a small fraction of codegen's cold 146-525 ms for
a whole 5-function pipeline, though not the literal "zero" the interpreter
gets for reusing existing steps in a new arrangement.

---

## 5. Does it stay in one kernel? Does it survive the disk cache?

**Fusion — confirmed structurally, decider2's own unmodified compiler.**
`build_driver(SearchForViableTerm.steps, group_ids=[0], ...)` returns
**exactly one segment**, `driver.segments[0].kind == "compiled"`
(`CompiledSegment`, not `FallbackSegment` — the `@cfunc` step functions
never tripped `_FALLBACK_TRIGGERS`), `type(kernel_fn) is numba.core.
registry.CPUDispatcher`. Matches §V's structural check exactly, on the
real compiler rather than a simulation.

**Cache — the opposite finding from §V, and this is the good news.**
`cache_check.py` against a persistent `__pycache__` (never a tempdir),
across a real process restart, `NUMBA_DEBUG_CACHE=1`:

| | elapsed (import + build + first call) |
|---|---|
| cold (fresh `__pycache__`) | 1026-1045 ms |
| warm (fresh process, same `__pycache__`) | **373-392 ms** (~2.7x faster, reproduced across 2 independent warm runs) |

Zero `NumbaWarning`s about caching or dynamic globals, on every run.
`NUMBA_DEBUG_CACHE=1` on a warm run shows, explicitly, for every one of the
five step `@cfunc`s AND for `run_program` itself:
```
[cache] index loaded from '.../steps_nested.is_high_income-22.py314.nbi'
[cache] data loaded from '.../steps_nested.is_high_income-22.py314.1.nbc'
...
[cache] index loaded from '.../interpreter.run_program-133.py314.nbi'
[cache] data loaded from '.../interpreter.run_program-133.py314.1.nbc'
```
— genuine cache **hits**, not silent recompiles, and the `.nbi`/`.nbc`
file mtimes are byte-identical (`stat`-verified) before and after a further
warm run: nothing was rewritten.

**Why this differs from §V.** §V's kernel referenced a `ctypes.CDLL`
symbol as a **module-level global** — numba specialises the compiled
kernel's own machine code on that specific Python object, which is exactly
what "dynamic globals" means and why it can never be cached. `run_program`
never references any cfunc, `ctypes` object, or address as a global at
all — `ptr_table` is an **ordinary `uint64[:]` argument**, ordinary DATA,
like every other array `run_program` takes. There is no dynamic global to
void the cache, and the empirical result confirms it: **this is a genuine,
substantive point in this mechanism's favour over §V/§U's Rust C-ABI
approach** — it keeps decider2's on-disk cache guarantee (doc 05 §4.2's
seven conditions, `decider2 build --verify`) intact, which §V's approach
silently does not.

**One caveat not tested here, flagged for honesty**: a `@cfunc` that
raises inside its body has no Python exception to propagate across the C
ABI boundary — numba's own documentation warns this is undefined behaviour,
architecturally the same class of risk §U/§V found for an unprotected Rust
`panic!()` (a process-ending failure, not a catchable one), though this
experiment did not reproduce or measure it. A production version of this
approach would need the same discipline §V recommends for Rust: every step
cfunc wrapped defensively, with nothing enforcing that a future step
remembers to.

---

## 6. Params — how do they arrive through a pointer table?

decider2's whole config story (doc 08 §2) is that a threshold is a kernel
**argument**, never an emitted literal, so retuning is a value swap, not a
recompile. `nested_program.py`'s `param_template()` answers this
concretely: params occupy their own fixed register slots (`REG_FLOOR`,
`REG_MICRO_STEPS`, `REG_JUMP`, `REG_STEP_SIZE`), written **once per kernel
invocation** (never touched inside the row loop), and every row starts
from `regs = param_template.copy()`. Retuning is `param_template[REG_FLOOR]
= new_value` — a plain float assignment in Python, nothing about
`run_program` changes.

`compare_vs_codegen.py` demonstrates this working, for **both** engines,
on the same retune (`term_still_short`'s `floor`: 40.0 → 25.0), on the
same 100k rows:

| | output changed by the retune | signatures before | signatures after |
|---|---|---|---|
| codegen (`ResolvedParams` value change) | yes (every row where `floor` was the binding constraint) | 1 | **1** |
| interpreter (`param_template` value change) | yes, identically | 1 | **1** |

Both engines' output changes (the retune has a real effect), neither
engine's compiled-signature count grows (`driver.signatures` and
`run_program.signatures` both stay at 1 — asserted, not just observed),
and **the two engines still agree exactly** (`np.array_equal`) on the
retuned output. A value-only retune is structurally incapable of
retriggering numba's type-based specialisation either way, but this
confirms it rather than assuming it, exactly as the brief asked.

---

## Recommendation

**Do not replace decider2's source-text codegen with this generic
cfunc-pointer interpreter.** Two things it was asked to settle come back
genuinely positive and are worth keeping on record: the raw-pointer-table
call mechanism itself is free (§1 — indistinguishable from numba's own
blessed `ctypes`-global call, in every signature/origin combination
tested), and — unlike §U/§V's Rust C-ABI approach — it does not cost
decider2 its on-disk compile cache (§5), because the pointer table is
ordinary runtime data rather than a compile-time global. Those are real,
previously-unmeasured results and they close the gap the brief identified
between §S and §V cleanly: a raw pointer is not "the same idea as §S, just
worse-measured" — it is a materially cheaper mechanism.

But the thing that actually matters — running decider2's own nested
`Loop(Branch(...))` pipeline — costs **8.8x** what codegen costs, stable
across five reruns, because a fully generic interpreter calls through the
pointer table once per control-flow node, and a realistic pipeline visits
many nodes per row; §1's near-zero mechanism tax gets paid that many
times over. That lands this approach **worse than §Q's plain numba array
interpreter (2.0-2.7x)**, which gets the same headline properties this
experiment was built to test — one generic kernel, zero recompiles per new
shape, no ~500-line cap, no CPython indentation limit — using ordinary
njit code, with no `@intrinsic`/LLVM-IR authoring layer, no `@cfunc` ABI
contract, and none of §5's unresolved exception-safety caveat. There is no
case in which this experiment's mechanism beats §Q's simpler one on
runtime cost, and §Q already answers the maintainability complaint that
motivated this whole line of investigation. If decider2 ever needs to
retire the codegen path, §Q's array interpreter remains the better
starting point; this experiment's contribution is settling — not
reopening — the question of whether raw pointers would have changed that
conclusion. They don't.
