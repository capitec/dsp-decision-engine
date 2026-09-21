# Rust via the C ABI, called from inside an njit kernel — the sixth option, re-decided

**Question:** §U (EXPERIMENTS.md) measured a PyO3 Rust tree interpreter and
recommended against it, on the strength of one argument — *"a Rust tree
cannot be inlined into a fused njit kernel."* That argument has already been
corrected in EXPERIMENTS.md: it is true of PyO3 and **false of the C ABI**.
A quick C-standing-in-for-Rust probe (cited in the correction, not reproduced
here) found the C-ABI call 26% *faster* than the pure numba walker and the
boundary itself costing ~3.84 ns. This experiment redoes that properly, with
real Rust, and re-decides the whole question on cost of ownership — since
the architectural objection that drove §U's verdict is gone.

**Answer: the architectural correction holds, with real Rust — the C ABI is
measurably faster than the numba array walker AND stays fused. It is not,
however, free: it permanently loses numba's on-disk kernel cache, requires a
new `unsafe`-and-`catch_unwind` discipline to avoid turning a bad row into a
dead process, and the packaging win is real but narrower than "no cost" —
one `.so` per platform, still per-`libc`-ABI. Weighed against the numba array
walker (the realistic alternative, not codegen — see "What this would
replace" below), the array walker still gets ~90% of the maintainability
win decider2's owner actually asked for, for none of this cost. Do not adopt
by default; the correction to §U stands as a documented, working option for
a throughput-bound case, not a default recommendation.**

Everything below is under `experimentation/rust-cabi-in-kernel/`.
`decider2/src/` was read, never modified. Row generation, tree shapes, and
the numba array walker are the **unmodified** files from
`experimentation/tree-codegen-vs-interpreted/` (`tree_shapes.py`,
`interpreted_kernel.py`), imported via `sys.path`. The Rust walk logic
(`walk_one_inner`) is lifted **unchanged** from
`experimentation/rust-tree-interpreter/rust_tree_interpreter/src/lib.rs` —
only the binding around it is new (no PyO3, no `numpy` crate — raw pointers
and lengths, loaded with `ctypes.CDLL`).

## What was built

`rust_cabi_tree/` — a plain `cdylib` (`cargo build --release`, cold 11.84 s,
cached 0.03 s, incremental 0.43 s), **zero Python dependencies at compile
time**: `[dependencies] regex = "1.12"` only. Exposes ten `#[no_mangle] pub
extern "C" fn` entry points:

- `walk_row` / `walk_row_unprotected` — one row per call, the fusion-
  preserving shape. `walk_row` wraps the body in `catch_unwind`; the
  `_unprotected` twin exists only for the panic-safety demo.
- `walk_batch` — one call for the whole batch, loop inside Rust,
  `catch_unwind`-wrapped once around the whole loop.
- `noop` / `trivial` — call-overhead baselines.
- `regex_compile` / `regex_is_match` / `regex_free` — compile-once,
  call-many via an opaque handle; no `Vec<String>` marshalling anywhere.
- `deliberately_panic_protected` / `deliberately_panic_unprotected` — the
  minimal panic-safety demo pair.

Every array argument is `*const T`/`*mut T` on the Rust side and
`ctypes.c_void_p` on the Python side, with every length crossing as a
separate `i32`/`i64` — numba's ctypes bridge rejects `c_char_p` and accepts
`c_void_p` (§T), confirmed again here. Inside each function, raw pointers
are turned into Rust **slices** (`slice::from_raw_parts`, bounds-checked
`[]` indexing from then on) — never dereferenced as raw pointer arithmetic
— which is what makes the panic-safety section below a demonstration of a
*controlled* failure rather than silent memory corruption.

`tree_walk_cabi.py` / `regex_walk_cabi.py` hold the `ctypes` wiring and the
`@njit` wrapper functions bench_perf.py, the fusion check, and the panic
demo all call — one binding, reused everywhere, never duplicated.

---

## 1. ns/row at 100k rows — against every existing option

Three independent full reruns of `bench_perf.py` (`run1.log`/`run2.log`/
`run3.log`), 7 timed calls per shape per run (21 total), **range reported,
not mean** — §Q found codegen's variance mattered more than its mean, and
that discipline is kept here.

| shape | leaves | numba walker (measured here) | **C-ABI, per-row** | C-ABI, per-batch | codegen (§Q, cited) |
|---|---:|---:|---:|---:|---:|
| credit tree | 20 | 52.5–64.6 | **46.0–53.7** | 43.2–51.1 | 20.7–22.9 |
| full-binary d5 | 32 | 59.7–81.8 | **49.5–64.2** | 48.9–60.0 | 24.5–26.8 |
| full-binary d7 | 128 | 85.0–105.1 | **66.5–73.7** | 66.9–75.7 | 42.2–49.7 |
| full-binary d9 | 512 | 120.0–141.5 | **81.5–94.8** | 88.2–98.3 | 82.8–127.0 (unstable, §Q) |
| one-sided chain, 100 | 101 | 34.1–42.3 | **28.1–35.1** | 26.5–32.7 | 28.4–28.8 |

Answers asserted identical across all three engines measured here (numba ==
C-ABI/row == C-ABI/batch), every shape, every run, before any timing
counted.

**The correction holds with real Rust: the C-ABI walker beats the pure
numba array walker on every shape, by roughly 12–35% depending on shape** —
consistent with the corrected §U probe's 26%-faster full-binary-d7 result
(here: numba 85.0–105.1 vs. C-ABI/row 66.5–73.7, i.e. ~14–30% faster
depending which end of each range you compare). Cross-checked against §U's
own aggregate PyO3 numbers (Rust interpreter 46.0–88.7 ns/row, numba array
walker 52.5–120.7 ns/row across the same five shapes) — the C-ABI numbers
sit inside the same band the PyO3 crate already occupied, which is the
expected result: `walk_one_inner` is the *same code*, so the C-ABI version
should cost about what PyO3's did minus PyO3's own marshalling, and that is
what it costs.

**Codegen still wins outright at every shape except d9** — the C-ABI
approach does not change that conclusion at all; it only replaces "the
numba array walker" as the interpreted alternative's fastest variant.

### The full six(seven)-way picture, one shape (full-binary d7, 100k rows)

d7 is the one shape every prior tree-engine experiment measured, so it is
the only point where all seven options line up:

| option | ns/row | source |
|---|---:|---|
| codegen | 42.2–49.7 | §Q, cited |
| numba array walker | 85.0–105.1 | measured here (§Q: 85.5–100.7; §T: 73.3 on a different seed — same order) |
| jitclass node-object walk | **708.5** | §T, cited (9.66× the struct-of-arrays walk) |
| composed njit callables (`typed.List[FunctionType]`) | not directly comparable — §S cites 14–20× the array interpreter's own ns/row | §S, cited as a multiplier, not re-derived into an absolute number |
| Rust via PyO3 (§U) | 46.0–88.7 (aggregate across shapes, not d7-specific) | §U, cited |
| **Rust via C-ABI, per-row (this experiment)** | **66.5–73.7** | measured here |
| **Rust via C-ABI, per-batch (this experiment)** | **66.9–75.7** | measured here |

jitclass and composed callables are not competitive with anything else in
this table — they were never in contention for a production tree engine,
and this experiment's numbers confirm §T/§S's conclusion rather than
revisit it. The real contest, at every shape, is codegen vs. {numba array
walker, Rust/PyO3, Rust/C-ABI} — and codegen wins it except at d9.

---

## 2. Does the fused kernel stay fused? — yes, confirmed structurally

Built a real, unmodified-`decider2.compile.driver.build_driver` pipeline
(`fusion_check/pipeline_module.py`): three ordinary steps —
`scale_features` (plain arithmetic) → `tree_decision` (the C-ABI tree walk,
via the SAME `walk_row_cabi` `bench_perf.py` times) → `apply_bonus`
(reads both prior steps' outputs by name) — all in one `fuse()`-group
(`group_ids = [0, 0, 0]`).

**Result:** `build_driver` returns exactly **one segment**, and it is a
`CompiledSegment` (`isinstance(seg, CompiledSegment)` — not a
`FallbackSegment`; the C-ABI step did not trip `_FALLBACK_TRIGGERS`).
`decider2.compile.codegen.emit_kernel_source(plan)` — the exact source text
`decider2.compile.cache.get_or_build` writes to disk and numba compiles —
contains **exactly one** `def kernel(`. Quoted verbatim
(`fusion_check/generated_kernel_source.py`):

```python
"""Generated kernel for group 'g0_scale_features_tree_decision_apply_bonus'.
...
"""
from __future__ import annotations

from numba import njit, prange

from pipeline_module import scale_features as _src_scale_features
from pipeline_module import tree_decision as _src_tree_decision
from pipeline_module import apply_bonus as _src_apply_bonus

_compiled_scale_features = njit(cache=True)(_src_scale_features)
_compiled_tree_decision = njit(cache=True)(_src_tree_decision)
_compiled_apply_bonus = njit(cache=True)(_src_apply_bonus)

@njit(cache=True)
def kernel(arr_f0, arr_f1, arr_f2, arr_f3, out_apply_bonus):
    n = arr_f0.shape[0]
    for i in range(n):
        v_f0 = arr_f0[i]
        v_f1 = arr_f1[i]
        v_f2 = arr_f2[i]
        v_f3 = arr_f3[i]
        v_scale_features = _compiled_scale_features(v_f0, v_f1)
        v_tree_decision = _compiled_tree_decision(v_f0, v_f1, v_f2, v_f3)
        v_apply_bonus = _compiled_apply_bonus(v_tree_decision, v_scale_features)
        out_apply_bonus[i] = v_apply_bonus
```

**One row loop, one `@njit` function, all three steps called inline in
written order — including `_compiled_tree_decision`, whose body makes the
`extern "C"` call into `rust_cabi_tree::walk_row`.** There is no split
around the C-ABI step: `scale_features`'s and `apply_bonus`'s local
variables (`v_scale_features`, `v_tree_decision`) live and die inside this
one function's registers, the same "lives in the kernel's registers and
dies there" property doc 03 §7 describes for an ordinary all-numba group —
extended here across a step that happens to call into Rust. Driver output
was verified against an independent oracle (`interpreted_kernel._walk_one`
plus the same arithmetic, computed by hand in the test) and matched exactly
on 5 rows.

**This is the whole correction to §U, demonstrated rather than argued: the
fused driver does not fragment around a C-ABI call.**

### A real cost found along the way: the numba on-disk cache is gone for any kernel that touches this

Every compile of `tree_decision`/`kernel` above prints:

```
NumbaWarning: Cannot cache compiled function "tree_decision" as it uses
dynamic globals (such as ctypes pointers and large global arrays)
```

Confirmed **not** an artifact of this experiment's harness (a
`tempfile.TemporaryDirectory` build_dir would trivially explain "no cache
hit" on its own) by rerunning `time_one_build.py` against a **persistent**
`build_dir` for two pipelines side by side:

| pipeline | run 1 (cold) | run 2 | run 3 | run 4 |
|---|---:|---:|---:|---:|
| control (pure numba, no ctypes) | 0.301 s | **0.159 s** | 0.152 s | 0.150 s |
| C-ABI (`pipeline_module`) | 0.468 s | 0.471 s | 0.475 s | 0.479 s |

The pure-numba control gets numba's on-disk cache hit from run 2 onward
(~2× faster, persistent build_dir, same process each time it would have
mattered). **The C-ABI pipeline never does — every run pays the full
compile, every time**, because any njit function that references a
ctypes-loaded symbol (the function pointer itself, not the tree-array
globals — a minimal `njit` function doing nothing but calling
`trivial(a, b)` triggers the identical warning) is, structurally, uncacheable
to numba. This is a real, previously-unmeasured cost of the C-ABI approach:
decider2.compile.cache's whole design point (doc 05 §4.1/§4.2: content-
addressed, persists across process restarts) does not apply to any group
containing a C-ABI call. For this toy 3-step kernel the absolute cost is
~0.3 s per process start; for a large real pipeline it would be larger and
compounding across every fused group that touches the C ABI.

---

## 3. Per-row vs. per-batch calling — the fusion tax turns out to be near zero

Per-shape minimums from the table in §1 (per-row is the fusion-preserving
shape; per-batch reintroduces a boundary, one call for 100k rows instead of
100k calls):

| shape | C-ABI per-row (min) | C-ABI per-batch (min) | per-row penalty |
|---|---:|---:|---:|
| credit tree | 46.0 | 43.2 | +6.5% |
| full-binary d5 | 49.5 | 48.9 | +1.2% |
| full-binary d7 | 66.5 | 66.9 | **−0.6%** (row faster) |
| full-binary d9 | 81.5 | 88.2 | **−7.6%** (row faster) |
| one-sided chain, 100 | 28.1 | 26.5 | +6.0% |

**The per-row shape is not reliably slower than per-batch at all** — it
wins outright on two of five shapes and is within a few percent everywhere
else, well inside this experiment's own run-to-run noise (§1's ranges
overlap by more than this gap in every row). The isolated call-overhead
measurement (below) explains why: at ~2.4–3.84 ns/call against a 40–100 ns
walk, the per-row boundary is 3–9% of the walk itself, and batching removes
only that — it does not remove the walk. **This is the opposite of the
assumption this experiment set out to test**: giving up per-batch's raw
speed to keep fusion costs close to nothing at these tree depths, so there
is no real throughput reason to prefer the non-fusable shape here.

Isolated call overhead, real Rust (`bench_call_overhead.py`, 5M iterations,
mirrors the original C-stand-in probe):

| | ns/call |
|---|---:|
| `njit` → Rust `extern "C"` (real Rust, this crate) | 2.40–3.84 |
| `njit` → `njit` (fully inlinable) | 0.00 |
| *(cited: C-stand-in probe, identical ABI)* | *3.84* |

The real-Rust number matches the C-stand-in probe's number closely (both
in the 2.4–3.84 ns/call band) — the probe's "C standing in for Rust,
identical ABI" premise holds.

---

## 4. Regex via the C ABI — compile once, raw `(pointer, length)` per call

Compile-once-handle pattern (`regex_compile` → opaque handle → `regex_is_match(handle, ptr, len)`
per call), zero marshalling on the hot path — no `Vec<String>` anywhere.

| case | ns/call | cited baselines |
|---|---:|---|
| per-category (n=12), single Python→njit dispatch | 866.4–1094.2 | *(dominated by one-time dispatch overhead at this tiny n — not a fair "per call" number; see below)* |
| per-category (n=12), amortised (5,000× repeat, same dispatch) | **48.94–59.84** | §T numba→libc: 60.3 · §U Rust pure: 25.6 |
| per-row (n=100,000) | **37.56–45.72** | §T numba→libc: 60.3 · §U Rust pure: 25.6 · §U Rust/PyO3 incl. marshalling: 104.4 |

The single-dispatch per-category number is reported for completeness but
is not representative: dividing one Python→numba call's ~1–2 µs own fixed
dispatch cost by only 12 iterations dominates the measurement, the same
"single record is noise" effect §U found for its own per-call figures.
Amortising the same scan 5,000× inside one dispatch (`regex_match_variable_repeated`)
removes that artifact and lands, as expected, between §U's pure-Rust
number (no boundary at all) and §T's numba→libc number (a real boundary,
via a different C library) — **the C-ABI regex boundary itself costs
roughly the "extra" ~13–20 ns/call between those two**, consistent with
the ~2.4–3.84 ns bare call overhead measured in §3 plus the actual match
cost.

**The per-row number is the clean result: 37.56–45.72 ns/call beats both
§T's numba→libc (60.3) and §U's naive-PyO3-marshalled number (104.4)
outright**, and sits only modestly above §U's pure-compute-only citation
(25.6, which excludes any boundary at all). This is exactly what "raw byte
pointers, no marshalling" was supposed to buy, and it measures out that
way: the boundary cost is small, the match itself is the real cost, same
conclusion §U reached for numbers, now confirmed for strings via the C ABI
specifically.

---

## 5. Panic safety — demonstrated, not just asserted

rustc **1.95.0** — well past 1.71, the release that made an `extern "C"
fn` panicking **without** unwinding across the boundary a *defined* abort
rather than undefined behaviour (RFC 2945, `extern "C-unwind"`). This
crate deliberately does **not** set `panic = "abort"` at the crate level
(that would make `catch_unwind` a no-op everywhere), and instead wraps
every production entry point (`walk_row`, `walk_batch`, `regex_is_match`,
`regex_compile`) in `std::panic::catch_unwind` — the other option the brief
names, applied per-boundary rather than crate-wide.

Four subprocess runs (`panic_demo.py`, each mode isolated in its own
process so the expected crashes cannot take the harness down):

| mode | trigger | result |
|---|---|---|
| `trivial_unprotected` | `panic!()`, no `catch_unwind` | **killed, signal 6 (SIGABRT)** — stderr: `thread caused non-unwinding panic. aborting.` |
| `trivial_protected` | same `panic!()`, through `catch_unwind` | exit 0 — returned `NaN`, then a subsequent good call returned `42.0` |
| `tree_unprotected` | **realistic trigger**: `left[0]` corrupted to `999999` on a real tree, walked by `walk_row_unprotected` | **killed, signal 6 (SIGABRT)** — stderr: `thread caused non-unwinding panic. aborting.` (the out-of-bounds *slice* index panicked cleanly — not a raw-pointer segfault — see the `tree_slices` note in §0) |
| `tree_protected` | same corrupted tree, via `walk_row` (`catch_unwind`-protected) | exit 0 — returned `NaN` (stderr shows Rust's own panic message: `index out of bounds: the len is 7 but the index is 999999`), then a **second call on an uncorrupted tree, same process**, returned the correct answer (`1.0`) |

**Three things worth stating plainly.** First, the failure mode without
`catch_unwind` is not the classic FFI horror story of silent memory
corruption or a segfault — rustc 1.95's guaranteed abort-at-the-boundary
turns it into a clean, deterministic `SIGABRT`, every time, for both the
synthetic panic and the realistic corrupted-tree trigger. Second, that is
still a **whole-process crash** — every other request that process was
serving dies with it, which is categorically worse than an in-process
Python exception a caller can catch and route (doc 03 §1's own
`MissingInputPolicy` machinery exists precisely to avoid this class of
failure for ordinary bad input). Third, `catch_unwind` at the boundary
converts that into exactly the graceful degradation a bank's credit kernel
needs: one bad row returns a sentinel, the process — and every other
request it is serving — keeps running. **This is the `unsafe` surface the
brief asked about, made concrete: it is real, it is exactly this large, and
it is exactly this fixable — but only if every single boundary function
remembers to wrap itself, which is a discipline this crate had to state
and enforce by hand, not something the type system enforces for you.**

---

## Packaging — confirmed empirically, not just structurally

**§U's finding, re-verified:** the PyO3 `.so` (from `rust-tree-interpreter/`,
still on disk) exports `PyInit_rust_tree_interpreter` — a CPython C-API
entry point whose expected struct layouts are version-specific without the
`abi3` feature (not enabled in that crate). This C-ABI `.so` exports
**no** `PyInit_*` symbol at all — `nm -D` confirms it, and it should not,
because it is not a Python extension module; it is loaded by `ctypes.CDLL`,
which does not care what produced the file.

**Empirical proof, not inference.** This machine has Python 3.14.5 (the
`.venv`) and a separate Python 3.12.12 install:

```
$ python3.12 -c "
import ctypes
lib = ctypes.CDLL('.../librust_cabi_tree.so')
lib.trivial.argtypes=[ctypes.c_int32, ctypes.c_int32]; lib.trivial.restype = ctypes.c_int32
print('trivial(3,4) =', lib.trivial(3,4))"
trivial(3,4) = 7                          # SAME .so file, no rebuild, different Python minor version

$ python3.12 -c "import rust_tree_interpreter"     # the PyO3 .so, compiled as cp314
ImportError: .../rust_tree_interpreter.so: undefined symbol: Py_TYPE
```

**The C-ABI `.so` loads and runs correctly under a Python minor version it
was never built against. The PyO3 `.so`, built as `cp314`, fails immediately
under 3.12 with an ABI-mismatch symbol error.** This directly confirms §U's
question: a `cdylib` loaded by `ctypes` is **one shared object per platform,
not per Python version** — a materially smaller packaging surface than
PyO3's wheel matrix.

**What did not change:** both `.so` files link only `libgcc_s`, `libc`,
`ld-linux` — neither links `libpython`, so that specific axis was never a
PyO3-vs-C-ABI differentiator. Both target the same `GLIBC_2.34` floor (the
same `manylinux_2_34` tier §U measured) — the OS/glibc axis is genuinely
unchanged; only the Python-ABI axis is gone. Build cost is lower here too
(cold 11.84 s / cached 0.03 s / incremental 0.43 s, against §U's cold 29.0 s
/ cached 14.7 s / incremental 0.9–1.5 s) — but that is mostly PyO3's own
proc-macro compile-time cost, not a structural property of the C ABI; a
leaner PyO3 crate would close most of this gap. The packaging win is the
real one: **one `.so` per (OS, architecture, libc-ABI) combination is a
genuinely smaller ongoing cost** — no wheel-per-Python-version matrix, no
`abi3` feature flag to remember, no `cibuildwheel`-style CI needed purely
for Python-version coverage (a real, cross-platform `.so` build matrix is
still needed for OS/arch, same as any native dependency).

---

## Cost of ownership

- **A second language and toolchain, unchanged from §U.** rustc/cargo,
  contributors who can read and write `unsafe extern "C" fn`, CI needing a
  Rust build step (11.84 s cold here — cheap, but still new).
- **The numba on-disk cache is gone for any kernel that calls through the
  C ABI** (§2's finding) — a real, measured, previously-undocumented cost
  distinct from anything §U raised, small here (~0.3 s) but structural: it
  applies to every fused group touching the C ABI, forever, on every
  process start.
- **A hand-maintained, compiler-unchecked FFI boundary.** `ctypes.argtypes`
  on the Python side and the Rust `extern "C" fn` signature must be kept in
  sync by hand — no proc macro, no build-time check, generates the wrong
  answer or a segfault silently if they drift. PyO3 at least generates its
  binding from the Rust signature; the C ABI does not generate anything.
- **The `unsafe`/panic surface is real and now demonstrated, not
  hypothetical** (§5): every entry point must remember `catch_unwind` or a
  single bad row takes the whole process down, and that discipline is
  enforced by convention (a docstring, a review), not by the compiler.
- **Debugging across the boundary** still needs both a Python debugger and
  gdb/`RUST_BACKTRACE` fluency — unchanged from §U, and this experiment's
  panic demo shows the stderr trace is at least legible (`thread panicked
  at src/lib.rs:54:11: index out of bounds...`), which is better than
  nothing but still a second toolset a Python-and-numba team does not
  otherwise need.

### What this would replace

decider2's owner's stated motivation for retiring codegen is
**maintainability**, not speed: codegen has produced one silent wrong
answer (sibling conditions sharing a parameter name, `5 < x < 10` silently
never satisfiable), a ~500-line cap, a fan-out wall, a CPython indentation
limit, and — §Q's own finding — unstable p99 past ~2,000 emitted lines. The
**numba array walker already fixes every one of those**, in the language
decider2 is already written in, with no new toolchain and no new failure
mode. Measured here, the C-ABI approach beats that array walker by
12–35% per row (§1) and adds nothing to codegen's list of fixed problems —
it is faster than the thing that already solved the maintainability
complaint, not a further fix to a complaint the array walker leaves open.
The comparison that matters is Rust-via-C-ABI vs. **the numba array
walker**, not vs. codegen — and on that comparison, both engines clear the
20–100 ms budget by a wide margin at every shape measured (worst case here,
d9 per-row: 94.8 ns/row × 100k = 9.5 ms), so the 12–35% gap is real but not
decision-relevant at decider2's realtime or batch scale.

## Recommendation

**Do not adopt the C ABI by default; keep it as a documented, working
correction to §U's verdict, revisited only if a future measurement shows
the numba array walker's speed — not its architecture — is an actual
bottleneck.** The architectural objection that drove §U's original
recommendation is gone: this experiment confirms, with real Rust and a real
decider2 driver, that a per-row C-ABI call stays inside one fused njit
kernel, costs ~2.4–3.84 ns to cross, and at realistic tree depths the
"fusion tax" of calling per-row instead of per-batch is close to zero
(§3) — the single biggest reason to prefer per-batch turns out barely to
apply here. Packaging is a genuinely smaller cost than PyO3's, confirmed
empirically rather than assumed (a `.so` loads across Python minor versions
with zero rebuild; a PyO3 extension does not). But this experiment also
found a cost §U did not measure — the numba on-disk cache disappears for
any kernel touching the C ABI — and reconfirmed every cost §U already
found that the correction didn't touch: a second language, a
compiler-unchecked FFI surface that must be hand-kept in sync, and a panic
discipline that, absent `catch_unwind` on every single boundary function,
turns one bad row into a dead process serving every other request too
(now demonstrated, not hypothesised). Against those costs, the numba array
walker still delivers the actual, stated maintainability win — one generic
kernel, no line cap, no indentation limit, no per-shape compile, and it
stays inside decider2's one existing language and one existing cache
model — for about 90% of the C ABI's measured speed and none of its new
failure modes. That is a perfectly good outcome, and it is the likeliest
one this experiment was always going to land on; the correction to §U
changes *why* Rust-via-C-ABI is not chosen (cost of ownership, freely
conceding the architecture works) rather than *whether* it is chosen.
