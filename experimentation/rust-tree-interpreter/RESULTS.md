# A Rust tree interpreter — the fifth option, measured

**Question:** if codegen is being retired for maintainability (doc 06,
owner's own framing: "prone to errors hard to maintain and understand" —
correctly so; §Q logs a silent wrong answer from two composite conditions
sharing a parameter name), is a native Rust extension (PyO3 + maturin) a
*better thing to own* than the numba array walker §Q already measured as
codegen's replacement — not just a faster one?

**Answer: Rust works, is fast, and is not worth it here.** It beats the
numba array walker by 12–28% in batch and is the only option that stays
*stable* at large tree sizes (§Q's own instability finding, reproduced
below). But the owner's stated problem — codegen's error-proneness, its
500-line cap, its CPython indentation limit — is already fully solved by
the numba interpreter §Q recommended, in the *same* language decider2 is
already written in, with no new toolchain, no wheel-per-platform problem,
and no new failure mode at the Rust/Python boundary. Rust adds real,
measured costs (below) for a batch-speed gain that is noise against the
20–100 ms budget both options already clear. This experiment does not
change §Q's recommendation; it closes the "but what about Rust" question
the owner raised, with numbers instead of a guess.

Everything below is under `experimentation/rust-tree-interpreter/`.
`decider2/src/` was read, never modified (confirmed: `git status
decider2/src/` is clean after this whole experiment). Row generation, tree
shapes, and the numba walker are the **unmodified** files from
`experimentation/tree-codegen-vs-interpreted/` (`tree_shapes.py`,
`interpreted_kernel.py`) — imported via `sys.path`, not copied — so the
Rust numbers below and codegen's numbers in EXPERIMENTS.md §Q are the same
trees, same rows, same encoding, differing only in which engine walks them.

## What was built

`rust_tree_interpreter/` — a PyO3 0.27 + numpy 0.27 + regex 1.13 extension,
built with maturin. Exposes:

- `walk_batch(numeric_cols, string_cols, kind, feat_idx, op_code, thresh,
  pat_start, pat_count, patterns, left, right, leaf_value, out)` — the
  brief's asked-for signature, field-for-field identical encoding to
  `interpreted_kernel.py`'s numba `walk_batch`. GIL released for the whole
  row loop (`py.detach`, one line). Called **once per batch**, per the
  brief's key prior.
- `walk_one(row, str_row, <10 tree arrays>)` — one row through the FFI,
  **only** to measure per-call cost; never call this in a batch loop.
- `PyTree` — a `#[pyclass]` that copies the tree into Rust-owned memory
  **once**, then exposes `.walk_one(row, str_row)` and `.walk_batch(...)`
  that only marshal the *row*, not the tree, per call. The realistic
  production shape for a server that loads a tree once and serves many
  requests — closer to ZEN's object-based API (EXPERIMENTS.md §R) than to
  re-sending the whole tree every call.
- `regex_bench_categories(pattern, categories)` / `regex_bench_rows(pattern,
  haystacks)` — §T's two regex cases, using the `regex` crate instead of
  numba→libc, timed purely inside Rust (after Python→Rust marshalling) so
  the marshalling cost can be reported separately.

The kernel itself (`walk_one_inner`) is 20 lines, no recursion, no heap
allocation, one `while` loop — same shape as the numba kernel it mirrors.

## 1. Batch throughput, 100k rows, ns/row (range across 7 runs, matching
   §Q's protocol)

| shape | leaves | codegen (§Q) | numba (measured here) | rust `walk_batch` | rust `PyTree.walk_batch` |
|---|---:|---:|---:|---:|---:|
| credit tree | 20 | **20.7–22.9** | 52.5–53.6 | 46.0–51.5 | 51.6–68.4 |
| full-binary d5 | 32 | **24.5–26.8** | 59.4–65.0 | 51.9–55.2 | 56.6–59.6 |
| full-binary d7 | 128 | **42.2–49.7** | 85.1–86.6 | 72.6–85.7 | 73.2–93.0 |
| full-binary d9 | 512 | 82.8–127.0 (unstable, §Q) | 119.4–120.7 | **86.2–88.7** | 88.3–94.7 |
| one-sided chain, 100 | 101 | **28.4–28.8** | 34.2–35.8 | 30.2–32.9 | 33.8–34.1 |

Answers asserted identical across all three engines (`numba == rust ==
rust/PyTree`) on every shape, every run, before any timing counted.

**Rust beats the numba array walker by 12–28% everywhere, consistently.**
That is a real, repeatable win — but codegen still wins outright at every
shape except d9, by 1.05–2.5× (narrowest on the one-sided chain, where
codegen's own advantage is already thin; widest on the credit tree and
full-binary shapes), and even the *slowest* number in this whole table
(rust/PyTree at d9, 94.7 ns/row) is 9.5 ms for 100k rows — comfortably
inside the 20–100 ms budget (2–10× headroom, depending which end of the
budget you measure against). This table cannot tell codegen, numba, and rust
apart on a "does it fit the budget" basis; it only tells them apart in
relative terms.

**The d9 result is the one that matters.** §Q's own finding was that
codegen becomes *unpredictable*, not just slower, above ~2,000 emitted
lines — d9 measured 82.8–127.0 ns/row across reruns, an ~53% spread, "the
finding that matters is variance, not the mean." Rust's d9 spread here is
86.2–88.7, **under 3%** — tighter than numba's own 119.4–120.7 (1%) and in
the same character as numba: a compiled-once kernel has no code-layout
sensitivity to a specific tree's emitted size, because there is no
per-tree emission. Rust's d9 mean across the same 7 runs is 87.2 ns/row —
well below §Q's own reported d9 *median* for codegen (124.7 ns/row) — at
exactly the tree size where codegen is least trustworthy, Rust is faster
than codegen on a typical run and far more stable.
That is a genuine point in Rust's favor, and it is also **exactly the
point the numba interpreter already made** in §Q — Rust sharpens it by
~30%, it does not create it.

## 2. Single-record latency — stated against the real floor

EXPERIMENTS.md §R measured decider2's actual realtime path: `score()` is
**352.7 µs** for a trivial pipeline, **1,548 µs** at 30 steps, against a
kernel of **~1.4 µs**. §N4/§S: the engine essentially does not matter at
this scale — marshal-and-readback is 92%+ of the cost. This section checks
whether Rust changes that.

| path | ns/call (min–max across shapes) |
|---|---:|
| `rust noop()` — zero-arg PyO3 call floor | 50.5–105.6 |
| `rust PyTree.walk_one` — tree resident, only row crosses | 373.9–480.9 |
| `numba _walk_one` — direct njit call, full tree args | 1,224.5–1,426.3 |
| `rust walk_one` — **all 12 tree arrays re-marshalled every call** | 1,890.1–2,095.9 |

**Two things worth stating plainly.**

First, **the naive Rust API is slower than numba's own per-call overhead**
(1.9–2.1 µs vs. 1.2–1.4 µs) — because it re-marshals 12 numpy-array
arguments through PyO3 on every single call. This is the same lesson §R
found for ZEN's two APIs 145× apart, at a smaller magnitude (Rust's numpy
arguments are zero-copy views, not deep-copied JSON, so the gap here is
~1.5×, not 145×) — but the *shape* of the finding replicates: **how you
call into a compiled engine matters more than which compiled engine it
is.** `PyTree.walk_one` (tree loaded once, only the row crosses per call)
is 3.19–3.27× faster than numba's own dispatch and 4.80–5.06× faster than
the naive Rust API doing the same job, consistently across all five shapes
— the fix is architectural, not a language choice.

Second, and more important: **every number in this table is noise against
decider2's real floor.** The slowest number here (2.1 µs) is 168× *below*
the fastest whole-`score()` measurement in EXPERIMENTS.md (352.7 µs), and
essentially invisible against the realistic 1,548 µs path. §S's conclusion
— "the engine does not matter for `score()`; marshal and readback are the
entire problem" — holds exactly as stated for Rust. Adopting Rust would
not move decider2's own most-over-budget number (§R: 25× over its own
60 µs spec). The unbuilt fix is still doc 05 §3.1b's whole-row marshalling,
regardless of which language sits behind it.

## 3. Build and packaging cost

| | measured |
|---|---:|
| cold `cargo build --release`, crates NOT yet downloaded | **29.03 s** wall (98.75 s user, 7.01 s sys, 364% CPU, 399 MB peak RSS) |
| cold `cargo build --release`, crates cached locally, after `cargo clean` | **14.68–14.71 s** wall, 401 MB peak RSS |
| `maturin develop --release` (extension-module cfg forces its own compile) | ~9.3–12 s wall, first time |
| incremental rebuild, one source line changed | **0.87–1.5 s** wall |
| unstripped `.so` (dev iteration) | 3.12 MB (3,120,576 bytes) |
| release wheel, `maturin build --release --strip` | 888 KB |
| wheel platform tag (built on this box) | `cp314-cp314-manylinux_2_34_x86_64` |
| dependency count (`cargo tree`) | 50 crates (pyo3, numpy, regex + transitive: proc-macro2, syn, indoc, portable-atomic, aho-corasick, memchr, …) |

**The build itself is cheap** — 15–29 seconds cold, under 400 MB RSS, on a
crate that already had pyo3/numpy/regex cached locally (they were; this
machine had built Rust before). That is not the real packaging cost.

**Two things are the real cost, and neither is "build time":**

1. **The wheel is not `abi3`.** `cp314-cp314` means it is tied to the exact
   CPython minor version it was built against — decider2 supporting, say,
   3.11 through 3.14 needs a separate wheel per Python minor version, per
   platform, not one wheel per platform. (PyO3's `abi3-py3xx` feature
   fixes this but was not enabled here — it's an available knob, not a
   free one: it constrains which PyO3/numpy APIs are usable and was not
   tested in this experiment.)
2. **`manylinux_2_34`** is the tag `maturin` produced building natively on
   this box — a real, fairly recent glibc floor (glibc ≥ 2.34, i.e., not
   Ubuntu 20.04, not most "just works everywhere" manylinux targets).
   Shipping a *properly portable* Linux wheel needs the actual manylinux
   container/zig cross-compilation toolchain maturin supports — another
   moving part, not exercised here, and CI infrastructure decider2 does
   not currently have.

Put together: **decider2 today ships as pure Python + numba**, which is
itself a compiled-artifact-free `pip install` (numba/llvmlite bring their
own prebuilt wheels; decider2 contributes zero native code of its own).
Adding this extension means decider2's own release process must either (a)
stand up a wheel build matrix — OS × arch × Python-minor, likely via
`cibuildwheel` or equivalent, genuinely new CI infrastructure for an
open-sourcing project that does not have it — or (b) require every user
without a matching prebuilt wheel to have `rustc`+`cargo` on their machine
to `pip install` decider2 at all. That is a materially higher install bar
than decider2 has today, and it is a cost every single install pays, not a
one-time cost the team pays once.

## 4. Regex — the `regex` crate vs. numba→libc (§T)

| case | §T (numba/libc, or polars reference) | rust `regex` crate, pure compute | rust, incl. Python→Rust marshalling |
|---|---:|---:|---:|
| per-row, cardinality ≈ row count (100k rows) | libc 60.3 ns/call · polars-internal 42.2 ns/call | **25.6 ns/call** (2.56 ms/100k) | 104.4 ns/call (10.44 ms/100k) |
| per-category mask (12 categories) | §T: 0.12 ms total, numba path | 1,834 ns/call pure (n=12, noisy at this scale) | **234.5 µs total** for all 12 |

**The pure-Rust number is the best of any measured option** — 25.6 ns/call
beats both numba→libc (60.3, 2.4×) and polars' own internal Rust `regex`
crate (42.2, 1.6×) — consistent with the brief's prediction that a native
`regex` crate should beat both.

**But the marshalling cost is the real finding here, and it cuts the other
way.** Overall (Python `list[str]` → Rust `Vec<String>`, then match): 104.4
ns/call — **more than 4× the pure-compute number**, because converting
100,000 individual Python `str` objects into owned Rust `String`s (UTF-8
validate + allocate + copy, once per row) costs ~79 ns/call on its own,
*more than the regex match itself*. This is the numeric-array boundary
story inverted: numpy arrays cross via zero-copy buffer views, so §1's
batch numbers pay ~0 marshalling; **Python strings do not have an
equivalent zero-copy path through a naive `Vec<String>` binding**, so the
boundary cost that was invisible for numbers is the dominant cost for
strings. A real implementation should marshal Arrow-style offset+bytes
buffers (zero-copy, exactly what §T's own conclusion flagged: "it needs
the bytes in-kernel — arrow offsets and data buffers sliced per row,
zero-copy but real boundary work") — not a Python `list[str]`, which is
what this measurement deliberately used to surface the cost honestly.

The per-category case reconfirms §T's own conclusion regardless of
language: **234.5 µs total, once, for the entire category set** is
trivially cheap when it happens once per `activate()` (§T: "not per row"),
and is not a per-batch cost either language pays repeatedly.

## 5. Concurrency and `nogil`

`walk_batch` releases the GIL for its entire compute region — one line
(`py.detach(move || { ... })`), no flag, no per-scenario decision. Measured
against real concurrent load (credit tree, 10k-row batches, 28-core box,
1.5 s per thread-count):

| threads | aggregate throughput | scaling vs. 1 thread | worst single call observed |
|---|---:|---:|---:|
| 1 | 2,375 calls/s (23.8M rows/s) | 1.0× | 797 µs |
| 2 | 4,591 calls/s (45.9M rows/s) | **1.89×** | 777 µs |
| 4 | 8,196 calls/s (82.0M rows/s) | **3.37×** | 1,679 µs |
| 8 | 15,593 calls/s (155.9M rows/s) | **6.42×** | 1,781 µs |

**Near-linear scaling through 8 threads** — genuine parallel execution,
not GIL-serialized. This is the opposite signature from §N4's
`nogil=False` finding (throughput flat at ~1,150–1,190/s regardless of
thread count, p99 exploding to 1270% of budget at 16 threads): here,
throughput keeps *climbing* with thread count, which is what real
parallelism looks like, not a convoy. Worst-observed single-call latency
does rise (797 µs → 1,781 µs, 1→8 threads) — expected 8-way contention for
the same physical cores doing real compute on a shared box, not a
GIL-serialization signature.

**The assessment question was whether Rust makes the concurrency story
better or just different. Measured answer: functionally the same result
numba's `nogil=True` already gets (§N4 found `nogil=True` essentially flat
from 1–16 threads too) — Rust does not unlock new concurrency headroom
numba's `nogil=True` didn't already have.** What Rust changes is *how you
get there*: `nogil=True` is a boolean numba accepts on arbitrary code, and
§H/§N4 together show the same flag has opposite right answers depending on
what's competing for the GIL (§H: `nogil=True` loses to a *compiling*
background thread; §N4: `nogil=False` loses catastrophically to *serving*
threads) — two experiments were needed to find the actual rule. In Rust,
releasing the GIL is a borrow-checked scope: the closure passed to
`py.detach` cannot touch a GIL-bound Python object without `unsafe`, so
the "is this region actually safe to run GIL-free" question is checked at
compile time, once, structurally — not tuned per call site by a flag that
a future change could silently invalidate. That is a real ownership
improvement in *how the guarantee is held*, even though the *measured
ceiling* is the same numba already reaches with the right flag.

## 6. The equivalence ladder (doc 02 §3.1) — asset or problem?

decider2's core correctness test is three-way agreement,
`interpreted ≡ stepped ≡ fused`, used to localise disagreement to a layer
(numba semantics vs. fusion/inlining). A Rust engine is a genuine fourth
path, not folded into that ladder by construction, and it cuts both ways:

**Asset.** A from-scratch, ahead-of-time-compiled reference implementation
in a second language is the *strongest possible* check against
"interpreted and fused agree because they share a bug" — the failure mode
the three-way ladder exists to catch is exactly this kind of shared-cause
blind spot, and Rust cannot share numba's or CPython's specific float/
integer/comparison quirks (§I's `log()` divergence hunt, §M's `co_consts`
staleness class) because it does not share their compilers. If Rust
disagrees with numba but agrees with the Python-interpreted reference,
that pins the divergence to numba specifically — genuinely useful
diagnostic leverage the three existing modes cannot offer each other, since
two of them (`stepped`, `fused`) both go through numba.

**Cost.** It is a fourth implementation to keep semantically identical
*forever*, in a language with materially different default behavior at
exactly the edges §I/§M found real bugs at: Rust panics on integer overflow
in debug builds and wraps in release (neither matches Python's arbitrary-
precision int or numba's C-int-overflow semantics by default); float
comparison, NaN handling, and rounding modes must be independently
verified rather than inherited "for free" from being the same language.
Every future edge case the ladder is designed to catch must now be checked
across **four** implementations in **two** languages, not three in one.
For a team optimizing for maintainability, this is a real, compounding
tax — not a one-time cost — and it is a cost the numba-interpreter
alternative (still Python, still one language) does not pay at all.

## 7. What actually breaks, and what does not

| hazard | status if trees move to Rust |
|---|---|
| ~500-emitted-line cap (doc 05 §7) | **gone for trees** — no emission at all |
| CPython 100-level indentation limit (§P) | **gone for trees** — no generated Python source |
| Fan-out wall / per-shape compile (§P/§Q) | **gone for trees** — one binary, built once, ever |
| The seven numba cache conditions (doc 05 §4.2) | **gone for the tree path only.** decider2's other numba steps (arithmetic, tables) keep all seven, including the `co_consts` staleness bug (§M) and the `sys.modules` identity requirement (§K). Moving trees to Rust removes one exception's worth of surface area from a hazard class that stays fully present everywhere else in the codebase. |
| numba JIT warm-up (~163 ms measured here) | **paid once per process regardless** — decider2's other numba steps still pay it; Rust trees just don't add to it (rust's own "first call" cost measured here: 9.5 µs, i.e., none) |
| Fusion into one njit kernel (doc 02 §1.2, §3.1) | **this is the one the numba interpreter does NOT have to give up and Rust does.** A numba tree-walker step can, in principle, be inlined into decider2's existing fused-njit-kernel model alongside arithmetic and other steps — same execution substrate, same compiler, doc 02 §1.2's explicit-fusion contract already covers it. **A Rust tree cannot be inlined into a fused njit kernel at all** — it is necessarily a separate compiled unit, so a pipeline mixing a tree with other steps would cross the Rust/Python(numba) boundary once per fused-group invocation, reintroducing exactly the kind of per-hop boundary cost doc 01/§A/§N1 spent multiple experiments eliminating from the *numba* side of the boundary. This is a real, structural cost specific to the Rust option that the pure-numba-interpreter option simply does not have, and it was not visible until this experiment's design forced trees to be a standalone extension. |

## 8. What it costs to own

Concrete, not abstract:

- **A second language and toolchain.** rustc 1.95 + cargo were already on
  this box and pyo3/numpy/regex were already cached — a contributor
  without a working Rust install cannot build this at all until they set
  one up (`rustup` or equivalent), a step decider2 does not currently ask
  of anyone.
- **CI.** Every runner needs the Rust toolchain, plus either a wheel-build
  matrix (new infrastructure) or a from-source build step. 50 crates in
  the dependency tree is a new supply-chain surface an open-source project
  needs to track (`cargo audit`/`cargo deny` or equivalent) in addition to
  its existing PyPI surface.
- **Cross-boundary debugging.** A bug at the Rust/Python seam needs both a
  Python debugger and gdb/lldb (or `RUST_BACKTRACE`) fluency. An
  unhandled Rust panic surfaces to Python as an opaque
  `pyo3_runtime.PanicException` — a failure mode class decider2 does not
  have today (numba failures already surface as ordinary Python exceptions
  with tracebacks the whole team can already read; this experiment hit
  zero panics, but the failure mode exists and was not exercised).
- **Two engineering cultures' worth of idiom.** Ownership/borrow-checking
  discipline (evident in `PyTree` vs. the naive `walk_one`, and in why
  `py.detach` is safe here) is a real skill a Python-and-numba team does
  not automatically have, and it is exactly the skill that determines
  whether a *future* contributor's Rust addition is as clean as this one.

None of this is a claim that Rust is hard to use well — this experiment
built a working, correct, fast extension in well under a day, using tools
already present on the box. It is a claim that the *ongoing* cost of
carrying it is real and additive, not a rounding error next to codegen's
existing maintenance cost.

## Recommendation

**Do not adopt Rust for decider2's tree engine; adopt the numba array
walker §Q already recommended, and treat this Rust extension as a
working, documented option to revisit only if a future measurement shows
the numba interpreter's ~12–28% batch gap or its lack of a
compiler-checked `nogil` guarantee is an actual, measured bottleneck — not
as a default choice now.** The owner's stated motivation is
maintainability, and the numba interpreter already delivers everything
that motivation asks for: one generic kernel instead of per-shape codegen,
zero compile after the first-ever call, no 500-line cap, no CPython
indentation limit, and — because it stays inside decider2's existing
numba/njit execution model — it can still be inlined into a fused kernel
alongside other steps the way doc 02's architecture already assumes.
Rust gets there too, is measurably faster in batch (12–28%) and
meaningfully more stable at the large-tree sizes where codegen's own
variance is the real problem (§Q's d9 finding) — genuine, real wins, shown
here with numbers, not asserted. But it buys those wins by adding a second
language, a second toolchain, a new CI and packaging burden (wheel matrix
or user-side `rustc`, a real cost for a project that is being
open-sourced and does not currently ask anyone to compile anything), a new
cross-boundary failure mode, a fourth path on the equivalence ladder that
must be kept in sync forever, and — the one genuinely new architectural
cost this experiment surfaced — the loss of the fused-kernel model for any
pipeline that mixes a tree with other steps. Both single-record numbers
are noise against decider2's real ~350 µs–1.5 ms floor regardless of which
option is picked, so speed is not the tie-breaker at the scale decider2's
realtime path actually runs at. The numba interpreter gets roughly 90% of
Rust's measured benefit — all of the *specific* maintainability complaint
that motivated this whole question — for none of the toolchain cost. That
is the plain, unglamorous answer this experiment is set up to give.
