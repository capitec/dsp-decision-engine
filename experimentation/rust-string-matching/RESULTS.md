# Rust string matching at the node — lazy, over Arrow buffers, through the C ABI

**Question.** decider2 dictionary-encodes strings at the boundary, so a kernel
sees an int32 code and `UnaryStringMatch` is restricted to `match_type="exact"`;
prefix/suffix/substring/regex raise `UnsupportedInKernel` and the answer today is
"precompute in the frame tier" (`pl.col(...).str.contains(...)`, O(rows), always).
The owner wants the pattern evaluated **lazily at the node** — `ft1 > 5 AND ft2 >
10 AND ft3 ~ /^dog/` should evaluate the regex for the rows that reach it, which
under decider2's short-circuiting AND may be ~1% or none. Can a Rust `cdylib`,
called from inside the njit kernel via the C ABI over polars' string buffers, do
that — and does the kernel still disk-cache?

**Answer.** It works, it is correct, it is panic-safe, and **it disk-caches**
(pointer-as-argument, §W's distinction, verified cold→warm across two
processes). On the owner's own shape it beats the frame tier by 4–5× at ≤1%
selectivity on a high-cardinality column, and the crossover where the frame tier
wins again is ~65–75% selectivity. **But it does not win where it matters most,
and the win where it does win is bounded**: (a) at low cardinality — the normal
case for a business string feature — an O(distinct) per-category mask beats every
Rust strategy at every selectivity, and that mask needs no Rust at all (12 regex
calls in Python per batch); (b) the brief's two load-bearing premises did not
survive measurement: polars' `_get_buffers()` is **not** zero-copy on polars 1.41
(it materialises Utf8View into offsets/values at 6–15 ns/row per batch, which is
the lazy strategy's floor), and the regex crate costs **~40 ns/call, not 25.6**,
on 13–22-byte strings; (c) the maximum saving is one frame-tier pass per string
node per batch — ~35 ns/row, i.e. 35 ms per million rows — against a second
language in a bank's credit path. **Recommendation: do not adopt Rust. Extend
`UnaryStringMatch` past `exact` by resolving the pattern against the category
dictionary per batch (O(distinct), plain Python, cache-safe) for
Categorical/low-cardinality columns; keep the frame tier for high-cardinality
ones; keep this experiment as the documented working option if a profile ever
shows a selective regex node over a high-cardinality column.**

Everything is under `experimentation/rust-string-matching/`. `decider2/src/` was
read, never modified (`git status --short decider2/src` is empty; the tree is
still at 549 green tests). Reproduce with `bash run_all.sh` from anywhere
(builds the crate, runs the smoke test, cold/warm cache check in two processes,
both benchmarks, the panic demo; ~3 min).

## What was built

`rust_string_match/` — a plain `cdylib` (`regex` + `memchr`, no PyO3, no
Python ABI; `cargo build --release` 11 s cold), six `extern "C"` entry points:

| entry point | role |
|---|---|
| `pattern_compile(kind, ptr, len) -> id` | compile ONCE at bind time into a lock-free append-only registry; `kind` mirrors decider2's `TStringMatchType` (`exact`/`starts_with`/`contains`/`ends_with`/`regex`), so only `regex` touches the regex crate |
| `match_one(id, offsets, n, values, values_len, idx) -> 0/1/-1` | **the lazy shape** — one string, one call, from inside the kernel, only when the walk reaches the node |
| `match_rows(id, offsets, n, values, values_len, row_codes_or_null, n_out, out_mask) -> 0/-1` | **one function, both batch strategies**: hand it the COLUMN → per-row mask, O(rows); hand it the CATEGORY DICTIONARY → per-category mask, O(distinct), kernel does `mask[code[i]]`; hand it dictionary + codes → per-row via the dictionary |
| `match_one_unprotected`, `deliberately_panic_*`, `trivial_i32` | panic demo and call-overhead probe only |

Every pointer argument crosses as a raw `*const T` plus an explicit length and
becomes a bounds-checked Rust **slice** on the far side; every boundary is
`catch_unwind`-wrapped. Regex matching is `regex::bytes::Regex` — polars strings
are valid UTF-8 by construction, so the per-call `from_utf8` pass the
`rust-cabi-in-kernel` binding paid is skipped.

`binding.py` — `ctypes.CDLL` load, the Arrow-buffer grab (`StringBuffers`, with
`to_numpy(allow_copy=False)` asserting the numpy views are zero-copy), and the
`@intrinsic` (`inttoptr` + `call`, lifted from
`cfunc-pointer-interpreter/call_ptr.py`) that lets an `@njit(cache=True)` kernel
call `match_one` through an address read from a `uint64[:]` **argument**, never a
ctypes global.

`kernels.py` — the owner's shape as five `@njit(cache=True)` kernels sharing the
same short-circuiting `if ft1 > c1: if ft2 > c2: if <string test>` body and
differing only in the string test: `frame_tier` reads a precomputed bool,
`per_category` reads `mask[codes[i]]`, `lazy_rust` calls `match_one` on the
column's buffers at row `i`, `lazy_rust_via_dict` calls it on the dictionary's
buffers at `codes[i]`. `smoke_test.py` asserts all of them, all five match kinds,
and all three `match_rows` modes agree with Python `re` on both cardinalities;
`bench.py` re-asserts identical outputs at every cell it times.

## Measurement

1,000,000 rows, `ft1`/`ft2` uniform; `ft3` is an employer-name column at two
cardinalities: **low** = 12 distinct employers, 3 of which start with "dog"
(mean 12.9 bytes), **high** = one distinct string per row, 25% starting with
"dog" (mean 22 bytes). Selectivity = fraction of rows passing `ft1 > c1 AND ft2 >
c2`, i.e. reaching the string node. Each cardinality runs in its own process
(polars' global string cache otherwise leaks one run's dictionary into the
other's). i7-14700HX, polars 1.41.2 (28-thread pool, but a one-chunk `Series`
string op ran effectively single-threaded — its numbers match single-threaded
Rust on the same crate), numba 0.67, median of 7.

Every row is **end to end for one batch**: the strategy's own prep (polars
expression / `cast(Categorical)` + mask / buffer grab) plus its kernel. The last
two italic rows are the kernel alone. The pattern is run in both forms decider2's
`match_type` distinguishes: `regex` `^dog` and `starts_with` `dog`.

`extern "C"` call through the pointer-table argument: **2.6–3.1 ns/call**
(§V/§W's 2.4–7.2 ns, same mechanism). At 50% selectivity every kernel pays an
extra ~7 ns/row of branch misprediction on `ft1 > c1` (the no-string baseline
vectorises so it does not show it); it is data, not strategy.

#### regex `^dog` — cardinality **low** (12 distinct)

prep, ns/row: polars `str.contains` 43.7 · `cast(Categorical)` 37.5 · Rust mask over dict 0.010 · column buffers 15.2 · Rust `match_rows` over column 39.8

| strategy (end to end, ns/row) | 0% | 1% | 50% | 100% |
|---|---:|---:|---:|---:|
| 1. frame tier (polars expr + kernel) | 44.4 | 44.5 | 52.4 | 45.0 |
| 1b. Rust batch (buffers + match_rows column + kernel) | 55.8 | 55.9 | 63.7 | 56.4 |
| 2. per-category, Utf8 in (cast + mask + kernel) | 38.3 | 38.5 | 45.3 | 39.2 |
| **2b. per-category, Categorical in (mask + kernel)** | **0.8** | **1.0** | **7.9** | **1.7** |
| 3. lazy Rust over column (buffers + kernel) | 16.1 | 17.3 | 44.1 | 58.1 |
| 3b. lazy Rust over dict, Categorical in (kernel) | 0.9 | 1.6 | 29.4 | 42.7 |
| *kernel only: lazy Rust* | *0.8* | *2.1* | *28.9* | *42.9* |
| *kernel only: no string node* | *0.7* | *0.7* | *0.7* | *0.7* |

#### regex `^dog` — cardinality **high** (1,000,000 distinct)

prep, ns/row: polars `str.contains` 34.0 · `cast(Categorical)` 262.8 · Rust mask over dict 39.4 · column buffers 5.9 · Rust `match_rows` over column 36.3

| strategy (end to end, ns/row) | 0% | 1% | 50% | 100% |
|---|---:|---:|---:|---:|
| 1. frame tier (polars expr + kernel) | 34.6 | 34.9 | 41.5 | **35.4** |
| 1b. Rust batch (buffers + match_rows column + kernel) | 42.8 | 43.1 | 49.7 | 43.5 |
| 2. per-category, Utf8 in (cast + mask + kernel) | 303.0 | 303.3 | 310.0 | 303.9 |
| 2b. per-category, Categorical in (mask + kernel) | 40.1 | 40.4 | 47.1 | 41.1 |
| **3. lazy Rust over column (buffers + kernel)** | **6.7** | **7.8** | **33.0** | 44.8 |
| 3b. lazy Rust over dict, Categorical in (kernel) | 0.9 | 2.1 | 27.1 | 38.9 |
| *kernel only: lazy Rust* | *0.8* | *1.8* | *27.1* | *38.9* |
| *kernel only: no string node* | *0.6* | *0.8* | *0.5* | *0.7* |

#### starts_with `dog` — cardinality **low** (12 distinct)

prep, ns/row: polars `str.starts_with` 20.5 · `cast(Categorical)` 37.5 · Rust mask over dict 0.011 · column buffers 15.2 · Rust `match_rows` over column 13.8

| strategy (end to end, ns/row) | 0% | 1% | 50% | 100% |
|---|---:|---:|---:|---:|
| 1. frame tier (polars expr + kernel) | 21.3 | 21.5 | 29.6 | 21.9 |
| 1b. Rust batch (buffers + match_rows column + kernel) | 29.9 | 30.1 | 38.2 | 30.5 |
| 2. per-category, Utf8 in (cast + mask + kernel) | 38.4 | 38.6 | 45.9 | 39.2 |
| **2b. per-category, Categorical in (mask + kernel)** | **0.9** | **1.2** | **8.4** | **1.8** |
| 3. lazy Rust over column (buffers + kernel) | 16.2 | 17.1 | 30.4 | 30.9 |
| 3b. lazy Rust over dict, Categorical in (kernel) | 1.0 | 1.6 | 14.9 | 15.7 |
| *kernel only: lazy Rust* | *1.0* | *1.9* | *15.1* | *15.7* |
| *kernel only: no string node* | *1.0* | *0.9* | *0.6* | *0.5* |

#### starts_with `dog` — cardinality **high** (1,000,000 distinct)

prep, ns/row: polars `str.starts_with` 16.1 · `cast(Categorical)` 262.8 · Rust mask over dict 12.6 · column buffers 5.9 · Rust `match_rows` over column 12.4

| strategy (end to end, ns/row) | 0% | 1% | 50% | 100% |
|---|---:|---:|---:|---:|
| 1. frame tier (polars expr + kernel) | 16.8 | 17.0 | **23.6** | **18.1** |
| 1b. Rust batch (buffers + match_rows column + kernel) | 18.9 | 19.1 | 25.8 | 20.3 |
| 2. per-category, Utf8 in (cast + mask + kernel) | 276.2 | 276.4 | 284.4 | 277.9 |
| 2b. per-category, Categorical in (mask + kernel) | 13.4 | 13.6 | 21.5 | 15.0 |
| **3. lazy Rust over column (buffers + kernel)** | **6.9** | **7.5** | 23.3 | 23.6 |
| 3b. lazy Rust over dict, Categorical in (kernel) | 0.9 | 2.0 | 19.1 | 17.4 |
| *kernel only: lazy Rust* | *0.9* | *1.6* | *17.4* | *17.7* |
| *kernel only: no string node* | *0.5* | *0.7* | *0.8* | *0.6* |

### Where the crossovers actually are

Lazy Rust's cost is linear in selectivity: `buffers + s × per_call`, with
`per_call` ≈ 39–43 ns for the regex crate and ≈ 16–18 ns for `starts_with` (the
kernel-only row at 100%), and `buffers` = 6–15 ns/row of per-batch
materialisation (next section). The frame tier is flat. Solving:

| column | pattern form | lazy Rust beats the frame tier when rows reaching the node are below… | at 1%, lazy vs frame |
|---|---|---|---|
| high cardinality, Utf8 | regex `^dog` | **~75%** | 7.8 vs 34.9 (**4.5×**) |
| high cardinality, Utf8 | starts_with `dog` | ~65% | 7.5 vs 17.0 (2.3×) |
| low cardinality, Utf8 | regex `^dog` | ~65% (and ~55% vs strategy 2) | 17.3 vs 44.5 (2.6×) |
| low cardinality, Utf8 | starts_with `dog` | ~40% | 17.1 vs 21.5 (1.3×) |
| low cardinality, **Categorical** | either | **never** — per-category mask (2b) is ≤ 1.8 ns/row at every selectivity; lazy-over-dict (3b) only ties it at 0% | 1.6 vs **1.0** (mask wins) |

**Cardinality inverts the answer, as expected, but in a way that removes Rust
from the winning cell.** At low cardinality the O(distinct) mask is the answer
and it is not a Rust answer: 12 regex evaluations per batch cost 10 µs whether
they run in Rust or in Python's `re`, and the kernel then does one `mask[code]`
read. The only reason "2. per-category, Utf8 in" is not equally cheap is the
`cast(Categorical)` (37.5 ns/row) — which is the dictionary-encoding cost decider2
**already pays today** for any string column entering a kernel, so from decider2's
point of view it is a sunk cost and the mask is free on top of it. At high
cardinality the dictionary is the column, `cast(Categorical)` alone is 263
ns/row, and the per-category route is out; that is the one regime where lazy
Rust wins, by ~27 ns/row at ≤1% and shrinking to zero by ~75%.

### Two premises of the brief that did not survive measurement

**`_get_buffers()` is not zero-copy on polars 1.41.** polars stores `String`
as Utf8View (16-byte views, ≤12-byte strings inlined) and `_get_buffers()`
converts it into a fresh offsets/values pair on every call: the values pointer
differs between two consecutive calls, and the cost scales with rows —
**15.2 ns/row** for the 12.9-byte low-card strings (inlined views, per-row
copy) and **5.9 ns/row** for the 22-byte high-card strings (out-of-line, closer
to a memcpy). The numpy views of the returned buffers *are* zero-copy
(`allow_copy=False` succeeds) — that is what the earlier verification saw — but
the buffers themselves are a per-batch O(rows) conversion. This is the lazy
strategy's floor: at 0% selectivity, where the owner's argument is strongest,
**all 6.7–16.1 ns/row of "lazy Rust over column" is this conversion**; the
kernel itself is 0.8. Going at the Utf8View buffers directly would remove it,
but polars' Python API does not expose them (`_get_buffers` is the only buffer
accessor; pyarrow is not installed here) — a Rust matcher over the view layout is
possible but is a different, larger build and was not measured. Note the same
tax applies to strategy 1b, which is why "Rust batch" never beats polars'
own expression: same crate, plus a conversion.

**The regex crate's `is_match` is ~40 ns/call here, not 25.6.** `^dog` costs
39.8 ns/string in Rust `match_rows` and 43.7 in polars `str.contains` (same
crate) on 13-byte strings, 36.3 / 34.0 on 22-byte strings; `(?i)^dog`, `^d` and
`dog` (unanchored) are all within ±5 ns of it. The regex crate's per-call fixed
cost (cache pool checkout, `Input` construction, prefilter dispatch) dominates
on haystacks this short and does not depend much on the pattern. §V's 25.6 ns
was on a different corpus; it does not transfer to short business strings.
`starts_with` as a `memcmp` is 12–14 ns/string in Rust and 16–20 in polars — and
that is the honest cost of what `^dog` asks for, which is why the tables show
both forms. The per-row FFI shape adds only ~3 ns on top of either (kernel-only
at 100% vs `match_rows` over the column: 42.9 vs 39.8, 17.7 vs 12.4).

## Does the kernel still disk-cache? Yes.

`kernels.lazy_rust` is `@njit(cache=True)`; it calls Rust through
`ptr_table[PTR_MATCH_ONE]`, a `uint64[:]` argument. `cache_check.py` was run in
two separate processes against the persistent `__pycache__/` next to
`kernels.py` (cleared before the cold run), with `NUMBA_DEBUG_CACHE=1`:

```
$ rm -rf __pycache__; NUMBA_DEBUG_CACHE=1 python cache_check.py cold
[cache] index saved to '.../rust-string-matching/__pycache__/kernels.lazy_rust-72.py314.nbi'
[cache] data saved to '.../rust-string-matching/__pycache__/kernels.lazy_rust-72.py314.1.nbc'
[cold] import+build+first call: 757.4 ms; hits=1243; cache warnings=none

$ NUMBA_DEBUG_CACHE=1 python cache_check.py warm        # a NEW process
[cache] index loaded from '.../rust-string-matching/__pycache__/kernels.lazy_rust-72.py314.nbi'
[cache] data loaded from '.../rust-string-matching/__pycache__/kernels.lazy_rust-72.py314.1.nbc'
[warm] import+build+first call: 553.2 ms; hits=1243; cache warnings=none
$ grep -c saved cache_warm.log
0
```

Genuine `[cache] ... loaded`, zero `saved`, no `Cannot cache compiled function`
warning, identical answer. The `.so` is loaded at whatever address the warm
process gets and the kernel does not care, because the address is data. §W's
distinction holds for the Rust route exactly as it predicted: **a pointer passed
as an argument caches; a symbol captured as a global does not.** (§V's "can
never be disk-cached" applied to the ctypes-global binding, and would apply here
too if `binding._lib.match_one` were referenced inside the kernel — it is not.)
The elapsed delta is modest only because the harness time is dominated by
importing polars/numba; the log lines are the evidence.

## Panic safety — demonstrated

`panic_demo.py` runs each mode in its own subprocess (`panic_demo.log`). Every
call is made **from inside njit** through the pointer table, which is the real
calling context. Two realistic hostile inputs plus the synthetic pair:

```
=== oob_protected: returncode=0
  match_one(idx=10**9) from inside njit, n_strings=1000 ...
  returned -1
  subsequent good call in the SAME process: match_one(idx=0) -> 0 (string='capitec bank')
  stderr| thread '<unnamed>' panicked at src/lib.rs:93:21:
  stderr| index out of bounds: the len is 1001 but the index is 1000000000
=== oob_unprotected: returncode=-6 KILLED by SIGABRT
  stderr| thread caused non-unwinding panic. aborting.

=== short_protected: returncode=0          # values_len lies: 4 bytes, offsets say 12962..12977
  returned -1
  subsequent good call in the SAME process: match_one(idx=0) -> 0
  stderr| range start index 12962 out of range for slice of length 4
=== short_unprotected: returncode=-6 KILLED by SIGABRT

=== synthetic_protected: returncode=0      (returned -1, next call fine)
=== synthetic_unprotected: returncode=-6 KILLED by SIGABRT
```

Same finding as §V, with the mechanism made explicit: the protection is two
disciplines that must both hold at every entry point — **slice indexing, not raw
pointer arithmetic** (turns a bad index or a lying length into a Rust panic
instead of a silent out-of-bounds read, which no `catch_unwind` can help with)
and **`catch_unwind` at the boundary** (turns that panic into `-1` instead of
`SIGABRT` — rustc ≥ 1.71 aborts on a panic reaching an `extern "C"` frame).
Nothing enforces either; a forgotten wrapper is a dead process on a bad row.
The kernel side must also treat `-1` as "not matched, and something is wrong"
rather than as false — the kernels here compare `== 1`.

## What it costs to own

Against the frame tier, which is architecturally free, the Rust route adds:
a second language and toolchain (`cargo`, an 11 s build, a 2.7 MB `.so`); a
wheel matrix of one `.so` per platform/libc (§V: no Python-version axis, which
is the C ABI's real packaging win over PyO3, but the OS/glibc axis stays); an
`unsafe` boundary with the two-discipline panic contract above and a `-1`
sentinel the kernel must honour; a per-batch buffer materialisation that
polars' Python API gives no way around today; and the process-lifetime pattern
registry (patterns are never freed — fine for decider2's bind-once contract,
worth knowing). The cache risk that §V called "close to fatal" is **not** on
this list: it is solved by construction and verified above.

## Recommendation

**Don't adopt Rust for strings; adopt the per-category mask.** The owner's
concern — do not evaluate the pattern for every row when a selective branch
reaches only a few — is fully met at low cardinality by evaluating it once per
*category* per batch: 12 regex calls, ~10 µs, plain Python `re` against the
dictionary decider2 already extracts (`_extract_codes` returns the categories
today), and then the existing `isin`-shaped code comparison in the kernel. That
lifts `UnaryStringMatch` past `exact` for every Categorical/Enum/low-cardinality
column with no new dependency, no cache risk, no panic surface, and it beats
every Rust strategy measured here at every selectivity (≤ 1.8 ns/row). For
high-cardinality free-text columns the dictionary is the column and the frame
tier remains the right answer; lazy Rust does beat it there — 4.5× at 1%
selectivity, crossover ~75% — but the whole prize is one frame-tier pass per
string node per batch (~35 ns/row, 35 ms per million rows), a quarter of it
eaten by polars' own buffer conversion, and that is not enough to justify a
second language in a bank's credit path. Keep this directory as the documented,
working, cache-safe option in case a profile ever shows a selective regex over a
high-cardinality column; it is a day's work to wire in, and the evidence that
it caches is the part that was in doubt.
