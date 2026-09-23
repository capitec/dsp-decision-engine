# Reading polars strings through nanoarrow instead of hand-decoding them

*Written for someone with no prior context. The question: a working prototype
(`../arrow-strings-in-tree/`) lets a numba decision tree match a string
straight out of polars' memory, but it does so by hand-decoding Arrow's
Utf8View ("binview") layout — 16 bytes per row, an inline/reference split at
12 bytes, a `(buffer_index, offset)` pair — and the owner would rather "use
existing libraries" so that "if specs change we can always just pull the
latest". Also, "90% of the use cases might be single record batches", so
per-call cost at batch size 1 is what matters, not ns/row at a million.
Three approaches were built, checked for identical answers, and measured on
this box (28 cores, shared with other agents: timings carry a 5–20% band, two
passes are in `results.jsonl`, `./run_all.sh` reproduces everything).*

---

## The recommendation

**Keep the hand-decoded numba kernel (A) and slim its Python glue. Do not
adopt nanoarrow for this, and drop the "validate once per batch" idea (C)
entirely.** Revisit nanoarrow only if (a) the project decides it is willing to
ship a compiled C artefact anyway for some other reason, and (b) nanoarrow
gains bounds validation for string-view arrays — it has none today, in 0.9.0
or on `main`, and no upstream issue tracks it.

The single-record number that drives this: **at batch size 1, every approach
costs 9–45 µs per call, against a 352.7 µs incumbent single-record path
(EXPERIMENTS.md §R)** — 3–13% of it. Performance cannot pick a winner here;
the per-row code that distinguishes the approaches is *noise* at batch size 1
(2–3 µs of numba dispatch, and that is all of it). What differs is the import
handshake, and that is Python-glue cost any approach can lean out.

| approach, batch size 1 | per call, as written | per call, lean floor (same effort each) | lines of Arrow-layout code the project owns | corrupt input (5 constructed cases) | needs a compiled artefact |
|---|---|---|---|---|---|
| **A** hand-decoded numba (incumbent) | 42.6–45.1 µs | **21.0 µs** | 42 (Python) | error leaf on 4/5, silent on 1/5 | no |
| **B** nanoarrow called per row | 19.3–20.6 µs | **9.3 µs** | **0** (a 9-line C wrapper) | **SIGSEGV on 2/5**, silent wrong answer on 2/5, refused 1/5 | **yes** |
| **B-checked** B + hand-written bounds check in C | 19.3–20.5 µs | ~9.3 µs | 37 (C) | error leaf on 4/5, silent on 1/5 | **yes** |
| **C** nanoarrow `ValidateFull` per batch, then A's numba | 33.3–35.9 µs | 17.2 µs | 42 (Python) + shim | same as A — the validation catches nothing A did not | **yes** |

Read across the rows and the owner's two constraints pull against each other:

* **The only approach with zero hand-written layout code (B, unsafe) segfaults
  on a corrupt buffer index or offset.** nanoarrow's accessor is called
  `ArrowArrayViewGetStringUnsafe` and means it; and `ArrowArrayViewValidate`
  at every level, including `FULL`, does *no per-element check* on string-view
  arrays (§4, §5). Making B safe means writing the bounds checks yourself
  (B-checked): 37 lines of C that encode the same spec facts as A's 42 lines
  of Python, now in a language the project does not otherwise ship, plus a
  build step. That is not a maintainability win over A; it is A moved into C.
* **C's hypothesis is false.** "Validate up front so the per-row path is
  provably in bounds" — nanoarrow's `ValidateFull` costs a flat 1.4 µs from 1
  row to 1,000,000 rows (§3) *precisely because it validates nothing per row
  for this type*. A kernel that trusts it (`C-trusting`) crashes exactly where
  B does. C = A plus 1.5 µs plus a compiled artefact, for no safety.
* **"Pull the latest" is mechanically true but does not buy what the owner
  wants.** nanoarrow ships two releases a year, the amalgamation has to be
  regenerated with its `bundle.py` from the tarball, and the shim recompiled.
  All fine. But what is pulled is only as good as what nanoarrow checks, and
  for string view it checks buffer *counts*, not element bounds. Meanwhile the
  layout A encodes is *frozen*: Arrow format changes are additive; Utf8View
  has not changed since it was added (Arrow 15, Jan 2024) and cannot change
  without a new format string, which A refuses by name (`vu` only).
* **Where B genuinely wins — a producer switching to `u`/`U`/`vu` — B needs
  zero changes and A refuses loudly (§5).** polars cannot produce anything but
  `vu` for a String column today, and if it ever does, A needs ~15 lines for
  the offsets layout, not a rewrite.
* **The 90% case costs the same either way.** A-lean at 21 µs vs B-pooled at
  9 µs is 12 µs on a 353 µs path. The compiled artefact is the whole price of
  B, and it buys 3% of the single-record path.

So: **A, with its glue leaned out** (`arrowc.export` → one ctypes loop, no
`np.ctypeslib.as_array`, pattern table hoisted per tree; 45 → 21 µs, shown in
`bench_lean.py`). Its 42 lines of layout knowledge are listed line by line in
§4 so the ownership cost is exactly visible; they already refuse every
corruption we could construct except the one nobody catches (§6).

---

## 1. What was built

```
experimentation/nanoarrow-accessors/
  vendor/nanoarrow.c, vendor/nanoarrow/nanoarrow.h   nanoarrow 0.9.0 amalgamation (Apache-2.0), 9,183 lines,
                                                      generated with the release tarball's ci/scripts/bundle.py;
                                                      sha512 recorded in vendor/VERSION
  c/nashim.c                                           the shim: 17 functions, one ctypes call for the whole
                                                      polars->nanoarrow import, sm_get_string / sm_get_string_checked
  c/build.sh                                           gcc -O2 -shared  ->  c/libnashim.so (76 KB, 1.7 s)
  nashim.py                                            ctypes loader + the numba @intrinsic that calls sm_get_string
                                                      through a function pointer passed as an ARGUMENT (§W rule)
  kernel_b.py                                          approach B: STR node = one call into nanoarrow + byte match
  kernel_c.py                                          approach C: nanoarrow validate, then A's walk_chunk (imported
                                                      unchanged) or walk_chunk_trusting (A minus its bounds checks)
  test_all.py                                          40 tests: A, B, B-checked, C at 3 levels, C-trusting all equal a
                                                      Python reference on single/two-chunk, sliced, 1-row, 0-row,
                                                      all-null, 3000 random rows
  bench.py / bench_lean.py / ablate_import.py / validate_cost.py / corrupt.py / spec_change.py / cache_check.py /
  build_cost.sh / count_layout_lines.py               the probes; every number below is in results.jsonl
```

Approach A is imported from `../arrow-strings-in-tree/` (`kernel.py`,
`arrowc.py`) exactly as it is; nothing there was changed.

**The per-row access code, side by side** (this is what the owner is judging):

```python
# A — kernel.py, hand-decoded. Every line here is Arrow Utf8View spec knowledge.
ln, bi, off = _view(va, i)                 # u32 at +0, +8, +12 of the 16-byte view at va + 16*i
if ln <= 12:
    s_addr = va + np.uint64(16 * i + 4)    # short string: bytes are inline at +4
else:
    if bi < 0 or bi >= str_n_data[col]:                     ERR_LEAF; break
    if off < 0 or off + ln > str_data_size[slot]:           ERR_LEAF; break
    s_addr = str_data_addr[slot] + np.uint64(off)           # long string: data buffer bi, at offset off
```

```python
# B — kernel_b.py. No layout knowledge; nanoarrow decides what a row is.
s_addr, ln = call_get_string(get_string_addr, str_view_addr[col], i)   # int64 sm_get_string(view*, i, &data)
if ln == -1:  pc = else_[pc]; continue     # null
```
```c
/* B — c/nashim.c, the whole shim function. Wraps nanoarrow; zero spec facts. */
int64_t sm_get_string(const struct ArrowArrayView* view, int64_t i, const uint8_t** data) {
  if (ArrowArrayViewIsNull(view, i)) { *data = NULL; return -1; }
  struct ArrowStringView sv = ArrowArrayViewGetStringUnsafe(view, i);   /* "Unsafe": no bounds checks */
  *data = (const uint8_t*)sv.data;  return sv.size_bytes;
}
```

```python
# C — kernel_c.py: nanoarrow validates the batch, then A's code above runs unchanged.
nano_view.validate(FULL)                   # ArrowArrayViewValidate(view, NANOARROW_VALIDATION_LEVEL_FULL)
walk_chunk(...)                            # A's kernel, imported
```

Both B and C also need the import side: `NanoView(series)` calls
`sm_import_single`, which pulls the schema and the chunk from polars'
`ArrowArrayStream`, runs `ArrowArrayViewInitFromSchema` +
`ArrowArrayViewSetArrayMinimal`, and hands back a `struct ArrowArrayView*`
that the kernel receives as a `uint64` argument.

---

## 2. Cost per call, by batch size

All approaches give identical answers (40 tests, `test_all.py`) before any of
this was timed. Blocks of k calls, median and min over 21–25 blocks, fresh
one-row Series cycled from a pool of 1,000 so polars' own frame construction
is outside the timed region. Strings are realistic merchant descriptors
(mostly > 12 bytes, so the reference path, not the inline path; 5% null).

### 2.1 Batch size 1 — the 90% case — broken into parts

| stage (µs per call, median of two passes) | A | B | C-full |
|---|---|---|---|
| polars `Series.__arrow_c_stream__()` (build the capsule) | 0.5 | 0.5 | 0.5 |
| get the Arrow data out: A `arrowc.export` (Python ctypes loop) / B,C `NanoView` (one C call, but 5 ctypes objects allocated per call) | **26.2** | **11.5–12.0** | 11.5–12.0 |
| nanoarrow validation (`FULL`; `MINIMAL`/`DEFAULT` are the same) | — | — | 1.5 |
| build kernel argument tables (`string_tables`, 7 numpy arrays) | 4.3–4.5 | — | 4.3–4.5 |
| pattern table (`pattern_table`, should be hoisted per tree) | 2.8–3.0 | (hoisted) | (hoisted) |
| kernel call: numba dispatch + the row itself | 3.3 (dispatch floor 3.3, 21 args) | 2.2 (dispatch floor 2.2, 16 args) | 3.3 |
| **end to end, as written** | **42.6–45.1** | **19.3–20.6** | **33.3–35.9** |
| **end to end, lean floor** (`bench_lean.py`: pooled structs, hoisted tables, sizes read via ctypes) | **21.0** | **9.3** | **17.2** |
| fraction of the 352.7 µs incumbent single-record path (as written / lean) | 12–13% / 6% | 5.5–6% / 2.6% | 9–10% / 4.9% |
| fraction of the 60 µs single-record spec (doc 05 §3.1b) | 71–75% / 35% | 32–34% / 16% | 56–60% / 29% |

**What dominates at batch size 1 is the import handshake, in every approach,
and none of it is the per-row work.** The match itself is invisible: the
kernel call equals its own dispatch floor (n=0) to the timer's resolution.

Where A's 26 µs export goes (`ablate_import.py`): `__arrow_c_stream__` 0.5 →
`PyCapsule_GetPointer` 1.3 → the ctypes loop (get_schema, two get_next,
buffers, releases) 14.3 → `np.ctypeslib.as_array` over the sizes buffer
**22.4** (reading the same 8 bytes with `(c_int64*n).from_address` is 15.8) →
dataclasses and capsule bookkeeping 26.2. So 10 µs of A's export is one numpy
call and some object construction, not the C Data Interface. B's `NanoView`
is 11.7 µs of which 6.2 is the actual stream + C import + releases and 5.5 is
allocating three ctypes structs and two buffers per call (1.7 µs) plus Python
attribute plumbing; pooled, 6.2 µs.

The B-checked accessor costs nothing measurable over B (19.3 vs 19.3 µs).

### 2.2 The crossover

| rows | A as written | B | B-checked | C-full | A lean | B pooled |
|---|---|---|---|---|---|---|
| 1 | 42.6–45.1 µs | 19.3–20.6 | 19.3–20.5 | 33.3–35.9 | 21.0 | 9.3 |
| 10 | 50.3 | 20.8 | 21.8 | 41.5 | | |
| 100 | 55.6 | 31.4 | 27.8 | 47.7 | | |
| 1,000 | 114 | 88.9 | 89.3 | 102 | 81.6 | 74.4 |
| 1,000,000 | 60.1–66.0 ms | 64.1–81.1 ms | 63.7–68.4 ms | 60.1–60.6 ms | | |

Kernel-only at 1M rows: A 59.9–63.5 ms, B 63.8–70.5 ms — **nanoarrow's
per-row call costs 4–7 ns/row more than the inlined decode** (a real function
call through a pointer per row, vs. three loads). B's fixed saving (~25 µs at
n=1 as written, ~12 µs lean) is therefore repaid at roughly **5,000–6,000
rows**; above that A is ahead, by 6–10% at a million. For the stated
workload (90% single record) the crossover is irrelevant in both directions:
the difference is a few percent of a 353 µs call either way.

---

## 3. nanoarrow validation cost vs. array length — and why it is flat

The task asked whether `ValidateFull` is O(rows), which would sink C at batch
size 1. It is not — it is O(1) — but for the wrong reason.

| rows | `MINIMAL` | `DEFAULT` | `FULL` | ctypes call floor |
|---|---|---|---|---|
| 1 | 1.43 µs | 1.43 | 1.44 | 0.71 |
| 100 | 1.43 | 1.43 | 1.44 | 0.71 |
| 10,000 | 1.41 | 1.41 | 1.43 | 0.71 |
| 1,000,000 | 1.62 | 1.58 | 1.58 | 0.85 |

Read `vendor/nanoarrow.c` for `ArrowArrayViewValidateFull` (line 4099): it
walks offsets buffers for `u`/`U`, type ids for unions, run ends, list views —
and has **no case for `STRING_VIEW`/`BINARY_VIEW`**. `ValidateMinimal` checks
the validity and views buffers are long enough for `length` (and refuses a
null validity buffer with `null_count > 0`); the variadic data buffers and the
per-element `(buffer_index, offset, length)` are `continue`d over. Confirmed
on `main` too (`src/nanoarrow/common/array.c`), and no issue tracks it. So
"validate once, then the per-row path is provably in bounds" does not hold
for the one type polars produces. The corruption matrix (§6) is the proof:
`C-trusting` passes `ValidateFull` and then segfaults.

---

## 4. Lines of Arrow-layout knowledge the project would own

Counted by `count_layout_lines.py`, which prints every counted line so the
count can be audited. A line counts if it encodes a fact from the Arrow
columnar spec; ctypes struct definitions of the C Data Interface (which every
approach needs and which are the *published ABI*, not the layout), the
byte-load intrinsics, and the tree walk do not count.

| | lines | where |
|---|---|---|
| **A** hand-decoded numba | **42** | `kernel.py`: `_view` 9, `_is_valid` 7, STR-node inline/reference split + bounds 13, slice handling in `string_tables` 3; `arrowc.py`: buffer roles of a `vu` array 10 |
| **B** nanoarrow per row (unsafe) | **0** | `c/nashim.c` `sm_get_string`: 9 lines, all wrapper |
| **B-checked** | **37** | `c/nashim.c` between `BEGIN/END layout knowledge`: the inline-≤12 rule, `buffer_index` vs `n_variadic_buffers`, `offset + n` vs `variadic_buffer_sizes[]`, and the offsets convention for `u` and `U` |
| **C** | **42** + shim | A's lines, unchanged, plus the import/validate wrappers |
| vendored nanoarrow (a dependency — not owned, but shipped and built) | 9,183 | `vendor/` |

Plainly: **the only zero-line option is the one that crashes on corrupt
input.** The moment a bounds check is wanted, it has to be written, and it
comes out the same size in C as in Python — because nanoarrow exposes the
decoded fields but does not check them.

---

## 5. What happens when the spec, or polars, changes

Built by hand with ctypes (no pyarrow installed) exactly as another Arrow
producer would hand them over; each case in its own subprocess
(`spec_change.py`).

| producer sends | A | B | B-checked | C-full |
|---|---|---|---|---|
| `u` utf8, int32 offsets | refused: `expected 'vu', got 'u'` | **correct, no code change** | correct (after `DEFAULT` validation resolves the data-buffer size) | refused (A's check) |
| `U` large_utf8, int64 offsets | refused | **correct, no code change** | correct | refused |
| `n` — a real polars column, `pl.Series([None])` infers dtype Null | refused, badly: `IndexError` inside `arrowc.export` | refused, clearly: `Expected array with 0 buffer(s) but found 1` | same | same |
| `q` — a format nanoarrow 0.9.0 does not know | refused | refused: `Unknown format: 'q'` | refused | refused |
| `vz` — schema says binary_view over utf8 buffers (a lie, not a spec change) | refused (format check) | **SIGSEGV** | error leaf on every row | refused (A's check) |

What A would need for `u`/`U`: the offsets-buffer decode is ~10 lines (read
`offsets[i]`, `offsets[i+1]`, bounds against the data buffer) plus a format
dispatch, and `arrowc.py` already records those buffers. It is not nothing,
but it is bounded and the formats are enumerable.

**Is "if specs change we can always just pull the latest" true here?**

* *Mechanically, yes.* nanoarrow is versioned semver-style (0.6.0 Oct 2024,
  0.7.0 Jun 2025, 0.8.0 Feb 2026, 0.9.0 Jul 2026 — about two releases a
  year), `NANOARROW_VERSION` is in the header, the C API has been stable
  across those. The release tarball does **not** ship a prebuilt
  amalgamation: it is produced by `ci/scripts/bundle.py` from the tarball
  (`vendor/VERSION` records the 5-step procedure and the sha512). Then the
  shim recompiles; struct sizes are read at runtime (`sm_sizeof_*`) so the
  Python side does not depend on nanoarrow's struct layout.
* *For the thing the owner cares about, no.* The Arrow columnar format is
  versioned additively; a layout, once published, does not change (Utf8View
  is unchanged since Arrow 15). "Spec changes" therefore means *new formats*,
  and the risk with A is a producer sending one A does not know — which A
  refuses by name. nanoarrow handles the new format only once nanoarrow
  supports it (string view arrived in nanoarrow 0.6, eight months after Arrow
  15) *and* the accessor is safe for it, which for string view it is not.
* *The nanoarrow PyPI wheel does not remove the compile step.* Its shared
  objects export the namespaced non-inline functions
  (`PythonPkgArrowArrayViewValidate`, `...SetArrayMinimal`,
  `...InitFromSchema`), so approach C's validation could be ctypes-loaded
  from the wheel with no C at all — but `ArrowArrayViewGetStringUnsafe` is
  `static inline` in the header and exists in no `.so`; B cannot avoid
  compiling.

---

## 6. Corruption behaviour — with evidence

Each (case, approach) is its own subprocess (`corrupt.py`, `corrupt_child.py`);
the corruption is applied by writing into the real polars-exported buffers or
`ArrowArray` struct, then (for B/C) re-importing through nanoarrow so its
validation gets its chance. Row 1 is a 37-byte string (reference path).
`CLEAN = [1,1,0,0,1,1]` for `contains "dog"`.

| case | A | B (nanoarrow, unsafe) | B-checked | C-full (validate + A) | C-trusting (validate, no checks) |
|---|---|---|---|---|---|
| clean | correct | correct | correct | correct | correct |
| sliced, Arrow `offset` = 3 (legit, zero-copy `Series.slice`) | correct | correct | correct | correct | correct |
| `buffer_index` = 1000 (of 1) | **error leaf** `[1,-1,0,0,1,1]` | **crash, SIGSEGV** (rc −11) | error leaf | error leaf | **crash, SIGSEGV** |
| `offset` = 2³¹−1 | error leaf | **crash, SIGSEGV** | error leaf | error leaf | **crash, SIGSEGV** |
| `length` = 2³⁰, pattern present early | error leaf | answered by luck (match found before the overrun) | error leaf | error leaf | answered by luck |
| `length` = 2³⁰, pattern absent (must scan) | error leaf | **wrong answer** `[0,1,0,0,0,0]` — walked past the buffer and found `zzz` somewhere in process memory | error leaf | error leaf | **wrong answer** |
| validity buffer NULL, `null_count` = 3 | answered (ignores `null_count`; the data has no nulls) | **refused at import**: `Expected string_view array buffer 0 to have size >= 1 bytes` | refused | refused | refused |
| `n_buffers` = 2 (data + sizes buffers dropped) | error leaf | **wrong answer** — nanoarrow computes `n_variadic = −1`, `SetArrayMinimal` passes, the accessor reads the buffer that still happens to be there | error leaf (`buffer_index >= −1`) | refused (by accident: numpy rejects shape −1 in the Python glue) | refused (same accident) |
| sizes buffer lies (claims 1 TB), `offset` 64 MB past the real buffer | **wrong answer** `[1,0,0,0,1,1]` | wrong answer | wrong answer | wrong answer | wrong answer |

Exit codes are in `results.jsonl` (`probe: "corrupt"`). Findings:

1. **nanoarrow's import-time validation catches exactly one of the five real
   corruptions** (the inconsistent null count), and its `n_buffers`
   arithmetic accepts an impossible negative variadic count — worth an
   upstream report, not a reason to trust it.
2. **A refuses everything it can know about.** It ignores `null_count` (the
   validity pointer is the source of truth, which is defensible) and it, like
   everyone, trusts the producer's sizes buffer. Nobody can catch a lying
   sizes buffer without knowing the allocation, which the C Data Interface
   does not expose.
3. **B-checked matches A case for case** — with 37 lines of hand-written C
   doing what A's 42 lines of Python do.

---

## 7. Disk caching (B and C)

`cache_check.py`: three processes share a fresh `NUMBA_CACHE_DIR` with
`NUMBA_DEBUG_CACHE=1`; numba's own lines are counted.

| run | data saved | data loaded | `sm_get_string` address in that process |
|---|---|---|---|
| cold | 3 | 0 | `0x7fec3c0d69e0` |
| warm | **0** | 3 | `0x7f19bd2d69e0` |
| warm2 | **0** | 3 | `0x7f11468d69e0` |

The function pointer and the `ArrowArrayView*` are kernel *arguments*
(`uint64`), never captured globals, so the cached object contains no
process-specific address; `libnashim.so` lands somewhere different in every
process and the warm runs still load and save nothing. The three cached
functions are `walk_chunk_b`, `walk_chunk_trusting` and A's `walk_chunk`.

---

## 8. Build cost (B and C)

| | measured |
|---|---|
| cold build, `gcc -O2 -fPIC -shared` (shim + `nanoarrow.c`) | **1.63–1.66 s** (nanoarrow.c 1.46 s of it; the shim alone 0.12 s) |
| object | **76,296 bytes** (68,080 stripped); 17 exported `sm_*` symbols |
| dependencies | none beyond libc (nanoarrow is C99 with no dependencies) |
| toolchain here | gcc 16.1.1, Fedora 44 |

What Windows / macOS would need: nanoarrow itself is fine — the 0.9.0 PyPI
release has 58 wheels covering win32, win_amd64, macOS x86_64 and arm64,
manylinux, musllinux and wasm32, so the C compiles with MSVC and clang. The
cost is entirely on the project's side: today it ships pure Python + numba
and no compiled artefact. Adding one means a build backend (setuptools
`Extension` / meson-python / scikit-build-core), a CI matrix producing wheels
per OS × Python version (cibuildwheel), and a fallback story for a platform
without a wheel (compile on install → needs a compiler on the target). That
is a permanent operational surface, and it is the actual price of B or C —
1.7 seconds and 76 KB are not.

---

## 9. Reproduce

```bash
cd experimentation/nanoarrow-accessors
./run_all.sh          # ~6 min: builds the shim, 40 tests, all probes, second timing pass; writes results.jsonl
```

Requirements: gcc, the repo venv (polars 1.41.2, numba 0.67.0, numpy 2.4.6,
llvmlite 0.49.0; no pyarrow), ≥ 8 GB free. The vendored nanoarrow is 0.9.0,
checksum and regeneration procedure in `vendor/VERSION`.

Individual probes: `bench.py [sizes...]` (staged per-call cost),
`bench_lean.py N` (equal-effort floors), `ablate_import.py` (where the
batch-1 import goes), `validate_cost.py`, `corrupt.py`, `spec_change.py`,
`cache_check.py`, `build_cost.sh`, `count_layout_lines.py`.
