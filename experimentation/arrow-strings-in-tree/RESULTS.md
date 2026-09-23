# Strings straight from polars into the tree, through Arrow

*Written for someone with no prior context. The question was: can a decision
tree read a string column directly out of polars' memory and match it at the
node — no dictionary encoding, no preprocessing pass — even if that costs a
little more, because it is a simpler pattern than turning every string into a
number? Everything below was measured on this box (28 cores, shared with
other agents, so timings carry a ±20–30% noise band; every number is in
`results.jsonl` and `./run_all.sh` reproduces it).*

---

## The recommendation

**Yes. Do it.** polars 1.41 hands its String buffers over through the Arrow C
Data Interface with no copy at all, a pure-numba tree node can match a row's
bytes in place for `exact` / `prefix` / `suffix` / `contains`, it disk-caches,
and end to end it is *cheaper* than today's encode-then-compare — not "slightly
more expensive". The dictionary-encoding step at the boundary can go for
String columns.

**The simplicity verdict, plainly:**

| | today | with the STR node |
|---|---|---|
| a string column reaches the kernel as | an int32 code, made by `cast(pl.Categorical)` at the boundary, plus a category list the code indexes into | the bytes polars already holds |
| a preprocessing pass over the column | yes, every batch (the cast) | none — one O(1) handshake per column |
| `match_type` other than exact | refused; author must precompute a bool column in polars | `exact`, `starts_with`, `ends_with`, `contains` at the node |
| a pattern in a rule is | resolved to a code against *this batch's* category list at param-resolution time | bytes in a pattern table, a kernel argument like any threshold |
| what a maintainer has to understand | codes, category lists, code stability across frames, the float64 slot the code rides in, and the frame-tier workaround | 16 bytes per row: a length, and either the bytes inline or (buffer, offset) |
| nulls | NaN in the float slot | validity bit read per row; a null never matches |
| still float-encoded? | yes | no — a string stays a string |

**What still has to stay (the honest part, §6):** `regex` is not covered (no
regex engine in nopython numba; the sibling strands' verdict on C/Rust stands).
`Categorical`/`Enum` columns arrive from upstream *already* as codes and would
keep the existing `CodesPlan` — that is a second representation, but it is
polars' distinction, not one the boundary invents, and no cast happens in
either case. `isin` over a large set becomes k byte-compares, fine for a
handful of values, not for thousands.

---

## 1. Is polars → Arrow zero-copy for a String column? Yes — and the evidence

polars 1.41.2 stores a String column as Arrow **Utf8View** ("binview"): one
16-byte *view* per row — a `u32` length, then either the bytes themselves
(≤ 12 bytes) or a 4-byte prefix plus `(buffer_index, offset)` into one of
several variable-size data buffers. Four routes were tested for handing that
memory over without copying. pyarrow is **not** installed and was not
installed; nothing here depends on it.

| route | copies? | same address on two calls | 1e5 rows | 1e6 rows | 4e6 rows |
|---|---|---|---|---|---|
| **`Series.__arrow_c_stream__()`** (Arrow PyCapsule interface, read with ctypes, `arrowc.py`) | **no** | **yes** | 19–29 µs | 16–30 µs | 66–72 µs |
| **`Series._export_arrow_to_c(ptr, ptr)`** (private, single chunk) | **no** | **yes** — and the same address as the stream route | 4 µs | 4 µs | 7–17 µs |
| `Series._get_buffers()` — what the sibling strands and the typed-features branch use | **yes** | no | 1.7 ms | 18–28 ms | 75–95 ms |
| `Series._get_buffer_info()` | — | — | refuses: ``not implemented for non-physical type str`` | | |
| `Series.to_arrow()` | untested | | needs pyarrow | | |

The two zero-copy routes are **O(1)** in rows (their small growth is the count
of variadic data buffers, 8 → 13); `_get_buffers()` is **17–28 ns/row**,
O(rows), confirming the overnight finding that it materialises a
`large_utf8` offsets+values pair every call. Asking the stream for
`large_utf8` via `requested_schema` is ignored — polars returns `vu` anyway.

Stronger witnesses that these are polars' *own* buffers and not a per-Series
cached conversion: seven access paths to the same column — `df["s"]`,
`df.get_column`, a lazy `select` collected, `select(...alias)`, `df.clone()`,
`Series.clone()`, and `with_columns` of an unrelated column — all export the
identical `views` and data addresses; `str.to_uppercase()` (a real
transformation) gives new memory. And the export hands over an owned
reference: dropping the DataFrame and thrashing the heap, the kernel still
reads the right bytes (`probe=ownership`). The layout was decoded by hand and
checked against the strings (`"exactly12chr"` sits inline; 13+ bytes go
through a buffer index).

So the earlier statement "polars string buffers are zero-copy" was *right
about the memory and wrong about the accessor*. `_get_buffers()` copies
because it converts to a different layout; the C Data Interface does not,
because binview is what polars already has.

## 2. Matching a string at the node — what was built

`kernel.py`: a `walk_tree`-shaped loop with node kinds `LEAF` / `CMP` /
`IS_TRUE` / **`STR`**. A `STR` node names a string column (`feat_idx`), a
pattern (`thr_slot` into a flat byte table) and a mode (`op` =
EXACT/PREFIX/SUFFIX/CONTAINS). Per row it: reads the validity bit (null →
`else_`); reads the 16-byte view; takes the inline bytes or bounds-checks
`(buffer_index, offset, length)` against the buffer table and takes the
address; compares bytes. A malformed view writes `ERR_LEAF` (−1) and stops
that row — it never reads outside a buffer.

Pure numba, no C, no Rust. The only "unsafe" surface is two eight-line
`@intrinsic`s (`inttoptr` + `load`) that read one byte / one `u32` at an
address. Every address is a **uint64 argument** (arrays of them, one entry
per string column per chunk), never a captured global — §W's rule.

It reads the binview layout in-kernel directly; the materialised
`(offsets, values)` form is not used anywhere.

**Correctness** (`test_kernel.py`, 38 tests, all green): the four modes ×
eight patterns against a plain-Python reference over inline strings, 100+
byte strings, empty strings, nulls, multi-byte UTF-8 (`"dög"`, `"ö"` as a
pattern — byte-wise prefix/suffix/contains are exact on valid UTF-8), an
all-null column, a zero-row column, and:

- **chunked frames** — `pl.concat` of two and three frames. The stream
  exports one `ArrowArray` per polars chunk and the kernel runs chunk by
  chunk with a row cursor; a 5+5 concat matching `"cat"` gives
  `[0]*5 + [1]*5`, so it is provably not reading only chunk 0. On 1M rows the
  3-chunk answer is byte-identical to the single-chunk one.
- **sliced frames** — `df[a:b]` is zero-copy in polars and exports a chunk
  with a non-zero Arrow `offset`; the views address moves by `16·offset` and
  the validity *bit* offset travels separately (it cannot be re-based to a
  byte). Tested at four offsets against a column with nulls.
- **columns of one frame with different chunk layouts** — polars does not
  promise alignment: `concat` (2 chunks) then `with_columns(pl.Series(...))`
  (1 chunk) is a real case. Walking per chunk would silently pair row *i* of
  one column with row *j* of another. `run_tree` compares the layouts and
  raises, naming them; `df.rechunk()` (a copy) fixes it and the test proves
  the answer is then right.

`_get_buffers()` on a chunked series, for the record, does not read only the
first chunk either — it rechunks, i.e. copies everything.

## 3. What it costs — 1M rows, same data for every route

Two independent runs, ns per row, **boundary + kernel = total**. "Boundary"
is whatever must happen to the column before the kernel can run. Low
cardinality: 12 employer names (about half longer than 12 bytes); high: a
distinct string per row (~25 bytes); ~1% nulls in both.

**Low cardinality (13 distinct)**

| route | mode | boundary | kernel | **total** |
|---|---|---|---|---|
| today: `cast(Categorical)` + `== code` | exact | 145–162 | 23–33 | **178–185** |
| today: polars `str.starts_with` → bool → `is_true` | prefix | 58–61 | 19–21 | **78–82** |
| " `str.ends_with` | suffix | 48–62 | 18–25 | **66–87** |
| " `str.contains` | contains | 96–132 | 23 | **120–155** |
| per-category mask (SUMMARY §4), encode + mask | prefix/suffix/contains | 135–164 | 14–23 | **150–187** |
| **arrow / STR node** | exact | **0.1** | 57–65 | **57–65** |
| **arrow / STR node** | prefix | 0.1 | 58–75 | **58–75** |
| **arrow / STR node** | suffix | 0.1 | 56–64 | **56–64** |
| **arrow / STR node** | contains | 0.1 | 98–110 | **98–110** |

**High cardinality (989,691 distinct)**

| route | mode | boundary | kernel | **total** |
|---|---|---|---|---|
| today: `cast(Categorical)` + `== code` | exact | 776–1092 | 19–23 | **799–1111** |
| today: polars str op → `is_true` | prefix | 33–36 | 21 | **53–57** |
| " | suffix | 26–32 | 16–18 | **42–50** |
| " | contains | 73–86 | 19–21 | **91–107** |
| **arrow / STR node** | exact | 0.1 | 38–41 | **38–41** |
| **arrow / STR node** | prefix | 0.1 | 51–54 | **51–55** |
| **arrow / STR node** | suffix | 0.1 | 39–44 | **39–44** |
| **arrow / STR node** | contains | 0.1 | 57–92 | **57–93** |

**Gated — `x > 0.99` at the root, ~1% of rows reach the string node**

| | low, prefix | low, contains | high, prefix | high, contains |
|---|---|---|---|---|
| today (polars pass over every row, then gated `is_true`) | 86–95 | 186–198 | 62–65 | 84–112 |
| **arrow / STR node** (only the 1% are matched) | **17–31** | **27–31** | **22–33** | **28–29** |

Reading it:

- **The STR node beats today's exact path end to end** — 57–65 vs 178–185
  at low cardinality, 38–41 vs 800–1100 at high — because the encode *is*
  the cost. The kernel-only comparison is the honest per-node price: a STR
  node costs **25–40 ns more than a `CMP` node** (57–65 vs 23–33). That is
  the "slightly higher cost" the owner said they would pay, and it turns out
  they don't have to, because the boundary cost disappears.
- Against **polars' vectorised string ops** (today's non-exact workaround)
  on an ungated full-column match, the STR node is level: within the noise
  band for prefix/suffix, somewhat better for contains. polars' Rust is
  faster per byte; the STR node has no pass at all. They cancel.
- **Gated, the STR node is 3–6× ahead**, because a tree short-circuits and a
  frame-tier pass cannot. This is the case the owner predicted in the earlier
  strands, and it holds with no C and no Rust.
- **The per-category mask is the worst route here** at 150–187 ns/row. The
  overnight summary's 1.07 ns/row for it was *conditional on the column
  arriving already encoded*; when the encode is paid per batch it loses to
  everything. That condition is exactly what this experiment removes.

**What each route asks a person to understand and maintain**

*Today, exact:* the boundary casts every String column to Categorical each
batch; the kernel sees an int32 code sitting in a float64 slot; a rule's
pattern is resolved against *this batch's* category list at param time, so
the same rule compiles to a different number on every batch; a code
comparison on a categorical is only meaningful for `==`/`!=` and nothing
stops `<`. Nulls are NaN. An author wanting `starts_with` gets an error
telling them to go and precompute a column in polars.

*Today, non-exact:* the author writes the string logic *outside* the tree in
polars, produces a bool column, and the tree branches on `is_true`. Two
places to read to understand one rule; the pattern lives in pipeline code,
not in the rule document; the pass runs over every row whether or not the
tree would have asked.

*STR node:* the boundary does one O(1) export per string column per batch;
the kernel gets addresses in arrays; the rule holds the pattern as bytes;
nulls are a validity bit. A maintainer needs to know the 16-byte view
layout (six lines of docstring), that chunks are walked in order, and that
a slice carries an offset. Nothing is a number that used to be a string.

## 4. Does it disk-cache? Yes

Two processes, fresh `NUMBA_CACHE_DIR`, `NUMBA_DEBUG_CACHE=1`, counting
numba's own lines (`cache_cold.log`, `cache_warm.log`):

| | data saved | data loaded | answer |
|---|---|---|---|
| cold | 1 (`kernel.walk_chunk-125.py314.1.nbc`) | 0 | `[1, 0, 0, 0, 1]` |
| warm | **0** | 1 | `[1, 0, 0, 0, 1]` |

One entry because the helpers are `inline="always"` and fold into
`walk_chunk`. The buffer addresses are arguments; the intrinsics are plain
LLVM with no dynamic globals; nothing captured.

## 5. A detour worth recording: 150 ns/row that was not the string matching

The first three versions of the kernel cost **100–250 ns/row** for every
mode, exact included. A flat micro-kernel doing the identical binview read
and byte compare cost **11–13 ns/row** (`ablate.py`), so the matching was
not the problem. `ablate2.py` found it: a flat loop whose only difference
was calling the `inline="always"` helpers *with array arguments* cost 170
ns/row and had 8 `NRT_incref` / 23 `NRT_decref` calls in its IR. **numba's
IR-level inlining keeps a refcount pair for every array argument at every
call**, and the pattern table and buffer tables have real meminfos, so those
are atomic operations, per row, per helper. Rewriting every helper to take
only integers (addresses, lengths) and indexing the tables in the walker's
own body gave zero increfs in the row loop and the numbers in §3.

This is the same wall the Numba+C strand hit at 365 ns/row and the
typed-features strand hit at 2× — a third independent encounter. The rule
is sharper than "do not put a non-inlined layer between the kernel and the
walker": **do not pass an array into any per-row helper, inlined or not;
pass scalars.** It is documented at the top of `kernel.py`'s helper block.

## 6. Can the dictionary encoding actually go? Mostly — here is exactly what stays

| need | does the int32 code path have to stay for it? |
|---|---|
| `== "dog"` speed | **No.** STR exact at 38–65 ns/row total beats encode + `== code` at 178–1100. If a column *arrives* Categorical from upstream, `== code` is 20–30 ns against STR's 40–60 — a real but small gap, and no cast is involved either way. |
| `isin` over a small set | No — k exact compares of a few bytes each. |
| `isin` over thousands of values | Not built. Would need a hash or a sorted table over bytes; codes are the easy answer today. Say so before promising it. |
| `starts_with` / `ends_with` / `contains` | **No** — this is the case codes could never do. |
| `regex` | Still not in the kernel. Nothing here changes the earlier verdict (PCRE2 backtracks; Rust is a second language). Keep the frame-tier workaround for regex only. |
| `case_sensitive=False`, `trim_whitespace` | Not built. ASCII case-folding and whitespace trimming are a few lines on the address/length pair; Unicode case-folding is not. |
| output / results | Unaffected — the kernel returns leaf indices; strings are never written by it. |
| **`Categorical` / `Enum` input columns** | **Yes, `CodesPlan` stays for those.** They are codes in polars' own memory (`COPY` tier, no cast). Their Arrow export is a dictionary array (codes + a small string dictionary); a STR node could match the *dictionary* once per batch and look up per row — which is the per-category mask, applied only where the input is already categorical. Not built here. |

So: **the preprocessing pass goes** — no String column is cast to
Categorical at the boundary any more, no pattern is turned into a batch-
specific integer, and prefix/suffix/contains stop being refused. What
remains is polars' own String-vs-Categorical distinction, which the boundary
mirrors rather than invents. That is two `ColumnPlan` variants for two
polars dtypes, not two representations of one dtype — a materially better
answer than "both have to exist".

## 7. How this fits the typed-features branch

`TYPED_FEATURES.md` §5 reserves a raw-string slot whose row array holds
`(start, end)` read off polars' `offsets` buffer via `_get_buffers()`. That
design is right about the slot and wrong about the buffer shape: `_get_buffers()`
is the 17–28 ns/row copy. The change is small and makes the slot cheaper:

- `RawStringPlan.extract()` returns, per chunk, `(views_addr, validity_addr,
  arrow_offset, data_addrs[], data_sizes[])` from `arrowc.export`, and keeps
  the `StringView` alive for the batch (it owns an Arc reference; `release()`
  when done).
- `_fill_spans` becomes unnecessary: the STR node reads the view for row `i`
  itself. The per-row string state is one `uint64` address per string
  column, which is exactly what keeps the refcount trap of §5 out of the
  walker.
- `walk_tree` gains the `STR` kind from `kernel.py` (the kind switch is
  already there); `encode_string_match` stops raising for
  `starts_with`/`ends_with`/`contains` and writes the pattern into a byte
  table instead of resolving it to a code.

## 8. Limits, stated

- **Assumes format `"vu"`.** `run_tree` refuses any other format by name
  rather than misreading it. If a future polars exports
  `large_utf8`, the `(offsets, values)` reader is the fallback — and it is
  then zero-copy too, because that would be polars' own layout.
- **Chunk alignment across columns is checked, not solved.** A misaligned
  frame raises and asks for `df.rechunk()`. A cursor-based walker that
  advances each column's chunk independently would remove the copy; not
  needed to answer the question.
- **Sliced frames carry an Arrow `offset`**; handled and tested, but it is
  the kind of thing that must not be forgotten when this is ported.
- **Timing noise is ±20–30%** on this shared box; the ranges in §3 are two
  runs, best-of-5 each. Kernel-only numbers are the stable ones.
- **No pyarrow was installed.** `to_arrow()` is therefore untested; it is a
  pyarrow wrapper over the same export and cannot be *more* zero-copy than
  the raw interface measured here.

## Files and reproduce

```
cd experimentation/arrow-strings-in-tree && ./run_all.sh     # ~10 min, writes results.jsonl
```

| file | what |
|---|---|
| `arrowc.py` | ctypes consumer of `__arrow_c_stream__`; per-chunk buffer addresses; no pyarrow |
| `kernel.py` | the walker with the `STR` node, the two intrinsics, table builders, `run_tree` |
| `test_kernel.py` | 38 correctness tests (modes, nulls, empty, UTF-8, chunked, sliced, misaligned) |
| `probe_zero_copy.py` | Q1: addresses and O(1) scaling at 1e5 / 1e6 / 4e6 rows |
| `bench.py` | Q3: the three routes, both cardinalities, gated and chunked |
| `ablate.py`, `ablate2.py` | §5: where the first kernel's 150 ns went |
| `cache_check.py` | Q4: two processes, numba's own saved/loaded lines |
| `kernel_v3_arrays_in_helpers.py` | the slow version, kept so §5 is reproducible |
| `results.jsonl` | every measurement, in order, with `probe` tags (`bench_v1..v4_*` are the superseded kernel versions) |
